//! Legalis-backed policy engine.
//!
//! [`LegalisPolicyEngine`] evaluates a set of [`legalis_core::Statute`] rules
//! for a given jurisdiction string and translates the matching effects into
//! [`LegalContext`] fields.
//!
//! Effect-to-field mapping:
//! - [`EffectType::Prohibition`] → `blocked_regions` (adds jurisdiction)
//! - [`EffectType::Grant`] with description containing "transfer" → `data_transfer_allowed = true`
//! - [`EffectType::Obligation`] with description containing "audit" → `audit_required = true`
//! - [`EffectType::Obligation`] with description containing "cert" → `required_certifications`
//!
//! When no statutes match, the engine falls back to
//! [`LegalContext::from_legalis_jurisdiction`], which provides sensible defaults
//! for well-known jurisdiction strings (EU/GDPR, US-CA/CCPA, BR/LGPD).

#[cfg(feature = "legal")]
use alloc::vec::Vec;

#[cfg(feature = "legal")]
use legalis_core::{Effect, EffectType, Statute};

#[cfg(feature = "legal")]
use super::{LegalContext, sensor::PolicyEngine};

/// Policy engine backed by `legalis_core::Statute` rules.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "legal")]
/// # {
/// use oxirouter::context::LegalisPolicyEngine;
/// use oxirouter::context::PolicyEngine;
///
/// let engine = LegalisPolicyEngine::gdpr();
/// let ctx = engine.evaluate_for_jurisdiction("EU");
/// assert!(ctx.map(|c| c.gdpr_region).unwrap_or(false));
/// # }
/// ```
#[cfg(feature = "legal")]
pub struct LegalisPolicyEngine {
    /// The statute set this engine evaluates.
    statutes: Vec<Statute>,
}

#[cfg(feature = "legal")]
impl LegalisPolicyEngine {
    /// Create from an explicit list of statutes.
    #[must_use]
    pub fn new(statutes: Vec<Statute>) -> Self {
        Self { statutes }
    }

    /// Pre-built GDPR statute set for EU/EEA jurisdictions.
    #[must_use]
    pub fn gdpr() -> Self {
        Self::new(Self::gdpr_statutes())
    }

    /// Pre-built CCPA statute set for California.
    #[must_use]
    pub fn ccpa() -> Self {
        Self::new(Self::ccpa_statutes())
    }

    /// Pre-built LGPD statute set for Brazil.
    #[must_use]
    pub fn lgpd() -> Self {
        Self::new(Self::lgpd_statutes())
    }

    /// Minimal GDPR statute set.
    ///
    /// Covers: prohibition on personal data transfer outside EU/EEA without an
    /// adequacy decision (GDPR Art. 44), audit obligation (Art. 5(2)), and the
    /// data-subject access grant (Art. 15).
    fn gdpr_statutes() -> Vec<Statute> {
        alloc::vec![
            Statute::new(
                "gdpr-art44-transfer",
                "GDPR Art. 44 – Personal Data Transfer Restriction",
                Effect::prohibition(
                    "personal data transfer outside EU/EEA without adequacy decision"
                ),
            )
            .with_jurisdiction("EU"),
            Statute::new(
                "gdpr-art5-accountability",
                "GDPR Art. 5(2) – Accountability Obligation",
                Effect::obligation("audit and demonstrate compliance"),
            )
            .with_jurisdiction("EU"),
            Statute::new(
                "gdpr-art15-access",
                "GDPR Art. 15 – Right of Access",
                Effect::grant("data subject access to personal data"),
            )
            .with_jurisdiction("EU"),
        ]
    }

    /// Minimal CCPA statute set.
    fn ccpa_statutes() -> Vec<Statute> {
        alloc::vec![
            Statute::new(
                "ccpa-1798-100",
                "CCPA § 1798.100 – Consumer Right to Know",
                Effect::grant("consumer right to know about personal information collected"),
            )
            .with_jurisdiction("US-CA"),
            Statute::new(
                "ccpa-1798-120",
                "CCPA § 1798.120 – Right to Opt-Out",
                Effect::grant("consumer right to opt-out of sale of personal information"),
            )
            .with_jurisdiction("US-CA"),
            Statute::new(
                "ccpa-audit",
                "CCPA – Privacy Notice Obligation",
                Effect::obligation("audit privacy practices and provide notice at collection"),
            )
            .with_jurisdiction("US-CA"),
        ]
    }

    /// Minimal LGPD statute set.
    fn lgpd_statutes() -> Vec<Statute> {
        alloc::vec![
            Statute::new(
                "lgpd-art46",
                "LGPD Art. 46 – Data Security Obligation",
                Effect::obligation(
                    "audit and implement technical and administrative security measures"
                ),
            )
            .with_jurisdiction("BR"),
            Statute::new(
                "lgpd-art33",
                "LGPD Art. 33 – International Transfer",
                Effect::prohibition("international personal data transfer without legal basis"),
            )
            .with_jurisdiction("BR"),
        ]
    }

    /// Translate a matched statute's effect into the appropriate `LegalContext` field mutations.
    fn apply_effect(ctx: &mut LegalContext, effect: &Effect, statute_jurisdiction: Option<&str>) {
        match effect.effect_type {
            EffectType::Prohibition => {
                // Prohibition on transfers → add the jurisdiction as a blocked region.
                if let Some(jur) = statute_jurisdiction {
                    // The prohibition applies *from* this jurisdiction; other regions
                    // that haven't been explicitly allowed are effectively blocked.
                    // We record the statute jurisdiction in blocked_regions so that
                    // callers can detect that cross-border transfer is prohibited.
                    ctx.blocked_regions.insert(jur.to_string());
                }
                ctx.data_transfer_allowed = false;
            }
            EffectType::Grant => {
                // A Grant that mentions "transfer" re-enables data transfer.
                if effect.description.to_lowercase().contains("transfer") {
                    ctx.data_transfer_allowed = true;
                }
                // A Grant that mentions "access" sets user_consent (opt-in model).
                if effect.description.to_lowercase().contains("access") {
                    ctx.user_consent = true;
                }
            }
            EffectType::Obligation => {
                if effect.description.to_lowercase().contains("audit") {
                    ctx.audit_required = true;
                }
                if effect.description.to_lowercase().contains("cert") {
                    // Extract certification name from effect parameter "cert_name" if present.
                    let cert_name = effect
                        .get_parameter("cert_name")
                        .cloned()
                        .unwrap_or_else(|| "REQUIRED".to_string());
                    ctx.required_certifications.insert(cert_name);
                }
            }
            // Revoke, MonetaryTransfer, StatusChange, Custom — no mapping defined.
            _ => {}
        }
    }
}

#[cfg(feature = "legal")]
impl PolicyEngine for LegalisPolicyEngine {
    /// Evaluate compliance for the given jurisdiction string.
    ///
    /// 1. Start from [`LegalContext::from_legalis_jurisdiction`] to get correct
    ///    `gdpr_region` / `ccpa_applies` / `lgpd_applies` flags.
    /// 2. Walk statutes whose `jurisdiction` matches the input (case-insensitive).
    /// 3. Apply each matched statute's effect to mutate the context.
    ///
    /// Always returns `Some(_)` — unknown jurisdictions return the permissive default.
    fn evaluate_for_jurisdiction(&self, jurisdiction: &str) -> Option<LegalContext> {
        // Populate regulatory flags from the well-known jurisdiction mapping.
        let mut ctx = LegalContext::from_legalis_jurisdiction(jurisdiction);

        let jur_upper = jurisdiction.to_uppercase();

        // Walk statutes that declare this jurisdiction (or have no jurisdiction
        // restriction, which means they apply universally).
        for statute in &self.statutes {
            let matches = statute
                .jurisdiction
                .as_deref()
                .map(|j| j.to_uppercase() == jur_upper)
                .unwrap_or(true); // no jurisdiction → applies everywhere

            if matches {
                Self::apply_effect(&mut ctx, &statute.effect, statute.jurisdiction.as_deref());
            }
        }

        Some(ctx)
    }
}
