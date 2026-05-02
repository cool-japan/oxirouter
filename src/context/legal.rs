//! Legal/compliance context from legalis (Social brain)

use hashbrown::HashSet;
use serde::{Deserialize, Serialize};

/// Legal/compliance context for regulation-aware routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LegalContext {
    /// Whether data transfer is allowed (for cross-border queries)
    pub data_transfer_allowed: bool,

    /// Whether in GDPR-regulated region
    pub gdpr_region: bool,

    /// Whether CCPA applies
    pub ccpa_applies: bool,

    /// Whether LGPD (Brazil) applies
    pub lgpd_applies: bool,

    /// Required data residency regions
    pub required_regions: Option<HashSet<String>>,

    /// Allowed data residency regions
    pub allowed_regions: Option<HashSet<String>>,

    /// Blocked regions (cannot query from these)
    pub blocked_regions: HashSet<String>,

    /// Required compliance certifications
    pub required_certifications: HashSet<String>,

    /// Data classification level
    pub data_classification: DataClassification,

    /// Purpose limitation flags
    pub purpose_flags: HashSet<String>,

    /// Whether user has consented to data processing
    pub user_consent: bool,

    /// Audit logging required
    pub audit_required: bool,

    /// Data retention policy in days (0 = no retention)
    pub retention_days: u32,
}

impl LegalContext {
    /// Create a new legal context with default (permissive) settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_transfer_allowed: true,
            gdpr_region: false,
            ccpa_applies: false,
            lgpd_applies: false,
            required_regions: None,
            allowed_regions: None,
            blocked_regions: HashSet::new(),
            required_certifications: HashSet::new(),
            data_classification: DataClassification::Public,
            purpose_flags: HashSet::new(),
            user_consent: true,
            audit_required: false,
            retention_days: 0,
        }
    }

    /// Create context for GDPR region
    #[must_use]
    pub fn gdpr() -> Self {
        let mut allowed = HashSet::new();
        // EU/EEA countries + adequacy decision countries
        for code in [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "IS",
            "LI", "NO", "CH", "GB", "JP", "KR", "CA", "NZ", "AR", "IL",
        ] {
            allowed.insert(code.to_string());
        }

        Self {
            data_transfer_allowed: true,
            gdpr_region: true,
            ccpa_applies: false,
            lgpd_applies: false,
            required_regions: None,
            allowed_regions: Some(allowed),
            blocked_regions: HashSet::new(),
            required_certifications: HashSet::new(),
            data_classification: DataClassification::Internal,
            purpose_flags: HashSet::new(),
            user_consent: false, // Requires explicit consent
            audit_required: true,
            retention_days: 0,
        }
    }

    /// Create context for CCPA (California)
    #[must_use]
    pub fn ccpa() -> Self {
        Self {
            data_transfer_allowed: true,
            gdpr_region: false,
            ccpa_applies: true,
            lgpd_applies: false,
            required_regions: None,
            allowed_regions: None,
            blocked_regions: HashSet::new(),
            required_certifications: HashSet::new(),
            data_classification: DataClassification::Internal,
            purpose_flags: HashSet::new(),
            user_consent: true, // Opt-out model
            audit_required: true,
            retention_days: 0,
        }
    }

    /// Block a specific region
    pub fn block_region(&mut self, region: impl Into<String>) {
        self.blocked_regions.insert(region.into());
    }

    /// Require a specific region for data residency
    pub fn require_region(&mut self, region: impl Into<String>) {
        let region = region.into();
        if self.required_regions.is_none() {
            self.required_regions = Some(HashSet::new());
        }
        if let Some(ref mut regions) = self.required_regions {
            regions.insert(region);
        }
    }

    /// Check if a region is allowed
    #[must_use]
    pub fn is_region_allowed(&self, region: &str) -> bool {
        // Check blocked list first
        if self.blocked_regions.contains(region) {
            return false;
        }

        // Check required regions (must be in one of these)
        if let Some(ref required) = self.required_regions {
            if !required.contains(region) {
                return false;
            }
        }

        // Check allowed regions (if specified, must be in list)
        if let Some(ref allowed) = self.allowed_regions {
            if !allowed.contains(region) {
                return false;
            }
        }

        true
    }

    /// Check if data transfer to a region is allowed
    #[must_use]
    pub fn can_transfer_to(&self, region: &str) -> bool {
        if !self.data_transfer_allowed {
            return false;
        }
        self.is_region_allowed(region)
    }

    /// Get the most restrictive regulation that applies
    #[must_use]
    pub fn dominant_regulation(&self) -> Regulation {
        if self.gdpr_region {
            Regulation::Gdpr
        } else if self.lgpd_applies {
            Regulation::Lgpd
        } else if self.ccpa_applies {
            Regulation::Ccpa
        } else {
            Regulation::None
        }
    }

    /// Check if processing is allowed for a given purpose
    #[must_use]
    pub fn is_purpose_allowed(&self, purpose: &str) -> bool {
        // If no purpose flags are set, all purposes are allowed
        if self.purpose_flags.is_empty() {
            return true;
        }
        self.purpose_flags.contains(purpose)
    }

    /// Add an allowed purpose
    pub fn allow_purpose(&mut self, purpose: impl Into<String>) {
        self.purpose_flags.insert(purpose.into());
    }

    /// Get compliance score (0.0 = blocked, 1.0 = fully compliant)
    #[must_use]
    pub fn compliance_score(&self) -> f32 {
        let mut score = 1.0;

        // No consent = major penalty
        if !self.user_consent {
            score *= 0.1;
        }

        // Missing certifications = penalty
        if !self.required_certifications.is_empty() {
            score *= 0.5; // Would need to check actual certs
        }

        // High classification = stricter requirements
        match self.data_classification {
            DataClassification::Public => {}
            DataClassification::Internal => score *= 0.9,
            DataClassification::Confidential => score *= 0.7,
            DataClassification::Restricted => score *= 0.3,
        }

        score
    }
}

/// Data classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DataClassification {
    /// Publicly available data
    #[default]
    Public,
    /// Internal use only
    Internal,
    /// Confidential/sensitive data
    Confidential,
    /// Highly restricted data
    Restricted,
}

/// Applicable regulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regulation {
    /// No specific regulation
    None,
    /// EU GDPR
    Gdpr,
    /// California CCPA
    Ccpa,
    /// Brazil LGPD
    Lgpd,
    /// China PIPL
    Pipl,
    /// Japan APPI
    Appi,
}

/// Integration with legalis-core
#[cfg(feature = "legal")]
impl LegalContext {
    /// Create from legalis jurisdiction
    pub fn from_legalis_jurisdiction(jurisdiction: &str) -> Self {
        match jurisdiction.to_uppercase().as_str() {
            "EU" | "EEA" => Self::gdpr(),
            "US-CA" | "CALIFORNIA" => Self::ccpa(),
            "BR" | "BRAZIL" => {
                let mut ctx = Self::new();
                ctx.lgpd_applies = true;
                ctx.audit_required = true;
                ctx
            }
            _ => Self::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legal_context() {
        let ctx = LegalContext::gdpr();
        assert!(ctx.gdpr_region);
        assert!(ctx.audit_required);
        assert!(ctx.is_region_allowed("DE"));
        assert!(ctx.is_region_allowed("JP")); // Adequacy decision
    }

    #[test]
    fn test_blocked_regions() {
        let mut ctx = LegalContext::new();
        ctx.block_region("XX");

        assert!(ctx.is_region_allowed("US"));
        assert!(!ctx.is_region_allowed("XX"));
    }

    #[test]
    fn test_required_regions() {
        let mut ctx = LegalContext::new();
        ctx.require_region("EU");
        ctx.require_region("US");

        assert!(ctx.is_region_allowed("EU"));
        assert!(ctx.is_region_allowed("US"));
        assert!(!ctx.is_region_allowed("JP"));
    }

    #[test]
    fn test_data_transfer() {
        let gdpr = LegalContext::gdpr();
        assert!(gdpr.can_transfer_to("DE"));
        assert!(gdpr.can_transfer_to("JP"));
        assert!(!gdpr.can_transfer_to("XX"));

        let mut blocked = LegalContext::new();
        blocked.data_transfer_allowed = false;
        assert!(!blocked.can_transfer_to("US"));
    }

    #[test]
    fn test_dominant_regulation() {
        assert_eq!(LegalContext::gdpr().dominant_regulation(), Regulation::Gdpr);
        assert_eq!(LegalContext::ccpa().dominant_regulation(), Regulation::Ccpa);
        assert_eq!(LegalContext::new().dominant_regulation(), Regulation::None);
    }

    #[test]
    fn test_compliance_score() {
        let full_consent = LegalContext::new();
        assert!(full_consent.compliance_score() > 0.9);

        let mut no_consent = LegalContext::new();
        no_consent.user_consent = false;
        assert!(no_consent.compliance_score() < 0.2);
    }
}
