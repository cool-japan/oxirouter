//! Tests for the Legalis-backed policy engine.

#[cfg(feature = "legal")]
mod policy_engine_tests {
    use oxirouter::{LegalisPolicyEngine, context::sensor::PolicyEngine};

    // ----- GDPR tests -----

    #[test]
    fn gdpr_engine_eu_jurisdiction_sets_gdpr_region() {
        let engine = LegalisPolicyEngine::gdpr();
        let ctx = engine
            .evaluate_for_jurisdiction("EU")
            .expect("GDPR engine must return Some for EU");
        assert!(
            ctx.gdpr_region,
            "gdpr_region should be true for EU jurisdiction"
        );
    }

    #[test]
    fn gdpr_engine_audit_required_for_eu() {
        let engine = LegalisPolicyEngine::gdpr();
        let ctx = engine
            .evaluate_for_jurisdiction("EU")
            .expect("must return Some");
        assert!(
            ctx.audit_required,
            "GDPR mandates audit logging (Art. 5(2))"
        );
    }

    #[test]
    fn gdpr_engine_data_transfer_restricted_for_eu() {
        let engine = LegalisPolicyEngine::gdpr();
        let ctx = engine
            .evaluate_for_jurisdiction("EU")
            .expect("must return Some");
        // Art. 44 prohibition sets data_transfer_allowed = false.
        assert!(
            !ctx.data_transfer_allowed,
            "GDPR restricts data transfer outside EU without adequacy decision"
        );
    }

    #[test]
    fn gdpr_engine_eea_jurisdiction_sets_gdpr_region() {
        let engine = LegalisPolicyEngine::gdpr();
        let ctx = engine
            .evaluate_for_jurisdiction("EEA")
            .expect("must return Some");
        assert!(ctx.gdpr_region, "EEA is treated as EU/GDPR jurisdiction");
    }

    // ----- CCPA tests -----

    #[test]
    fn ccpa_engine_us_ca_sets_ccpa_applies() {
        let engine = LegalisPolicyEngine::ccpa();
        let ctx = engine
            .evaluate_for_jurisdiction("US-CA")
            .expect("CCPA engine must return Some for US-CA");
        assert!(ctx.ccpa_applies, "ccpa_applies should be true for US-CA");
    }

    #[test]
    fn ccpa_engine_audit_required_for_california() {
        let engine = LegalisPolicyEngine::ccpa();
        let ctx = engine
            .evaluate_for_jurisdiction("US-CA")
            .expect("must return Some");
        assert!(ctx.audit_required, "CCPA mandates privacy practice audit");
    }

    // ----- LGPD tests -----

    #[test]
    fn lgpd_engine_br_sets_lgpd_applies() {
        let engine = LegalisPolicyEngine::lgpd();
        let ctx = engine
            .evaluate_for_jurisdiction("BR")
            .expect("LGPD engine must return Some for BR");
        assert!(ctx.lgpd_applies, "lgpd_applies should be true for BR");
    }

    #[test]
    fn lgpd_engine_audit_required_for_brazil() {
        let engine = LegalisPolicyEngine::lgpd();
        let ctx = engine
            .evaluate_for_jurisdiction("BR")
            .expect("must return Some");
        assert!(ctx.audit_required, "LGPD Art. 46 mandates security audit");
    }

    // ----- Unknown jurisdiction tests -----

    #[test]
    fn unknown_jurisdiction_returns_sensible_default() {
        let engine = LegalisPolicyEngine::gdpr();
        // Should not panic — returns permissive default for unknown jurisdictions.
        let ctx = engine.evaluate_for_jurisdiction("ZZZZ");
        assert!(
            ctx.is_some(),
            "engine must always return Some, even for unknown jurisdictions"
        );
        let ctx = ctx.expect("already checked is_some");
        // Unknown jurisdiction: all three regulation flags should be false.
        assert!(!ctx.gdpr_region);
        assert!(!ctx.ccpa_applies);
        assert!(!ctx.lgpd_applies);
    }

    #[test]
    fn empty_jurisdiction_string_does_not_panic() {
        let engine = LegalisPolicyEngine::ccpa();
        let ctx = engine.evaluate_for_jurisdiction("");
        assert!(
            ctx.is_some(),
            "must not panic or return None for empty string"
        );
    }

    #[test]
    fn jurisdiction_matching_is_case_insensitive() {
        let engine = LegalisPolicyEngine::gdpr();
        let upper = engine
            .evaluate_for_jurisdiction("EU")
            .expect("must return Some for EU");
        let lower_engine = LegalisPolicyEngine::gdpr();
        let lower = lower_engine
            .evaluate_for_jurisdiction("eu")
            .expect("must return Some for eu");
        // Both should produce the same gdpr_region flag.
        assert_eq!(upper.gdpr_region, lower.gdpr_region);
    }

    // ----- Mixed statute tests -----

    #[test]
    fn custom_statute_set_evaluates_correctly() {
        use legalis_core::{Effect, Statute};

        let statutes = alloc::vec![
            Statute::new(
                "custom-prohibition",
                "Custom Prohibition",
                Effect::prohibition("data processing"),
            )
            .with_jurisdiction("XY"),
            Statute::new("custom-grant", "Custom Grant", Effect::grant("read access"),)
                .with_jurisdiction("XY"),
        ];

        let engine = LegalisPolicyEngine::new(statutes);
        let ctx = engine
            .evaluate_for_jurisdiction("XY")
            .expect("must return Some for custom jurisdiction");

        // The prohibition should have set data_transfer_allowed = false.
        assert!(
            !ctx.data_transfer_allowed,
            "Prohibition statute should disable data transfer"
        );
    }

    #[test]
    fn statute_without_jurisdiction_applies_universally() {
        use legalis_core::{Effect, Statute};

        // A statute with no jurisdiction field applies to every jurisdiction.
        let statutes = alloc::vec![Statute::new(
            "universal-audit",
            "Universal Audit Obligation",
            Effect::obligation("audit all data processing"),
        )]; // No .with_jurisdiction() call

        let engine = LegalisPolicyEngine::new(statutes);

        for jur in ["EU", "US-CA", "JP", "UNKNOWN", "XY"] {
            let ctx = engine
                .evaluate_for_jurisdiction(jur)
                .expect("must return Some");
            assert!(
                ctx.audit_required,
                "universal statute should set audit_required for {jur}"
            );
        }
    }
}

extern crate alloc;
