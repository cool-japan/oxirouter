//! Ensemble classifier tests.
//!
//! Verifies that the ensemble combines component predictions correctly,
//! and that serialization roundtrip preserves predictions.

#[cfg(feature = "ml")]
mod tests {
    use oxirouter::ml::{
        EnsembleClassifier, FeatureVector, Model, ModelPersistence, NaiveBayesClassifier,
    };

    fn make_features(dim: usize) -> FeatureVector {
        let mut fv = FeatureVector::new();
        for i in 0..dim {
            fv.add(format!("f{i}"), 0.5_f32);
        }
        fv
    }

    fn make_initialized_nb(dim: usize, sources: &[String]) -> NaiveBayesClassifier {
        let mut nb = NaiveBayesClassifier::new(dim);
        let refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&refs);
        nb
    }

    #[test]
    fn ensemble_empty_returns_error() {
        let ensemble = EnsembleClassifier::new(4);
        let sources = vec!["a".to_string()];
        let refs: Vec<&String> = sources.iter().collect();
        let features = make_features(4);
        let result = ensemble.predict(&features, &refs);
        assert!(result.is_err(), "Empty ensemble should return error");
    }

    #[test]
    fn single_component_ensemble_same_as_component() {
        let source_ids = vec!["a".to_string(), "b".to_string()];
        let refs: Vec<&String> = source_ids.iter().collect();
        let features = make_features(4);

        // Standalone NB prediction
        let nb1 = make_initialized_nb(4, &source_ids);
        let standalone_pred = nb1.predict(&features, &refs).unwrap();

        // Ensemble with single NB component (weight=1.0)
        let nb2 = make_initialized_nb(4, &source_ids);
        let ensemble = EnsembleClassifier::new(4)
            .add_component(Box::new(nb2), 1.0, "nb")
            .unwrap();
        let ens_pred = ensemble.predict(&features, &refs).unwrap();

        assert_eq!(standalone_pred.len(), ens_pred.len());
        for ((id_a, conf_a), (id_b, conf_b)) in standalone_pred.iter().zip(ens_pred.iter()) {
            assert_eq!(id_a, id_b, "Source IDs should match");
            assert!(
                (conf_a - conf_b).abs() < 1e-4,
                "Confidences diverged: {conf_a} vs {conf_b}"
            );
        }
    }

    #[test]
    fn two_component_ensemble_averages_predictions() {
        let source_ids = vec!["a".to_string(), "b".to_string()];
        let refs: Vec<&String> = source_ids.iter().collect();
        let features = make_features(4);

        // Two identical NB models — equal weights — should give same result as single
        let nb1 = make_initialized_nb(4, &source_ids);
        let nb2 = make_initialized_nb(4, &source_ids);
        let nb_ref = make_initialized_nb(4, &source_ids);

        let ensemble = EnsembleClassifier::new(4)
            .add_component(Box::new(nb1), 0.5, "nb1")
            .unwrap()
            .add_component(Box::new(nb2), 0.5, "nb2")
            .unwrap();

        let ens_pred = ensemble.predict(&features, &refs).unwrap();
        let ref_pred = nb_ref.predict(&features, &refs).unwrap();

        assert_eq!(ens_pred.len(), ref_pred.len());
        for ((_, ca), (_, cb)) in ens_pred.iter().zip(ref_pred.iter()) {
            assert!(
                (ca - cb).abs() < 1e-4,
                "Two identical components should give same result: {ca} vs {cb}"
            );
        }
    }

    #[test]
    fn feature_dim_mismatch_returns_error() {
        let source_ids = vec!["a".to_string()];
        let nb = make_initialized_nb(8, &source_ids); // dim=8 ≠ ensemble dim=4
        let result = EnsembleClassifier::new(4).add_component(Box::new(nb), 1.0, "nb");
        assert!(result.is_err(), "Feature dim mismatch should return error");
    }

    #[test]
    fn all_zero_weights_returns_error() {
        let source_ids = vec!["a".to_string()];
        let refs: Vec<&String> = source_ids.iter().collect();
        let features = make_features(4);

        let nb = make_initialized_nb(4, &source_ids);
        let ensemble = EnsembleClassifier::new(4)
            .add_component(Box::new(nb), 0.0, "nb")
            .unwrap();

        let result = ensemble.predict(&features, &refs);
        assert!(result.is_err(), "Zero-weight ensemble should return error");
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let source_ids = vec!["x".to_string(), "y".to_string()];
        let refs: Vec<&String> = source_ids.iter().collect();
        let features = make_features(4);

        let nb = make_initialized_nb(4, &source_ids);
        let ensemble = EnsembleClassifier::new(4)
            .add_component(Box::new(nb), 0.6, "nb")
            .unwrap();

        // Serialize
        let bytes = ModelPersistence::to_bytes(&ensemble);
        assert!(!bytes.is_empty());

        // Deserialize
        let restored = EnsembleClassifier::from_bytes(&bytes).unwrap();

        // Predictions must match
        let pred_orig = ensemble.predict(&features, &refs).unwrap();
        let pred_restored = restored.predict(&features, &refs).unwrap();

        assert_eq!(pred_orig.len(), pred_restored.len());
        for ((id_a, ca), (id_b, cb)) in pred_orig.iter().zip(pred_restored.iter()) {
            assert_eq!(id_a, id_b, "Source IDs must match after roundtrip");
            assert!(
                (ca - cb).abs() < 1e-4,
                "Confidences diverged after roundtrip: {ca} vs {cb}"
            );
        }
    }

    #[test]
    fn from_bytes_wrong_version_returns_error() {
        let mut bad_bytes = vec![0u8; 12];
        // Write version=99
        bad_bytes[0..4].copy_from_slice(&99u32.to_le_bytes());
        let result = EnsembleClassifier::from_bytes(&bad_bytes);
        assert!(result.is_err(), "Wrong version should fail");
    }

    #[test]
    fn component_count_correct() {
        let source_ids = vec!["a".to_string()];
        let nb1 = make_initialized_nb(4, &source_ids);
        let nb2 = make_initialized_nb(4, &source_ids);
        let ensemble = EnsembleClassifier::new(4)
            .add_component(Box::new(nb1), 1.0, "nb1")
            .unwrap()
            .add_component(Box::new(nb2), 1.0, "nb2")
            .unwrap();

        assert_eq!(ensemble.component_count(), 2);
    }
}
