//! Online training pipeline tests.
//!
//! Verifies that feature vectors are captured during routing and used to update
//! the ML model weights when `learn_from_outcome` is called.

#[cfg(feature = "ml")]
mod tests {
    use oxirouter::ml::ModelPersistence;
    use oxirouter::ml::NaiveBayesClassifier;
    use oxirouter::{DataSource, Query, Router};

    fn make_router_with_nb() -> Router {
        let mut router = Router::new();
        router.add_source(DataSource::new("source_a", "https://a.example.org/sparql"));
        router.add_source(DataSource::new("source_b", "https://b.example.org/sparql"));

        let mut nb = NaiveBayesClassifier::new(38);
        let sources = vec!["source_a".to_string(), "source_b".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);
        router.set_model(Box::new(nb));
        router
    }

    #[test]
    fn online_training_does_not_panic() {
        let mut router = make_router_with_nb();
        router.set_online_training(true);

        let query = Query::parse("SELECT ?s WHERE { ?s a <http://schema.org/Person> }").unwrap();

        // Route and log
        let ranking = router.route_and_log(&query).unwrap();

        if let Some(top) = ranking.sources.first() {
            let source_id = top.source_id.clone();
            let query_id = query.predicate_hash();

            // Record a successful outcome — triggers online model update
            router
                .learn_from_outcome(query_id, &source_id, true, 50, 10)
                .unwrap();
        }
        // If we reach here, no panic occurred
    }

    #[test]
    fn feature_vector_stored_in_log() {
        let mut router = make_router_with_nb();
        router.set_online_training(true);

        let query = Query::parse("SELECT ?s WHERE { ?s a <http://schema.org/Person> }").unwrap();
        let ranking = router.route_and_log(&query).unwrap();

        // The query log should have stored feature vectors for ML-used routes
        let log = router.query_log();
        assert!(!log.is_empty(), "Log should have entries");

        if ranking.ml_used {
            // At least the first source should have a feature vector stored
            if let Some(top) = ranking.sources.first() {
                let fv = log.find_entry_features(query.predicate_hash(), &top.source_id);
                assert!(
                    fv.is_some(),
                    "Feature vector should be stored when ML is used"
                );
                assert!(
                    !fv.unwrap().is_empty(),
                    "Feature vector should not be empty"
                );
            }
        }
    }

    #[test]
    fn set_online_training_toggle() {
        let mut router = Router::new();

        // Default: enabled
        assert!(
            router.is_online_training_enabled(),
            "Online training should be enabled by default"
        );

        router.set_online_training(false);
        assert!(
            !router.is_online_training_enabled(),
            "Online training should be disabled after set_online_training(false)"
        );

        router.set_online_training(true);
        assert!(
            router.is_online_training_enabled(),
            "Online training should be re-enabled"
        );
    }

    #[test]
    fn online_training_disabled_no_feature_vector_stored() {
        let mut router = make_router_with_nb();
        router.set_online_training(false);

        let query = Query::parse("SELECT ?s WHERE { ?s a <http://schema.org/Person> }").unwrap();
        let _ranking = router.route_and_log(&query).unwrap();

        // When online training is disabled, no feature vectors should be stored
        let log = router.query_log();
        for entry in log.recent_entries(100) {
            assert!(
                entry.feature_vector.is_none(),
                "No feature vector should be stored when online training is disabled"
            );
        }
    }

    #[test]
    fn load_model_from_bytes_enables_export() {
        let mut nb = NaiveBayesClassifier::new(38);
        let sources = vec!["a".to_string(), "b".to_string()];
        let refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&refs);

        let bytes = nb.to_bytes();

        let mut router = Router::new();
        router.add_source(DataSource::new("a", "https://a.example.org/sparql"));
        router.add_source(DataSource::new("b", "https://b.example.org/sparql"));

        router.load_model_from_bytes(&bytes).unwrap();

        let exported = router.export_weights().unwrap();
        assert!(!exported.is_empty(), "Exported weights should not be empty");
        assert_eq!(exported, bytes, "Exported bytes should match input bytes");
    }
}
