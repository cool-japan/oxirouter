//! Federated weight sharing tests.
//!
//! Verifies the merge strategies work correctly, and that incompatible models
//! produce informative errors.

#[cfg(feature = "ml")]
mod tests {
    use oxirouter::OxiRouterError;
    use oxirouter::ml::{
        MergeStrategy, ModelPersistence, ModelState, NaiveBayesClassifier, merge_states,
    };

    fn make_nb_state(feature_dim: usize, sources: &[&str]) -> ModelState {
        let mut nb = NaiveBayesClassifier::new(feature_dim);
        let source_strings: Vec<String> = sources.iter().map(|&s| s.to_string()).collect();
        let refs: Vec<&String> = source_strings.iter().collect();
        nb.initialize_sources(&refs);
        nb.to_state()
    }

    #[test]
    fn average_merge_yields_midpoint() {
        let sources = ["a", "b"];
        let mut local = make_nb_state(4, &sources);
        let mut remote = make_nb_state(4, &sources);

        // Force weights to 0.0 and 1.0 respectively
        for w in local.weights.iter_mut() {
            *w = 0.0;
        }
        for w in remote.weights.iter_mut() {
            *w = 1.0;
        }

        merge_states(&mut local, &remote, MergeStrategy::Average).unwrap();

        for w in &local.weights {
            assert!(
                (w - 0.5).abs() < 1e-4,
                "Expected 0.5 after average merge, got {w}"
            );
        }
    }

    #[test]
    fn weighted_average_w0_is_identity() {
        let sources = ["a"];
        let mut local = make_nb_state(4, &sources);
        let remote = make_nb_state(4, &sources);

        for w in local.weights.iter_mut() {
            *w = 0.7;
        }
        let mut remote2 = remote;
        for w in remote2.weights.iter_mut() {
            *w = 0.3;
        }

        // w=0.0 means "keep local"
        merge_states(&mut local, &remote2, MergeStrategy::WeightedAverage(0.0)).unwrap();
        for w in &local.weights {
            assert!(
                (w - 0.7).abs() < 1e-4,
                "w=0.0 should keep local: expected 0.7, got {w}"
            );
        }
    }

    #[test]
    fn weighted_average_w1_is_replace() {
        let sources = ["a"];
        let mut local = make_nb_state(4, &sources);
        let remote = make_nb_state(4, &sources);

        for w in local.weights.iter_mut() {
            *w = 0.7;
        }
        let mut remote2 = remote;
        for w in remote2.weights.iter_mut() {
            *w = 0.3;
        }

        // w=1.0 means "replace with remote"
        merge_states(&mut local, &remote2, MergeStrategy::WeightedAverage(1.0)).unwrap();
        for w in &local.weights {
            assert!(
                (w - 0.3).abs() < 1e-4,
                "w=1.0 should replace with remote: expected 0.3, got {w}"
            );
        }
    }

    #[test]
    fn keep_latest_chooses_higher_iteration() {
        let sources = ["a"];
        let mut local = make_nb_state(4, &sources);
        local.iterations = 5;
        for w in local.weights.iter_mut() {
            *w = 1.0;
        }

        let mut remote = make_nb_state(4, &sources);
        remote.iterations = 10;
        for w in remote.weights.iter_mut() {
            *w = 99.0;
        }

        merge_states(&mut local, &remote, MergeStrategy::KeepLatest).unwrap();
        assert_eq!(
            local.iterations, 10,
            "Should keep the remote with higher iterations"
        );
        assert!(
            (local.weights[0] - 99.0).abs() < 1e-4,
            "Weights should be replaced with remote's"
        );
    }

    #[test]
    fn keep_latest_no_change_if_local_newer() {
        let sources = ["a"];
        let mut local = make_nb_state(4, &sources);
        local.iterations = 20;
        for w in local.weights.iter_mut() {
            *w = 7.0;
        }

        let mut remote = make_nb_state(4, &sources);
        remote.iterations = 5;
        for w in remote.weights.iter_mut() {
            *w = 99.0;
        }

        merge_states(&mut local, &remote, MergeStrategy::KeepLatest).unwrap();
        // Local has more iterations, should not be replaced
        assert_eq!(local.iterations, 20, "Local should be kept");
        assert!(
            (local.weights[0] - 7.0).abs() < 1e-4,
            "Weights should remain local"
        );
    }

    #[test]
    fn keep_best_chooses_higher_reward() {
        let sources = ["a"];
        let mut local = make_nb_state(4, &sources);
        local.extra_params.push(0.3); // last extra_param = total_reward

        let mut remote = make_nb_state(4, &sources);
        remote.extra_params.push(0.9); // remote has higher reward
        for w in remote.weights.iter_mut() {
            *w = 55.0;
        }

        merge_states(&mut local, &remote, MergeStrategy::KeepBest).unwrap();
        assert!(
            (local.weights[0] - 55.0).abs() < 1e-4,
            "Should switch to remote with higher reward"
        );
    }

    #[test]
    fn incompatible_feature_dim_returns_error() {
        let mut local = make_nb_state(4, &["a"]);
        let mut remote = make_nb_state(8, &["a"]); // different dim

        // Force same source IDs so only dim mismatch triggers
        remote.source_ids = local.source_ids.clone();

        let result = merge_states(&mut local, &remote, MergeStrategy::Average);
        match result {
            Err(OxiRouterError::IncompatibleModel { reason }) => {
                assert!(
                    reason.contains("feature_dim"),
                    "Error reason should mention feature_dim, got: {reason}"
                );
            }
            other => panic!("Expected IncompatibleModel error, got: {other:?}"),
        }
    }

    #[test]
    fn incompatible_source_ids_returns_error() {
        let mut local = make_nb_state(4, &["a"]);
        let remote = make_nb_state(4, &["b"]); // different source

        let result = merge_states(&mut local, &remote, MergeStrategy::Average);
        match result {
            Err(OxiRouterError::IncompatibleModel { reason }) => {
                assert!(
                    reason.contains("source_ids"),
                    "Error reason should mention source_ids, got: {reason}"
                );
            }
            other => panic!("Expected IncompatibleModel error, got: {other:?}"),
        }
    }

    #[test]
    fn router_load_model_then_export_weights() {
        use oxirouter::{DataSource, Router};

        let mut nb = NaiveBayesClassifier::new(38);
        let sources = vec!["a".to_string(), "b".to_string()];
        let refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&refs);

        let original_bytes = nb.to_bytes();

        let mut router = Router::new();
        router.add_source(DataSource::new("a", "https://a.example.org/sparql"));
        router.add_source(DataSource::new("b", "https://b.example.org/sparql"));

        router.load_model_from_bytes(&original_bytes).unwrap();
        let exported = router.export_weights().unwrap();
        assert_eq!(
            exported, original_bytes,
            "Exported bytes should match loaded bytes"
        );
    }

    #[test]
    fn router_merge_weights_average() {
        use oxirouter::{DataSource, Router};

        let mut nb_local = NaiveBayesClassifier::new(38);
        let mut nb_remote = NaiveBayesClassifier::new(38);
        let sources = vec!["a".to_string(), "b".to_string()];
        let refs: Vec<&String> = sources.iter().collect();
        nb_local.initialize_sources(&refs);
        nb_remote.initialize_sources(&refs);

        let local_bytes = nb_local.to_bytes();
        let remote_bytes = nb_remote.to_bytes();

        let mut router = Router::new();
        router.add_source(DataSource::new("a", "https://a.example.org/sparql"));
        router.add_source(DataSource::new("b", "https://b.example.org/sparql"));

        router.load_model_from_bytes(&local_bytes).unwrap();
        router
            .merge_weights(&remote_bytes, MergeStrategy::Average)
            .unwrap();

        // After merge, should still be able to export
        let merged_bytes = router.export_weights().unwrap();
        assert!(
            !merged_bytes.is_empty(),
            "Merged model bytes should not be empty"
        );
    }

    #[test]
    fn export_without_load_returns_error() {
        let router = oxirouter::Router::new();
        let result = router.export_weights();
        assert!(
            result.is_err(),
            "export_weights without load should return error"
        );
    }
}
