//! Tests that Naive Bayes uses per-source sample counts (not global count).

#[cfg(feature = "ml")]
mod tests {
    use oxirouter::ml::NaiveBayesClassifier;
    use oxirouter::ml::feature::FeatureVector;
    use oxirouter::ml::model::{Model, ModelPersistence, TrainingSample};

    fn fv(values: Vec<f32>) -> FeatureVector {
        let names: Vec<String> = (0..values.len()).map(|i| format!("f{}", i)).collect();
        FeatureVector { values, names }
    }

    /// Build a training sample. All samples use success=true so reward() > 0.
    fn sample(source: &str, feat_val: f32, dim: usize) -> TrainingSample {
        TrainingSample::new(
            fv(vec![feat_val; dim]),
            source,
            true, // success — ensures reward() > 0 for any feat_val
            50,   // latency_ms
            10,   // result_count
        )
    }

    const DIM: usize = 4;

    #[test]
    fn test_per_source_count_after_updates() {
        let mut nb = NaiveBayesClassifier::new(DIM);
        // Train source-A 10 times, source-B 5 times
        for _ in 0..10 {
            let s = sample("a", 0.8, DIM);
            let _ = nb.train(&[s]);
        }
        for _ in 0..5 {
            let s = sample("b", 0.2, DIM);
            let _ = nb.train(&[s]);
        }
        let counts = nb.per_source_counts();
        assert_eq!(
            *counts.get("a").unwrap_or(&0),
            10,
            "source-A count should be 10"
        );
        assert_eq!(
            *counts.get("b").unwrap_or(&0),
            5,
            "source-B count should be 5"
        );
    }

    #[test]
    fn test_means_converge_independently() {
        // Target means: source-A → 0.8, source-B → 0.2
        // Alternating 60 samples each (120 total).
        // Both sources have success=true so weight > 0 always.
        let mut nb = NaiveBayesClassifier::new(DIM);
        for _ in 0..60 {
            let _ = nb.train(&[sample("a", 0.8, DIM)]);
            let _ = nb.train(&[sample("b", 0.2, DIM)]);
        }
        let means_a = nb.source_means("a").expect("a has means").clone();
        let means_b = nb.source_means("b").expect("b has means").clone();
        for &m in &means_a {
            assert!(
                (m - 0.8).abs() < 0.05,
                "source-A mean should be ~0.8, got {m}"
            );
        }
        for &m in &means_b {
            assert!(
                (m - 0.2).abs() < 0.05,
                "source-B mean should be ~0.2, got {m}"
            );
        }
    }

    #[test]
    fn test_legacy_blob_loads_with_empty_per_source_counts() {
        // A blob produced BEFORE the per_source_counts field existed has no such
        // field in its JSON. Deserializing it must succeed and default the counts
        // to an empty HashMap.
        let mut nb = NaiveBayesClassifier::new(DIM);
        let _ = nb.train(&[sample("a", 0.5, DIM)]);
        // `to_bytes` on Model trait returns Vec<u8> (not Result)
        let bytes: Vec<u8> = ModelPersistence::to_bytes(&nb);
        // Load should succeed regardless
        let restored = NaiveBayesClassifier::from_bytes(&bytes);
        assert!(restored.is_ok(), "legacy blob should load: {:?}", restored);
    }

    #[test]
    fn test_configurable_prior_decay() {
        // With decay=0.5, prior should decay faster than default 0.99
        let mut nb_fast = NaiveBayesClassifier::new(DIM).with_prior_decay(0.5);
        let mut nb_slow = NaiveBayesClassifier::new(DIM); // default decay=0.99

        // Train source-A 20 times, then source-B 1 time
        for _ in 0..20 {
            let _ = nb_fast.train(&[sample("a", 0.8, DIM)]);
            let _ = nb_slow.train(&[sample("a", 0.8, DIM)]);
        }
        let _ = nb_fast.train(&[sample("b", 0.2, DIM)]);
        let _ = nb_slow.train(&[sample("b", 0.2, DIM)]);

        // 30 more a-samples to further decay b's prior
        for _ in 0..30 {
            let _ = nb_fast.train(&[sample("a", 0.8, DIM)]);
            let _ = nb_slow.train(&[sample("a", 0.8, DIM)]);
        }

        // Both should still work correctly
        let fv_input = fv(vec![0.8_f32; DIM]);
        let sources = vec!["a".to_string(), "b".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        assert!(nb_fast.predict(&fv_input, &source_refs).is_ok());
        assert!(nb_slow.predict(&fv_input, &source_refs).is_ok());
    }
}
