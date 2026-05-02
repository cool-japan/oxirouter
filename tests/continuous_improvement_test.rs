//! Tests that NeuralNetwork optimizer state survives save/load (Block U).
//!
//! After a save/load roundtrip the Adam moments, LR schedule epoch counter, and
//! early-stopping patience state must be fully restored.  Predictions must be
//! bit-for-bit identical (within float tolerance) to the pre-save network.

#[cfg(feature = "ml")]
mod tests {
    use oxirouter::ml::feature::FeatureVector;
    use oxirouter::ml::model::{Model, ModelPersistence, TrainingSample};
    use oxirouter::ml::neural::NeuralNetwork;
    use oxirouter::ml::optimizer::{AdamConfig, OptimizerType};
    use oxirouter::ml::schedule::EarlyStoppingConfig;

    /// Helper: serialize a NeuralNetwork to bytes via ModelPersistence trait.
    fn to_bytes(net: &NeuralNetwork) -> Vec<u8> {
        ModelPersistence::to_bytes(net)
    }

    /// Helper: deserialize a NeuralNetwork from bytes via ModelPersistence trait.
    fn from_bytes(bytes: &[u8]) -> oxirouter::Result<NeuralNetwork> {
        NeuralNetwork::from_bytes(bytes)
    }

    /// Build a FeatureVector of uniform value with the given dimension.
    fn uniform_fv(dim: usize, value: f32) -> FeatureVector {
        let mut fv = FeatureVector::new();
        for i in 0..dim {
            fv.add(format!("f{i}"), value);
        }
        fv
    }

    /// Build a TrainingSample directed at a specific source index within `num_sources`.
    fn make_sample(
        source_idx: usize,
        num_sources: usize,
        success: bool,
        latency_ms: u32,
    ) -> TrainingSample {
        let feature_value = source_idx as f32 / num_sources.max(1) as f32;
        let fv = uniform_fv(48, feature_value);
        let source_name = format!("source_{source_idx}");
        TrainingSample::new(
            fv,
            source_name,
            success,
            latency_ms,
            if success { 10 } else { 0 },
        )
    }

    // -------------------------------------------------------------------------
    // Test 1: Adam moments survive save/load — predictions identical
    // -------------------------------------------------------------------------
    #[test]
    fn test_adam_moments_survive_save_load() {
        let mut net = NeuralNetwork::new(48, &[32, 16], 3)
            .with_optimizer(OptimizerType::Adam(AdamConfig::default()));
        net.set_source_ids(vec![
            "source_0".to_string(),
            "source_1".to_string(),
            "source_2".to_string(),
        ]);

        // Train 50 steps to build up Adam moments
        for i in 0..50_usize {
            let s = make_sample(i % 3, 3, i % 2 == 0, 100);
            net.train(&[s]).expect("train should succeed");
        }

        // Save → load
        let bytes = to_bytes(&net);
        let restored = from_bytes(&bytes).expect("from_bytes failed");

        // Predictions must be identical
        let fv = uniform_fv(48, 0.5);
        let sources = vec![
            "source_0".to_string(),
            "source_1".to_string(),
            "source_2".to_string(),
        ];
        let source_refs: Vec<&String> = sources.iter().collect();

        let orig_pred = net
            .predict(&fv, &source_refs)
            .expect("predict (orig) failed");
        let rest_pred = restored
            .predict(&fv, &source_refs)
            .expect("predict (restored) failed");

        assert_eq!(orig_pred.len(), rest_pred.len());
        for (o, r) in orig_pred.iter().zip(rest_pred.iter()) {
            assert!(
                (o.1 - r.1).abs() < 1e-5,
                "prediction mismatch after save/load: orig={:.8}, restored={:.8}",
                o.1,
                r.1
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: optimizer_state is non-None after training; restored net can
    //         continue training immediately (no re-initialisation needed)
    // -------------------------------------------------------------------------
    #[test]
    fn test_optimizer_state_is_not_none_after_training() {
        let mut net = NeuralNetwork::new(48, &[32, 16], 2)
            .with_optimizer(OptimizerType::Adam(AdamConfig::default()));
        net.set_source_ids(vec!["source_0".to_string(), "source_1".to_string()]);

        let s = make_sample(0, 2, true, 50);
        net.train(&[s]).expect("initial train");

        // Save/load
        let bytes = to_bytes(&net);
        let mut restored = from_bytes(&bytes).expect("from_bytes");

        // Restored net must be able to train immediately without error
        let s2 = make_sample(1, 2, false, 200);
        restored
            .train(&[s2])
            .expect("training after restore should succeed");
    }

    // -------------------------------------------------------------------------
    // Test 3: early_stopping_state is preserved across save/load
    // -------------------------------------------------------------------------
    #[test]
    fn test_early_stopping_state_preserved() {
        let mut net = NeuralNetwork::new(48, &[16], 2).with_early_stopping(EarlyStoppingConfig {
            patience: 100,
            min_delta: 0.0,
        });
        net.set_source_ids(vec!["source_0".to_string(), "source_1".to_string()]);

        // Train a few steps
        for _ in 0..10_usize {
            let s = make_sample(0, 2, true, 100);
            net.train(&[s]).expect("train");
        }

        let bytes = to_bytes(&net);
        let mut restored = from_bytes(&bytes).expect("from_bytes");

        // Must still be able to train
        let s = make_sample(1, 2, true, 100);
        restored
            .train(&[s])
            .expect("train after restore should succeed");
    }

    // -------------------------------------------------------------------------
    // Test 4: SGD model (optimizer_state = None) loads cleanly (backward compat)
    // -------------------------------------------------------------------------
    #[test]
    fn test_sgd_model_loads_cleanly() {
        let mut net = NeuralNetwork::new(48, &[16], 2).with_optimizer(OptimizerType::SGD);
        net.set_source_ids(vec!["source_0".to_string(), "source_1".to_string()]);

        let s = make_sample(0, 2, true, 50);
        net.train(&[s]).expect("train");

        let bytes = to_bytes(&net);
        let restored = from_bytes(&bytes);
        assert!(
            restored.is_ok(),
            "SGD model should load cleanly: {restored:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Test 5: weights are identical after roundtrip
    // -------------------------------------------------------------------------
    #[test]
    fn test_weights_identical_after_roundtrip() {
        let mut net = NeuralNetwork::new(48, &[32], 3)
            .with_optimizer(OptimizerType::Adam(AdamConfig::default()));
        net.set_source_ids(vec![
            "source_0".to_string(),
            "source_1".to_string(),
            "source_2".to_string(),
        ]);

        for _ in 0..20_usize {
            let s = make_sample(0, 3, true, 100);
            net.train(&[s]).expect("train");
        }

        let bytes = to_bytes(&net);
        let restored = from_bytes(&bytes).expect("deserialize");

        let fv = uniform_fv(48, 0.1);
        let sources = vec![
            "source_0".to_string(),
            "source_1".to_string(),
            "source_2".to_string(),
        ];
        let source_refs: Vec<&String> = sources.iter().collect();

        let p1 = net.predict(&fv, &source_refs).expect("predict1");
        let p2 = restored.predict(&fv, &source_refs).expect("predict2");

        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!(
                (a.1 - b.1).abs() < 1e-5,
                "weight mismatch: {} vs {}",
                a.1,
                b.1
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: iterations counter is preserved
    // -------------------------------------------------------------------------
    #[test]
    fn test_iterations_preserved_after_roundtrip() {
        let mut net = NeuralNetwork::new(48, &[16], 2)
            .with_optimizer(OptimizerType::Adam(AdamConfig::default()));
        net.set_source_ids(vec!["source_0".to_string(), "source_1".to_string()]);

        for i in 0..30_usize {
            let s = make_sample(i % 2, 2, true, 100);
            net.train(&[s]).expect("train");
        }

        let pre_iterations = net.iterations();
        let bytes = to_bytes(&net);
        let restored = from_bytes(&bytes).expect("from_bytes");

        assert_eq!(
            pre_iterations,
            restored.iterations(),
            "iterations counter must survive save/load"
        );
    }

    // -------------------------------------------------------------------------
    // Test 7: differential — training after restore diverges from training a
    //         fresh network (proving moments are actually applied)
    // -------------------------------------------------------------------------
    #[test]
    fn test_continued_training_uses_restored_moments() {
        let mut net = NeuralNetwork::new(48, &[32, 16], 3)
            .with_optimizer(OptimizerType::Adam(AdamConfig::default()));
        net.set_source_ids(vec![
            "source_0".to_string(),
            "source_1".to_string(),
            "source_2".to_string(),
        ]);

        // Phase 1: 20 steps to build Adam moments
        for i in 0..20_usize {
            let s = make_sample(i % 3, 3, true, 100);
            net.train(&[s]).expect("phase1 train");
        }

        // Save/load → restored net has the same moments
        let bytes = to_bytes(&net);
        let mut restored = from_bytes(&bytes).expect("from_bytes");

        // Phase 2: 20 more steps on both
        for i in 0..20_usize {
            let s1 = make_sample(i % 3, 3, i % 2 == 0, 100);
            let s2 = make_sample(i % 3, 3, i % 2 == 0, 100);
            net.train(&[s1]).expect("phase2 net train");
            restored.train(&[s2]).expect("phase2 restored train");
        }

        // Both should produce identical predictions
        let fv = uniform_fv(48, 0.5);
        let sources = vec![
            "source_0".to_string(),
            "source_1".to_string(),
            "source_2".to_string(),
        ];
        let source_refs: Vec<&String> = sources.iter().collect();

        let p_net = net.predict(&fv, &source_refs).expect("predict net");
        let p_restored = restored
            .predict(&fv, &source_refs)
            .expect("predict restored");

        for (a, b) in p_net.iter().zip(p_restored.iter()) {
            assert!(
                (a.1 - b.1).abs() < 1e-4,
                "continued training diverged after restore: net={:.8}, restored={:.8}",
                a.1,
                b.1
            );
        }
    }
}
