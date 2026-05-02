use super::*;
use crate::ml::activation::Activation;
use crate::ml::layer::{Layer, LayerGradients};
use crate::ml::optimizer::AdamConfig;
use crate::ml::schedule::{EarlyStoppingConfig, LearningRateSchedule};

fn create_test_features(values: &[f32]) -> FeatureVector {
    let mut fv = FeatureVector::new();
    for (i, &v) in values.iter().enumerate() {
        fv.add(format!("f{}", i), v);
    }
    fv
}

#[test]
fn test_activation_functions() {
    assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
    assert_eq!(Activation::ReLU.apply(1.0), 1.0);

    let sigmoid_val = Activation::Sigmoid.apply(0.0);
    assert!((sigmoid_val - 0.5).abs() < 0.001);

    assert_eq!(Activation::Linear.apply(5.0), 5.0);
}

#[test]
fn test_layer_forward() {
    let layer = Layer::new(3, 2, Activation::ReLU);
    let input = vec![1.0, 0.5, 0.0];
    let output = layer.forward(&input);

    assert_eq!(output.len(), 2);
}

#[test]
fn test_neural_network_creation() {
    let nn = NeuralNetwork::new(10, &[8, 4], 3);
    assert_eq!(nn.feature_dim(), 10);
    assert_eq!(nn.layers.len(), 3); // 2 hidden + 1 output
}

#[test]
fn test_neural_network_forward() {
    let nn = NeuralNetwork::new(5, &[4], 3);
    let features = create_test_features(&[0.5, 0.5, 0.5, 0.5, 0.5]);

    let sources = vec!["src1".to_string(), "src2".to_string(), "src3".to_string()];
    let source_refs: Vec<&String> = sources.iter().collect();

    let predictions = nn.predict(&features, &source_refs).unwrap();

    assert_eq!(predictions.len(), 3);
    let total: f32 = predictions.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 0.01); // Softmax should sum to 1
}

#[test]
fn test_neural_network_training() {
    let mut nn = NeuralNetwork::new(3, &[4], 2);
    nn.set_source_ids(vec!["src1".to_string(), "src2".to_string()]);

    let samples = vec![
        TrainingSample::new(
            create_test_features(&[0.9, 0.9, 0.9]),
            "src1",
            true,
            100,
            10,
        ),
        TrainingSample::new(
            create_test_features(&[0.1, 0.1, 0.1]),
            "src2",
            true,
            100,
            10,
        ),
    ];

    let initial_iterations = nn.iterations;
    nn.train(&samples).unwrap();
    assert!(nn.iterations > initial_iterations);
}

#[test]
fn test_softmax_stability() {
    let nn = NeuralNetwork::new(3, &[], 3);

    // Test with large values (should not overflow)
    let large_logits = vec![1000.0, 999.0, 998.0];
    let probs = nn.softmax(&large_logits);

    assert!(probs[0] > 0.0 && probs[0] <= 1.0);
    let total: f32 = probs.iter().sum();
    assert!((total - 1.0).abs() < 0.01);
}

#[test]
fn test_neural_network_serialization_roundtrip() {
    use super::{ModelPersistence, ModelState};

    let mut nn = NeuralNetwork::new(5, &[8, 4], 3);
    nn.set_source_ids(vec![
        "src1".to_string(),
        "src2".to_string(),
        "src3".to_string(),
    ]);

    // Train with some samples
    let samples = vec![
        TrainingSample::new(
            create_test_features(&[0.9, 0.8, 0.7, 0.6, 0.5]),
            "src1",
            true,
            100,
            10,
        ),
        TrainingSample::new(
            create_test_features(&[0.1, 0.2, 0.3, 0.4, 0.5]),
            "src2",
            true,
            100,
            10,
        ),
        TrainingSample::new(
            create_test_features(&[0.5, 0.5, 0.5, 0.5, 0.5]),
            "src3",
            true,
            150,
            5,
        ),
    ];
    nn.train(&samples).unwrap();

    // Serialize
    let state = nn.to_state();
    let bytes = state.to_bytes();

    // Deserialize
    let restored_state = ModelState::from_bytes(&bytes).unwrap();
    let restored_nn = NeuralNetwork::from_state(restored_state).unwrap();

    // Verify architecture
    assert_eq!(nn.layers.len(), restored_nn.layers.len());
    assert_eq!(nn.feature_dim(), restored_nn.feature_dim());
    assert_eq!(nn.iterations, restored_nn.iterations);
    assert_eq!(nn.learning_rate, restored_nn.learning_rate);
    assert_eq!(nn.regularization, restored_nn.regularization);

    // Compare predictions
    let sources = vec!["src1".to_string(), "src2".to_string(), "src3".to_string()];
    let source_refs: Vec<&String> = sources.iter().collect();
    let test_features = create_test_features(&[0.6, 0.6, 0.6, 0.6, 0.6]);

    let original_pred = nn.predict(&test_features, &source_refs).unwrap();
    let restored_pred = restored_nn.predict(&test_features, &source_refs).unwrap();

    // Predictions should match
    assert_eq!(original_pred.len(), restored_pred.len());
    for (orig, rest) in original_pred.iter().zip(restored_pred.iter()) {
        assert_eq!(orig.0, rest.0);
        assert!(
            (orig.1 - rest.1).abs() < 1e-5,
            "Prediction mismatch: {} vs {}",
            orig.1,
            rest.1
        );
    }
}

#[test]
fn test_neural_network_to_bytes_from_bytes() {
    use super::ModelPersistence;

    let mut nn = NeuralNetwork::new(4, &[6], 2)
        .with_learning_rate(0.05)
        .with_regularization(0.002);
    nn.set_source_ids(vec!["alpha".to_string(), "beta".to_string()]);

    // Train
    let samples = vec![
        TrainingSample::new(
            create_test_features(&[0.1, 0.2, 0.3, 0.4]),
            "alpha",
            true,
            50,
            20,
        ),
        TrainingSample::new(
            create_test_features(&[0.9, 0.8, 0.7, 0.6]),
            "beta",
            true,
            100,
            15,
        ),
    ];
    nn.train(&samples).unwrap();

    // Use convenience methods
    let bytes = ModelPersistence::to_bytes(&nn);
    let restored = NeuralNetwork::from_bytes(&bytes).unwrap();

    assert_eq!(nn.feature_dim(), restored.feature_dim());
    assert_eq!(nn.iterations, restored.iterations);
    assert_eq!(nn.learning_rate, restored.learning_rate);
    assert_eq!(nn.regularization, restored.regularization);

    // Verify layer structure
    for (orig_layer, rest_layer) in nn.layers.iter().zip(restored.layers.iter()) {
        assert_eq!(orig_layer.input_dim, rest_layer.input_dim);
        assert_eq!(orig_layer.output_dim, rest_layer.output_dim);
        assert_eq!(orig_layer.activation, rest_layer.activation);
        assert_eq!(orig_layer.weights.len(), rest_layer.weights.len());
        assert_eq!(orig_layer.biases.len(), rest_layer.biases.len());

        // Check weights match
        for (ow, rw) in orig_layer.weights.iter().zip(rest_layer.weights.iter()) {
            assert!((ow - rw).abs() < 1e-6);
        }
        // Check biases match
        for (ob, rb) in orig_layer.biases.iter().zip(rest_layer.biases.iter()) {
            assert!((ob - rb).abs() < 1e-6);
        }
    }
}

#[test]
fn test_activation_serialization() {
    assert_eq!(
        Activation::from_byte(Activation::ReLU.to_byte()),
        Activation::ReLU
    );
    assert_eq!(
        Activation::from_byte(Activation::Sigmoid.to_byte()),
        Activation::Sigmoid
    );
    assert_eq!(
        Activation::from_byte(Activation::Tanh.to_byte()),
        Activation::Tanh
    );
    assert_eq!(
        Activation::from_byte(Activation::Linear.to_byte()),
        Activation::Linear
    );
    // Unknown byte defaults to Linear
    assert_eq!(Activation::from_byte(255), Activation::Linear);
}

#[test]
fn test_activation_derivatives() {
    // ReLU derivative
    assert_eq!(Activation::ReLU.derivative(1.0), 1.0);
    assert_eq!(Activation::ReLU.derivative(-1.0), 0.0);

    // Sigmoid derivative at 0 should be 0.25
    let sigmoid_deriv = Activation::Sigmoid.derivative(0.0);
    assert!((sigmoid_deriv - 0.25).abs() < 0.001);

    // Linear derivative is always 1
    assert_eq!(Activation::Linear.derivative(5.0), 1.0);
}

#[test]
fn test_layer_forward_with_cache() {
    let layer = Layer::new(3, 2, Activation::ReLU);
    let input = vec![1.0, 0.5, 0.0];
    let cache = layer.forward_with_cache(&input);

    assert_eq!(cache.input.len(), 3);
    assert_eq!(cache.pre_activation.len(), 2);
    assert_eq!(cache.post_activation.len(), 2);
}

#[test]
fn test_multi_layer_gradients() {
    // Test that gradients are computed for all layers, not just output
    let mut nn = NeuralNetwork::new(4, &[8, 4], 2);
    nn.set_source_ids(vec!["src1".to_string(), "src2".to_string()]);

    // Store initial weights for all layers
    let initial_weights: Vec<Vec<f32>> = nn.layers.iter().map(|l| l.weights.clone()).collect();

    // Train
    let features = create_test_features(&[0.5, 0.5, 0.5, 0.5]);
    nn.train_step(&features, 0, 1.0);

    // Verify all layers have been updated
    for (i, layer) in nn.layers.iter().enumerate() {
        let weights_changed = layer
            .weights
            .iter()
            .zip(&initial_weights[i])
            .any(|(w, iw)| (*w - *iw).abs() > 1e-10);
        assert!(
            weights_changed,
            "Layer {i} weights should have been updated"
        );
    }
}

#[test]
fn test_momentum_optimizer() {
    let mut nn = NeuralNetwork::new(3, &[4], 2).with_momentum(0.9);
    nn.set_source_ids(vec!["src1".to_string(), "src2".to_string()]);

    let features = create_test_features(&[0.5, 0.5, 0.5]);

    // Train multiple times
    for _ in 0..5 {
        nn.train_step(&features, 0, 1.0);
    }

    // Verify optimizer state was created
    assert!(nn.optimizer_state.is_some());
    let state = nn.optimizer_state.as_ref().unwrap();
    assert!(!state.weight_velocities.is_empty());
}

#[test]
fn test_adam_optimizer() {
    let mut nn = NeuralNetwork::new(3, &[4], 2).with_adam(AdamConfig::default());
    nn.set_source_ids(vec!["src1".to_string(), "src2".to_string()]);

    let features = create_test_features(&[0.5, 0.5, 0.5]);

    // Train multiple times
    for _ in 0..5 {
        nn.train_step(&features, 0, 1.0);
    }

    // Verify optimizer state was created
    assert!(nn.optimizer_state.is_some());
    let state = nn.optimizer_state.as_ref().unwrap();
    assert!(!state.weight_m.is_empty());
    assert!(!state.weight_v.is_empty());
    assert_eq!(state.t, 5);
}

#[test]
fn test_batch_training() {
    let mut nn = NeuralNetwork::new(3, &[4], 2);
    nn.set_source_ids(vec!["src1".to_string(), "src2".to_string()]);

    let samples = vec![
        TrainingSample::new(
            create_test_features(&[0.9, 0.9, 0.9]),
            "src1",
            true,
            100,
            10,
        ),
        TrainingSample::new(
            create_test_features(&[0.1, 0.1, 0.1]),
            "src2",
            true,
            100,
            10,
        ),
        TrainingSample::new(
            create_test_features(&[0.5, 0.5, 0.5]),
            "src1",
            true,
            100,
            10,
        ),
    ];

    let loss = nn.train_batch(&samples).unwrap();
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_learning_rate_decay() {
    let nn = NeuralNetwork::new(3, &[4], 2)
        .with_learning_rate(0.1)
        .with_lr_decay(0.95);

    // Initial learning rate
    assert!((nn.current_learning_rate() - 0.1).abs() < 1e-6);

    // After simulating epoch
    let mut nn = nn;
    nn.epoch = 10;
    let expected_lr = 0.1 * pow_f32(0.95, 10.0);
    assert!((nn.current_learning_rate() - expected_lr).abs() < 1e-6);
}

#[test]
fn test_step_decay_schedule() {
    let mut nn = NeuralNetwork::new(3, &[4], 2)
        .with_learning_rate(0.1)
        .with_lr_schedule(LearningRateSchedule::StepDecay {
            drop: 0.5,
            step_size: 10,
        });

    // Initial
    assert!((nn.current_learning_rate() - 0.1).abs() < 1e-6);

    // After 10 epochs
    nn.epoch = 10;
    assert!((nn.current_learning_rate() - 0.05).abs() < 1e-6);

    // After 20 epochs
    nn.epoch = 20;
    assert!((nn.current_learning_rate() - 0.025).abs() < 1e-6);
}

#[test]
fn test_cosine_annealing() {
    let mut nn = NeuralNetwork::new(3, &[4], 2)
        .with_learning_rate(0.1)
        .with_lr_schedule(LearningRateSchedule::CosineAnnealing {
            lr_min: 0.001,
            t_max: 100,
        });

    // At start (epoch 0)
    assert!((nn.current_learning_rate() - 0.1).abs() < 1e-5);

    // At middle (epoch 50)
    nn.epoch = 50;
    let mid_lr = nn.current_learning_rate();
    assert!(mid_lr > 0.001 && mid_lr < 0.1);

    // At end (epoch 100)
    nn.epoch = 100;
    assert!((nn.current_learning_rate() - 0.001).abs() < 1e-5);
}

#[test]
fn test_early_stopping() {
    let mut nn = NeuralNetwork::new(3, &[4], 2).with_early_stopping(EarlyStoppingConfig {
        patience: 3,
        min_delta: 0.01,
    });

    // Simulate improving loss
    assert!(!nn.update_early_stopping(1.0));
    assert!(!nn.update_early_stopping(0.8));
    assert!(!nn.update_early_stopping(0.6));

    // Simulate stagnating loss
    assert!(!nn.update_early_stopping(0.6));
    assert!(!nn.update_early_stopping(0.6));
    assert!(nn.update_early_stopping(0.6)); // Should trigger after patience=3

    assert!(nn.should_stop());
}

#[test]
fn test_training_convergence() {
    // Create a simple classification task
    let mut nn = NeuralNetwork::new(2, &[4], 2)
        .with_learning_rate(0.1)
        .with_momentum(0.9);

    let samples = vec![
        // Class 0: high values
        TrainingSample::new(create_test_features(&[0.9, 0.9]), "class0", true, 100, 10),
        TrainingSample::new(create_test_features(&[0.8, 0.85]), "class0", true, 100, 10),
        TrainingSample::new(create_test_features(&[0.95, 0.8]), "class0", true, 100, 10),
        // Class 1: low values
        TrainingSample::new(create_test_features(&[0.1, 0.1]), "class1", true, 100, 10),
        TrainingSample::new(create_test_features(&[0.15, 0.2]), "class1", true, 100, 10),
        TrainingSample::new(create_test_features(&[0.2, 0.15]), "class1", true, 100, 10),
    ];

    // Train for multiple epochs
    let mut losses = Vec::new();
    for _ in 0..50 {
        let loss = nn.train_batch(&samples).unwrap();
        losses.push(loss);
    }

    // Verify loss is decreasing (comparing first 10 vs last 10)
    let first_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let last_avg: f32 = losses[40..].iter().sum::<f32>() / 10.0;
    assert!(
        last_avg < first_avg,
        "Loss should decrease: first_avg={first_avg}, last_avg={last_avg}"
    );
}

#[test]
fn test_momentum_helps_convergence() {
    let samples = vec![
        TrainingSample::new(create_test_features(&[0.9, 0.9]), "class0", true, 100, 10),
        TrainingSample::new(create_test_features(&[0.1, 0.1]), "class1", true, 100, 10),
    ];

    // Train without momentum
    let mut nn_sgd = NeuralNetwork::new(2, &[4], 2).with_learning_rate(0.1);
    for _ in 0..20 {
        let _ = nn_sgd.train_batch(&samples);
    }
    let loss_sgd = nn_sgd.train_batch(&samples).unwrap();

    // Train with momentum (fresh network with same architecture)
    let mut nn_momentum = NeuralNetwork::new(2, &[4], 2)
        .with_learning_rate(0.1)
        .with_momentum(0.9);
    for _ in 0..20 {
        let _ = nn_momentum.train_batch(&samples);
    }
    let loss_momentum = nn_momentum.train_batch(&samples).unwrap();

    // Both should converge reasonably
    assert!(loss_sgd < 2.0);
    assert!(loss_momentum < 2.0);
}

#[test]
fn test_layer_gradients_accumulate() {
    let layer = Layer::new(3, 2, Activation::ReLU);
    let mut grads1 = LayerGradients::zeros(&layer);
    grads1.weight_gradients[0] = 1.0;
    grads1.bias_gradients[0] = 0.5;

    let mut grads2 = LayerGradients::zeros(&layer);
    grads2.weight_gradients[0] = 2.0;
    grads2.bias_gradients[0] = 0.3;

    grads1.accumulate(&grads2);

    assert!((grads1.weight_gradients[0] - 3.0).abs() < 1e-6);
    assert!((grads1.bias_gradients[0] - 0.8).abs() < 1e-6);
}

#[test]
fn test_layer_gradients_scale() {
    let layer = Layer::new(3, 2, Activation::ReLU);
    let mut grads = LayerGradients::zeros(&layer);
    grads.weight_gradients[0] = 4.0;
    grads.bias_gradients[0] = 2.0;

    grads.scale(0.5);

    assert!((grads.weight_gradients[0] - 2.0).abs() < 1e-6);
    assert!((grads.bias_gradients[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_restore_best_weights() {
    let mut nn = NeuralNetwork::new(3, &[4], 2).with_early_stopping(EarlyStoppingConfig {
        patience: 5,
        min_delta: 0.01,
    });

    // Record initial weights
    let _initial_weights = nn.layers[0].weights.clone();

    // Simulate training with improving loss
    nn.update_early_stopping(1.0);
    let _best_weights_at_1 = nn.layers[0].weights.clone();

    // Modify weights
    nn.layers[0].weights[0] = 999.0;

    // Simulate worse loss
    nn.update_early_stopping(0.5); // Better loss, saves new best

    // Modify weights again
    nn.layers[0].weights[0] = -999.0;

    // Restore best weights
    nn.restore_best_weights();

    // Weights should be restored (to the state at loss 0.5, not 1.0)
    assert!(nn.layers[0].weights[0] != -999.0);
}

#[test]
fn test_model_persistence() {
    let mut nn = NeuralNetwork::new(3, &[4, 2], 2)
        .with_learning_rate(0.05)
        .with_regularization(0.002);
    nn.set_source_ids(vec!["src1".to_string(), "src2".to_string()]);

    // Train a bit
    let features = create_test_features(&[0.5, 0.5, 0.5]);
    nn.train_step(&features, 0, 1.0);

    // Serialize
    let state = nn.to_state();
    let bytes = state.to_bytes();

    // Deserialize
    let restored_state = ModelState::from_bytes(&bytes).unwrap();
    let restored = NeuralNetwork::from_state(restored_state).unwrap();

    // Verify
    assert_eq!(restored.layers.len(), nn.layers.len());
    assert_eq!(restored.learning_rate, nn.learning_rate);
    assert_eq!(restored.regularization, nn.regularization);
    assert_eq!(restored.iterations, nn.iterations);

    for (orig, rest) in nn.layers.iter().zip(&restored.layers) {
        assert_eq!(orig.input_dim, rest.input_dim);
        assert_eq!(orig.output_dim, rest.output_dim);
        assert_eq!(orig.weights, rest.weights);
        assert_eq!(orig.biases, rest.biases);
    }
}
