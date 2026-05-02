//! Lightweight neural network for source selection with full backpropagation

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use super::activation::Activation;
use super::feature::FeatureVector;
use super::layer::{Layer, LayerCache, LayerGradients};
use super::model::{Model, ModelConfig, ModelPersistence, ModelState, ModelType, TrainingSample};
use super::optimizer::{AdamConfig, OptimizerState, OptimizerType};
#[cfg(feature = "dropout")]
use super::schedule::{DropoutConfig, DropoutState};
use super::schedule::{EarlyStoppingConfig, EarlyStoppingState, LearningRateSchedule};
use crate::core::error::{OxiRouterError, Result};

/// Lightweight neural network (MLP) for source selection with full backpropagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// Network layers
    layers: Vec<Layer>,
    /// Source ID mapping (index -> source_id)
    source_ids: Vec<String>,
    /// Learning rate
    learning_rate: f32,
    /// L2 regularization strength
    regularization: f32,
    /// Training iteration count
    iterations: u64,
    /// Current epoch for learning rate scheduling
    epoch: u64,
    /// Optimizer type
    optimizer: OptimizerType,
    /// Optimizer state (velocities, moments)
    #[serde(skip)]
    optimizer_state: Option<OptimizerState>,
    /// Learning rate schedule
    lr_schedule: LearningRateSchedule,
    /// Early stopping configuration
    early_stopping: Option<EarlyStoppingConfig>,
    /// Early stopping state
    #[serde(skip)]
    early_stopping_state: Option<EarlyStoppingState>,
    /// Dropout configuration
    #[cfg(feature = "dropout")]
    dropout: Option<DropoutConfig>,
    /// Dropout state
    #[cfg(feature = "dropout")]
    #[serde(skip)]
    dropout_state: Option<DropoutState>,
    /// Whether the network is in training mode
    #[serde(skip)]
    training: bool,
}

impl NeuralNetwork {
    /// Create a new neural network
    ///
    /// # Arguments
    ///
    /// * `feature_dim` - Input feature dimension
    /// * `hidden_sizes` - Sizes of hidden layers
    /// * `num_sources` - Number of output sources
    #[must_use]
    pub fn new(feature_dim: usize, hidden_sizes: &[usize], num_sources: usize) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = feature_dim;

        // Hidden layers with ReLU
        for &hidden_size in hidden_sizes {
            layers.push(Layer::new(prev_dim, hidden_size, Activation::ReLU));
            prev_dim = hidden_size;
        }

        // Output layer with Linear (softmax applied separately)
        layers.push(Layer::new(prev_dim, num_sources, Activation::Linear));

        Self {
            layers,
            source_ids: Vec::new(),
            learning_rate: 0.01,
            regularization: 0.001,
            iterations: 0,
            epoch: 0,
            optimizer: OptimizerType::default(),
            optimizer_state: None,
            lr_schedule: LearningRateSchedule::default(),
            early_stopping: None,
            early_stopping_state: None,
            #[cfg(feature = "dropout")]
            dropout: None,
            #[cfg(feature = "dropout")]
            dropout_state: None,
            training: false,
        }
    }

    /// Create from configuration
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        // Default architecture: input -> 32 -> 16 -> output
        let hidden_sizes = [32, 16];
        Self::new(config.feature_dim, &hidden_sizes, config.num_classes)
            .with_learning_rate(config.learning_rate)
            .with_regularization(config.regularization)
    }

    /// Set learning rate
    #[must_use]
    pub const fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set regularization strength
    #[must_use]
    pub const fn with_regularization(mut self, reg: f32) -> Self {
        self.regularization = reg;
        self
    }

    /// Set source IDs
    pub fn set_source_ids(&mut self, source_ids: Vec<String>) {
        self.source_ids = source_ids;
    }

    /// Set momentum coefficient (enables momentum optimizer)
    #[must_use]
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.optimizer = OptimizerType::Momentum {
            coefficient: momentum,
        };
        self.optimizer_state = None;
        self
    }

    /// Set Adam optimizer
    #[must_use]
    pub fn with_adam(mut self, config: AdamConfig) -> Self {
        self.optimizer = OptimizerType::Adam(config);
        self.optimizer_state = None;
        self
    }

    /// Set optimizer type
    #[must_use]
    pub fn with_optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer = optimizer;
        self.optimizer_state = None;
        self
    }

    /// Set learning rate decay (exponential decay per epoch)
    #[must_use]
    pub fn with_lr_decay(mut self, decay: f32) -> Self {
        self.lr_schedule = LearningRateSchedule::ExponentialDecay { decay };
        self
    }

    /// Set learning rate schedule
    #[must_use]
    pub fn with_lr_schedule(mut self, schedule: LearningRateSchedule) -> Self {
        self.lr_schedule = schedule;
        self
    }

    /// Enable early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = Some(config);
        self.early_stopping_state = Some(EarlyStoppingState::new());
        self
    }

    /// Enable dropout (feature-gated)
    #[cfg(feature = "dropout")]
    #[must_use]
    pub fn with_dropout(mut self, config: DropoutConfig) -> Self {
        self.dropout = Some(config);
        self.dropout_state = Some(DropoutState::default());
        self
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get current effective learning rate based on schedule
    #[must_use]
    pub fn current_learning_rate(&self) -> f32 {
        match &self.lr_schedule {
            LearningRateSchedule::Constant => self.learning_rate,
            LearningRateSchedule::ExponentialDecay { decay } => {
                self.learning_rate * pow_f32(*decay, self.epoch as f32)
            }
            LearningRateSchedule::StepDecay { drop, step_size } => {
                let steps = self.epoch / step_size;
                self.learning_rate * pow_f32(*drop, steps as f32)
            }
            LearningRateSchedule::CosineAnnealing { lr_min, t_max } => {
                let progress = (self.epoch as f32) / (*t_max as f32);
                let cos_val = cos_f32(core::f32::consts::PI * progress);
                lr_min + 0.5 * (self.learning_rate - lr_min) * (1.0 + cos_val)
            }
        }
    }

    /// Increment epoch counter (call after each training epoch)
    pub fn step_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Get current epoch
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Check if early stopping should trigger
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.early_stopping_state
            .as_ref()
            .is_some_and(|s| s.should_stop)
    }

    /// Update early stopping state with validation loss
    /// Returns true if early stopping is triggered
    pub fn update_early_stopping(&mut self, validation_loss: f32) -> bool {
        let Some(config) = &self.early_stopping else {
            return false;
        };
        let state = self
            .early_stopping_state
            .get_or_insert_with(EarlyStoppingState::new);

        if validation_loss < state.best_loss - config.min_delta {
            state.best_loss = validation_loss;
            state.epochs_without_improvement = 0;
            state.best_weights = Some(self.layers.iter().map(|l| l.weights.clone()).collect());
            state.best_biases = Some(self.layers.iter().map(|l| l.biases.clone()).collect());
        } else {
            state.epochs_without_improvement += 1;
            if state.epochs_without_improvement >= config.patience {
                state.should_stop = true;
            }
        }

        state.should_stop
    }

    /// Restore best weights from early stopping
    pub fn restore_best_weights(&mut self) {
        if let Some(state) = &self.early_stopping_state {
            if let (Some(weights), Some(biases)) = (&state.best_weights, &state.best_biases) {
                for (i, layer) in self.layers.iter_mut().enumerate() {
                    if let Some(w) = weights.get(i) {
                        layer.weights.clone_from(w);
                    }
                    if let Some(b) = biases.get(i) {
                        layer.biases.clone_from(b);
                    }
                }
            }
        }
    }

    /// Get number of training iterations
    #[must_use]
    pub const fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Get the layers (for testing/inspection)
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Forward pass through the network
    fn forward(&self, features: &FeatureVector) -> Vec<f32> {
        let mut current = features.values.clone();

        for layer in &self.layers {
            current = layer.forward(&current);
        }

        // Apply softmax to output
        self.softmax(&current)
    }

    /// Softmax normalization
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = logits
            .iter()
            .map(|&x| {
                #[cfg(feature = "ml")]
                {
                    libm::expf(x - max_logit)
                }
                #[cfg(not(feature = "ml"))]
                {
                    (x - max_logit).exp()
                }
            })
            .sum();

        logits
            .iter()
            .map(|&x| {
                #[cfg(feature = "ml")]
                {
                    libm::expf(x - max_logit) / exp_sum
                }
                #[cfg(not(feature = "ml"))]
                {
                    (x - max_logit).exp() / exp_sum
                }
            })
            .collect()
    }

    /// Forward pass with caching for backpropagation
    fn forward_with_cache(&self, features: &FeatureVector) -> (Vec<LayerCache>, Vec<f32>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut current = features.values.clone();

        #[cfg(feature = "dropout")]
        let dropout_active = self.training && self.dropout.is_some();

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = layer.forward_with_cache(&current);

            #[cfg(feature = "dropout")]
            {
                if dropout_active && i < self.layers.len() - 1 {
                    if let Some(state) = &self.dropout_state {
                        if let Some(mask) = state.masks.get(i) {
                            let scale = 1.0 / (1.0 - self.dropout.as_ref().map_or(0.0, |d| d.rate));
                            current = cache
                                .post_activation
                                .iter()
                                .zip(mask.iter())
                                .map(|(&a, &keep)| if keep { a * scale } else { 0.0 })
                                .collect();
                            caches.push(LayerCache {
                                input: cache.input,
                                pre_activation: cache.pre_activation,
                                post_activation: current.clone(),
                            });
                            continue;
                        }
                    }
                }
            }
            let _ = i;

            current.clone_from(&cache.post_activation);
            caches.push(cache);
        }

        let output = self.softmax(&current);
        (caches, output)
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, output: &[f32], target_idx: usize) -> f32 {
        let prob = output.get(target_idx).copied().unwrap_or(1e-7).max(1e-7);
        -ln_f32(prob)
    }

    /// Full backpropagation through all layers
    fn backpropagate(
        &self,
        caches: &[LayerCache],
        output: &[f32],
        target_idx: usize,
        reward: f32,
    ) -> Vec<LayerGradients> {
        let mut gradients = Vec::with_capacity(self.layers.len());

        let output_delta: Vec<f32> = output
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let target = if i == target_idx { 1.0 } else { 0.0 };
                reward * (p - target)
            })
            .collect();

        let mut delta = output_delta;

        for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
            let cache = &caches[layer_idx];
            let mut layer_grads = LayerGradients::zeros(layer);

            for i in 0..layer.output_dim {
                layer_grads.bias_gradients[i] = delta[i];

                for j in 0..layer.input_dim.min(cache.input.len()) {
                    let weight_idx = i * layer.input_dim + j;
                    layer_grads.weight_gradients[weight_idx] = delta[i] * cache.input[j];
                    layer_grads.weight_gradients[weight_idx] +=
                        self.regularization * layer.weights[weight_idx];
                }
            }

            gradients.push(layer_grads);

            if layer_idx > 0 {
                let prev_cache = &caches[layer_idx - 1];
                let prev_layer = &self.layers[layer_idx - 1];
                let mut new_delta = vec![0.0; layer.input_dim];

                for i in 0..layer.output_dim {
                    for j in 0..layer.input_dim {
                        let weight_idx = i * layer.input_dim + j;
                        new_delta[j] += delta[i] * layer.weights[weight_idx];
                    }
                }

                for (j, d) in new_delta.iter_mut().enumerate() {
                    if j < prev_cache.pre_activation.len() {
                        *d *= prev_layer
                            .activation
                            .derivative(prev_cache.pre_activation[j]);
                    }
                }

                delta = new_delta;
            }
        }

        gradients.reverse();
        gradients
    }

    /// Apply gradients using the configured optimizer
    fn apply_gradients(&mut self, gradients: &[LayerGradients]) {
        let lr = self.current_learning_rate();

        if self.optimizer_state.is_none() {
            self.optimizer_state = Some(OptimizerState::new(&self.layers, &self.optimizer));
        }

        match &self.optimizer {
            OptimizerType::SGD => {
                self.apply_sgd_gradients(gradients, lr);
            }
            OptimizerType::Momentum { coefficient } => {
                let momentum = *coefficient;
                self.apply_momentum_gradients(gradients, lr, momentum);
            }
            OptimizerType::Adam(config) => {
                let config = config.clone();
                self.apply_adam_gradients(gradients, lr, &config);
            }
        }
    }

    /// Apply gradients using vanilla SGD
    fn apply_sgd_gradients(&mut self, gradients: &[LayerGradients], lr: f32) {
        for (layer, grads) in self.layers.iter_mut().zip(gradients.iter()) {
            for (w, g) in layer.weights.iter_mut().zip(&grads.weight_gradients) {
                *w -= lr * g;
            }
            for (b, g) in layer.biases.iter_mut().zip(&grads.bias_gradients) {
                *b -= lr * g;
            }
        }
    }

    /// Apply gradients using momentum
    fn apply_momentum_gradients(&mut self, gradients: &[LayerGradients], lr: f32, momentum: f32) {
        let state = self
            .optimizer_state
            .as_mut()
            .expect("optimizer state must be initialized before apply_momentum_gradients");

        for (layer_idx, (layer, grads)) in self.layers.iter_mut().zip(gradients.iter()).enumerate()
        {
            for (i, (w, g)) in layer
                .weights
                .iter_mut()
                .zip(&grads.weight_gradients)
                .enumerate()
            {
                let v = &mut state.weight_velocities[layer_idx][i];
                *v = momentum * *v - lr * g;
                *w += *v;
            }

            for (i, (b, g)) in layer
                .biases
                .iter_mut()
                .zip(&grads.bias_gradients)
                .enumerate()
            {
                let v = &mut state.bias_velocities[layer_idx][i];
                *v = momentum * *v - lr * g;
                *b += *v;
            }
        }
    }

    /// Apply gradients using Adam optimizer
    fn apply_adam_gradients(&mut self, gradients: &[LayerGradients], lr: f32, config: &AdamConfig) {
        let state = self
            .optimizer_state
            .as_mut()
            .expect("optimizer state must be initialized before apply_adam_gradients");
        state.t += 1;
        let t = state.t as f32;

        let bias_correction1 = 1.0 - pow_f32(config.beta1, t);
        let bias_correction2 = 1.0 - pow_f32(config.beta2, t);

        for (layer_idx, (layer, grads)) in self.layers.iter_mut().zip(gradients.iter()).enumerate()
        {
            for (i, (w, g)) in layer
                .weights
                .iter_mut()
                .zip(&grads.weight_gradients)
                .enumerate()
            {
                let m = &mut state.weight_m[layer_idx][i];
                *m = config.beta1 * *m + (1.0 - config.beta1) * g;

                let v = &mut state.weight_v[layer_idx][i];
                *v = config.beta2 * *v + (1.0 - config.beta2) * g * g;

                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;

                *w -= lr * m_hat / (sqrt_f32(v_hat) + config.epsilon);
            }

            for (i, (b, g)) in layer
                .biases
                .iter_mut()
                .zip(&grads.bias_gradients)
                .enumerate()
            {
                let m = &mut state.bias_m[layer_idx][i];
                *m = config.beta1 * *m + (1.0 - config.beta1) * g;

                let v = &mut state.bias_v[layer_idx][i];
                *v = config.beta2 * *v + (1.0 - config.beta2) * g * g;

                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;

                *b -= lr * m_hat / (sqrt_f32(v_hat) + config.epsilon);
            }
        }
    }

    /// Full training step with backpropagation through all layers
    fn train_step(&mut self, features: &FeatureVector, target_idx: usize, reward: f32) -> f32 {
        self.training = true;

        #[cfg(feature = "dropout")]
        if let Some(dropout_config) = &self.dropout {
            let state = self.dropout_state.get_or_insert_with(DropoutState::default);
            let seed = dropout_config.seed.wrapping_add(self.iterations);
            state.generate_masks(&self.layers, dropout_config.rate, seed);
        }

        let (caches, output) = self.forward_with_cache(features);
        let loss = self.compute_loss(&output, target_idx);
        let gradients = self.backpropagate(&caches, &output, target_idx, reward);
        self.apply_gradients(&gradients);

        self.iterations += 1;
        self.training = false;

        loss
    }

    /// Train on a batch of samples and return average loss
    ///
    /// # Errors
    ///
    /// Returns error if training fails
    pub fn train_batch(&mut self, samples: &[TrainingSample]) -> Result<f32> {
        if samples.is_empty() {
            return Ok(0.0);
        }

        self.training = true;

        let mut source_index = hashbrown::HashMap::new();
        for sample in samples {
            if !source_index.contains_key(&sample.selected_source) {
                let idx = source_index.len();
                source_index.insert(sample.selected_source.clone(), idx);
            }
        }

        let mut accumulated_gradients: Vec<LayerGradients> =
            self.layers.iter().map(LayerGradients::zeros).collect();
        let mut total_loss = 0.0;
        let batch_size = samples.len();

        for sample in samples {
            if let Some(&target_idx) = source_index.get(&sample.selected_source) {
                let reward = sample.reward();

                #[cfg(feature = "dropout")]
                if let Some(dropout_config) = &self.dropout {
                    let state = self.dropout_state.get_or_insert_with(DropoutState::default);
                    let seed = dropout_config.seed.wrapping_add(self.iterations);
                    state.generate_masks(&self.layers, dropout_config.rate, seed);
                }

                let (caches, output) = self.forward_with_cache(&sample.features);
                total_loss += self.compute_loss(&output, target_idx);
                let gradients = self.backpropagate(&caches, &output, target_idx, reward);

                for (acc, grad) in accumulated_gradients.iter_mut().zip(&gradients) {
                    acc.accumulate(grad);
                }

                self.iterations += 1;
            }
        }

        let scale = 1.0 / batch_size as f32;
        for grads in &mut accumulated_gradients {
            grads.scale(scale);
        }

        self.apply_gradients(&accumulated_gradients);

        self.training = false;

        Ok(total_loss / batch_size as f32)
    }
}

impl Model for NeuralNetwork {
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, features, source_ids),
            fields(input_dim = features.values.len())
        )
    )]
    fn predict(
        &self,
        features: &FeatureVector,
        source_ids: &[&String],
    ) -> Result<Vec<(String, f32)>> {
        if source_ids.is_empty() {
            return Err(OxiRouterError::ModelError(
                "No sources provided".to_string(),
            ));
        }

        let dim = self.feature_dim();
        if dim > 0 && features.values.len() != dim {
            return Err(OxiRouterError::FeatureDimMismatch {
                expected: dim,
                found: features.values.len(),
            });
        }

        #[cfg(all(feature = "observability", feature = "std"))]
        let predict_start = std::time::Instant::now();

        let probabilities = self.forward(features);

        #[cfg(all(feature = "observability", feature = "std"))]
        {
            let elapsed_us = predict_start.elapsed().as_micros() as f64;
            metrics::histogram!("oxirouter.ml.predict.duration_us", "model" => "neural")
                .record(elapsed_us);
        }

        // Map probabilities to source IDs
        let mut results: Vec<(String, f32)> = source_ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let prob = probabilities.get(i).copied().unwrap_or(0.0);
                ((*id).clone(), prob)
            })
            .collect();

        // Sort by probability (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        Ok(results)
    }

    fn name(&self) -> &str {
        "NeuralNetwork"
    }

    fn feature_dim(&self) -> usize {
        if let Some(first_layer) = self.layers.first() {
            first_layer.input_dim
        } else {
            0
        }
    }

    fn train(&mut self, samples: &[TrainingSample]) -> Result<()> {
        // Build source ID index
        let mut source_index = hashbrown::HashMap::new();
        for sample in samples {
            if !source_index.contains_key(&sample.selected_source) {
                let idx = source_index.len();
                source_index.insert(sample.selected_source.clone(), idx);
            }
        }

        // Train on samples with full backpropagation
        for sample in samples {
            if let Some(&idx) = source_index.get(&sample.selected_source) {
                let reward = sample.reward();
                let _loss = self.train_step(&sample.features, idx, reward);
            }
        }

        Ok(())
    }

    fn update(&mut self, features: &FeatureVector, source_id: &str, reward: f32) -> Result<()> {
        // Find source index
        let idx = self
            .source_ids
            .iter()
            .position(|s| s == source_id)
            .unwrap_or(0);

        self.train_step(features, idx, reward);
        Ok(())
    }

    fn to_bytes(&self) -> Vec<u8> {
        <Self as ModelPersistence>::to_bytes(self)
    }

    fn model_type(&self) -> &'static str {
        "neural"
    }
}

impl ModelPersistence for NeuralNetwork {
    fn to_state(&self) -> ModelState {
        // Collect all weights from all layers (flattened)
        let mut weights = Vec::new();
        for layer in &self.layers {
            weights.extend_from_slice(&layer.weights);
        }

        // Collect all biases in extra_params
        let mut extra_params = Vec::new();
        // First, add learning_rate and regularization
        extra_params.push(self.learning_rate);
        extra_params.push(self.regularization);
        // Then add all biases
        for layer in &self.layers {
            extra_params.extend_from_slice(&layer.biases);
        }

        // Collect layer dimensions
        let layer_dims: Vec<(usize, usize)> = self
            .layers
            .iter()
            .map(|l| (l.input_dim, l.output_dim))
            .collect();

        // Collect activation types
        let activation_types: Vec<u8> =
            self.layers.iter().map(|l| l.activation.to_byte()).collect();

        // Determine feature_dim and num_classes
        let feature_dim = self.layers.first().map(|l| l.input_dim).unwrap_or(0);
        let num_classes = self.layers.last().map(|l| l.output_dim).unwrap_or(0);

        let config = ModelConfig {
            model_type: ModelType::NeuralNetwork,
            feature_dim,
            num_classes,
            learning_rate: self.learning_rate,
            regularization: self.regularization,
        };

        ModelState {
            config,
            weights,
            source_ids: self.source_ids.clone(),
            iterations: self.iterations,
            extra_params,
            layer_dims,
            activation_types,
            optimizer_type: Some(self.optimizer.clone()),
            optimizer_state: self.optimizer_state.clone(),
            lr_schedule: Some(self.lr_schedule.clone()),
            epoch: self.epoch,
            early_stopping_config: self.early_stopping.clone(),
            early_stopping_state: self.early_stopping_state.clone(),
        }
    }

    fn from_state(state: ModelState) -> Result<Self> {
        if state.config.model_type != ModelType::NeuralNetwork {
            return Err(OxiRouterError::ModelError(format!(
                "Expected NeuralNetwork model type, got {:?}",
                state.config.model_type
            )));
        }

        if state.layer_dims.is_empty() {
            return Err(OxiRouterError::ModelError(
                "No layer dimensions in model state".to_string(),
            ));
        }

        if state.layer_dims.len() != state.activation_types.len() {
            return Err(OxiRouterError::ModelError(format!(
                "Layer dims count ({}) != activation types count ({})",
                state.layer_dims.len(),
                state.activation_types.len()
            )));
        }

        // Extract learning_rate and regularization from extra_params
        let learning_rate = state.extra_params.first().copied().unwrap_or(0.01);
        let regularization = state.extra_params.get(1).copied().unwrap_or(0.001);

        // Reconstruct layers
        let mut layers = Vec::with_capacity(state.layer_dims.len());
        let mut weight_pos = 0;
        let mut bias_pos = 2; // Skip learning_rate and regularization

        for (i, &(input_dim, output_dim)) in state.layer_dims.iter().enumerate() {
            let weight_count = input_dim * output_dim;
            let bias_count = output_dim;

            // Validate we have enough weights
            if weight_pos + weight_count > state.weights.len() {
                return Err(OxiRouterError::ModelError(format!(
                    "Not enough weights for layer {}: need {} at pos {}, have {}",
                    i,
                    weight_count,
                    weight_pos,
                    state.weights.len()
                )));
            }

            // Validate we have enough biases
            if bias_pos + bias_count > state.extra_params.len() {
                return Err(OxiRouterError::ModelError(format!(
                    "Not enough biases for layer {}: need {} at pos {}, have {}",
                    i,
                    bias_count,
                    bias_pos,
                    state.extra_params.len()
                )));
            }

            let layer_weights = state.weights[weight_pos..weight_pos + weight_count].to_vec();
            let layer_biases = state.extra_params[bias_pos..bias_pos + bias_count].to_vec();
            let activation = Activation::from_byte(state.activation_types[i]);

            layers.push(Layer::from_weights(
                input_dim,
                output_dim,
                activation,
                layer_weights,
                layer_biases,
            ));

            weight_pos += weight_count;
            bias_pos += bias_count;
        }

        Ok(Self {
            layers,
            source_ids: state.source_ids,
            learning_rate,
            regularization,
            iterations: state.iterations,
            epoch: state.epoch,
            optimizer: state.optimizer_type.unwrap_or_default(),
            optimizer_state: state.optimizer_state,
            lr_schedule: state.lr_schedule.unwrap_or_default(),
            early_stopping: state.early_stopping_config,
            early_stopping_state: state.early_stopping_state,
            #[cfg(feature = "dropout")]
            dropout: None,
            #[cfg(feature = "dropout")]
            dropout_state: None,
            training: false,
        })
    }
}

// Math helper functions (no_std compatible using libm)

#[inline]
fn ln_f32(x: f32) -> f32 {
    #[cfg(feature = "ml")]
    {
        libm::logf(x)
    }
    #[cfg(not(feature = "ml"))]
    {
        x.ln()
    }
}

#[inline]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "ml")]
    {
        libm::sqrtf(x)
    }
    #[cfg(not(feature = "ml"))]
    {
        x.sqrt()
    }
}

#[inline]
fn pow_f32(base: f32, exp: f32) -> f32 {
    #[cfg(feature = "ml")]
    {
        libm::powf(base, exp)
    }
    #[cfg(not(feature = "ml"))]
    {
        base.powf(exp)
    }
}

#[inline]
fn cos_f32(x: f32) -> f32 {
    #[cfg(feature = "ml")]
    {
        libm::cosf(x)
    }
    #[cfg(not(feature = "ml"))]
    {
        x.cos()
    }
}

#[cfg(test)]
#[path = "neural_tests.rs"]
mod tests;
