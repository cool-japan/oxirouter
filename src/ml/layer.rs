//! Neural network layer types and forward-pass caching

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use serde::{Deserialize, Serialize};

use super::activation::Activation;

/// A single layer in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Weight matrix (output_size x input_size, flattened)
    pub weights: Vec<f32>,
    /// Bias vector
    pub biases: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with random initialization
    #[must_use]
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        // Xavier/Glorot initialization
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        let mut weights = Vec::with_capacity(input_dim * output_dim);
        let mut biases = Vec::with_capacity(output_dim);

        // Simple PRNG for initialization (no rand dependency)
        let mut seed = 12345u64;
        for _ in 0..(input_dim * output_dim) {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
            weights.push(rand_val * scale);
        }

        for _ in 0..output_dim {
            biases.push(0.0);
        }

        Self {
            weights,
            biases,
            input_dim,
            output_dim,
            activation,
        }
    }

    /// Forward pass through the layer
    #[must_use]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.output_dim);

        for i in 0..self.output_dim {
            let mut sum = self.biases[i];
            for j in 0..self.input_dim.min(input.len()) {
                sum += self.weights[i * self.input_dim + j] * input[j];
            }
            output.push(self.activation.apply(sum));
        }

        output
    }

    /// Forward pass returning both pre-activation (z) and post-activation (a) values
    #[must_use]
    pub fn forward_with_cache(&self, input: &[f32]) -> LayerCache {
        let mut pre_activation = Vec::with_capacity(self.output_dim);
        let mut post_activation = Vec::with_capacity(self.output_dim);

        for i in 0..self.output_dim {
            let mut sum = self.biases[i];
            for j in 0..self.input_dim.min(input.len()) {
                sum += self.weights[i * self.input_dim + j] * input[j];
            }
            pre_activation.push(sum);
            post_activation.push(self.activation.apply(sum));
        }

        LayerCache {
            input: input.to_vec(),
            pre_activation,
            post_activation,
        }
    }

    /// Get weight for specific connection
    #[must_use]
    pub fn get_weight(&self, output_idx: usize, input_idx: usize) -> f32 {
        self.weights[output_idx * self.input_dim + input_idx]
    }

    /// Set weight for specific connection
    pub fn set_weight(&mut self, output_idx: usize, input_idx: usize, value: f32) {
        self.weights[output_idx * self.input_dim + input_idx] = value;
    }

    /// Create a layer from pre-existing weights and biases
    #[must_use]
    pub fn from_weights(
        input_dim: usize,
        output_dim: usize,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) -> Self {
        Self {
            weights,
            biases,
            input_dim,
            output_dim,
            activation,
        }
    }
}

/// Cached values from forward pass for backpropagation
#[derive(Debug, Clone)]
pub struct LayerCache {
    /// Input to this layer
    pub input: Vec<f32>,
    /// Pre-activation values (z = Wx + b)
    pub pre_activation: Vec<f32>,
    /// Post-activation values (a = activation(z))
    pub post_activation: Vec<f32>,
}

/// Gradients for a single layer
#[derive(Debug, Clone)]
pub struct LayerGradients {
    /// Weight gradients
    pub weight_gradients: Vec<f32>,
    /// Bias gradients
    pub bias_gradients: Vec<f32>,
}

impl LayerGradients {
    /// Create zero gradients for a layer
    #[must_use]
    pub fn zeros(layer: &Layer) -> Self {
        Self {
            weight_gradients: vec![0.0; layer.weights.len()],
            bias_gradients: vec![0.0; layer.biases.len()],
        }
    }

    /// Accumulate gradients from another set
    pub fn accumulate(&mut self, other: &Self) {
        for (g, o) in self
            .weight_gradients
            .iter_mut()
            .zip(&other.weight_gradients)
        {
            *g += o;
        }
        for (g, o) in self.bias_gradients.iter_mut().zip(&other.bias_gradients) {
            *g += o;
        }
    }

    /// Scale gradients by a factor
    pub fn scale(&mut self, factor: f32) {
        for g in &mut self.weight_gradients {
            *g *= factor;
        }
        for g in &mut self.bias_gradients {
            *g *= factor;
        }
    }
}
