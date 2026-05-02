//! Optimizer types and state for neural network training

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use serde::{Deserialize, Serialize};

use super::layer::Layer;

/// Optimizer type for neural network training
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    Momentum {
        /// Momentum coefficient (default 0.9)
        coefficient: f32,
    },
    /// Adam optimizer
    Adam(AdamConfig),
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::SGD
    }
}

/// Adam optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamConfig {
    /// Exponential decay rate for first moment (default 0.9)
    pub beta1: f32,
    /// Exponential decay rate for second moment (default 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (default 1e-8)
    pub epsilon: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Optimizer state for momentum-based methods
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Velocity vectors for momentum (one per layer)
    pub weight_velocities: Vec<Vec<f32>>,
    /// Bias velocities for momentum
    pub bias_velocities: Vec<Vec<f32>>,
    /// First moment estimates for Adam (m)
    pub weight_m: Vec<Vec<f32>>,
    /// First moment estimates for bias
    pub bias_m: Vec<Vec<f32>>,
    /// Second moment estimates for Adam (v)
    pub weight_v: Vec<Vec<f32>>,
    /// Second moment estimates for bias
    pub bias_v: Vec<Vec<f32>>,
    /// Time step for Adam bias correction
    pub t: u64,
}

impl OptimizerState {
    /// Initialize optimizer state for given layers
    #[must_use]
    pub fn new(layers: &[Layer], optimizer: &OptimizerType) -> Self {
        let mut state = Self::default();

        match optimizer {
            OptimizerType::SGD => {}
            OptimizerType::Momentum { .. } => {
                state.weight_velocities =
                    layers.iter().map(|l| vec![0.0; l.weights.len()]).collect();
                state.bias_velocities = layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
            }
            OptimizerType::Adam(_) => {
                state.weight_m = layers.iter().map(|l| vec![0.0; l.weights.len()]).collect();
                state.bias_m = layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
                state.weight_v = layers.iter().map(|l| vec![0.0; l.weights.len()]).collect();
                state.bias_v = layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
            }
        }

        state
    }
}
