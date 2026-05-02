//! Learning rate schedules, early stopping, and dropout configuration

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

#[cfg(feature = "dropout")]
use super::layer::Layer;

/// Learning rate schedule type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Exponential decay: lr = lr_0 * decay^epoch
    ExponentialDecay {
        /// Decay factor per epoch
        decay: f32,
    },
    /// Step decay: lr = lr_0 * drop^floor(epoch/step_size)
    StepDecay {
        /// Factor to multiply by at each step
        drop: f32,
        /// Number of epochs between drops
        step_size: u64,
    },
    /// Cosine annealing: lr = lr_min + 0.5*(lr_0 - lr_min)*(1 + cos(pi*epoch/T_max))
    CosineAnnealing {
        /// Minimum learning rate
        lr_min: f32,
        /// Maximum number of epochs
        t_max: u64,
    },
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::Constant
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait for improvement before stopping
    pub patience: u64,
    /// Minimum change to qualify as improvement
    pub min_delta: f32,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 1e-4,
        }
    }
}

/// Early stopping state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EarlyStoppingState {
    /// Best validation loss seen
    pub best_loss: f32,
    /// Number of epochs without improvement
    pub epochs_without_improvement: u64,
    /// Whether early stopping has been triggered
    pub should_stop: bool,
    /// Best model weights (serialized)
    pub best_weights: Option<Vec<Vec<f32>>>,
    /// Best model biases
    pub best_biases: Option<Vec<Vec<f32>>>,
}

impl EarlyStoppingState {
    /// Create new early stopping state
    #[must_use]
    pub fn new() -> Self {
        Self {
            best_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            should_stop: false,
            best_weights: None,
            best_biases: None,
        }
    }
}

/// Dropout configuration
#[cfg(feature = "dropout")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Dropout probability (0.0 to 1.0)
    pub rate: f32,
    /// Random seed for reproducibility
    pub seed: u64,
}

#[cfg(feature = "dropout")]
impl Default for DropoutConfig {
    fn default() -> Self {
        Self {
            rate: 0.5,
            seed: 42,
        }
    }
}

/// Dropout state during training
#[cfg(feature = "dropout")]
#[derive(Debug, Clone, Default)]
pub struct DropoutState {
    /// Dropout masks for each layer
    pub masks: Vec<Vec<bool>>,
    /// Current random state
    pub rng_state: u64,
}

#[cfg(feature = "dropout")]
impl DropoutState {
    /// Create dropout masks for all layers
    pub fn generate_masks(&mut self, layers: &[Layer], rate: f32, seed: u64) {
        self.rng_state = seed;
        self.masks.clear();

        for layer in layers {
            let mut mask = Vec::with_capacity(layer.output_dim);
            for _ in 0..layer.output_dim {
                // Simple LCG PRNG
                self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let rand_val = (self.rng_state as f32) / (u64::MAX as f32);
                mask.push(rand_val > rate);
            }
            self.masks.push(mask);
        }
    }
}
