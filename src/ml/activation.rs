//! Activation functions for neural network layers

use serde::{Deserialize, Serialize};

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// No activation (linear)
    Linear,
}

impl Activation {
    /// Apply activation function
    #[must_use]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => {
                #[cfg(feature = "ml")]
                {
                    1.0 / (1.0 + libm::expf(-x))
                }
                #[cfg(not(feature = "ml"))]
                {
                    1.0 / (1.0 + (-x).exp())
                }
            }
            Self::Tanh => {
                #[cfg(feature = "ml")]
                {
                    libm::tanhf(x)
                }
                #[cfg(not(feature = "ml"))]
                {
                    x.tanh()
                }
            }
            Self::Linear => x,
        }
    }

    /// Apply derivative for backpropagation
    /// Takes the pre-activation value (z)
    #[must_use]
    pub fn derivative(&self, z: f32) -> f32 {
        match self {
            Self::ReLU => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sigmoid => {
                let s = self.apply(z);
                s * (1.0 - s)
            }
            Self::Tanh => {
                let t = self.apply(z);
                1.0 - t * t
            }
            Self::Linear => 1.0,
        }
    }

    /// Convert activation to byte representation for serialization
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::ReLU => 0,
            Self::Sigmoid => 1,
            Self::Tanh => 2,
            Self::Linear => 3,
        }
    }

    /// Create activation from byte representation
    #[must_use]
    pub const fn from_byte(byte: u8) -> Self {
        match byte {
            0 => Self::ReLU,
            1 => Self::Sigmoid,
            2 => Self::Tanh,
            _ => Self::Linear,
        }
    }
}
