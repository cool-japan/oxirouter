//! Federated model weight sharing strategies.
//!
//! Provides utilities for merging ML model parameters across distributed edge nodes
//! without sharing raw training data (federated averaging paradigm).

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString};

use crate::core::error::{OxiRouterError, Result};
use crate::ml::{ModelState, ModelType};

/// Strategy for merging a remote model's weights into the local model.
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Average local and remote weights element-wise: `local = (local + remote) / 2`.
    Average,
    /// Weighted average: `local = (1-w)*local + w*remote`. `w` is clamped to [0, 1].
    WeightedAverage(f32),
    /// Replace local with remote if remote has more training iterations.
    KeepLatest,
    /// Replace local with remote if remote has higher accumulated reward.
    ///
    /// Convention: the last element of `extra_params` stores total accumulated reward.
    KeepBest,
}

/// Merge a remote model state into a local model state using the given strategy.
///
/// # Errors
///
/// Returns [`OxiRouterError::IncompatibleModel`] if:
/// - Model types differ between local and remote.
/// - Feature dimensions differ.
/// - Source ID lists differ (order-sensitive).
/// - Weight vector lengths differ (should not happen if types/dims match).
pub fn merge_states(
    local: &mut ModelState,
    remote: &ModelState,
    strategy: MergeStrategy,
) -> Result<()> {
    // Validate compatibility
    if local.config.model_type != remote.config.model_type {
        return Err(OxiRouterError::IncompatibleModel {
            reason: format!(
                "model_type mismatch: local={:?}, remote={:?}",
                local.config.model_type, remote.config.model_type
            ),
        });
    }

    // Ensemble-of-Ensemble merging is not supported (depth limit).
    if local.config.model_type == ModelType::Ensemble {
        return Err(OxiRouterError::IncompatibleModel {
            reason:
                "Merging Ensemble models is not supported; merge individual components separately"
                    .to_string(),
        });
    }

    if local.config.feature_dim != remote.config.feature_dim {
        return Err(OxiRouterError::IncompatibleModel {
            reason: format!(
                "feature_dim mismatch: local={}, remote={}",
                local.config.feature_dim, remote.config.feature_dim
            ),
        });
    }

    if local.source_ids != remote.source_ids {
        return Err(OxiRouterError::IncompatibleModel {
            reason: format!(
                "source_ids mismatch: local has {} sources, remote has {}",
                local.source_ids.len(),
                remote.source_ids.len()
            ),
        });
    }

    if local.weights.len() != remote.weights.len() {
        return Err(OxiRouterError::IncompatibleModel {
            reason: format!(
                "weights length mismatch: local={}, remote={}",
                local.weights.len(),
                remote.weights.len()
            ),
        });
    }

    match strategy {
        MergeStrategy::Average => {
            for (l, r) in local.weights.iter_mut().zip(&remote.weights) {
                *l = (*l + *r) * 0.5;
            }
            // Also average extra_params for things like smoothing / sample counts
            for (l, r) in local.extra_params.iter_mut().zip(&remote.extra_params) {
                *l = (*l + *r) * 0.5;
            }
        }
        MergeStrategy::WeightedAverage(w) => {
            let w = w.clamp(0.0, 1.0);
            for (l, r) in local.weights.iter_mut().zip(&remote.weights) {
                *l = (1.0 - w) * *l + w * *r;
            }
        }
        MergeStrategy::KeepLatest => {
            if remote.iterations > local.iterations {
                *local = remote.clone();
            }
        }
        MergeStrategy::KeepBest => {
            // Convention: last extra_param element stores total accumulated reward.
            let local_reward = local.extra_params.last().copied().unwrap_or(0.0);
            let remote_reward = remote.extra_params.last().copied().unwrap_or(0.0);
            if remote_reward > local_reward {
                *local = remote.clone();
            }
        }
    }

    Ok(())
}
