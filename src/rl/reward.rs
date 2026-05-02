//! Reward calculation for reinforcement learning

use serde::{Deserialize, Serialize};

use super::Feedback;

/// Reward signal from query execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Reward {
    /// Raw reward value (0.0 - 1.0)
    raw: f32,
    /// Latency component of reward
    latency_component: f32,
    /// Success component of reward
    success_component: f32,
    /// Result count component
    result_component: f32,
}

impl Reward {
    /// Create a new reward
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self {
            raw: value,
            latency_component: 0.0,
            success_component: value,
            result_component: 0.0,
        }
    }

    /// Create reward from feedback
    #[must_use]
    pub fn from_feedback(feedback: &Feedback) -> Self {
        if !feedback.success {
            return Self::failure(feedback.is_transient_failure());
        }

        // Success component (base 0.5)
        let success_component = 0.5;

        // Latency component (0.0 - 0.3)
        // Fast: < 100ms = 0.3
        // Medium: 100-1000ms = 0.1-0.3
        // Slow: > 1000ms = 0.0-0.1
        let latency_component = if feedback.latency_ms < 100 {
            0.3
        } else if feedback.latency_ms < 1000 {
            0.3 * (1.0 - (feedback.latency_ms - 100) as f32 / 900.0)
        } else {
            0.1 * (1.0 - ((feedback.latency_ms - 1000) as f32 / 9000.0).min(1.0))
        };

        // Result component (0.0 - 0.2)
        let result_component = if feedback.result_count > 0 {
            let log_results = (feedback.result_count as f32).ln();
            (log_results / 10.0).min(0.2)
        } else {
            0.0
        };

        let raw = (success_component + latency_component + result_component).min(1.0);

        Self {
            raw,
            latency_component,
            success_component,
            result_component,
        }
    }

    /// Create reward for a failure
    #[must_use]
    pub const fn failure(transient: bool) -> Self {
        // Transient failures get small reward (might work next time)
        // Permanent failures get zero
        let raw = if transient { 0.1 } else { 0.0 };
        Self {
            raw,
            latency_component: 0.0,
            success_component: raw,
            result_component: 0.0,
        }
    }

    /// Create maximum reward
    #[must_use]
    pub const fn max() -> Self {
        Self {
            raw: 1.0,
            latency_component: 0.3,
            success_component: 0.5,
            result_component: 0.2,
        }
    }

    /// Create minimum reward
    #[must_use]
    pub const fn min() -> Self {
        Self {
            raw: 0.0,
            latency_component: 0.0,
            success_component: 0.0,
            result_component: 0.0,
        }
    }

    /// Get the reward value
    #[must_use]
    pub const fn value(&self) -> f32 {
        self.raw
    }

    /// Get latency component
    #[must_use]
    pub const fn latency(&self) -> f32 {
        self.latency_component
    }

    /// Get success component
    #[must_use]
    pub const fn success(&self) -> f32 {
        self.success_component
    }

    /// Get result count component
    #[must_use]
    pub const fn results(&self) -> f32 {
        self.result_component
    }

    /// Check if reward is positive (good outcome)
    #[must_use]
    pub const fn is_positive(&self) -> bool {
        self.raw > 0.5
    }

    /// Check if reward is negative (bad outcome)
    #[must_use]
    pub const fn is_negative(&self) -> bool {
        self.raw < 0.5
    }

    /// Apply discount factor for temporal difference learning
    #[must_use]
    pub const fn discounted(&self, gamma: f32) -> f32 {
        self.raw * gamma
    }

    /// Combine with another reward (weighted average)
    #[must_use]
    pub fn combine(&self, other: &Self, weight: f32) -> Self {
        let w = weight.clamp(0.0, 1.0);
        Self {
            raw: self.raw * (1.0 - w) + other.raw * w,
            latency_component: self.latency_component * (1.0 - w) + other.latency_component * w,
            success_component: self.success_component * (1.0 - w) + other.success_component * w,
            result_component: self.result_component * (1.0 - w) + other.result_component * w,
        }
    }
}

impl Default for Reward {
    fn default() -> Self {
        Self::new(0.5) // Neutral reward
    }
}

/// Reward shaping configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct RewardConfig {
    /// Weight for latency in reward calculation
    pub latency_weight: f32,
    /// Weight for success/failure
    pub success_weight: f32,
    /// Weight for result count
    pub result_weight: f32,
    /// Discount factor for future rewards
    pub gamma: f32,
    /// Penalty for transient failures
    pub transient_penalty: f32,
    /// Penalty for permanent failures
    pub permanent_penalty: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            latency_weight: 0.3,
            success_weight: 0.5,
            result_weight: 0.2,
            gamma: 0.99,
            transient_penalty: 0.9,
            permanent_penalty: 0.0,
        }
    }
}

#[allow(dead_code)]
impl RewardConfig {
    /// Calculate reward using this configuration
    #[must_use]
    pub fn calculate(&self, feedback: &Feedback) -> Reward {
        if !feedback.success {
            let penalty = if feedback.is_transient_failure() {
                self.transient_penalty
            } else {
                self.permanent_penalty
            };
            return Reward::new(penalty);
        }

        // Normalize latency (0-10 seconds -> 0-1)
        let latency_norm = 1.0 - (feedback.latency_ms as f32 / 10000.0).min(1.0);

        // Normalize results (log scale)
        let result_norm = if feedback.result_count > 0 {
            ((feedback.result_count as f32).ln() / 10.0).min(1.0)
        } else {
            0.0
        };

        let raw = self.success_weight
            + self.latency_weight * latency_norm
            + self.result_weight * result_norm;

        Reward::new(raw.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_from_feedback() {
        let success = Feedback::success("src1", 1, 100, 50);
        let reward = Reward::from_feedback(&success);
        assert!(reward.is_positive());
        assert!(reward.value() > 0.5);
    }

    #[test]
    fn test_reward_failure() {
        let failure = Feedback::failure("src1", 1, "Error");
        let reward = Reward::from_feedback(&failure);
        assert!(reward.is_negative());
    }

    #[test]
    fn test_reward_latency_impact() {
        let fast = Feedback::success("src1", 1, 50, 100);
        let slow = Feedback::success("src1", 2, 5000, 100);

        let fast_reward = Reward::from_feedback(&fast);
        let slow_reward = Reward::from_feedback(&slow);

        assert!(fast_reward.value() > slow_reward.value());
    }

    #[test]
    fn test_reward_combine() {
        let high = Reward::new(1.0);
        let low = Reward::new(0.0);

        let combined = high.combine(&low, 0.5);
        assert!((combined.value() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_reward_config() {
        let config = RewardConfig::default();
        let feedback = Feedback::success("src1", 1, 100, 100);
        let reward = config.calculate(&feedback);

        assert!(reward.is_positive());
    }

    #[test]
    fn test_discount() {
        let reward = Reward::new(1.0);
        let discounted = reward.discounted(0.9);
        assert!((discounted - 0.9).abs() < 0.01);
    }
}
