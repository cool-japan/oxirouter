//! Policy update logic for reinforcement learning

#[cfg(feature = "alloc")]
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::reward::Reward;

/// Pseudo-random generator for policy exploration decisions.
#[derive(Debug, Clone)]
pub(crate) struct PolicyRng {
    #[cfg(feature = "std")]
    inner: rand::rngs::SmallRng,
    #[cfg(not(feature = "std"))]
    state: u64,
}

impl PolicyRng {
    /// Create a new RNG, seeded from entropy (std) or a constant (no_std).
    fn new() -> Self {
        #[cfg(feature = "std")]
        {
            use rand::SeedableRng;
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xcafe_babe_dead_beef);
            Self {
                inner: rand::rngs::SmallRng::seed_from_u64(seed),
            }
        }
        #[cfg(not(feature = "std"))]
        {
            Self {
                state: 0xdead_beef_cafe_babe,
            }
        }
    }

    /// Create with a fixed seed (for reproducible tests).
    pub(crate) fn with_seed(seed: u64) -> Self {
        #[cfg(feature = "std")]
        {
            use rand::SeedableRng;
            Self {
                inner: rand::rngs::SmallRng::seed_from_u64(seed),
            }
        }
        #[cfg(not(feature = "std"))]
        {
            Self {
                state: seed ^ 0xdead_beef_cafe_babe,
            }
        }
    }

    /// Sample a float in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        #[cfg(feature = "std")]
        {
            use rand::RngExt;
            self.inner.random::<f32>()
        }
        #[cfg(not(feature = "std"))]
        {
            // LCG step (Knuth multiplicative congruential)
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Extract 23 mantissa bits from upper bits (avoiding sign-bit ambiguity)
            let mantissa = ((self.state >> 41) as u32) & 0x007F_FFFF;
            f32::from_bits(0x3F80_0000 | mantissa) - 1.0
        }
    }

    /// Sample a uniform integer in [0, n).
    fn next_range(&mut self, n: usize) -> usize {
        let r = self.next_f32();
        (r * n as f32) as usize % n
    }
}

fn default_rng_state() -> core::cell::RefCell<PolicyRng> {
    core::cell::RefCell::new(PolicyRng::new())
}

/// Sample from Gamma(shape) using the Erlang method (sum of exponentials).
///
/// For alpha and beta values that are always positive integers >= 1 (as in our
/// Beta(alpha=successes+1, beta=failures+1) setup), this is exact. The Erlang
/// method avoids any trig dependency and stays fully no_std compatible.
fn sample_gamma(rng: &mut PolicyRng, shape: f32) -> f32 {
    // Round to nearest positive integer (our shape is always successes+1 or failures+1)
    let k = shape.max(1.0).round() as u32;
    let mut sum = 0.0_f32;
    for _ in 0..k {
        let u = rng.next_f32().max(f32::EPSILON);
        sum += -libm::logf(u);
    }
    sum
}

/// Policy for source selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Q-values for each source (estimated value)
    q_values: HashMap<String, f32>,
    /// Visit counts for each source (for UCB)
    visit_counts: HashMap<String, u32>,
    /// Total visits
    total_visits: u32,
    /// Learning rate
    learning_rate: f32,
    /// Exploration parameter (for UCB / epsilon-greedy)
    exploration_constant: f32,
    /// Discount factor
    gamma: f32,
    /// Policy type
    policy_type: PolicyType,
    /// Per-source success counts for Thompson sampling.
    #[serde(default)]
    visits_success: HashMap<String, u32>,
    /// Per-source failure counts for Thompson sampling.
    #[serde(default)]
    visits_failure: HashMap<String, u32>,
    /// Pseudo-random number generator for exploration. Skipped during serialization;
    /// re-seeded on deserialize.
    #[serde(skip, default = "default_rng_state")]
    rng_state: core::cell::RefCell<PolicyRng>,
}

impl Policy {
    /// Create a new policy
    #[must_use]
    pub fn new(policy_type: PolicyType) -> Self {
        Self {
            q_values: HashMap::new(),
            visit_counts: HashMap::new(),
            total_visits: 0,
            learning_rate: 0.1,
            exploration_constant: 2.0,
            gamma: 0.99,
            policy_type,
            visits_success: HashMap::new(),
            visits_failure: HashMap::new(),
            rng_state: core::cell::RefCell::new(PolicyRng::new()),
        }
    }

    /// Create with Thompson Sampling
    #[must_use]
    pub fn thompson_sampling() -> Self {
        Self::new(PolicyType::ThompsonSampling)
    }

    /// Create with UCB (Upper Confidence Bound)
    #[must_use]
    pub fn ucb() -> Self {
        Self::new(PolicyType::Ucb)
    }

    /// Create with epsilon-greedy
    #[must_use]
    pub fn epsilon_greedy(epsilon: f32) -> Self {
        let mut policy = Self::new(PolicyType::EpsilonGreedy);
        policy.exploration_constant = epsilon;
        policy
    }

    /// Set learning rate
    #[must_use]
    pub const fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set exploration constant
    #[must_use]
    pub const fn with_exploration(mut self, c: f32) -> Self {
        self.exploration_constant = c;
        self
    }

    /// Create with a fixed RNG seed for reproducible tests.
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        *self.rng_state.borrow_mut() = PolicyRng::with_seed(seed);
        self
    }

    /// Initialize source with default Q-value
    pub fn initialize_source(&mut self, source_id: impl Into<String>) {
        let id = source_id.into();
        if !self.q_values.contains_key(&id) {
            self.q_values.insert(id.clone(), 0.5); // Optimistic initialization
            self.visit_counts.insert(id, 0);
        }
    }

    /// Get Q-value for a source
    #[must_use]
    pub fn get_q_value(&self, source_id: &str) -> f32 {
        self.q_values.get(source_id).copied().unwrap_or(0.5)
    }

    /// Get visit count for a source
    #[must_use]
    pub fn get_visits(&self, source_id: &str) -> u32 {
        self.visit_counts.get(source_id).copied().unwrap_or(0)
    }

    /// Select a source based on policy
    #[must_use]
    pub fn select(&self, source_ids: &[&String]) -> Option<String> {
        if source_ids.is_empty() {
            return None;
        }

        match self.policy_type {
            PolicyType::Greedy => self.select_greedy(source_ids),
            PolicyType::EpsilonGreedy => self.select_epsilon_greedy(source_ids),
            PolicyType::Ucb => self.select_ucb(source_ids),
            PolicyType::ThompsonSampling => self.select_thompson(source_ids),
        }
    }

    /// Greedy selection (best Q-value)
    fn select_greedy(&self, source_ids: &[&String]) -> Option<String> {
        source_ids
            .iter()
            .max_by(|a, b| {
                let qa = self.get_q_value(a);
                let qb = self.get_q_value(b);
                qa.partial_cmp(&qb).unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|s| (*s).clone())
    }

    /// Epsilon-greedy selection using real RNG.
    fn select_epsilon_greedy(&self, source_ids: &[&String]) -> Option<String> {
        let r = self.rng_state.borrow_mut().next_f32();
        if r < self.exploration_constant {
            // Explore: pick uniformly at random
            #[cfg(feature = "observability")]
            {
                metrics::counter!("oxirouter.rl.explore.total").increment(1);
            }
            let idx = self.rng_state.borrow_mut().next_range(source_ids.len());
            source_ids.get(idx).map(|s| (*s).clone())
        } else {
            // Exploit: pick the greedy best
            #[cfg(feature = "observability")]
            {
                metrics::counter!("oxirouter.rl.exploit.total").increment(1);
            }
            self.select_greedy(source_ids)
        }
    }

    /// UCB selection
    fn select_ucb(&self, source_ids: &[&String]) -> Option<String> {
        let total = (self.total_visits + 1) as f32;

        source_ids
            .iter()
            .max_by(|a, b| {
                let ucb_a = self.ucb_value(a, total);
                let ucb_b = self.ucb_value(b, total);
                ucb_a
                    .partial_cmp(&ucb_b)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|s| (*s).clone())
    }

    /// Calculate UCB value for a source
    fn ucb_value(&self, source_id: &str, total: f32) -> f32 {
        let q = self.get_q_value(source_id);

        // UCB1 formula: Q + c * sqrt(ln(N) / n)
        #[cfg(feature = "std")]
        let exploration_bonus = {
            let n = self.get_visits(source_id).max(1) as f32;
            self.exploration_constant * ((total.ln()) / n).sqrt()
        };
        #[cfg(not(feature = "std"))]
        let exploration_bonus = {
            let n = self.get_visits(source_id).max(1) as f32;
            self.exploration_constant * libm::sqrtf(libm::logf(total) / n)
        };

        q + exploration_bonus
    }

    /// Thompson Sampling selection
    fn select_thompson(&self, source_ids: &[&String]) -> Option<String> {
        source_ids
            .iter()
            .max_by(|a, b| {
                let sample_a = self.thompson_sample(a);
                let sample_b = self.thompson_sample(b);
                sample_a
                    .partial_cmp(&sample_b)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|s| (*s).clone())
    }

    /// Sample from Beta(alpha, beta) via two Gamma samples (exact for integer shapes).
    fn thompson_sample(&self, source_id: &str) -> f32 {
        let alpha = (*self.visits_success.get(source_id).unwrap_or(&0) as f32) + 1.0;
        let beta = (*self.visits_failure.get(source_id).unwrap_or(&0) as f32) + 1.0;

        // Sample Beta(alpha, beta) = Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
        let mut rng = self.rng_state.borrow_mut();
        let x = sample_gamma(&mut rng, alpha);
        let y = sample_gamma(&mut rng, beta);
        if x + y < f32::EPSILON {
            return self.get_q_value(source_id);
        }
        (x / (x + y)).clamp(0.0, 1.0)
    }

    /// Update policy based on reward
    pub fn update(&mut self, source_id: &str, reward: Reward) {
        // Ensure source is initialized
        if !self.q_values.contains_key(source_id) {
            self.initialize_source(source_id);
        }

        // Increment visit counts
        *self.visit_counts.entry(source_id.to_string()).or_insert(0) += 1;
        self.total_visits += 1;

        // Update Q-value with exponential moving average
        let old_q = self.get_q_value(source_id);
        let new_q = old_q + self.learning_rate * (reward.value() - old_q);
        self.q_values.insert(source_id.to_string(), new_q);

        // Track success/failure for Thompson sampling
        if reward.value() >= 0.5 {
            *self
                .visits_success
                .entry(source_id.to_string())
                .or_insert(0) += 1;
        } else {
            *self
                .visits_failure
                .entry(source_id.to_string())
                .or_insert(0) += 1;
        }
    }

    /// Decay exploration over time
    pub fn decay_exploration(&mut self, decay_rate: f32) {
        match self.policy_type {
            PolicyType::EpsilonGreedy => {
                self.exploration_constant *= decay_rate;
                self.exploration_constant = self.exploration_constant.max(0.01);
            }
            PolicyType::Ucb => {
                self.exploration_constant *= decay_rate;
                self.exploration_constant = self.exploration_constant.max(0.5);
            }
            _ => {}
        }
    }

    /// Get source rankings based on current Q-values
    #[must_use]
    pub fn get_rankings(&self, source_ids: &[&String]) -> Vec<(String, f32)> {
        let mut rankings: Vec<_> = source_ids
            .iter()
            .map(|id| ((*id).clone(), self.get_q_value(id)))
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        rankings
    }

    /// Get exploration statistics
    #[must_use]
    pub fn stats(&self) -> PolicyStats {
        let avg_q = if self.q_values.is_empty() {
            0.5
        } else {
            self.q_values.values().sum::<f32>() / self.q_values.len() as f32
        };

        let max_q = self.q_values.values().copied().fold(0.0f32, f32::max);
        let min_q = self.q_values.values().copied().fold(1.0f32, f32::min);

        PolicyStats {
            total_visits: self.total_visits,
            source_count: self.q_values.len(),
            avg_q_value: avg_q,
            max_q_value: max_q,
            min_q_value: min_q,
            exploration_rate: self.exploration_constant,
        }
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self::ucb()
    }
}

/// Type of exploration/exploitation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyType {
    /// Always select best Q-value
    Greedy,
    /// Random with probability epsilon
    EpsilonGreedy,
    /// Upper Confidence Bound
    Ucb,
    /// Thompson Sampling (probability matching)
    ThompsonSampling,
}

/// Statistics about policy state
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PolicyStats {
    /// Total visits across all sources
    pub total_visits: u32,
    /// Number of tracked sources
    pub source_count: usize,
    /// Average Q-value
    pub avg_q_value: f32,
    /// Maximum Q-value
    pub max_q_value: f32,
    /// Minimum Q-value
    pub min_q_value: f32,
    /// Current exploration rate
    pub exploration_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    use alloc::vec;

    #[test]
    fn test_policy_creation() {
        let policy = Policy::ucb();
        assert_eq!(policy.policy_type, PolicyType::Ucb);
    }

    #[test]
    fn test_initialize_source() {
        let mut policy = Policy::new(PolicyType::Greedy);
        policy.initialize_source("src1");

        assert!(policy.q_values.contains_key("src1"));
        assert_eq!(policy.get_q_value("src1"), 0.5);
    }

    #[test]
    fn test_greedy_selection() {
        let mut policy = Policy::new(PolicyType::Greedy);
        policy.initialize_source("src1");
        policy.initialize_source("src2");

        // Make src1 better
        policy.q_values.insert("src1".to_string(), 0.8);
        policy.q_values.insert("src2".to_string(), 0.3);

        let sources = vec!["src1".to_string(), "src2".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        let selected = policy.select(&source_refs);

        assert_eq!(selected, Some("src1".to_string()));
    }

    #[test]
    fn test_policy_update() {
        let mut policy = Policy::ucb();
        policy.initialize_source("src1");

        let initial_q = policy.get_q_value("src1");
        policy.update("src1", Reward::new(1.0));

        // Q-value should increase toward reward
        assert!(policy.get_q_value("src1") > initial_q);
        assert_eq!(policy.get_visits("src1"), 1);
    }

    #[test]
    fn test_ucb_exploration() {
        let mut policy = Policy::ucb();
        policy.initialize_source("explored");
        policy.initialize_source("unexplored");

        // Simulate many visits to "explored"
        for _ in 0..100 {
            policy.update("explored", Reward::new(0.5));
        }

        // UCB should prefer unexplored source
        let sources = vec!["explored".to_string(), "unexplored".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        let selected = policy.select(&source_refs);

        // Unexplored should have higher UCB due to uncertainty bonus
        assert_eq!(selected, Some("unexplored".to_string()));
    }

    #[test]
    fn test_exploration_decay() {
        let mut policy = Policy::epsilon_greedy(0.5);
        policy.decay_exploration(0.9);

        assert!(policy.exploration_constant < 0.5);
    }

    #[test]
    fn test_rankings() {
        let mut policy = Policy::new(PolicyType::Greedy);
        policy.q_values.insert("best".to_string(), 0.9);
        policy.q_values.insert("middle".to_string(), 0.5);
        policy.q_values.insert("worst".to_string(), 0.1);

        let sources = vec![
            "best".to_string(),
            "middle".to_string(),
            "worst".to_string(),
        ];
        let source_refs: Vec<&String> = sources.iter().collect();
        let rankings = policy.get_rankings(&source_refs);

        assert_eq!(rankings[0].0, "best");
        assert_eq!(rankings[2].0, "worst");
    }
}
