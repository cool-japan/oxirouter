//! Query log for statistics-based routing
//!
//! Tracks routing decisions and their outcomes to support adaptive,
//! statistics-driven source selection. Provides per-source and per-query
//! analytics that feed back into heuristic and ML routing.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Maximum number of log entries to keep (ring-buffer style)
pub const DEFAULT_MAX_LOG_SIZE: usize = 10_000;

/// A single routing decision log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingLogEntry {
    /// Query identifier (hash)
    pub query_id: u64,
    /// Selected source ID
    pub source_id: String,
    /// Confidence score at routing time
    pub confidence: f32,
    /// Whether ML model was used
    pub ml_used: bool,
    /// Timestamp (Unix epoch ms)
    pub timestamp_ms: u64,
    /// Outcome (filled in after execution)
    pub outcome: Option<RoutingOutcome>,
    /// Feature vector used for routing (stored for online training). None if ML was not used.
    #[serde(default)]
    pub feature_vector: Option<Vec<f32>>,
}

/// Outcome of a routing decision after execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOutcome {
    /// Whether the query succeeded
    pub success: bool,
    /// Actual latency in milliseconds
    pub latency_ms: u32,
    /// Number of results returned
    pub result_count: u32,
    /// Reward signal computed from outcome
    pub reward: f32,
}

/// Aggregated statistics per data source
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceLogStats {
    /// Total routing decisions to this source
    pub total_routed: u64,
    /// Successful executions
    pub successes: u64,
    /// Sum of latencies for running average
    latency_sum: f64,
    /// Sum of reward signals for running average
    reward_sum: f64,
    /// Outcomes received (may be less than total_routed if some pending)
    pub outcomes_received: u64,
}

impl SourceLogStats {
    /// Record a routing outcome
    pub fn record(&mut self, outcome: &RoutingOutcome) {
        self.outcomes_received += 1;
        if outcome.success {
            self.successes += 1;
        }
        self.latency_sum += f64::from(outcome.latency_ms);
        self.reward_sum += f64::from(outcome.reward);
    }

    /// Average latency in milliseconds
    #[must_use]
    pub fn avg_latency_ms(&self) -> f64 {
        if self.outcomes_received == 0 {
            0.0
        } else {
            self.latency_sum / self.outcomes_received as f64
        }
    }

    /// Average reward signal (0.0 - 1.0)
    #[must_use]
    pub fn avg_reward(&self) -> f64 {
        if self.outcomes_received == 0 {
            0.5 // neutral prior
        } else {
            self.reward_sum / self.outcomes_received as f64
        }
    }

    /// Success rate (0.0 - 1.0)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.outcomes_received == 0 {
            1.0 // optimistic prior
        } else {
            self.successes as f64 / self.outcomes_received as f64
        }
    }
}

/// Query log for tracking routing history and computing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLog {
    /// Ring-buffer of routing entries (newest at back)
    entries: Vec<RoutingLogEntry>,
    /// Maximum number of entries to keep
    max_size: usize,
    /// Per-source aggregated statistics
    source_stats: HashMap<String, SourceLogStats>,
    /// Total entries ever recorded (including evicted)
    pub total_recorded: u64,
}

impl QueryLog {
    /// Create a new query log with the default capacity
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_LOG_SIZE)
    }

    /// Create a new query log with given capacity
    #[must_use]
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_size.min(1024)),
            max_size,
            source_stats: HashMap::new(),
            total_recorded: 0,
        }
    }

    /// Record a routing decision (before execution result is known).
    ///
    /// `feature_vector` is the ML feature vector used for this routing decision; stored
    /// so that `learn_from_outcome` can retrieve it for online model updates.
    pub fn record_routing(
        &mut self,
        query_id: u64,
        source_id: impl Into<String>,
        confidence: f32,
        ml_used: bool,
        feature_vector: Option<Vec<f32>>,
    ) {
        let source_id = source_id.into();

        // Increment routed count for this source
        self.source_stats
            .entry(source_id.clone())
            .or_default()
            .total_routed += 1;

        self.total_recorded += 1;

        let entry = RoutingLogEntry {
            query_id,
            source_id,
            confidence,
            ml_used,
            timestamp_ms: get_time_ms(),
            outcome: None,
            feature_vector,
        };

        // Evict oldest entry if at capacity
        if self.entries.len() >= self.max_size {
            self.entries.remove(0);
        }

        self.entries.push(entry);
    }

    /// Find the feature vector stored for a given routing decision.
    ///
    /// Returns `None` if the entry doesn't exist or has no feature vector.
    #[must_use]
    pub fn find_entry_features(&self, query_id: u64, source_id: &str) -> Option<Vec<f32>> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.query_id == query_id && e.source_id == source_id)
            .and_then(|e| e.feature_vector.clone())
    }

    /// Record the outcome of a routing decision
    ///
    /// Matches by `query_id` + `source_id` (most recent match).
    pub fn record_outcome(
        &mut self,
        query_id: u64,
        source_id: &str,
        success: bool,
        latency_ms: u32,
        result_count: u32,
        reward: f32,
    ) {
        let outcome = RoutingOutcome {
            success,
            latency_ms,
            result_count,
            reward,
        };

        // Update source stats
        if let Some(stats) = self.source_stats.get_mut(source_id) {
            stats.record(&outcome);
        }

        // Find the most recent pending entry for this query+source
        if let Some(entry) = self
            .entries
            .iter_mut()
            .rev()
            .find(|e| e.query_id == query_id && e.source_id == source_id && e.outcome.is_none())
        {
            entry.outcome = Some(outcome);
        }
    }

    /// Get aggregated statistics for a source
    #[must_use]
    pub fn source_stats(&self, source_id: &str) -> Option<&SourceLogStats> {
        self.source_stats.get(source_id)
    }

    /// Get the routing score for a source based on log history
    ///
    /// Returns a value in 0.0–1.0 representing how well this source
    /// has performed historically. Returns `None` if no history.
    #[must_use]
    pub fn routing_score(&self, source_id: &str) -> Option<f32> {
        let stats = self.source_stats.get(source_id)?;
        if stats.outcomes_received == 0 {
            return None;
        }

        // Combine success rate and reward, weighted
        let score = stats.success_rate() * 0.6 + stats.avg_reward() * 0.4;
        Some(score as f32)
    }

    /// Get the best source ID based on log history
    #[must_use]
    pub fn best_source(&self) -> Option<&str> {
        self.source_stats
            .iter()
            .filter(|(_, s)| s.outcomes_received > 0)
            .max_by(|(_, a), (_, b)| {
                let sa = a.success_rate() * 0.6 + a.avg_reward() * 0.4;
                let sb = b.success_rate() * 0.6 + b.avg_reward() * 0.4;
                sa.partial_cmp(&sb).unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.as_str())
    }

    /// Get all source IDs that have log history, sorted by score (descending)
    #[must_use]
    pub fn ranked_sources(&self) -> Vec<(String, f32)> {
        let mut ranked: Vec<_> = self
            .source_stats
            .iter()
            .filter(|(_, s)| s.outcomes_received > 0)
            .map(|(id, s)| {
                let score = s.success_rate() * 0.6 + s.avg_reward() * 0.4;
                (id.clone(), score as f32)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        ranked
    }

    /// Get recent log entries (up to `limit`)
    #[must_use]
    pub fn recent_entries(&self, limit: usize) -> &[RoutingLogEntry] {
        let len = self.entries.len();
        if len <= limit {
            &self.entries
        } else {
            &self.entries[len - limit..]
        }
    }

    /// Get the number of entries in the log
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the log is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the log (preserves source statistics)
    pub fn clear_entries(&mut self) {
        self.entries.clear();
    }

    /// Clear everything including statistics
    pub fn clear_all(&mut self) {
        self.entries.clear();
        self.source_stats.clear();
        self.total_recorded = 0;
    }

    /// Remove entries older than `max_age_ms` milliseconds
    ///
    /// Returns the number of entries removed.
    pub fn evict_older_than(&mut self, max_age_ms: u64) -> usize {
        let now = get_time_ms();
        let before = self.entries.len();
        self.entries
            .retain(|e| now.saturating_sub(e.timestamp_ms) <= max_age_ms);
        before - self.entries.len()
    }

    /// Get the number of sources with log history
    #[must_use]
    pub fn tracked_source_count(&self) -> usize {
        self.source_stats.len()
    }

    /// Compute a reliability score for a source: combines log stats and the
    /// provided current success_rate from `SourceStats`.
    ///
    /// `current_rate` should be `source.stats.success_rate`.
    /// If no log history, falls back to `current_rate`.
    #[must_use]
    pub fn combined_reliability(&self, source_id: &str, current_rate: f32) -> f32 {
        match self.routing_score(source_id) {
            Some(log_score) => {
                // Blend: 40% current stats, 60% log history (log is richer)
                0.4 * current_rate + 0.6 * log_score
            }
            None => current_rate,
        }
    }
}

impl Default for QueryLog {
    fn default() -> Self {
        Self::new()
    }
}

fn get_time_ms() -> u64 {
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
    #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    use alloc::{format, vec};

    #[test]
    fn test_query_log_basic() {
        let mut log = QueryLog::new();
        assert!(log.is_empty());

        log.record_routing(1, "src1", 0.9, false, None);
        log.record_routing(1, "src2", 0.7, false, None);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_record_outcome() {
        let mut log = QueryLog::new();

        log.record_routing(42, "src1", 0.8, false, None);
        log.record_outcome(42, "src1", true, 100, 50, 0.9);

        let stats = log.source_stats("src1").unwrap();
        assert_eq!(stats.outcomes_received, 1);
        assert_eq!(stats.successes, 1);
    }

    #[test]
    fn test_routing_score() {
        let mut log = QueryLog::new();

        // Good source
        for _ in 0..5 {
            log.record_routing(1, "good", 0.9, false, None);
            log.record_outcome(1, "good", true, 100, 50, 0.9);
        }

        // Bad source
        for _ in 0..5 {
            log.record_routing(2, "bad", 0.5, false, None);
            log.record_outcome(2, "bad", false, 5000, 0, 0.0);
        }

        let good_score = log.routing_score("good").unwrap();
        let bad_score = log.routing_score("bad").unwrap();
        assert!(good_score > bad_score);
    }

    #[test]
    fn test_best_source() {
        let mut log = QueryLog::new();

        log.record_routing(1, "fast", 0.9, false, None);
        log.record_outcome(1, "fast", true, 50, 100, 1.0);

        log.record_routing(2, "slow", 0.5, false, None);
        log.record_outcome(2, "slow", true, 3000, 10, 0.4);

        assert_eq!(log.best_source(), Some("fast"));
    }

    #[test]
    fn test_ranked_sources() {
        let mut log = QueryLog::new();

        log.record_routing(1, "a", 0.9, false, None);
        log.record_outcome(1, "a", true, 100, 50, 0.9);

        log.record_routing(2, "b", 0.5, false, None);
        log.record_outcome(2, "b", false, 2000, 0, 0.1);

        let ranked = log.ranked_sources();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, "a");
    }

    #[test]
    fn test_eviction() {
        let mut log = QueryLog::with_capacity(3);

        for i in 0..5u64 {
            log.record_routing(i, format!("src{}", i), 0.5, false, None);
        }

        assert_eq!(log.len(), 3); // Only 3 kept due to capacity
    }

    #[test]
    fn test_combined_reliability() {
        let mut log = QueryLog::new();

        // Source with no history
        let score = log.combined_reliability("unknown", 0.8);
        assert!((score - 0.8).abs() < 0.001);

        // Source with good history
        log.record_routing(1, "known", 0.9, false, None);
        log.record_outcome(1, "known", true, 100, 50, 0.9);

        let score = log.combined_reliability("known", 0.8);
        assert!(score > 0.7); // Should be high
    }

    #[test]
    fn test_clear() {
        let mut log = QueryLog::new();
        log.record_routing(1, "src1", 0.8, false, None);
        log.record_outcome(1, "src1", true, 100, 50, 0.9);

        log.clear_entries();
        assert!(log.is_empty());
        // Stats should be preserved
        assert!(log.source_stats("src1").is_some());

        log.clear_all();
        assert!(log.source_stats("src1").is_none());
    }

    #[test]
    fn test_feature_vector_storage_and_retrieval() {
        let mut log = QueryLog::new();
        let fv = vec![0.1_f32, 0.2, 0.3, 0.4];

        log.record_routing(99, "src1", 0.8, true, Some(fv.clone()));

        let retrieved = log.find_entry_features(99, "src1");
        assert_eq!(retrieved, Some(fv));
    }

    #[test]
    fn test_find_entry_features_none_when_not_ml() {
        let mut log = QueryLog::new();
        log.record_routing(100, "src1", 0.8, false, None);

        let retrieved = log.find_entry_features(100, "src1");
        assert!(retrieved.is_none());
    }
}
