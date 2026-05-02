//! Query result feedback collection

#[cfg(feature = "alloc")]
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use super::reward::Reward;

/// Feedback from a query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback {
    /// Source that was queried
    pub source_id: String,
    /// Query identifier (hash or ID)
    pub query_id: u64,
    /// Whether the query succeeded
    pub success: bool,
    /// Latency in milliseconds
    pub latency_ms: u32,
    /// Number of results returned
    pub result_count: u32,
    /// Size of results in bytes (if available)
    pub result_size_bytes: Option<u64>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Timestamp of feedback (Unix epoch ms)
    pub timestamp_ms: u64,
    /// Additional metadata
    pub metadata: FeedbackMetadata,
}

impl Feedback {
    /// Create new feedback for a successful query
    #[must_use]
    pub fn success(
        source_id: impl Into<String>,
        query_id: u64,
        latency_ms: u32,
        result_count: u32,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            query_id,
            success: true,
            latency_ms,
            result_count,
            result_size_bytes: None,
            error: None,
            timestamp_ms: Self::current_time_ms(),
            metadata: FeedbackMetadata::default(),
        }
    }

    /// Create new feedback for a failed query
    #[must_use]
    pub fn failure(source_id: impl Into<String>, query_id: u64, error: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            query_id,
            success: false,
            latency_ms: 0,
            result_count: 0,
            result_size_bytes: None,
            error: Some(error.into()),
            timestamp_ms: Self::current_time_ms(),
            metadata: FeedbackMetadata::default(),
        }
    }

    /// Create feedback for a timeout
    #[must_use]
    pub fn timeout(source_id: impl Into<String>, query_id: u64, timeout_ms: u32) -> Self {
        Self {
            source_id: source_id.into(),
            query_id,
            success: false,
            latency_ms: timeout_ms,
            result_count: 0,
            result_size_bytes: None,
            error: Some("Timeout".to_string()),
            timestamp_ms: Self::current_time_ms(),
            metadata: FeedbackMetadata {
                timed_out: true,
                ..Default::default()
            },
        }
    }

    /// Set the latency for failed queries
    #[must_use]
    pub const fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Set result size
    #[must_use]
    pub const fn with_result_size(mut self, size_bytes: u64) -> Self {
        self.result_size_bytes = Some(size_bytes);
        self
    }

    /// Set metadata
    #[must_use]
    pub const fn with_metadata(mut self, metadata: FeedbackMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Calculate the reward signal from this feedback
    #[must_use]
    pub fn to_reward(&self) -> Reward {
        Reward::from_feedback(self)
    }

    /// Check if feedback indicates a transient failure (retry-able)
    #[must_use]
    pub fn is_transient_failure(&self) -> bool {
        if self.success {
            return false;
        }

        // Timeouts are usually transient
        if self.metadata.timed_out {
            return true;
        }

        // Check error message for transient patterns
        if let Some(ref error) = self.error {
            let error_lower = error.to_lowercase();
            return error_lower.contains("timeout")
                || error_lower.contains("connection")
                || error_lower.contains("temporary")
                || error_lower.contains("overload")
                || error_lower.contains("rate limit")
                || error_lower.contains("503")
                || error_lower.contains("429");
        }

        false
    }

    /// Check if feedback indicates a permanent failure
    #[must_use]
    pub fn is_permanent_failure(&self) -> bool {
        if self.success {
            return false;
        }

        if let Some(ref error) = self.error {
            let error_lower = error.to_lowercase();
            return error_lower.contains("not found")
                || error_lower.contains("404")
                || error_lower.contains("forbidden")
                || error_lower.contains("403")
                || error_lower.contains("unauthorized")
                || error_lower.contains("401")
                || error_lower.contains("invalid");
        }

        false
    }

    /// Get current time in milliseconds
    fn current_time_ms() -> u64 {
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
}

/// Additional metadata for feedback
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackMetadata {
    /// Whether the query timed out
    pub timed_out: bool,
    /// Whether results were truncated
    pub truncated: bool,
    /// HTTP status code (if applicable)
    pub http_status: Option<u16>,
    /// Number of retries attempted
    pub retry_count: u8,
    /// Query complexity estimate
    pub complexity: Option<f32>,
    /// Was this a cache hit?
    pub cache_hit: bool,
}

/// Collection of feedback for batch processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct FeedbackBatch {
    /// Collected feedback items
    pub items: Vec<Feedback>,
    /// Start timestamp of batch
    pub start_time_ms: u64,
    /// End timestamp of batch
    pub end_time_ms: u64,
}

#[allow(dead_code)]
impl FeedbackBatch {
    /// Create a new empty batch
    #[must_use]
    pub fn new() -> Self {
        let now = Feedback::current_time_ms();
        Self {
            items: Vec::new(),
            start_time_ms: now,
            end_time_ms: now,
        }
    }

    /// Add feedback to the batch
    pub fn add(&mut self, feedback: Feedback) {
        self.end_time_ms = feedback.timestamp_ms.max(self.end_time_ms);
        self.items.push(feedback);
    }

    /// Get batch statistics
    #[must_use]
    pub fn stats(&self) -> BatchStats {
        let total = self.items.len() as u32;
        let successful = self.items.iter().filter(|f| f.success).count() as u32;
        let total_latency: u64 = self.items.iter().map(|f| u64::from(f.latency_ms)).sum();
        let total_results: u64 = self.items.iter().map(|f| u64::from(f.result_count)).sum();

        BatchStats {
            total_queries: total,
            successful_queries: successful,
            failed_queries: total - successful,
            avg_latency_ms: if total > 0 {
                (total_latency / u64::from(total)) as u32
            } else {
                0
            },
            total_results,
            success_rate: if total > 0 {
                successful as f32 / total as f32
            } else {
                0.0
            },
        }
    }

    /// Get feedback for a specific source
    #[must_use]
    pub fn for_source(&self, source_id: &str) -> Vec<&Feedback> {
        self.items
            .iter()
            .filter(|f| f.source_id == source_id)
            .collect()
    }

    /// Calculate aggregate reward for a source
    #[must_use]
    pub fn source_reward(&self, source_id: &str) -> f32 {
        let source_feedback: Vec<_> = self.for_source(source_id);
        if source_feedback.is_empty() {
            return 0.5; // Neutral if no data
        }

        let total_reward: f32 = source_feedback.iter().map(|f| f.to_reward().value()).sum();

        total_reward / source_feedback.len() as f32
    }

    /// Check if batch is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get batch size
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.items.clear();
        self.start_time_ms = Feedback::current_time_ms();
        self.end_time_ms = self.start_time_ms;
    }
}

/// Statistics for a feedback batch
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct BatchStats {
    /// Total number of queries
    pub total_queries: u32,
    /// Number of successful queries
    pub successful_queries: u32,
    /// Number of failed queries
    pub failed_queries: u32,
    /// Average latency in milliseconds
    pub avg_latency_ms: u32,
    /// Total results returned
    pub total_results: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_success_feedback() {
        let fb = Feedback::success("src1", 123, 100, 50);
        assert!(fb.success);
        assert_eq!(fb.latency_ms, 100);
        assert_eq!(fb.result_count, 50);
    }

    #[test]
    fn test_failure_feedback() {
        let fb = Feedback::failure("src1", 123, "Connection refused");
        assert!(!fb.success);
        assert!(fb.error.is_some());
    }

    #[test]
    fn test_timeout_feedback() {
        let fb = Feedback::timeout("src1", 123, 5000);
        assert!(!fb.success);
        assert!(fb.is_transient_failure());
        assert!(!fb.is_permanent_failure());
    }

    #[test]
    fn test_transient_detection() {
        let fb = Feedback::failure("src1", 123, "503 Service Unavailable");
        assert!(fb.is_transient_failure());

        let fb = Feedback::failure("src1", 123, "404 Not Found");
        assert!(!fb.is_transient_failure());
        assert!(fb.is_permanent_failure());
    }

    #[test]
    fn test_feedback_batch() {
        let mut batch = FeedbackBatch::new();
        batch.add(Feedback::success("src1", 1, 100, 10));
        batch.add(Feedback::success("src1", 2, 200, 20));
        batch.add(Feedback::failure("src2", 3, "error"));

        let stats = batch.stats();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.successful_queries, 2);
        assert_eq!(stats.failed_queries, 1);
    }

    #[test]
    fn test_source_reward() {
        let mut batch = FeedbackBatch::new();
        batch.add(Feedback::success("src1", 1, 100, 10));
        batch.add(Feedback::success("src1", 2, 100, 10));

        let reward = batch.source_reward("src1");
        assert!(reward > 0.5); // Should be positive for successes
    }
}
