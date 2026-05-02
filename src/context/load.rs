//! Load/task queue context from celers (Situation brain)

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Load/task queue context for workload-aware routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadContext {
    /// Target endpoint health status (endpoint -> health score 0.0-1.0)
    pub target_health: HashMap<String, f32>,

    /// Estimated response time per endpoint in milliseconds
    pub estimated_response_ms: HashMap<String, u32>,

    /// Current queue depth per endpoint
    pub queue_depth: HashMap<String, u32>,

    /// Active connections per endpoint
    pub active_connections: HashMap<String, u32>,

    /// Error rate per endpoint (0.0-1.0)
    pub error_rates: HashMap<String, f32>,

    /// Circuit breaker states
    pub circuit_breakers: HashMap<String, CircuitState>,

    /// Global load factor (0.0 = idle, 1.0 = overloaded)
    pub global_load: f32,

    /// Number of pending tasks in the system
    pub pending_tasks: u32,
}

impl LoadContext {
    /// Create a new load context
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_health: HashMap::new(),
            estimated_response_ms: HashMap::new(),
            queue_depth: HashMap::new(),
            active_connections: HashMap::new(),
            error_rates: HashMap::new(),
            circuit_breakers: HashMap::new(),
            global_load: 0.0,
            pending_tasks: 0,
        }
    }

    /// Set health for an endpoint
    pub fn set_health(&mut self, endpoint: impl Into<String>, health: f32) {
        self.target_health
            .insert(endpoint.into(), health.clamp(0.0, 1.0));
    }

    /// Set estimated response time
    pub fn set_response_time(&mut self, endpoint: impl Into<String>, ms: u32) {
        self.estimated_response_ms.insert(endpoint.into(), ms);
    }

    /// Set queue depth
    pub fn set_queue_depth(&mut self, endpoint: impl Into<String>, depth: u32) {
        self.queue_depth.insert(endpoint.into(), depth);
    }

    /// Set circuit breaker state
    pub fn set_circuit_state(&mut self, endpoint: impl Into<String>, state: CircuitState) {
        self.circuit_breakers.insert(endpoint.into(), state);
    }

    /// Get health score for an endpoint
    #[must_use]
    pub fn get_health(&self, endpoint: &str) -> f32 {
        self.target_health.get(endpoint).copied().unwrap_or(1.0)
    }

    /// Get estimated response time for an endpoint
    #[must_use]
    pub fn get_response_time(&self, endpoint: &str) -> u32 {
        self.estimated_response_ms
            .get(endpoint)
            .copied()
            .unwrap_or(1000)
    }

    /// Check if endpoint is available (circuit not open)
    #[must_use]
    pub fn is_available(&self, endpoint: &str) -> bool {
        match self.circuit_breakers.get(endpoint) {
            Some(CircuitState::Open) => false,
            _ => true,
        }
    }

    /// Get overall availability score for an endpoint (0.0-1.0)
    #[must_use]
    pub fn availability_score(&self, endpoint: &str) -> f32 {
        if !self.is_available(endpoint) {
            return 0.0;
        }

        let health = self.get_health(endpoint);
        let error_rate = self.error_rates.get(endpoint).copied().unwrap_or(0.0);
        let queue_factor = self
            .queue_depth
            .get(endpoint)
            .map(|&d| 1.0 - (d as f32 / 1000.0).min(1.0))
            .unwrap_or(1.0);

        health * (1.0 - error_rate) * queue_factor
    }

    /// Get the best endpoints sorted by availability
    #[must_use]
    pub fn best_endpoints(&self, limit: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<_> = self
            .target_health
            .keys()
            .map(|ep| (ep.clone(), self.availability_score(ep)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    /// Check if system is overloaded
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        self.global_load > 0.9 || self.pending_tasks > 1000
    }

    /// Recommended concurrency based on load
    #[must_use]
    pub fn recommended_concurrency(&self) -> u32 {
        if self.is_overloaded() {
            1
        } else if self.global_load > 0.7 {
            2
        } else if self.global_load > 0.5 {
            4
        } else {
            8
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing fast)
    Open,
    /// Circuit is half-open (testing recovery)
    HalfOpen,
}

impl Default for CircuitState {
    fn default() -> Self {
        Self::Closed
    }
}

/// Integration with celers-core
#[cfg(feature = "load")]
impl LoadContext {
    /// Create from celers worker stats.
    ///
    /// Maps `WorkerStats` fields to `LoadContext`:
    ///
    /// | `WorkerStats` field | `LoadContext` field |
    /// |---------------------|---------------------|
    /// | `active_tasks` | `pending_tasks` |
    /// | `loadavg[0]` (1-min) | `global_load` (clamped 0-1) |
    /// | `pool.available` | informs `queue_depth` for the `"__pool__"` key |
    /// | `broker.connected` | circuit state for the `"__broker__"` endpoint |
    /// | error rate estimate | `error_rates["__global__"]` |
    ///
    /// Fields without a direct mapping (`target_health`, `estimated_response_ms`,
    /// `active_connections`) remain empty — callers should augment the returned
    /// context with per-endpoint data if available.
    pub fn from_celers_stats(stats: &celers_core::WorkerStats) -> Self {
        let mut ctx = Self::new();

        // Active task count is the primary pending_tasks indicator.
        ctx.pending_tasks = stats.active_tasks;

        // Map 1-minute load average to global_load (normalise to 0-1).
        // A loadavg of 1.0 per core maps to ~100% utilisation; we normalise
        // by the reported pool concurrency when available, otherwise cap at 1.0.
        if let Some(loadavg) = stats.loadavg {
            let one_min = loadavg[0];
            let max_concurrency = stats
                .pool
                .as_ref()
                .map(|p| f64::from(p.max_concurrency))
                .filter(|&c| c > 0.0)
                .unwrap_or(1.0);
            ctx.global_load = (one_min / max_concurrency).clamp(0.0, 1.0) as f32;
        }

        // Pool availability -> queue_depth for the virtual "__pool__" endpoint.
        if let Some(ref pool) = stats.pool {
            let busy = pool.pool_size.saturating_sub(pool.available);
            ctx.queue_depth.insert("__pool__".to_string(), busy);
            ctx.active_connections
                .insert("__pool__".to_string(), pool.pool_size);
        }

        // Broker connectivity -> circuit breaker for the virtual "__broker__" endpoint.
        if let Some(ref broker) = stats.broker {
            let state = if broker.connected {
                CircuitState::Closed
            } else {
                CircuitState::Open
            };
            ctx.circuit_breakers.insert("__broker__".to_string(), state);
            // Also record health: connected = 1.0, disconnected = 0.0.
            let health = if broker.connected { 1.0_f32 } else { 0.0_f32 };
            ctx.target_health.insert("__broker__".to_string(), health);
        }

        // Global error rate estimate from failed/total to avoid division by zero.
        if stats.total_tasks > 0 {
            let error_rate = stats.failed as f32 / stats.total_tasks as f32;
            ctx.error_rates
                .insert("__global__".to_string(), error_rate.clamp(0.0, 1.0));
        }

        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_context() {
        let mut ctx = LoadContext::new();
        ctx.set_health("endpoint1", 0.9);
        ctx.set_response_time("endpoint1", 100);

        assert!(ctx.get_health("endpoint1") > 0.8);
        assert!(ctx.is_available("endpoint1"));
    }

    #[test]
    fn test_circuit_breaker() {
        let mut ctx = LoadContext::new();
        ctx.set_circuit_state("endpoint1", CircuitState::Open);

        assert!(!ctx.is_available("endpoint1"));
        assert_eq!(ctx.availability_score("endpoint1"), 0.0);
    }

    #[test]
    fn test_best_endpoints() {
        let mut ctx = LoadContext::new();
        ctx.set_health("good", 0.9);
        ctx.set_health("bad", 0.3);
        ctx.set_health("medium", 0.6);

        let best = ctx.best_endpoints(2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].0, "good");
    }

    #[test]
    fn test_overload_detection() {
        let mut ctx = LoadContext::new();
        assert!(!ctx.is_overloaded());

        ctx.global_load = 0.95;
        assert!(ctx.is_overloaded());
    }
}
