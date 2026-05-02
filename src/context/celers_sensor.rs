//! Celers-backed load sensor.
//!
//! [`CelersLoadSensor`] wraps a closure that yields a `celers_core::WorkerStats`
//! snapshot. The closure approach decouples the sensor from any specific celers
//! transport (AMQP, Redis, in-memory) and allows the caller to bridge any API surface.

#[cfg(feature = "load")]
use super::{LoadContext, sensor::LoadSensor};

/// Load sensor backed by a `celers_core::WorkerStats` closure.
///
/// The closure is called on every `sense()` invocation; expensive broker I/O
/// should be cached or rate-limited inside the closure itself.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "load")]
/// # {
/// use oxirouter::context::CelersLoadSensor;
/// use celers_core::WorkerStats;
///
/// let sensor = CelersLoadSensor::new(|| {
///     // Replace with real broker inspection call
///     WorkerStats {
///         total_tasks: 1000,
///         active_tasks: 5,
///         succeeded: 990,
///         failed: 5,
///         retried: 2,
///         uptime: 3600.0,
///         loadavg: Some([0.3, 0.25, 0.2]),
///         memory_usage: Some(256 * 1024 * 1024),
///         pool: None,
///         broker: None,
///         clock: None,
///     }
/// });
/// # }
/// ```
#[cfg(feature = "load")]
pub struct CelersLoadSensor<F>
where
    F: Fn() -> celers_core::WorkerStats + Send + Sync,
{
    /// Closure that yields a fresh `WorkerStats` snapshot.
    stats_fn: F,
}

#[cfg(feature = "load")]
impl<F> CelersLoadSensor<F>
where
    F: Fn() -> celers_core::WorkerStats + Send + Sync,
{
    /// Create a new sensor with the given stats-fetching closure.
    #[must_use]
    pub fn new(stats_fn: F) -> Self {
        Self { stats_fn }
    }
}

#[cfg(feature = "load")]
impl<F> LoadSensor for CelersLoadSensor<F>
where
    F: Fn() -> celers_core::WorkerStats + Send + Sync,
{
    fn sense(&self) -> Option<LoadContext> {
        let stats = (self.stats_fn)();
        Some(LoadContext::from_celers_stats(&stats))
    }
}
