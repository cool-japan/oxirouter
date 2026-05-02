//! Context provider trait and combined context

use serde::{Deserialize, Serialize};

#[cfg(feature = "geo")]
use super::GeoContext;

#[cfg(any(feature = "device", feature = "std"))]
use super::DeviceContext;

#[cfg(any(feature = "load", feature = "std"))]
use super::LoadContext;

#[cfg(any(feature = "legal", feature = "std"))]
use super::LegalContext;

/// Combined context from all 4 brains
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CombinedContext {
    /// Geographic/physical context (oxigdal)
    #[cfg(feature = "geo")]
    pub geo: Option<GeoContext>,

    /// Device/hardware context (mielin)
    #[cfg(any(feature = "device", feature = "std"))]
    pub device: Option<DeviceContext>,

    /// Load/task queue context (celers)
    #[cfg(any(feature = "load", feature = "std"))]
    pub load: Option<LoadContext>,

    /// Legal/compliance context (legalis)
    #[cfg(any(feature = "legal", feature = "std"))]
    pub legal: Option<LegalContext>,

    /// Timestamp when context was collected (Unix epoch seconds)
    pub timestamp: u64,
}

impl CombinedContext {
    /// Create a new empty combined context
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "geo")]
            geo: None,
            #[cfg(any(feature = "device", feature = "std"))]
            device: None,
            #[cfg(any(feature = "load", feature = "std"))]
            load: None,
            #[cfg(any(feature = "legal", feature = "std"))]
            legal: None,
            timestamp: 0,
        }
    }

    /// Set the geo context
    #[cfg(feature = "geo")]
    #[must_use]
    pub fn with_geo(mut self, geo: GeoContext) -> Self {
        self.geo = Some(geo);
        self
    }

    /// Set the device context
    #[cfg(any(feature = "device", feature = "std"))]
    #[must_use]
    pub fn with_device(mut self, device: DeviceContext) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the load context
    #[cfg(any(feature = "load", feature = "std"))]
    #[must_use]
    pub fn with_load(mut self, load: LoadContext) -> Self {
        self.load = Some(load);
        self
    }

    /// Set the legal context
    #[cfg(any(feature = "legal", feature = "std"))]
    #[must_use]
    pub fn with_legal(mut self, legal: LegalContext) -> Self {
        self.legal = Some(legal);
        self
    }

    /// Set the timestamp
    #[must_use]
    pub const fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Check if the context is stale (older than max_age_secs)
    #[must_use]
    pub const fn is_stale(&self, current_time: u64, max_age_secs: u64) -> bool {
        if self.timestamp == 0 {
            return true;
        }
        current_time.saturating_sub(self.timestamp) > max_age_secs
    }

    /// Get the overall quality score of the context (0.0 - 1.0)
    #[must_use]
    pub fn quality_score(&self) -> f32 {
        #[cfg(any(
            feature = "geo",
            feature = "device",
            feature = "load",
            feature = "legal",
            feature = "std",
        ))]
        let mut score = 0.0_f32;
        #[cfg(not(any(
            feature = "geo",
            feature = "device",
            feature = "load",
            feature = "legal",
            feature = "std",
        )))]
        let score = 0.0_f32;

        #[cfg(any(
            feature = "geo",
            feature = "device",
            feature = "load",
            feature = "legal",
            feature = "std",
        ))]
        let mut count = 0.0_f32;
        #[cfg(not(any(
            feature = "geo",
            feature = "device",
            feature = "load",
            feature = "legal",
            feature = "std",
        )))]
        let count = 0.0_f32;

        #[cfg(feature = "geo")]
        {
            if self.geo.is_some() {
                score += 1.0;
            }
            count += 1.0;
        }

        #[cfg(any(feature = "device", feature = "std"))]
        {
            if self.device.is_some() {
                score += 1.0;
            }
            count += 1.0;
        }

        #[cfg(any(feature = "load", feature = "std"))]
        {
            if self.load.is_some() {
                score += 1.0;
            }
            count += 1.0;
        }

        #[cfg(any(feature = "legal", feature = "std"))]
        {
            if self.legal.is_some() {
                score += 1.0;
            }
            count += 1.0;
        }

        if count > 0.0 { score / count } else { 0.0 }
    }
}

/// Trait for providing context information
pub trait ContextProvider: Send + Sync {
    /// Get the combined context from all available sources
    fn get_combined_context(&self) -> CombinedContext;

    /// Get geo context (physical brain)
    #[cfg(feature = "geo")]
    fn get_geo_context(&self) -> Option<GeoContext> {
        None
    }

    /// Get device context (body brain)
    #[cfg(any(feature = "device", feature = "std"))]
    fn get_device_context(&self) -> Option<DeviceContext> {
        None
    }

    /// Get load context (situation brain)
    #[cfg(any(feature = "load", feature = "std"))]
    fn get_load_context(&self) -> Option<LoadContext> {
        None
    }

    /// Get legal context (social brain)
    #[cfg(any(feature = "legal", feature = "std"))]
    fn get_legal_context(&self) -> Option<LegalContext> {
        None
    }
}

/// Default context provider that returns no context
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultContextProvider;

impl ContextProvider for DefaultContextProvider {
    fn get_combined_context(&self) -> CombinedContext {
        #[cfg(any(
            feature = "geo",
            feature = "device",
            feature = "load",
            feature = "legal",
            feature = "std",
        ))]
        let mut ctx = CombinedContext::new();
        #[cfg(not(any(
            feature = "geo",
            feature = "device",
            feature = "load",
            feature = "legal",
            feature = "std",
        )))]
        let ctx = CombinedContext::new();

        #[cfg(feature = "geo")]
        {
            ctx.geo = self.get_geo_context();
        }

        #[cfg(any(feature = "device", feature = "std"))]
        {
            ctx.device = self.get_device_context();
        }

        #[cfg(any(feature = "load", feature = "std"))]
        {
            ctx.load = self.get_load_context();
        }

        #[cfg(any(feature = "legal", feature = "std"))]
        {
            ctx.legal = self.get_legal_context();
        }

        // Set timestamp
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            ctx.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
        }

        ctx
    }
}

/// Ecosystem context provider integrating all 4 brains via injectable sensors.
///
/// Each sensor slot is optional. When `None`, the corresponding brain returns
/// `None` — identical to the previous stub behavior. Wire in concrete sensors
/// (e.g., [`crate::context::oxigdal_sensor::StaticOxigdalGeoSensor`],
/// [`crate::context::mielin_sensor::MielinDeviceSensor`]) to get real context.
///
/// # Cache
///
/// Combined context is cached internally with a configurable TTL. The cache
/// uses a `Mutex<Option<CombinedContext>>` for interior mutability under `&self`.
/// On Mutex poison (a prior panic inside the lock), the poisoned guard is
/// recovered via `PoisonError::into_inner` rather than propagating panic.
/// On `try_lock` contention, context is recomputed without caching.
#[cfg(feature = "ecosystem")]
pub struct EcosystemContextProvider {
    /// Optional geographic sensor (physical brain).
    #[cfg(feature = "geo")]
    geo_sensor: Option<Box<dyn crate::context::sensor::GeoSensor>>,

    /// Optional device sensor (body brain).
    #[cfg(any(feature = "device", feature = "std"))]
    device_sensor: Option<Box<dyn crate::context::sensor::DeviceSensor>>,

    /// Optional load sensor (situation brain).
    #[cfg(any(feature = "load", feature = "std"))]
    load_sensor: Option<Box<dyn crate::context::sensor::LoadSensor>>,

    /// Optional policy engine (social brain).
    #[cfg(any(feature = "legal", feature = "std"))]
    policy_engine: Option<Box<dyn crate::context::sensor::PolicyEngine>>,

    /// Cache TTL in seconds.
    cache_ttl: u64,

    /// Mutex-guarded cache of the last combined context.
    #[cfg(feature = "std")]
    cached: std::sync::Mutex<Option<CombinedContext>>,
}

#[cfg(feature = "ecosystem")]
impl EcosystemContextProvider {
    /// Create a new ecosystem context provider with no sensors wired in.
    ///
    /// All brains return `None` until sensors are attached via the builder methods.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "geo")]
            geo_sensor: None,
            #[cfg(any(feature = "device", feature = "std"))]
            device_sensor: None,
            #[cfg(any(feature = "load", feature = "std"))]
            load_sensor: None,
            #[cfg(any(feature = "legal", feature = "std"))]
            policy_engine: None,
            cache_ttl: 60,
            #[cfg(feature = "std")]
            cached: std::sync::Mutex::new(None),
        }
    }

    /// Set the cache TTL in seconds (default: 60).
    #[must_use]
    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.cache_ttl = ttl_seconds;
        self
    }

    /// Attach a geographic sensor (physical brain).
    #[cfg(feature = "geo")]
    #[must_use]
    pub fn with_geo_sensor(mut self, sensor: Box<dyn crate::context::sensor::GeoSensor>) -> Self {
        self.geo_sensor = Some(sensor);
        self
    }

    /// Attach a device sensor (body brain).
    #[cfg(any(feature = "device", feature = "std"))]
    #[must_use]
    pub fn with_device_sensor(
        mut self,
        sensor: Box<dyn crate::context::sensor::DeviceSensor>,
    ) -> Self {
        self.device_sensor = Some(sensor);
        self
    }

    /// Attach a load sensor (situation brain).
    #[cfg(any(feature = "load", feature = "std"))]
    #[must_use]
    pub fn with_load_sensor(mut self, sensor: Box<dyn crate::context::sensor::LoadSensor>) -> Self {
        self.load_sensor = Some(sensor);
        self
    }

    /// Attach a policy engine (social brain).
    #[cfg(any(feature = "legal", feature = "std"))]
    #[must_use]
    pub fn with_policy_engine(
        mut self,
        engine: Box<dyn crate::context::sensor::PolicyEngine>,
    ) -> Self {
        self.policy_engine = Some(engine);
        self
    }

    /// Invalidate the internal cache, forcing recomputation on the next call.
    pub fn invalidate_cache(&self) {
        #[cfg(feature = "std")]
        {
            let mut guard = self
                .cached
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *guard = None;
        }
    }

    /// Query the legal context for the inferred jurisdiction.
    ///
    /// The jurisdiction string is derived by querying the geo sensor for a
    /// country code. Falls back to `"UNKNOWN"` if no geo sensor is wired in
    /// or if the sensor cannot determine a country.
    #[cfg(feature = "legal")]
    fn compute_legal_context(&self) -> Option<LegalContext> {
        let engine = self.policy_engine.as_ref()?;
        let jurisdiction = {
            #[cfg(feature = "geo")]
            {
                self.geo_sensor
                    .as_ref()
                    .and_then(|g| g.sense())
                    .and_then(|gc| gc.country_code)
                    .unwrap_or_else(|| "UNKNOWN".to_string())
            }
            #[cfg(not(feature = "geo"))]
            {
                "UNKNOWN".to_string()
            }
        };
        engine.evaluate_for_jurisdiction(&jurisdiction)
    }

    /// Build a fresh `CombinedContext` by querying all wired sensors.
    fn build_fresh(&self) -> CombinedContext {
        let mut ctx = CombinedContext::new();

        #[cfg(feature = "geo")]
        {
            ctx.geo = self.geo_sensor.as_ref().and_then(|s| s.sense());
        }

        #[cfg(any(feature = "device", feature = "std"))]
        {
            ctx.device = self.device_sensor.as_ref().and_then(|s| s.sense());
        }

        #[cfg(any(feature = "load", feature = "std"))]
        {
            ctx.load = self.load_sensor.as_ref().and_then(|s| s.sense());
        }

        #[cfg(feature = "legal")]
        {
            ctx.legal = self.compute_legal_context();
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            ctx.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
        }

        ctx
    }
}

#[cfg(feature = "ecosystem")]
impl Default for EcosystemContextProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "ecosystem")]
impl ContextProvider for EcosystemContextProvider {
    fn get_combined_context(&self) -> CombinedContext {
        #[cfg(feature = "std")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};

            // Fast path: try_lock to avoid blocking on contention.
            if let Ok(guard) = self.cached.try_lock() {
                if let Some(ref cached) = *guard {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    if !cached.is_stale(now, self.cache_ttl) {
                        return cached.clone();
                    }
                }
                // Cache miss or stale — drop the try_lock guard before building.
                drop(guard);
            }

            // Build fresh context outside the lock to avoid holding it during I/O.
            let fresh = self.build_fresh();

            // Store in cache — recover from poison rather than propagating.
            let mut guard = self
                .cached
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *guard = Some(fresh.clone());

            fresh
        }

        #[cfg(not(feature = "std"))]
        {
            // no_std: always build fresh (no Mutex, no time).
            self.build_fresh()
        }
    }

    #[cfg(feature = "geo")]
    fn get_geo_context(&self) -> Option<GeoContext> {
        self.geo_sensor.as_ref().and_then(|s| s.sense())
    }

    #[cfg(any(feature = "device", feature = "std"))]
    fn get_device_context(&self) -> Option<DeviceContext> {
        self.device_sensor.as_ref().and_then(|s| s.sense())
    }

    #[cfg(any(feature = "load", feature = "std"))]
    fn get_load_context(&self) -> Option<LoadContext> {
        self.load_sensor.as_ref().and_then(|s| s.sense())
    }

    #[cfg(any(feature = "legal", feature = "std"))]
    fn get_legal_context(&self) -> Option<LegalContext> {
        #[cfg(feature = "legal")]
        {
            Self::compute_legal_context(self)
        }
        #[cfg(not(feature = "legal"))]
        {
            self.policy_engine
                .as_ref()
                .and_then(|e| e.evaluate_for_jurisdiction("UNKNOWN"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_provider() {
        let provider = DefaultContextProvider;
        let ctx = provider.get_combined_context();
        assert!(ctx.timestamp > 0 || cfg!(not(feature = "std")));
    }

    #[test]
    fn test_combined_context_quality() {
        let ctx = CombinedContext::new();
        assert!(ctx.quality_score() >= 0.0);
    }
}
