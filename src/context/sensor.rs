//! Sensor traits for the four-brain ecosystem context integration.
//!
//! These traits define the interface for injecting real hardware/ecosystem
//! context into [`EcosystemContextProvider`]. Null implementations return
//! `None`, preserving existing behavior when no real sensor is wired in.
//!
//! [`EcosystemContextProvider`]: crate::EcosystemContextProvider

#[cfg(feature = "geo")]
use super::GeoContext;

#[cfg(any(feature = "device", feature = "std"))]
use super::DeviceContext;

#[cfg(any(feature = "load", feature = "std"))]
use super::LoadContext;

#[cfg(any(feature = "legal", feature = "std"))]
use super::LegalContext;

// --- GeoSensor ---------------------------------------------------------------

/// Provides geographic context from a hardware or data source.
///
/// Implementations bridge real geospatial backends (e.g., `oxigdal_core`)
/// into the routing system. The `sense()` method should be cheap to call;
/// heavy I/O should be cached by the sensor itself.
#[cfg(feature = "geo")]
pub trait GeoSensor: Send + Sync {
    /// Sense the current geographic context. Returns `None` if unavailable.
    fn sense(&self) -> Option<GeoContext>;
}

/// Null geo sensor that always returns `None`.
///
/// Used as a placeholder when no real geographic sensor is configured.
#[cfg(feature = "geo")]
pub struct NullGeoSensor;

#[cfg(feature = "geo")]
impl GeoSensor for NullGeoSensor {
    fn sense(&self) -> Option<GeoContext> {
        None
    }
}

// --- DeviceSensor ------------------------------------------------------------

/// Provides hardware/device context from the local platform.
///
/// Implementations bridge hardware-detection libraries (e.g., `mielin_hal`)
/// into the routing system.
#[cfg(any(feature = "device", feature = "std"))]
pub trait DeviceSensor: Send + Sync {
    /// Sense the current device context. Returns `None` if unavailable.
    fn sense(&self) -> Option<DeviceContext>;
}

/// Null device sensor that always returns `None`.
///
/// Used as a placeholder when no real device sensor is configured.
#[cfg(any(feature = "device", feature = "std"))]
pub struct NullDeviceSensor;

#[cfg(any(feature = "device", feature = "std"))]
impl DeviceSensor for NullDeviceSensor {
    fn sense(&self) -> Option<DeviceContext> {
        None
    }
}

// --- LoadSensor --------------------------------------------------------------

/// Provides system load context (task queue, node health).
///
/// Implementations bridge distributed-task libraries (e.g., `celers_core`)
/// into the routing system.
#[cfg(any(feature = "load", feature = "std"))]
pub trait LoadSensor: Send + Sync {
    /// Sense the current load context. Returns `None` if unavailable.
    fn sense(&self) -> Option<LoadContext>;
}

/// Null load sensor that always returns `None`.
///
/// Used as a placeholder when no real load sensor is configured.
#[cfg(any(feature = "load", feature = "std"))]
pub struct NullLoadSensor;

#[cfg(any(feature = "load", feature = "std"))]
impl LoadSensor for NullLoadSensor {
    fn sense(&self) -> Option<LoadContext> {
        None
    }
}

// --- PolicyEngine ------------------------------------------------------------

/// Evaluates legal/compliance context for a given jurisdiction.
///
/// Implementations bridge legal-rule engines (e.g., `legalis_core`) into
/// the routing system. The jurisdiction string uses BCP 47-style codes:
/// `"EU"`, `"US-CA"`, `"BR"`, etc.
#[cfg(any(feature = "legal", feature = "std"))]
pub trait PolicyEngine: Send + Sync {
    /// Evaluate compliance for the given jurisdiction string (e.g., `"EU"`, `"US-CA"`).
    ///
    /// Returns `None` if the jurisdiction is not covered by this engine.
    fn evaluate_for_jurisdiction(&self, jurisdiction: &str) -> Option<LegalContext>;
}

/// Null policy engine that always returns `None`.
///
/// Used as a placeholder when no real policy engine is configured.
#[cfg(any(feature = "legal", feature = "std"))]
pub struct NullPolicyEngine;

#[cfg(any(feature = "legal", feature = "std"))]
impl PolicyEngine for NullPolicyEngine {
    fn evaluate_for_jurisdiction(&self, _jurisdiction: &str) -> Option<LegalContext> {
        None
    }
}
