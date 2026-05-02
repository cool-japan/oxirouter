//! Context providers for the 4 brains integration
//!
//! This module provides context-awareness through integration with:
//! - **Physical brain** (oxigdal): Geo/location context
//! - **Body brain** (mielin): Device/hardware context
//! - **Situation brain** (celers): Load/task queue context
//! - **Social brain** (legalis): Legal/regulatory compliance

mod provider;

/// Sensor traits and null implementations for all four brains.
pub mod sensor;

#[cfg(feature = "geo")]
mod geo;

#[cfg(feature = "geo")]
mod oxigdal_sensor;

#[cfg(feature = "device")]
mod mielin_sensor;

#[cfg(feature = "load")]
mod celers_sensor;

#[cfg(feature = "legal")]
mod legalis_sensor;

#[cfg(any(feature = "device", feature = "std"))]
mod device;

#[cfg(any(feature = "load", feature = "std"))]
mod load;

#[cfg(any(feature = "legal", feature = "std"))]
mod legal;

pub use provider::{CombinedContext, ContextProvider, DefaultContextProvider};

// Sensor traits
#[cfg(feature = "geo")]
pub use sensor::GeoSensor;
#[cfg(feature = "geo")]
pub use sensor::NullGeoSensor;

#[cfg(any(feature = "device", feature = "std"))]
pub use sensor::DeviceSensor;
#[cfg(any(feature = "device", feature = "std"))]
pub use sensor::NullDeviceSensor;

#[cfg(any(feature = "load", feature = "std"))]
pub use sensor::LoadSensor;
#[cfg(any(feature = "load", feature = "std"))]
pub use sensor::NullLoadSensor;

#[cfg(any(feature = "legal", feature = "std"))]
pub use sensor::NullPolicyEngine;
#[cfg(any(feature = "legal", feature = "std"))]
pub use sensor::PolicyEngine;

// Concrete sensor types
#[cfg(feature = "geo")]
pub use oxigdal_sensor::{DynamicOxigdalGeoSensor, StaticOxigdalGeoSensor};

#[cfg(feature = "device")]
pub use mielin_sensor::MielinDeviceSensor;

#[cfg(feature = "load")]
pub use celers_sensor::CelersLoadSensor;

#[cfg(feature = "legal")]
pub use legalis_sensor::LegalisPolicyEngine;

#[cfg(feature = "geo")]
pub use geo::GeoContext;

/// Re-export `DeviceContext` and its supporting type enums for downstream use.
#[cfg(any(feature = "device", feature = "std"))]
pub use device::{DeviceContext, DeviceType, NetworkType};

#[cfg(any(feature = "load", feature = "std"))]
pub use load::{CircuitState, LoadContext};

#[cfg(any(feature = "legal", feature = "std"))]
pub use legal::LegalContext;

#[cfg(feature = "ecosystem")]
pub use provider::EcosystemContextProvider;
