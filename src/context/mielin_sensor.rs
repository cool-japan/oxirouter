//! MielinOS-backed device sensor.
//!
//! [`MielinDeviceSensor`] uses `mielin_hal` to detect CPU, memory, and platform
//! information. Battery and network state are **not** provided by `mielin_hal`
//! and intentionally remain `None`/`Unknown` — these require separate sensors
//! (e.g., browser Battery Status API, OS-level network APIs).

#[cfg(feature = "device")]
use super::{DeviceContext, sensor::DeviceSensor};

/// Device sensor backed by `mielin_hal`.
///
/// Provides:
/// - CPU core count (as a load proxy mapped to `cpu_load`)
/// - Available memory in MB from `system::MemoryInfo`
/// - Device type from `platform::detect_platform()`
///
/// Does **not** provide battery or network state — `mielin_hal` does not
/// expose those interfaces; they require OS/browser-specific APIs.
#[cfg(feature = "device")]
pub struct MielinDeviceSensor;

#[cfg(feature = "device")]
impl DeviceSensor for MielinDeviceSensor {
    fn sense(&self) -> Option<DeviceContext> {
        Some(DeviceContext::from_mielin_hal())
    }
}
