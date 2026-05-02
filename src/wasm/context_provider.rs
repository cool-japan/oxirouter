//! WASM-side context provider for JS-injected brain context.
//!
//! `WasmContextProvider` lets JavaScript set geographic, device, load, and legal
//! context directly, bypassing hardware sensor access (unavailable in browser).
//! The provider is accessed concurrently through `Arc<WasmContextProvider>` so all
//! setters use interior mutability without requiring `&mut self`.
//!
//! On `std` targets a `Mutex` is used; on WASM32 (single-threaded) a `RefCell`
//! wrapped in an `UnsafeSync` newtype provides the same interior-mutability
//! guarantee with zero runtime overhead.

use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use core::cell::RefCell;

#[cfg(any(feature = "legal", feature = "std"))]
use crate::context::LegalContext;
#[cfg(any(feature = "load", feature = "std"))]
use crate::context::LoadContext;
use crate::context::{CombinedContext, ContextProvider};
#[cfg(any(feature = "device", feature = "std"))]
use crate::context::{DeviceContext, DeviceType, NetworkType};

#[cfg(feature = "geo")]
use crate::context::GeoContext;

// ---------------------------------------------------------------------------
// Platform-compatible interior mutability
// ---------------------------------------------------------------------------
//
// On `std` builds we use `std::sync::Mutex<T>`.
// On no_std WASM32 (single-threaded) we use `core::cell::RefCell<T>` wrapped
// in a newtype whose `Sync` impl is sound because WASM32-unknown-unknown
// provides no true concurrency at the module level.

/// Platform-independent inner lock type.
#[cfg(feature = "std")]
struct WasmCell<T>(std::sync::Mutex<T>);

/// Platform-independent inner lock type (no_std / WASM32 variant).
#[cfg(not(feature = "std"))]
struct WasmCell<T>(RefCell<T>);

// Safety: wasm32-unknown-unknown is single-threaded; no data races can occur.
// This impl is only compiled when `std` is absent (i.e., on WASM no_std builds).
#[cfg(not(feature = "std"))]
unsafe impl<T> Sync for WasmCell<T> {}

impl<T> WasmCell<T> {
    fn new(val: T) -> Self {
        #[cfg(feature = "std")]
        {
            Self(std::sync::Mutex::new(val))
        }
        #[cfg(not(feature = "std"))]
        {
            Self(RefCell::new(val))
        }
    }

    /// Acquire a mutable borrow and execute `f` with the guard.
    fn with_borrow_mut<R, F: FnOnce(&mut T) -> R>(&self, f: F) -> R {
        #[cfg(feature = "std")]
        {
            // On poisoned lock, recover the inner value.
            let mut guard = self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            f(&mut *guard)
        }
        #[cfg(not(feature = "std"))]
        {
            f(&mut self.0.borrow_mut())
        }
    }
}

/// WASM context provider that stores JS-injected context values.
///
/// Uses `WasmCell<CombinedContext>` for interior mutability.
/// All setter methods borrow the cell mutably and update the relevant field.
pub struct WasmContextProvider {
    inner: WasmCell<CombinedContext>,
}

impl WasmContextProvider {
    /// Create a new provider with default (empty) context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: WasmCell::new(CombinedContext::default()),
        }
    }

    /// Set geographic context from JS-provided lon/lat and country code.
    ///
    /// `lon` and `lat` follow the `(longitude, latitude)` convention used by
    /// `GeoContext::position`.
    #[cfg(feature = "geo")]
    pub fn set_geo(&self, country_code: &str, lon: f64, lat: f64) {
        self.inner.with_borrow_mut(|guard| {
            let mut ctx = GeoContext::new();
            ctx.position = Some((lon, lat));
            if !country_code.is_empty() {
                ctx = ctx.with_country(country_code.to_string());
            }
            guard.geo = Some(ctx);
        });
    }

    /// Set device context from JS-provided values.
    ///
    /// `network_type_u8` mapping:
    /// - 0 → Offline, 1 → Wifi, 2 → Cellular4G, 3 → Cellular5G,
    /// - 4 → Cellular3G, 5 → Cellular2G, 6 → Ethernet, 7 → Satellite,
    /// - other → Unknown
    #[cfg(any(feature = "device", feature = "std"))]
    pub fn set_device(
        &self,
        battery_pct: u8,
        network_type_u8: u8,
        bandwidth_kbps: u32,
        rtt_ms: u32,
    ) {
        self.inner.with_borrow_mut(|guard| {
            let network_type = match network_type_u8 {
                0 => NetworkType::Offline,
                1 => NetworkType::Wifi,
                2 => NetworkType::Cellular4G,
                3 => NetworkType::Cellular5G,
                4 => NetworkType::Cellular3G,
                5 => NetworkType::Cellular2G,
                6 => NetworkType::Ethernet,
                7 => NetworkType::Satellite,
                _ => NetworkType::Unknown,
            };

            let mut ctx = DeviceContext::new();
            ctx.battery_pct = if battery_pct <= 100 {
                Some(battery_pct)
            } else {
                None
            };
            ctx.network_type = network_type;
            ctx.bandwidth_kbps = if bandwidth_kbps > 0 {
                Some(bandwidth_kbps)
            } else {
                None
            };
            ctx.rtt_ms = if rtt_ms > 0 { Some(rtt_ms) } else { None };
            ctx.device_type = DeviceType::Unknown;
            guard.device = Some(ctx);
        });
    }

    /// Set device context — no-op stub when neither `device` nor `std` feature is enabled.
    #[cfg(not(any(feature = "device", feature = "std")))]
    pub fn set_device(
        &self,
        _battery_pct: u8,
        _network_type_u8: u8,
        _bandwidth_kbps: u32,
        _rtt_ms: u32,
    ) {
    }

    /// Set load context from JS-provided values.
    ///
    /// `global_load` is clamped to `[0.0, 1.0]`.
    #[cfg(any(feature = "load", feature = "std"))]
    pub fn set_load(&self, global_load: f32, pending_tasks: u32) {
        self.inner.with_borrow_mut(|guard| {
            let mut ctx = LoadContext::new();
            ctx.global_load = global_load.clamp(0.0, 1.0);
            ctx.pending_tasks = pending_tasks;
            guard.load = Some(ctx);
        });
    }

    /// Set load context — no-op stub when neither `load` nor `std` feature is enabled.
    #[cfg(not(any(feature = "load", feature = "std")))]
    pub fn set_load(&self, _global_load: f32, _pending_tasks: u32) {}

    /// Set legal context from JS-provided values.
    ///
    /// `blocked_regions_csv` is a comma-separated list of ISO 3166-1 alpha-2
    /// country codes (e.g. `"CN,RU,KP"`).
    #[cfg(any(feature = "legal", feature = "std"))]
    pub fn set_legal(&self, gdpr_region: bool, ccpa_applies: bool, blocked_regions_csv: &str) {
        self.inner.with_borrow_mut(|guard| {
            let mut ctx = LegalContext::new();
            ctx.gdpr_region = gdpr_region;
            ctx.ccpa_applies = ccpa_applies;
            if !blocked_regions_csv.is_empty() {
                for region in blocked_regions_csv.split(',') {
                    let r = region.trim().to_string();
                    if !r.is_empty() {
                        ctx.block_region(r);
                    }
                }
            }
            guard.legal = Some(ctx);
        });
    }

    /// Set legal context — no-op stub when neither `legal` nor `std` feature is enabled.
    #[cfg(not(any(feature = "legal", feature = "std")))]
    pub fn set_legal(&self, _gdpr_region: bool, _ccpa_applies: bool, _blocked_regions_csv: &str) {}

    /// Stamp the current Unix-epoch second onto a `CombinedContext`.
    #[cfg(feature = "std")]
    fn stamp_timestamp(ctx: &mut CombinedContext) {
        use std::time::{SystemTime, UNIX_EPOCH};
        ctx.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// No-op timestamp stamp when `std` is not available.
    #[cfg(not(feature = "std"))]
    fn stamp_timestamp(_ctx: &mut CombinedContext) {}
}

impl Default for WasmContextProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextProvider for WasmContextProvider {
    fn get_combined_context(&self) -> CombinedContext {
        self.inner.with_borrow_mut(|guard| {
            Self::stamp_timestamp(guard);
            guard.clone()
        })
    }
}

// ---------------------------------------------------------------------------
// WasmRouterContextAdapter
// ---------------------------------------------------------------------------

/// Thin adapter that implements [`ContextProvider`] by delegating to a shared
/// [`WasmContextProvider`] via `Arc`.
///
/// This allows `Router<WasmRouterContextAdapter>` to hold a generic context
/// provider while the `OxiRouter` WASM struct retains an independent
/// `Arc<WasmContextProvider>` for the JS setter methods.
///
/// `Send + Sync` are automatically derived because `Arc<WasmContextProvider>`
/// is `Send + Sync` when `WasmContextProvider: Send + Sync` (guaranteed by
/// `WasmCell`'s unsafe impl).
///
/// Available on all platforms (not just wasm32) so that host-side integration
/// tests can construct `Router<WasmRouterContextAdapter>` directly.
pub struct WasmRouterContextAdapter(pub Arc<WasmContextProvider>);

impl ContextProvider for WasmRouterContextAdapter {
    fn get_combined_context(&self) -> CombinedContext {
        self.0.get_combined_context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_context_provider_default() {
        let provider = WasmContextProvider::new();
        let ctx = provider.get_combined_context();
        // timestamp is only set on std builds
        #[cfg(feature = "std")]
        assert!(ctx.timestamp > 0);
        #[cfg(not(feature = "std"))]
        let _ = ctx;
    }

    #[cfg(any(feature = "device", feature = "std"))]
    #[test]
    fn test_set_device_network_types() {
        use crate::context::NetworkType;

        let provider = WasmContextProvider::new();

        // Test Wifi mapping
        provider.set_device(80, 1, 50_000, 20);
        let ctx = provider.get_combined_context();
        let device = ctx.device.expect("device should be set");
        assert_eq!(device.network_type, NetworkType::Wifi);
        assert_eq!(device.battery_pct, Some(80));
        assert_eq!(device.bandwidth_kbps, Some(50_000));
        assert_eq!(device.rtt_ms, Some(20));

        // Test Offline mapping
        provider.set_device(50, 0, 0, 0);
        let ctx = provider.get_combined_context();
        let device = ctx.device.expect("device should be set");
        assert_eq!(device.network_type, NetworkType::Offline);
        assert_eq!(device.bandwidth_kbps, None); // 0 → None
        assert_eq!(device.rtt_ms, None); // 0 → None
    }

    #[cfg(any(feature = "load", feature = "std"))]
    #[test]
    fn test_set_load() {
        let provider = WasmContextProvider::new();
        provider.set_load(0.75, 42);
        let ctx = provider.get_combined_context();
        let load = ctx.load.expect("load should be set");
        assert!((load.global_load - 0.75).abs() < f32::EPSILON);
        assert_eq!(load.pending_tasks, 42);
    }

    #[cfg(any(feature = "load", feature = "std"))]
    #[test]
    fn test_set_load_clamping() {
        let provider = WasmContextProvider::new();
        provider.set_load(5.0, 0); // Out of range — should clamp to 1.0
        let ctx = provider.get_combined_context();
        let load = ctx.load.expect("load should be set");
        assert!((load.global_load - 1.0).abs() < f32::EPSILON);
    }

    #[cfg(any(feature = "legal", feature = "std"))]
    #[test]
    fn test_set_legal_blocked_regions() {
        let provider = WasmContextProvider::new();
        provider.set_legal(true, false, "CN,RU, KP");
        let ctx = provider.get_combined_context();
        let legal = ctx.legal.expect("legal should be set");
        assert!(legal.gdpr_region);
        assert!(!legal.ccpa_applies);
        assert!(legal.blocked_regions.contains("CN"));
        assert!(legal.blocked_regions.contains("RU"));
        assert!(legal.blocked_regions.contains("KP"));
    }

    #[cfg(any(feature = "legal", feature = "std"))]
    #[test]
    fn test_set_legal_empty_regions() {
        let provider = WasmContextProvider::new();
        provider.set_legal(false, true, "");
        let ctx = provider.get_combined_context();
        let legal = ctx.legal.expect("legal should be set");
        assert!(!legal.gdpr_region);
        assert!(legal.ccpa_applies);
        assert!(legal.blocked_regions.is_empty());
    }

    #[test]
    fn test_wasm_router_context_adapter() {
        let provider = Arc::new(WasmContextProvider::new());
        #[cfg(any(feature = "load", feature = "std"))]
        provider.set_load(0.5, 10);
        let adapter = WasmRouterContextAdapter(Arc::clone(&provider));
        let ctx = adapter.get_combined_context();
        #[cfg(any(feature = "load", feature = "std"))]
        {
            let load = ctx.load.expect("load from adapter");
            assert!((load.global_load - 0.5).abs() < f32::EPSILON);
        }
        #[cfg(not(any(feature = "load", feature = "std")))]
        let _ = ctx;
    }
}
