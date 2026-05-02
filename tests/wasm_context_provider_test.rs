//! Host-side tests for `WasmContextProvider`.
//!
//! These tests verify the pure-Rust logic of context injection without
//! requiring a `wasm32` target or a browser runtime. All tests compile
//! and run on any platform that supports the `wasm` feature flag.

#[cfg(feature = "wasm")]
mod tests {
    use oxirouter::context::ContextProvider;
    use oxirouter::wasm::WasmContextProvider;

    // -----------------------------------------------------------------------
    // Basic construction and timestamp
    // -----------------------------------------------------------------------

    #[test]
    fn wasm_context_provider_default_has_timestamp() {
        let provider = WasmContextProvider::new();
        let ctx = provider.get_combined_context();
        // Timestamp must be non-zero (wall-clock stamped on the host).
        assert!(
            ctx.timestamp > 0,
            "timestamp should be stamped by get_combined_context"
        );
    }

    #[test]
    fn wasm_context_provider_default_trait_object() {
        // Ensure WasmContextProvider can be used as a trait object.
        let provider: Box<dyn ContextProvider> = Box::new(WasmContextProvider::new());
        let ctx = provider.get_combined_context();
        assert!(ctx.timestamp > 0);
    }

    // -----------------------------------------------------------------------
    // Device context
    // -----------------------------------------------------------------------

    #[test]
    fn wasm_context_provider_set_device_wifi() {
        use oxirouter::context::NetworkType;

        let provider = WasmContextProvider::new();
        provider.set_device(75, 1 /* Wifi */, 50_000, 15);

        let ctx = provider.get_combined_context();
        let device = ctx.device.expect("device context should be set");
        assert_eq!(device.network_type, NetworkType::Wifi);
        assert_eq!(device.battery_pct, Some(75));
        assert_eq!(device.bandwidth_kbps, Some(50_000));
        assert_eq!(device.rtt_ms, Some(15));
    }

    #[test]
    fn wasm_context_provider_set_device_offline() {
        use oxirouter::context::NetworkType;

        let provider = WasmContextProvider::new();
        // network_type_u8 = 0 → Offline
        // bandwidth_kbps = 0 → stored as None
        // rtt_ms = 0 → stored as None
        provider.set_device(100, 0, 0, 0);

        let ctx = provider.get_combined_context();
        let device = ctx.device.expect("device context should be set");
        assert_eq!(device.network_type, NetworkType::Offline);
        assert_eq!(device.bandwidth_kbps, None, "zero bandwidth maps to None");
        assert_eq!(device.rtt_ms, None, "zero rtt maps to None");
    }

    #[test]
    fn wasm_context_provider_network_type_mapping() {
        use oxirouter::context::NetworkType;

        let cases: &[(u8, NetworkType)] = &[
            (0, NetworkType::Offline),
            (1, NetworkType::Wifi),
            (2, NetworkType::Cellular4G),
            (3, NetworkType::Cellular5G),
            (4, NetworkType::Cellular3G),
            (5, NetworkType::Cellular2G),
            (6, NetworkType::Ethernet),
            (7, NetworkType::Satellite),
            (255, NetworkType::Unknown),
        ];

        let provider = WasmContextProvider::new();
        for (code, expected) in cases {
            provider.set_device(50, *code, 1000, 10);
            let ctx = provider.get_combined_context();
            let device = ctx.device.expect("device should be set");
            assert_eq!(
                device.network_type, *expected,
                "network_type_u8={code} should map to {expected:?}",
            );
        }
    }

    #[test]
    fn wasm_context_provider_battery_over_100_clamps() {
        let provider = WasmContextProvider::new();
        // battery_pct > 100 should store None (invalid value)
        provider.set_device(200, 1, 0, 0);
        let ctx = provider.get_combined_context();
        let device = ctx.device.expect("device should be set");
        assert_eq!(device.battery_pct, None, "battery > 100 should be None");
    }

    // -----------------------------------------------------------------------
    // Load context
    // -----------------------------------------------------------------------

    #[test]
    fn wasm_context_provider_set_load() {
        let provider = WasmContextProvider::new();
        provider.set_load(0.6, 100);

        let ctx = provider.get_combined_context();
        let load = ctx.load.expect("load context should be set");
        assert!(
            (load.global_load - 0.6).abs() < f32::EPSILON,
            "global_load should be 0.6"
        );
        assert_eq!(load.pending_tasks, 100);
    }

    #[test]
    fn wasm_context_provider_load_clamped_to_zero() {
        let provider = WasmContextProvider::new();
        provider.set_load(-5.0, 0);

        let ctx = provider.get_combined_context();
        let load = ctx.load.expect("load context should be set");
        assert!(
            load.global_load >= 0.0 && load.global_load <= 1.0,
            "global_load must be in [0,1], got {}",
            load.global_load
        );
    }

    #[test]
    fn wasm_context_provider_load_clamped_to_one() {
        let provider = WasmContextProvider::new();
        provider.set_load(99.9, 42);

        let ctx = provider.get_combined_context();
        let load = ctx.load.expect("load context should be set");
        assert!(
            (load.global_load - 1.0).abs() < f32::EPSILON,
            "global_load should be clamped to 1.0, got {}",
            load.global_load
        );
        assert_eq!(load.pending_tasks, 42);
    }

    // -----------------------------------------------------------------------
    // Legal context
    // -----------------------------------------------------------------------

    #[test]
    fn wasm_context_provider_set_legal_gdpr() {
        let provider = WasmContextProvider::new();
        provider.set_legal(true, false, "");

        let ctx = provider.get_combined_context();
        let legal = ctx.legal.expect("legal context should be set");
        assert!(legal.gdpr_region, "gdpr_region should be true");
        assert!(!legal.ccpa_applies, "ccpa_applies should be false");
        assert!(legal.blocked_regions.is_empty(), "no blocked regions");
    }

    #[test]
    fn wasm_context_provider_set_legal_blocked_regions() {
        let provider = WasmContextProvider::new();
        provider.set_legal(false, true, "CN,RU, KP");

        let ctx = provider.get_combined_context();
        let legal = ctx.legal.expect("legal context should be set");
        assert!(legal.blocked_regions.contains("CN"));
        assert!(legal.blocked_regions.contains("RU"));
        assert!(
            legal.blocked_regions.contains("KP"),
            "whitespace-trimmed region"
        );
        assert_eq!(legal.blocked_regions.len(), 3);
    }

    #[test]
    fn wasm_context_provider_set_legal_empty_csv() {
        let provider = WasmContextProvider::new();
        provider.set_legal(true, true, "");

        let ctx = provider.get_combined_context();
        let legal = ctx.legal.expect("legal context should be set");
        assert!(
            legal.blocked_regions.is_empty(),
            "empty CSV → no blocked regions"
        );
    }

    // -----------------------------------------------------------------------
    // Multiple successive updates
    // -----------------------------------------------------------------------

    #[test]
    fn wasm_context_provider_successive_updates() {
        let provider = WasmContextProvider::new();

        // First update
        provider.set_load(0.2, 5);
        let ctx1 = provider.get_combined_context();
        assert!((ctx1.load.as_ref().unwrap().global_load - 0.2).abs() < f32::EPSILON);

        // Second update — should replace the previous value
        provider.set_load(0.8, 50);
        let ctx2 = provider.get_combined_context();
        assert!((ctx2.load.as_ref().unwrap().global_load - 0.8).abs() < f32::EPSILON);
        assert_eq!(ctx2.load.unwrap().pending_tasks, 50);
    }

    // -----------------------------------------------------------------------
    // Combined context in router integration
    // -----------------------------------------------------------------------

    #[test]
    fn wasm_context_provider_integrates_with_router() {
        use oxirouter::core::router::Router;
        use oxirouter::wasm::context_provider::WasmRouterContextAdapter;
        use std::sync::Arc;

        let provider = Arc::new(WasmContextProvider::new());
        provider.set_load(0.3, 7);

        let adapter = WasmRouterContextAdapter(Arc::clone(&provider));
        let router = Router::with_context_provider(adapter);

        // Router should be constructable and have zero sources initially.
        assert_eq!(router.source_count(), 0);

        // The context provider should reflect the load we set.
        let ctx = provider.get_combined_context();
        assert!((ctx.load.unwrap().global_load - 0.3).abs() < f32::EPSILON);
    }
}
