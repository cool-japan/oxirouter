//! Integration tests for Block X: RouterConfig serialization, Timeout struct
//! variant, and RouterState v1→v2 wire format upgrade.
#![cfg(feature = "std")]

use oxirouter::prelude::*;
use oxirouter::{CircuitBreakerConfig, OxiRouterError, QueryLog, RouterConfig, RouterState};

/// Version constants matching the wire format (literals to avoid relying on
/// `pub(crate)` constants from the state module).
const V1: u32 = 1;
const V2: u32 = 2;

// ─── X-1: RouterConfig round-trip ────────────────────────────────────────────

/// Serialise a non-default `RouterConfig` to JSON and confirm all scalar
/// fields survive the round-trip.
#[test]
fn test_router_config_roundtrip_json() {
    let cfg = RouterConfig {
        max_sources: 12,
        min_confidence: 0.42,
        use_ml: false,
        use_context: false,
        timeout_us: 99_999,
        history_weight: 0.11,
        vocab_weight: 0.22,
        geo_weight: 0.33,
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 3,
            cooldown_ms: 15_000,
            now_ms: None,
        },
        max_response_bytes: 64 * 1024 * 1024,
        #[cfg(feature = "cache")]
        cache_enabled: false,
        #[cfg(feature = "cache")]
        cache_ttl_ms: 60_000,
        #[cfg(feature = "cache")]
        cache_max_entries: 500,
    };

    let json = serde_json::to_string(&cfg).expect("serialize failed");
    let restored: RouterConfig = serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(restored.max_sources, 12);
    assert!((restored.min_confidence - 0.42).abs() < 1e-5);
    assert!(!restored.use_ml);
    assert!(!restored.use_context);
    assert_eq!(restored.timeout_us, 99_999);
    assert!((restored.history_weight - 0.11).abs() < 1e-5);
    assert!((restored.vocab_weight - 0.22).abs() < 1e-5);
    assert!((restored.geo_weight - 0.33).abs() < 1e-5);
    assert_eq!(restored.circuit_breaker.failure_threshold, 3);
    assert_eq!(restored.circuit_breaker.cooldown_ms, 15_000);
    // now_ms is skipped by serde; it will be restored to the default value.
    let _ = restored.circuit_breaker.now_ms;
}

/// Write a JSON config to a temp file, load it via `RouterConfig::from_config_file`,
/// and confirm the values are correct. Also verifies `Router::with_config_file`.
#[test]
fn test_router_with_config_file() {
    let cfg = RouterConfig {
        max_sources: 7,
        min_confidence: 0.25,
        use_ml: true,
        use_context: false,
        timeout_us: 50_000,
        history_weight: 0.5,
        vocab_weight: 0.3,
        geo_weight: 0.2,
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 2,
            cooldown_ms: 5_000,
            now_ms: None,
        },
        max_response_bytes: 64 * 1024 * 1024,
        #[cfg(feature = "cache")]
        cache_enabled: true,
        #[cfg(feature = "cache")]
        cache_ttl_ms: 120_000,
        #[cfg(feature = "cache")]
        cache_max_entries: 200,
    };

    let json = serde_json::to_string(&cfg).expect("serialize failed");
    let path = std::env::temp_dir().join("oxi_test_config_file.json");
    std::fs::write(&path, &json).expect("write temp file failed");

    // Load via RouterConfig::from_config_file
    let loaded = RouterConfig::from_config_file(&path).expect("from_config_file failed");
    assert_eq!(loaded.max_sources, 7);
    assert!((loaded.min_confidence - 0.25).abs() < 1e-5);
    assert!(!loaded.use_context);
    assert_eq!(loaded.circuit_breaker.failure_threshold, 2);

    // Also verify Router::with_config_file works end-to-end
    let router = Router::with_config_file(&path).expect("Router::with_config_file failed");
    assert_eq!(router.source_count(), 0); // no sources added, just config

    // Clean up
    let _ = std::fs::remove_file(&path);
}

// ─── X-2: Timeout struct variant ─────────────────────────────────────────────

/// Construct a `Timeout` error and confirm all four fields are accessible.
#[test]
fn test_timeout_error_carries_context() {
    let err = OxiRouterError::Timeout {
        source_id: "test-src".into(),
        operation: "execute".into(),
        elapsed_ms: 500,
        deadline_ms: 100,
    };

    match &err {
        OxiRouterError::Timeout {
            source_id,
            operation,
            elapsed_ms,
            deadline_ms,
        } => {
            assert_eq!(source_id, "test-src");
            assert_eq!(operation, "execute");
            assert_eq!(*elapsed_ms, 500);
            assert_eq!(*deadline_ms, 100);
        }
        _ => panic!("expected Timeout variant"),
    }
}

/// Format the `Timeout` error and assert that all four context values appear
/// in the resulting string.
#[test]
fn test_timeout_display_message() {
    let err = OxiRouterError::Timeout {
        source_id: "test-src".into(),
        operation: "execute".into(),
        elapsed_ms: 500,
        deadline_ms: 100,
    };

    let msg = err.to_string();
    assert!(msg.contains("test-src"), "expected source_id in: {msg}");
    assert!(msg.contains("execute"), "expected operation in: {msg}");
    assert!(msg.contains("500"), "expected elapsed_ms in: {msg}");
    assert!(msg.contains("100"), "expected deadline_ms in: {msg}");
}

// ─── X-3: RouterState v1 → v2 migration ──────────────────────────────────────

/// Hand-craft a v1 state blob (version byte = 1, no `config` field in JSON)
/// and confirm that `RouterState::from_bytes` accepts it with `config == None`
/// and that `Router::load_state` leaves the current config unchanged.
#[test]
fn test_state_v1_loads_with_default_config() {
    // Build the JSON body that a v1 encoder would have produced (no config key).
    #[derive(serde::Serialize)]
    struct V1Body<'a> {
        version: u32,
        sources: Vec<oxirouter::DataSource>,
        model_bytes: Option<Vec<u8>>,
        rl_bytes: Option<Vec<u8>>,
        query_log: &'a QueryLog,
    }

    let ql = QueryLog::new();
    let body = V1Body {
        version: V1,
        sources: Vec::new(),
        model_bytes: None,
        rl_bytes: None,
        query_log: &ql,
    };
    let body_json = serde_json::to_vec(&body).expect("serialize v1 body failed");

    // Prepend magic + v1 version bytes.
    let mut blob = Vec::with_capacity(8 + body_json.len());
    blob.extend_from_slice(b"OXIR");
    blob.extend_from_slice(&V1.to_le_bytes());
    blob.extend_from_slice(&body_json);

    // Verify RouterState API: v1 blob has no config.
    let state = RouterState::from_bytes(&blob).expect("v1 blob should load cleanly");
    assert!(
        state.config().is_none(),
        "v1 blob should produce config == None"
    );

    // Build a router with a non-default config and load the v1 blob.
    // The router's config must be preserved because the blob carries no config.
    let custom_cfg = RouterConfig {
        max_sources: 13,
        min_confidence: 0.77,
        use_ml: false,
        use_context: false,
        timeout_us: 42_000,
        history_weight: 0.1,
        vocab_weight: 0.5,
        geo_weight: 0.4,
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 7,
            cooldown_ms: 12_345,
            now_ms: None,
        },
        max_response_bytes: 64 * 1024 * 1024,
        #[cfg(feature = "cache")]
        cache_enabled: false,
        #[cfg(feature = "cache")]
        cache_ttl_ms: 10_000,
        #[cfg(feature = "cache")]
        cache_max_entries: 50,
    };
    let mut router = Router::with_config(custom_cfg, oxirouter::DefaultContextProvider);
    router
        .load_state(&blob)
        .expect("load_state from v1 blob failed");

    // Config must be unchanged — v1 blobs carry no config field.
    assert_eq!(
        router.config().max_sources,
        13,
        "v1 load must not clobber router config (max_sources)"
    );
    assert!(
        (router.config().min_confidence - 0.77).abs() < 1e-5,
        "v1 load must not clobber router config (min_confidence)"
    );
    assert_eq!(
        router.source_count(),
        0,
        "no sources in v1 blob, so count should be 0"
    );
}

/// Build a router with a non-default config, `save_state`, then load into a
/// different router and confirm the config is restored.
#[test]
fn test_state_v2_roundtrip_with_config() {
    let custom_cfg = RouterConfig {
        max_sources: 9,
        min_confidence: 0.33,
        use_ml: false,
        use_context: true,
        timeout_us: 200_000,
        history_weight: 0.6,
        vocab_weight: 0.2,
        geo_weight: 0.2,
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 10,
            cooldown_ms: 60_000,
            now_ms: None,
        },
        max_response_bytes: 64 * 1024 * 1024,
        #[cfg(feature = "cache")]
        cache_enabled: false,
        #[cfg(feature = "cache")]
        cache_ttl_ms: 30_000,
        #[cfg(feature = "cache")]
        cache_max_entries: 100,
    };

    // Build the original router with the custom config.
    let mut original_router = Router::with_config(custom_cfg, oxirouter::DefaultContextProvider);
    original_router.add_source(DataSource::new("src-x", "https://x.example.org/sparql"));

    // Save state — this is a v2 blob containing the config.
    let bytes = original_router.save_state().expect("save_state failed");

    // Check that the blob carries v2 version number.
    let version_in_blob = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        version_in_blob, V2,
        "saved blob must carry version 2, got {version_in_blob}"
    );

    // Load into a router that starts with the default config.
    let mut restored_router = Router::new();
    restored_router
        .load_state(&bytes)
        .expect("load_state v2 failed");

    // Config should have been restored to the custom values — via RouterState API.
    let state = RouterState::from_bytes(&bytes).expect("from_bytes failed");
    let restored_cfg = state.config().expect("v2 state must carry a config");
    assert_eq!(restored_cfg.max_sources, 9);
    assert!((restored_cfg.min_confidence - 0.33).abs() < 1e-5);
    assert!(!restored_cfg.use_ml);
    assert_eq!(restored_cfg.circuit_breaker.failure_threshold, 10);
    assert_eq!(restored_cfg.circuit_breaker.cooldown_ms, 60_000);

    // Also verify the live router's config was actually updated by load_state.
    assert_eq!(
        restored_router.config().max_sources,
        9,
        "Router::load_state must apply the config from the v2 blob"
    );
    assert!(
        (restored_router.config().min_confidence - 0.33).abs() < 1e-5,
        "Router::load_state must restore min_confidence"
    );
    assert!(
        !restored_router.config().use_ml,
        "Router::load_state must restore use_ml"
    );

    // Source should also be restored.
    assert_eq!(restored_router.source_count(), 1);
    assert!(restored_router.get_source("src-x").is_some());
}
