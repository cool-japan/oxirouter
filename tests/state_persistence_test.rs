//! Integration tests for `Router::save_state` / `Router::load_state` (Block M).
#![cfg(feature = "std")]

use oxirouter::RouterState;
use oxirouter::prelude::*;

/// Build a router with two named sources.
fn make_two_source_router() -> Router {
    let mut r = Router::new();
    r.add_source(
        DataSource::new("src-a", "https://a.example.org/sparql")
            .with_vocabulary("http://example.org/a/"),
    );
    r.add_source(
        DataSource::new("src-b", "https://b.example.org/sparql")
            .with_vocabulary("http://example.org/b/"),
    );
    r
}

/// 1. Sources survive a save/load round-trip.
#[test]
fn test_state_roundtrip_sources() {
    let router = make_two_source_router();

    let bytes = router.save_state().expect("save_state failed");

    let mut restored = Router::new();
    restored.load_state(&bytes).expect("load_state failed");

    assert_eq!(restored.source_count(), 2, "expected 2 sources after load");
    assert!(
        restored.get_source("src-a").is_some(),
        "src-a should be present after load"
    );
    assert!(
        restored.get_source("src-b").is_some(),
        "src-b should be present after load"
    );
}

/// 2. ML: save/load on a router without a model does not error (gated on feature).
#[cfg(feature = "ml")]
#[test]
fn test_state_roundtrip_with_ml() {
    let router = Router::new();

    let bytes = router
        .save_state()
        .expect("save_state with no model failed");

    let mut restored = Router::new();
    restored
        .load_state(&bytes)
        .expect("load_state with no model failed");

    // No assertions about model presence — just verify no error.
}

/// 3. Query log is preserved across save/load.
#[test]
fn test_state_roundtrip_query_log() {
    let mut router = make_two_source_router();

    // Issue a query so the log has at least one entry.
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse failed");
    let ranking = router.route_and_log(&query).expect("route_and_log failed");

    // Record an outcome so source_stats are populated.
    if let Some(best) = ranking.best() {
        let source_id = best.source_id.clone();
        router
            .learn_from_outcome(query.predicate_hash(), &source_id, true, 100, 10)
            .expect("learn_from_outcome failed");

        let bytes = router.save_state().expect("save_state failed");

        let mut restored = Router::new();
        restored.load_state(&bytes).expect("load_state failed");

        // The query log should have stats for the source that was used.
        let stats = restored.query_log().source_stats(&source_id);
        assert!(
            stats.is_some(),
            "expected source_stats for {source_id} after load"
        );
    } else {
        // No best source available — just verify save/load succeeds.
        let bytes = router.save_state().expect("save_state failed");
        let mut restored = Router::new();
        restored.load_state(&bytes).expect("load_state failed");
    }
}

/// 4. Magic-byte mismatch is rejected with IncompatibleModel.
#[test]
fn test_state_magic_mismatch() {
    let mut bad_bytes = [0u8; 16];
    bad_bytes[0..4].copy_from_slice(b"BAAD");
    // Leave the rest as zeros.

    let err = RouterState::from_bytes(&bad_bytes).expect_err("expected error on bad magic");
    assert!(
        matches!(err, oxirouter::OxiRouterError::IncompatibleModel { .. }),
        "expected IncompatibleModel, got {err:?}"
    );
}

/// 5. Version mismatch is rejected with IncompatibleModel.
#[test]
fn test_state_version_mismatch() {
    let router = Router::new();
    let mut bytes = router.save_state().expect("save_state failed");

    // Overwrite the version field (bytes[4..8]) with version 99.
    bytes[4..8].copy_from_slice(&99u32.to_le_bytes());

    let err = RouterState::from_bytes(&bytes).expect_err("expected error on version mismatch");
    assert!(
        matches!(err, oxirouter::OxiRouterError::IncompatibleModel { .. }),
        "expected IncompatibleModel, got {err:?}"
    );
}

/// 6. End-to-end smoke test: save → load → route works without error.
#[test]
fn test_state_roundtrip_no_context_change() {
    let router = make_two_source_router();

    let bytes = router.save_state().expect("save_state failed");

    let mut restored = Router::new();
    restored.load_state(&bytes).expect("load_state failed");

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse failed");
    let ranking = restored
        .route(&query)
        .expect("route failed after load_state");

    assert!(
        !ranking.is_empty(),
        "expected non-empty routing result after restore"
    );
}
