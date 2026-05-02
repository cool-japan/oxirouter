//! Integration tests for the router-side circuit breaker.
//!
//! Each test uses a dedicated static `AtomicU64` clock to avoid cross-test
//! races when `cargo test` runs in parallel.

use std::sync::atomic::{AtomicU64, Ordering};

use oxirouter::prelude::*;

// ---------------------------------------------------------------------------
// Per-test mock clocks (one static + one fn per test).
// ---------------------------------------------------------------------------

static CLOCK_T1: AtomicU64 = AtomicU64::new(0);
fn now_t1() -> u64 {
    CLOCK_T1.load(Ordering::SeqCst)
}

static CLOCK_T2: AtomicU64 = AtomicU64::new(0);
fn now_t2() -> u64 {
    CLOCK_T2.load(Ordering::SeqCst)
}

static CLOCK_T3: AtomicU64 = AtomicU64::new(0);
fn now_t3() -> u64 {
    CLOCK_T3.load(Ordering::SeqCst)
}

static CLOCK_T4: AtomicU64 = AtomicU64::new(0);
fn now_t4() -> u64 {
    CLOCK_T4.load(Ordering::SeqCst)
}

static CLOCK_T5: AtomicU64 = AtomicU64::new(0);
fn now_t5() -> u64 {
    CLOCK_T5.load(Ordering::SeqCst)
}

static CLOCK_T6A: AtomicU64 = AtomicU64::new(0);
fn now_t6() -> u64 {
    CLOCK_T6A.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a router with one source ("src-a"), a custom CB config, and a given
/// `now_ms` clock function.
fn make_router_single_source(threshold: u32, cooldown_ms: u64, now_fn: fn() -> u64) -> Router {
    let mut config = RouterConfig::default();
    config.circuit_breaker = CircuitBreakerConfig {
        failure_threshold: threshold,
        cooldown_ms,
        now_ms: Some(now_fn),
    };
    config.use_ml = false; // heuristic only
    config.use_context = false;
    let mut router = Router::with_config(config, oxirouter::context::DefaultContextProvider);
    router.add_source(DataSource::new("src-a", "http://example.com/sparql"));
    router
}

/// Build a router with two sources ("src-a", "src-b"), a custom CB config, and
/// a given `now_ms` clock function.
fn make_router_two_sources(threshold: u32, cooldown_ms: u64, now_fn: fn() -> u64) -> Router {
    let mut config = RouterConfig::default();
    config.circuit_breaker = CircuitBreakerConfig {
        failure_threshold: threshold,
        cooldown_ms,
        now_ms: Some(now_fn),
    };
    config.use_ml = false;
    config.use_context = false;
    let mut router = Router::with_config(config, oxirouter::context::DefaultContextProvider);
    router.add_source(DataSource::new("src-a", "http://a.example.com/sparql"));
    router.add_source(DataSource::new("src-b", "http://b.example.com/sparql"));
    router
}

/// Fail source `source_id` `count` times using `learn_from_outcome`.
fn fail_source(router: &mut Router, source_id: &str, count: u32) {
    let dummy_query_id = 0u64;
    for _ in 0..count {
        router
            .learn_from_outcome(dummy_query_id, source_id, false, 1000, 0)
            .expect("learn_from_outcome failed");
    }
}

/// Succeed source `source_id` once using `learn_from_outcome`.
fn succeed_source(router: &mut Router, source_id: &str) {
    let dummy_query_id = 0u64;
    router
        .learn_from_outcome(dummy_query_id, source_id, true, 100, 10)
        .expect("learn_from_outcome succeeded");
}

/// Return true if `source_id` appears in the routing result.
fn source_in_route(router: &Router, source_id: &str) -> bool {
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse query");
    let ranking = router.route(&query).expect("route succeeded");
    ranking.sources.iter().any(|s| s.source_id == source_id)
}

// ---------------------------------------------------------------------------
// Test 1: failures below threshold do NOT trip the circuit breaker
// ---------------------------------------------------------------------------

#[test]
fn test_cb_failure_accumulation() {
    CLOCK_T1.store(1000, Ordering::SeqCst);

    let mut router = make_router_single_source(5, 30_000, now_t1);

    // Fail 4 times (threshold = 5); source must still be routable.
    fail_source(&mut router, "src-a", 4);

    // Source stats should have 4 consecutive failures but not be tripped.
    let src = router.get_source("src-a").expect("source exists");
    assert_eq!(src.stats.consecutive_failures, 4);
    assert!(
        src.stats.tripped_until_ms.is_none(),
        "source must not be tripped after only 4 failures"
    );

    assert!(
        source_in_route(&router, "src-a"),
        "source-a must still appear in routing with 4/5 failures"
    );
}

// ---------------------------------------------------------------------------
// Test 2: exactly threshold failures trip the circuit
// ---------------------------------------------------------------------------

#[test]
fn test_cb_trips_on_threshold() {
    CLOCK_T2.store(1000, Ordering::SeqCst);

    let mut router = make_router_single_source(5, 30_000, now_t2);

    // Fail exactly threshold times.
    fail_source(&mut router, "src-a", 5);

    let src = router.get_source("src-a").expect("source exists");
    assert_eq!(src.stats.consecutive_failures, 5);
    assert!(
        src.stats.tripped_until_ms.is_some(),
        "source must be tripped after threshold failures"
    );
}

// ---------------------------------------------------------------------------
// Test 3: tripped source is excluded from routing during cooldown
// ---------------------------------------------------------------------------

#[test]
fn test_cb_source_skipped_during_cooldown() {
    CLOCK_T3.store(1000, Ordering::SeqCst);

    let mut router = make_router_single_source(5, 30_000, now_t3);

    // Trip the source.
    fail_source(&mut router, "src-a", 5);

    // Clock is still at 1000 ms (well within the 30-second cooldown).
    // The route() must return NoSources since the only source is tripped.
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse query");
    let result = router.route(&query);

    // Either NoSources error or an empty ranking (depending on implementation).
    // Our implementation returns NoSources only if self.sources.is_empty();
    // since the source is registered but filtered, heuristic returns empty
    // ranking. The `route()` does NOT return NoSources in that case — the
    // post-processing just yields an empty ranking.
    // We assert the source is NOT in the result.
    match result {
        Err(OxiRouterError::NoSources { .. }) => {
            // Acceptable: no sources passed the filter.
        }
        Ok(ranking) => {
            assert!(
                !ranking.sources.iter().any(|s| s.source_id == "src-a"),
                "tripped source must not appear in routing during cooldown"
            );
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Test 4: source auto-recovers after cooldown elapses
// ---------------------------------------------------------------------------

#[test]
fn test_cb_auto_recovery_after_cooldown() {
    CLOCK_T4.store(1000, Ordering::SeqCst);

    let mut router = make_router_single_source(5, 30_000, now_t4);

    // Trip the source (tripped_until = 1000 + 30_000 = 31_000).
    fail_source(&mut router, "src-a", 5);

    // Advance clock past the cooldown end.
    CLOCK_T4.store(32_000, Ordering::SeqCst);

    // Source should now be included in routing again.
    assert!(
        source_in_route(&router, "src-a"),
        "source must be included in routing after cooldown elapses"
    );
}

// ---------------------------------------------------------------------------
// Test 5: a success resets consecutive failure counter
// ---------------------------------------------------------------------------

#[test]
fn test_cb_success_resets_counter() {
    CLOCK_T5.store(1000, Ordering::SeqCst);

    let mut router = make_router_single_source(5, 30_000, now_t5);

    // Fail 3 times, then succeed.
    fail_source(&mut router, "src-a", 3);
    succeed_source(&mut router, "src-a");

    let src = router.get_source("src-a").expect("source exists");
    assert_eq!(
        src.stats.consecutive_failures, 0,
        "consecutive_failures must reset to 0 after success"
    );
    assert!(
        src.stats.tripped_until_ms.is_none(),
        "tripped_until_ms must be None after success"
    );

    // Failing again from 0 should re-count: 3 more failures = 3, not 6.
    fail_source(&mut router, "src-a", 3);
    let src2 = router.get_source("src-a").expect("source exists");
    assert_eq!(
        src2.stats.consecutive_failures, 3,
        "consecutive_failures must restart from 0 after reset"
    );
    assert!(
        src2.stats.tripped_until_ms.is_none(),
        "source must not be tripped with only 3/5 failures after reset"
    );
}

// ---------------------------------------------------------------------------
// Test 6: tripping source A does not affect source B
// ---------------------------------------------------------------------------

#[test]
fn test_cb_multi_source_isolation() {
    CLOCK_T6A.store(1000, Ordering::SeqCst);

    let mut router = make_router_two_sources(5, 30_000, now_t6);

    // Trip only src-a.
    fail_source(&mut router, "src-a", 5);

    // src-a should be tripped.
    let src_a = router.get_source("src-a").expect("source exists");
    assert!(
        src_a.stats.tripped_until_ms.is_some(),
        "src-a must be tripped"
    );

    // src-b should be healthy and appear in routing.
    let src_b = router.get_source("src-b").expect("source exists");
    assert_eq!(
        src_b.stats.consecutive_failures, 0,
        "src-b must have 0 consecutive failures"
    );
    assert!(
        src_b.stats.tripped_until_ms.is_none(),
        "src-b must not be tripped"
    );

    assert!(
        source_in_route(&router, "src-b"),
        "healthy src-b must still appear in routing even when src-a is tripped"
    );

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse query");
    let ranking = router.route(&query).expect("route succeeded");
    assert!(
        !ranking.sources.iter().any(|s| s.source_id == "src-a"),
        "tripped src-a must NOT appear in routing"
    );
}
