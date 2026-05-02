//! Tests for the `observability` feature gate: tracing spans and metrics.
//!
//! All tests in this file require `--features observability` to compile and run.
//! The default (no `observability`) build must still compile cleanly — verified
//! by the separate clippy/test runs in CI.

#[cfg(feature = "observability")]
mod obs_tests {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use oxirouter::prelude::*;

    /// Build a router with two sources for use in multiple tests.
    fn build_router() -> Router {
        let mut router = Router::new();
        let s1 = DataSource::new("src-a", "http://a.example.com/sparql");
        let s2 = DataSource::new("src-b", "http://b.example.com/sparql");
        router.add_source(s1);
        router.add_source(s2);
        router
    }

    /// Build a simple query referencing a predicate URI.
    fn build_query() -> Query {
        Query::parse("SELECT ?s WHERE { ?s <http://example.org/pred1> ?o }").expect("query parse")
    }

    /// Test 1: Tracing infrastructure compiles and runs without panic.
    ///
    /// We install a tracing subscriber for the duration of the call and assert
    /// that `Router::route()` executes without panicking. This validates that
    /// the `#[cfg_attr(feature = "observability", tracing::instrument(...))]`
    /// attributes compile and behave correctly at runtime.
    #[test]
    fn test_route_emits_tracing_span() {
        use tracing_subscriber::fmt;

        // Write to a sink that ignores output — we only care that no panic occurs.
        let subscriber = fmt::Subscriber::builder()
            .with_writer(std::io::sink)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            let router = build_router();
            let query = build_query();
            // Should succeed and not panic regardless of subscriber state.
            let result = router.route(&query);
            assert!(
                result.is_ok(),
                "route() failed under tracing subscriber: {result:?}"
            );
        });
    }

    /// Test 2: `route()` increments `oxirouter.route.total` counter.
    ///
    /// We use `metrics::with_local_recorder` for per-test isolation so that
    /// multiple tests can coexist without fighting over a global recorder.
    #[test]
    fn test_route_emits_counter() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            let router = build_router();
            let query = build_query();

            // Route three times to make the counter value unambiguous.
            for _ in 0..3 {
                let _ = router.route(&query);
            }

            let snapshot = snapshotter.snapshot();
            let data = snapshot.into_vec();

            // Find `oxirouter.route.total` counter.
            let route_total = data
                .iter()
                .find(|(ck, _, _, _)| ck.key().name() == "oxirouter.route.total");

            assert!(
                route_total.is_some(),
                "expected oxirouter.route.total counter but got metrics: {:?}",
                data.iter()
                    .map(|(ck, _, _, _)| ck.key().name())
                    .collect::<Vec<_>>()
            );

            if let Some((_, _, _, DebugValue::Counter(count))) = route_total {
                assert_eq!(*count, 3, "expected counter = 3 after 3 route() calls");
            } else {
                panic!("oxirouter.route.total is not a Counter");
            }
        });
    }

    /// Test 3: `route()` emits `oxirouter.route.total` across multiple calls,
    /// and `oxirouter.route.confidence` histogram receives observations.
    ///
    /// (A true federation test would require real HTTP; this validates the
    ///  metrics plumbing without network I/O.)
    #[test]
    fn test_federation_metrics_via_multiple_routes() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        metrics::with_local_recorder(&recorder, || {
            let router = build_router();
            let query = build_query();

            // Call route() 5 times.
            for _ in 0..5 {
                let _ = router.route(&query);
            }

            let snapshot = snapshotter.snapshot();
            let data = snapshot.into_vec();

            // oxirouter.route.total must be 5.
            let route_total = data
                .iter()
                .find(|(ck, _, _, _)| ck.key().name() == "oxirouter.route.total");
            assert!(route_total.is_some(), "oxirouter.route.total not found");
            if let Some((_, _, _, DebugValue::Counter(c))) = route_total {
                assert_eq!(*c, 5, "counter mismatch after 5 route() calls");
            }

            // oxirouter.route.confidence histogram must have observations.
            let conf_hist = data
                .iter()
                .find(|(ck, _, _, _)| ck.key().name() == "oxirouter.route.confidence");
            assert!(
                conf_hist.is_some(),
                "oxirouter.route.confidence histogram not found"
            );
            if let Some((_, _, _, DebugValue::Histogram(vals))) = conf_hist {
                assert!(
                    !vals.is_empty(),
                    "oxirouter.route.confidence histogram has no observations"
                );
            }
        });
    }

    /// Test 4: Compile-time verification that default (no `observability`) builds cleanly.
    ///
    /// This is a documentation / marker test.  The real check is the CI gate that runs:
    ///
    /// ```
    /// cargo clippy --features full,http,p2p,agent,sparql,void --all-targets -- -D warnings
    /// cargo nextest run --features full,http,p2p,agent,sparql,void --all-targets
    /// ```
    ///
    /// In both cases the `observability` feature is absent and every
    /// `#[cfg(feature = "observability")]` block is compiled out completely —
    /// zero tracing or metrics code leaks into the binary.
    #[test]
    fn test_no_observability_compile_guard() {
        // This test exists solely to document the compile-time contract:
        // the default build (without `observability`) must compile cleanly
        // because every tracing/metrics call is inside `#[cfg(feature = "observability")]`
        // blocks. The actual check is performed by the CI gate that runs clippy
        // and nextest without this feature enabled.
        let _ = "default build compiles without observability code";
    }
}
