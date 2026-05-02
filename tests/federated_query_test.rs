//! Integration tests for `Router::federated_query()` and `Router::route_and_execute()`.
//!
//! Each test binds to `127.0.0.1:0` (OS-assigned port), spawns a minimal HTTP/1.1
//! server thread, and calls the Router methods under test.

#![cfg(feature = "http")]

use oxirouter::core::source::SourceCapabilities;
use oxirouter::{AggregationStrategy, DataSource, Query, Router};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 server helpers
// ---------------------------------------------------------------------------

/// Minimal SPARQL JSON response with one binding.
const SPARQL_ONE_BINDING: &str = r#"{"head":{"vars":["name"]},"results":{"bindings":[{"name":{"type":"literal","value":"Test"}}]}}"#;

/// Bind a server that accepts `max_connections` connections, each served with
/// `response_status` ("200 OK" or "500 Internal Server Error") and the given body.
///
/// Returns (addr, shutdown_flag).
fn make_server(
    response_status: &'static str,
    response_body: &'static str,
    max_connections: usize,
) -> (SocketAddr, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind server");
    let addr = listener.local_addr().expect("local_addr");
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);

    std::thread::spawn(move || {
        let mut handled = 0usize;
        for stream in listener.incoming() {
            if shutdown_clone.load(Ordering::SeqCst) || handled >= max_connections {
                break;
            }
            let shutdown2 = Arc::clone(&shutdown_clone);
            if let Ok(mut s) = stream {
                std::thread::spawn(move || {
                    drain_http_request(&mut s);
                    if !shutdown2.load(Ordering::SeqCst) {
                        let content_type = if response_status.starts_with("200") {
                            "application/sparql-results+json"
                        } else {
                            "text/plain"
                        };
                        let response = format!(
                            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            response_status,
                            content_type,
                            response_body.len(),
                            response_body
                        );
                        let _ = s.write_all(response.as_bytes());
                    }
                });
                handled += 1;
            } else {
                break;
            }
        }
        shutdown_clone.store(true, Ordering::SeqCst);
    });

    (addr, shutdown)
}

/// Read and discard an HTTP request from the stream until end-of-headers.
fn drain_http_request(stream: &mut TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok();
    let mut buf = [0u8; 4096];
    let mut received = Vec::with_capacity(512);
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                received.extend_from_slice(&buf[..n]);
                if received.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
                if received.len() >= 4096 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build a DataSource pointing at a local address
// ---------------------------------------------------------------------------

fn make_source(id: &str, addr: SocketAddr) -> DataSource {
    DataSource::new(id, format!("http://{}/sparql", addr))
        .with_capabilities(SourceCapabilities::full())
}

fn make_query() -> Query {
    Query::parse("SELECT ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name } LIMIT 10")
        .expect("parse query")
}

// ---------------------------------------------------------------------------
// Test 1: First strategy returns a result from the first successful source
// ---------------------------------------------------------------------------

#[test]
fn test_federated_query_first_strategy() {
    // Two successful sources; max_connections=3 to survive any retry attempts.
    let (addr1, _s1) = make_server("200 OK", SPARQL_ONE_BINDING, 3);
    let (addr2, _s2) = make_server("200 OK", SPARQL_ONE_BINDING, 3);

    let mut router = Router::new();
    router.add_source(make_source("src1", addr1));
    router.add_source(make_source("src2", addr2));

    let query = make_query();
    let result = router
        .federated_query(&query, AggregationStrategy::First)
        .expect("federated_query should succeed");

    // First strategy: exactly one source contributes.
    assert_eq!(result.successful_sources, 1, "First strategy: one source");
    assert_eq!(result.failed_sources, 0, "First strategy: no failures");
    // row_count comes from parsing the JSON; should be 1 binding.
    assert_eq!(result.row_count, 1, "First strategy: one binding");
}

// ---------------------------------------------------------------------------
// Test 2: Union strategy merges results from all sources
// ---------------------------------------------------------------------------

#[test]
fn test_federated_query_union_strategy() {
    // Two successful sources, each returning 1 binding.
    let (addr1, _s1) = make_server("200 OK", SPARQL_ONE_BINDING, 3);
    let (addr2, _s2) = make_server("200 OK", SPARQL_ONE_BINDING, 3);

    let mut router = Router::new();
    router.add_source(make_source("src1", addr1));
    router.add_source(make_source("src2", addr2));

    let query = make_query();
    let result = router
        .federated_query(&query, AggregationStrategy::Union)
        .expect("union federated_query should succeed");

    // Union strategy: both sources contribute.
    assert_eq!(result.successful_sources, 2, "Union strategy: two sources");
    assert_eq!(result.failed_sources, 0, "Union strategy: no failures");
    // After deduplication (default=true), identical bindings merge → 1 row.
    // But we do not assert the exact row_count since dedup depends on config;
    // what matters is that data is non-empty and the call succeeded.
    assert!(!result.data.is_empty(), "Union result should have data");
}

// ---------------------------------------------------------------------------
// Test 3: All sources fail → Err (ExecutionError from aggregator)
// ---------------------------------------------------------------------------

#[test]
fn test_federated_query_all_sources_fail() {
    // Both servers return HTTP 500. Executor will retry (max_retries=2 default),
    // so each server needs 3 connections per source.
    let (addr1, _s1) = make_server("500 Internal Server Error", "error", 3);
    let (addr2, _s2) = make_server("500 Internal Server Error", "error", 3);

    let mut router = Router::new();
    router.add_source(make_source("src1", addr1));
    router.add_source(make_source("src2", addr2));

    let query = make_query();
    let result = router.federated_query(&query, AggregationStrategy::First);

    // When all sources fail, aggregate_first returns Err.
    assert!(
        result.is_err(),
        "Expected error when all sources fail, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Test 4: route_and_execute returns per-source results (one per source)
// ---------------------------------------------------------------------------

#[test]
fn test_route_and_execute_returns_per_source() {
    let (addr1, _s1) = make_server("200 OK", SPARQL_ONE_BINDING, 3);
    let (addr2, _s2) = make_server("200 OK", SPARQL_ONE_BINDING, 3);

    let mut router = Router::new();
    router.add_source(make_source("src1", addr1));
    router.add_source(make_source("src2", addr2));

    let query = make_query();
    let results = router
        .route_and_execute(&query)
        .expect("route_and_execute should succeed");

    // Both sources are registered; max_concurrency=4 covers both.
    assert_eq!(
        results.len(),
        2,
        "route_and_execute should return one result per source"
    );

    // Each result should carry a source_id.
    for r in &results {
        assert!(!r.source_id.is_empty(), "Each result must have a source_id");
        // Successful results have no error field.
        assert!(
            r.error.is_none(),
            "Expected successful result for {}",
            r.source_id
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: No sources registered → NoSources error from route()
// ---------------------------------------------------------------------------

#[test]
fn test_federated_query_no_sources() {
    let router = Router::new(); // no sources registered
    let query = make_query();

    let result = router.federated_query(&query, AggregationStrategy::First);

    assert!(
        result.is_err(),
        "Expected NoSources error, got: {:?}",
        result
    );

    // Verify it is the NoSources variant.
    let err = result.unwrap_err();
    assert!(
        matches!(err, oxirouter::OxiRouterError::NoSources { .. }),
        "Expected NoSources error variant, got: {:?}",
        err
    );
}
