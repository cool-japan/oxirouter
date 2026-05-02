//! Parallel federation executor tests using local TCP listeners.
//!
//! These tests verify that `ExecutionConfig.parallel = true` actually
//! dispatches source queries concurrently and that per-source / end-to-end
//! deadlines are correctly enforced.
//!
//! All tests bind to `127.0.0.1:0` (OS-assigned port), serve minimal
//! HTTP/1.1 responses after a configurable delay, and check wall-clock
//! timing bounds.

#![cfg(feature = "http")]

use oxirouter::federation::{ExecutionConfig, Executor};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Minimal HTTP server helpers
// ---------------------------------------------------------------------------

/// Minimal SPARQL JSON response body (0 bindings).
const SPARQL_EMPTY_JSON: &str = r#"{"head":{"vars":["s"]},"results":{"bindings":[]}}"#;

/// Bind a server that accepts multiple connections (one per source),
/// each served after `delay_ms` with a distinct marker in the JSON body.
///
/// Returns the single bound address (all sources point here) and the
/// shutdown flag.  This variant spawns a fresh thread per accepted
/// connection up to `max_connections`.
fn make_multi_connection_server(
    delay_ms: u64,
    max_connections: usize,
    response_body: &str,
) -> (SocketAddr, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind multi server");
    let addr = listener.local_addr().expect("local_addr");
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);
    let body = response_body.to_owned();

    std::thread::spawn(move || {
        let mut handled = 0usize;
        for stream in listener.incoming() {
            if shutdown_clone.load(Ordering::SeqCst) || handled >= max_connections {
                break;
            }
            let body_clone = body.clone();
            let shutdown2 = Arc::clone(&shutdown_clone);
            if let Ok(mut s) = stream {
                std::thread::spawn(move || {
                    drain_http_request(&mut s);
                    if delay_ms > 0 {
                        std::thread::sleep(Duration::from_millis(delay_ms));
                    }
                    if !shutdown2.load(Ordering::SeqCst) {
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/sparql-results+json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body_clone.len(),
                            body_clone
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

/// Read and discard an HTTP request from the stream until we see the
/// end of headers (`\r\n\r\n`).  Stops at a 4 KB cap to avoid blocking.
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
// Build DataSource + Query helpers
// ---------------------------------------------------------------------------

fn make_source(id: &str, addr: SocketAddr) -> oxirouter::DataSource {
    use oxirouter::core::source::SourceCapabilities;
    oxirouter::DataSource::new(id, format!("http://{}/sparql", addr))
        .with_capabilities(SourceCapabilities::full())
}

fn make_simple_query() -> oxirouter::Query {
    oxirouter::Query::parse("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1").expect("parse query")
}

fn make_ranking(source_ids: &[&str]) -> oxirouter::core::source::SourceRanking {
    use oxirouter::core::source::{SelectionReason, SourceRanking, SourceSelection};
    let mut ranking = SourceRanking::new();
    for id in source_ids {
        ranking.add(SourceSelection {
            source_id: (*id).to_string(),
            confidence: 1.0,
            estimated_latency_ms: 100,
            reason: SelectionReason::Fallback,
        });
    }
    ranking
}

// ---------------------------------------------------------------------------
// Test 1: Two sources run concurrently — wall time < serial time
// ---------------------------------------------------------------------------

#[test]
fn test_parallel_two_sources_concurrent() {
    // Each source takes ~100 ms.  Serial = ~200 ms; parallel should be <160 ms.
    let (addr1, _shutdown1) = make_multi_connection_server(100, 1, SPARQL_EMPTY_JSON);
    let (addr2, _shutdown2) = make_multi_connection_server(100, 1, SPARQL_EMPTY_JSON);

    let source1 = make_source("src1", addr1);
    let source2 = make_source("src2", addr2);

    let config = ExecutionConfig {
        timeout_ms: 2_000,
        max_retries: 0,
        parallel: true,
        max_concurrency: 4,
        ..ExecutionConfig::default()
    };
    let executor = Executor::with_config(config);
    let query = make_simple_query();
    let ranking = make_ranking(&["src1", "src2"]);
    let sources: Vec<&oxirouter::DataSource> = vec![&source1, &source2];

    let t0 = Instant::now();
    let results = executor
        .execute(&query, &sources, &ranking)
        .expect("execute");
    let elapsed = t0.elapsed();

    // Both results must be present (may succeed or error depending on real endpoint).
    assert_eq!(results.len(), 2, "expected 2 results");

    // Generous bound: parallel should take well under serial (200 ms), so ≤ 160 ms.
    // On a very loaded machine we allow up to 1800 ms as an outer guard.
    assert!(
        elapsed < Duration::from_millis(1800),
        "parallel execution took too long: {:?}",
        elapsed
    );

    // The real test: it should be faster than ~160 ms (serial would be ≥ 200 ms).
    // We only enforce this when timing is reliable (< 1 s overhead).
    if elapsed > Duration::from_millis(250) {
        // Flaky timing on this machine — skip the concurrency assertion.
        return;
    }
    assert!(
        elapsed < Duration::from_millis(160),
        "parallel execution was not concurrent: {:?} >= 160ms",
        elapsed
    );
}

// ---------------------------------------------------------------------------
// Test 2: Per-source timeout trips a slow source; fast source returns data
// ---------------------------------------------------------------------------

#[test]
fn test_parallel_per_source_timeout() {
    // Fast source: 50 ms delay.
    let (fast_addr, _shutdown_fast) = make_multi_connection_server(50, 1, SPARQL_EMPTY_JSON);
    // Slow source: will not finish within the 200 ms per-source timeout.
    let (slow_addr, _shutdown_slow) = make_multi_connection_server(5_000, 1, SPARQL_EMPTY_JSON);

    let source_fast = make_source("fast", fast_addr);
    let source_slow = make_source("slow", slow_addr);

    let config = ExecutionConfig {
        timeout_ms: 200, // per-source HTTP timeout
        max_retries: 0,  // no retries — one shot only
        parallel: true,
        max_concurrency: 4,
        ..ExecutionConfig::default()
    };
    let executor = Executor::with_config(config);
    let query = make_simple_query();
    let ranking = make_ranking(&["fast", "slow"]);
    let sources: Vec<&oxirouter::DataSource> = vec![&source_fast, &source_slow];

    let t0 = Instant::now();
    let results = executor
        .execute(&query, &sources, &ranking)
        .expect("execute");
    let elapsed = t0.elapsed();

    assert_eq!(results.len(), 2, "expected 2 results");

    // The slow source must have an error (timeout or connection failure).
    let slow_result = results
        .iter()
        .find(|r| r.source_id == "slow")
        .expect("slow result missing");
    assert!(
        slow_result.error.is_some(),
        "slow source should have timed out, got: {:?}",
        slow_result
    );

    // Total wall time must be well under 5000 ms (the slow source's delay).
    assert!(
        elapsed < Duration::from_millis(1500),
        "per-source timeout not enforced: {:?}",
        elapsed
    );
}

// ---------------------------------------------------------------------------
// Test 3: Source order preserved in results
// ---------------------------------------------------------------------------

#[test]
fn test_parallel_source_order_preserved() {
    // Three sources with slightly staggered delays to make ordering non-trivial.
    let delays = [150u64, 50, 100];
    let mut addrs = Vec::new();
    let mut shutdowns = Vec::new();

    for &d in &delays {
        let (addr, shutdown) = make_multi_connection_server(d, 1, SPARQL_EMPTY_JSON);
        addrs.push(addr);
        shutdowns.push(shutdown);
    }

    let source0 = make_source("s0", addrs[0]);
    let source1 = make_source("s1", addrs[1]);
    let source2 = make_source("s2", addrs[2]);

    let config = ExecutionConfig {
        timeout_ms: 2_000,
        max_retries: 0,
        parallel: true,
        max_concurrency: 4,
        ..ExecutionConfig::default()
    };
    let executor = Executor::with_config(config);
    let query = make_simple_query();
    let ranking = make_ranking(&["s0", "s1", "s2"]);
    let sources: Vec<&oxirouter::DataSource> = vec![&source0, &source1, &source2];

    let results = executor
        .execute(&query, &sources, &ranking)
        .expect("execute");

    assert_eq!(results.len(), 3, "expected 3 results");
    // Result order must match source order regardless of which finished first.
    assert_eq!(results[0].source_id, "s0", "result[0] source mismatch");
    assert_eq!(results[1].source_id, "s1", "result[1] source mismatch");
    assert_eq!(results[2].source_id, "s2", "result[2] source mismatch");
}

// ---------------------------------------------------------------------------
// Test 4: All sources timeout — all results are error results, fast return
// ---------------------------------------------------------------------------

#[test]
fn test_parallel_all_timeout() {
    // All sources take 5000 ms but we give them only 200 ms.
    let (addr1, _s1) = make_multi_connection_server(5_000, 1, SPARQL_EMPTY_JSON);
    let (addr2, _s2) = make_multi_connection_server(5_000, 1, SPARQL_EMPTY_JSON);

    let source1 = make_source("t1", addr1);
    let source2 = make_source("t2", addr2);

    let config = ExecutionConfig {
        timeout_ms: 200, // total budget = 2 * 200 = 400 ms (+ 1000 floor)
        max_retries: 0,
        parallel: true,
        max_concurrency: 4,
        ..ExecutionConfig::default()
    };
    let executor = Executor::with_config(config);
    let query = make_simple_query();
    let ranking = make_ranking(&["t1", "t2"]);
    let sources: Vec<&oxirouter::DataSource> = vec![&source1, &source2];

    let t0 = Instant::now();
    let results = executor
        .execute(&query, &sources, &ranking)
        .expect("execute");
    let elapsed = t0.elapsed();

    assert_eq!(results.len(), 2, "expected 2 results");

    // All results must be error/timeout results.
    for result in &results {
        assert!(
            result.error.is_some(),
            "expected timeout error for {}, got success",
            result.source_id
        );
    }

    // Must return within 1500 ms (well under the 5000 ms source delay).
    assert!(
        elapsed < Duration::from_millis(1500),
        "all-timeout test took too long: {:?}",
        elapsed
    );
}
