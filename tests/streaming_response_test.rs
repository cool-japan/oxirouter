//! Streaming HTTP response with bounded buffer tests.
//!
//! Verifies that `Executor` streams response bodies chunk-by-chunk and aborts
//! with an appropriate error when the body exceeds `ExecutionConfig::max_response_bytes`.
//!
//! All tests use a hand-rolled local TCP server so no real network is needed.

#![cfg(all(feature = "http", not(target_arch = "wasm32")))]

use oxirouter::OxiRouterError;
use oxirouter::federation::{ExecutionConfig, Executor};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::thread;

// ---------------------------------------------------------------------------
// Minimal single-connection HTTP server helper
// ---------------------------------------------------------------------------

/// Spawn a thread that accepts one connection, drains the HTTP request,
/// then streams a response body of exactly `body_size` bytes (space characters)
/// with the correct `Content-Length` header.
fn start_mock_server(body_size: usize) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
    let addr = listener.local_addr().expect("local_addr");

    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            // Drain the HTTP request headers
            stream
                .set_read_timeout(Some(std::time::Duration::from_millis(500)))
                .ok();
            let mut req_buf = [0u8; 4096];
            let mut received: Vec<u8> = Vec::with_capacity(512);
            loop {
                match stream.read(&mut req_buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        received.extend_from_slice(&req_buf[..n]);
                        if received.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                        if received.len() >= 4096 {
                            break;
                        }
                    }
                }
            }

            // Send HTTP response headers
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/sparql-results+json\r\nContent-Length: {body_size}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(header.as_bytes());

            // Stream the body in 4 KiB chunks
            let chunk = vec![b' '; 4096];
            let mut sent = 0usize;
            while sent < body_size {
                let to_send = (body_size - sent).min(4096);
                if stream.write_all(&chunk[..to_send]).is_err() {
                    break;
                }
                sent += to_send;
            }
        }
    });

    addr
}

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

fn make_source(id: &str, addr: SocketAddr) -> oxirouter::DataSource {
    use oxirouter::core::source::SourceCapabilities;
    oxirouter::DataSource::new(id, format!("http://{addr}/sparql"))
        .with_capabilities(SourceCapabilities::full())
}

fn make_query() -> oxirouter::Query {
    oxirouter::Query::parse("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1").expect("parse query")
}

fn make_ranking(source_id: &str) -> oxirouter::core::source::SourceRanking {
    use oxirouter::core::source::{SelectionReason, SourceRanking, SourceSelection};
    let mut ranking = SourceRanking::new();
    ranking.add(SourceSelection {
        source_id: source_id.to_string(),
        confidence: 1.0,
        estimated_latency_ms: 100,
        reason: SelectionReason::Fallback,
    });
    ranking
}

// ---------------------------------------------------------------------------
// Test 1: Small response succeeds (error must NOT be ResponseTooLarge)
// ---------------------------------------------------------------------------

/// A 1 KiB response against a 64 MiB limit should be fully buffered.
/// The result may be a JSON parse failure (the body is spaces, not valid JSON),
/// but the error must not mention "too large".
#[test]
fn test_small_response_succeeds() {
    let addr = start_mock_server(1024);
    let source = make_source("src_small", addr);
    let query = make_query();
    let ranking = make_ranking("src_small");

    let config = ExecutionConfig {
        timeout_ms: 5_000,
        max_retries: 0,
        max_response_bytes: 64 * 1024 * 1024, // 64 MiB — well above 1 KiB
        ..ExecutionConfig::default()
    };
    let executor = Executor::with_config(config);

    let results = executor
        .execute(&query, &[&source], &ranking)
        .expect("execute must return results vec");

    assert_eq!(results.len(), 1, "expected exactly one result");
    let result = &results[0];

    // The body (spaces) is not valid SPARQL JSON, so an error is expected —
    // but it must NOT be a ResponseTooLarge error.
    if let Some(err_msg) = &result.error {
        assert!(
            !err_msg.contains("too large"),
            "small response must not trigger ResponseTooLarge: {err_msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: Oversized response aborts with ResponseTooLarge in error string
// ---------------------------------------------------------------------------

/// A 2 MiB response against a 1 MiB limit must abort early.
#[test]
fn test_oversized_response_aborts() {
    const BODY_SIZE: usize = 2 * 1024 * 1024; // 2 MiB
    const LIMIT: u64 = 1024 * 1024; // 1 MiB

    let addr = start_mock_server(BODY_SIZE);
    let source = make_source("src_large", addr);
    let query = make_query();
    let ranking = make_ranking("src_large");

    let config = ExecutionConfig {
        timeout_ms: 5_000,
        max_retries: 0,
        max_response_bytes: LIMIT,
        ..ExecutionConfig::default()
    };
    let executor = Executor::with_config(config);

    let results = executor
        .execute(&query, &[&source], &ranking)
        .expect("execute must return results vec");

    assert_eq!(results.len(), 1, "expected exactly one result");
    let result = &results[0];

    assert!(
        result.error.is_some(),
        "oversized response must produce an error result"
    );
    let err_msg = result.error.as_deref().unwrap_or("");
    assert!(
        err_msg.contains("too large"),
        "error must mention 'too large', got: {err_msg}"
    );
    assert!(
        err_msg.contains("src_large"),
        "error must mention the source id, got: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Exact limit boundary
// ---------------------------------------------------------------------------

/// Body exactly at the limit must succeed (not trigger ResponseTooLarge).
/// Body one byte over the limit must produce ResponseTooLarge.
#[test]
fn test_exact_limit_boundary() {
    const LIMIT: u64 = 32 * 1024; // 32 KiB
    let limit_usize = LIMIT as usize;

    // --- Exactly at limit: must succeed (no "too large" error) ---
    let addr_exact = start_mock_server(limit_usize);
    let source_exact = make_source("src_exact", addr_exact);
    let query = make_query();
    let ranking_exact = make_ranking("src_exact");

    let config_exact = ExecutionConfig {
        timeout_ms: 5_000,
        max_retries: 0,
        max_response_bytes: LIMIT,
        ..ExecutionConfig::default()
    };
    let executor_exact = Executor::with_config(config_exact);

    let results_exact = executor_exact
        .execute(&query, &[&source_exact], &ranking_exact)
        .expect("execute must return results vec");

    assert_eq!(results_exact.len(), 1);
    if let Some(err_msg) = &results_exact[0].error {
        assert!(
            !err_msg.contains("too large"),
            "body exactly at limit must not trigger ResponseTooLarge: {err_msg}"
        );
    }

    // --- One byte over limit: must fail with ResponseTooLarge ---
    let addr_over = start_mock_server(limit_usize + 1);
    let source_over = make_source("src_over", addr_over);
    let ranking_over = make_ranking("src_over");

    let config_over = ExecutionConfig {
        timeout_ms: 5_000,
        max_retries: 0,
        max_response_bytes: LIMIT,
        ..ExecutionConfig::default()
    };
    let executor_over = Executor::with_config(config_over);

    let results_over = executor_over
        .execute(&query, &[&source_over], &ranking_over)
        .expect("execute must return results vec");

    assert_eq!(results_over.len(), 1);
    assert!(
        results_over[0].error.is_some(),
        "body 1 byte over limit must produce an error"
    );
    let err_msg_over = results_over[0].error.as_deref().unwrap_or("");
    assert!(
        err_msg_over.contains("too large"),
        "error must mention 'too large', got: {err_msg_over}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: ResponseTooLarge Display formatting
// ---------------------------------------------------------------------------

/// Construct the error variant directly and verify its Display output contains
/// the source_id, observed_bytes, and limit_bytes values.
#[test]
fn test_response_too_large_display() {
    let err = OxiRouterError::ResponseTooLarge {
        source_id: "my-endpoint".to_string(),
        observed_bytes: 1_048_577,
        limit_bytes: 1_048_576,
    };
    let msg = format!("{err}");

    assert!(
        msg.contains("my-endpoint"),
        "Display must include source_id, got: {msg}"
    );
    assert!(
        msg.contains("1048577"),
        "Display must include observed_bytes, got: {msg}"
    );
    assert!(
        msg.contains("1048576"),
        "Display must include limit_bytes, got: {msg}"
    );
}
