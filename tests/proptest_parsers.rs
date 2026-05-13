//! Property-based tests for parsers and wire formats.
//!
//! These tests use proptest to verify that parsers never panic and that
//! various invariants hold across random inputs. Run with:
//!
//! ```bash
//! cargo nextest run --features full,http,p2p,agent,sparql,void --test proptest_parsers
//! ```

#[cfg(feature = "sparql")]
use proptest::prelude::*;

// ── SPARQL parser properties ──────────────────────────────────────────────────

/// Property tests for the SPARQL parser (`Query::from_sparql`).
///
/// All tests are gated on the `sparql` feature because `from_sparql` is only
/// compiled when that feature is active.
#[cfg(feature = "sparql")]
mod sparql_parser_props {
    use super::*;
    use oxirouter::Query;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 256,
            ..Default::default()
        })]

        /// Any ASCII string of 0–256 bytes must not cause a panic in the SPARQL parser.
        ///
        /// The parser is allowed to return `Ok` or `Err`; the only requirement is
        /// that it does not panic or abort.
        #[test]
        fn parser_does_not_panic(s in "[\x00-\x7f]{0,256}") {
            let _ = Query::from_sparql(&s);
        }

        /// Adding whitespace around tokens in a known-valid SELECT query must not
        /// change the Ok/Err outcome (both must succeed).
        ///
        /// The base query is always valid, so inserting extra spaces must still
        /// yield `Ok`.  We assert that both calls succeed and that both produce
        /// the same `query_type`.
        #[test]
        fn whitespace_invariance(
            prefix_spaces in " {0,8}",
            infix_spaces in " {1,4}",
        ) {
            let base = "SELECT ?x WHERE { }";
            let padded = format!(
                "{prefix_spaces}SELECT{infix_spaces}?x{infix_spaces}WHERE{infix_spaces}{{ }}"
            );

            // Both must parse without error
            let base_result = Query::from_sparql(base);
            let padded_result = Query::from_sparql(&padded);

            prop_assert!(base_result.is_ok(), "base query must parse: {:?}", base_result);
            prop_assert!(padded_result.is_ok(), "padded query must parse: {:?}", padded_result);

            // Both must agree on query_type
            let base_q = base_result.unwrap();
            let padded_q = padded_result.unwrap();
            prop_assert_eq!(
                base_q.query_type,
                padded_q.query_type,
                "query_type must be invariant to whitespace"
            );
        }

        /// Two PREFIX declarations in either order must yield the same prefix key set.
        ///
        /// If both orderings parse successfully, the set of declared prefix names
        /// (bare labels without the trailing colon) must be identical.
        #[test]
        fn prefix_decl_order_invariance(
            label1 in "[a-z]{2,6}",
            label2 in "[a-z]{2,6}",
        ) {
            prop_assume!(label1 != label2);

            let order1 = format!(
                "PREFIX {label1}: <http://example.org/{label1}/> \
                 PREFIX {label2}: <http://example.org/{label2}/> \
                 SELECT ?x WHERE {{ }}"
            );
            let order2 = format!(
                "PREFIX {label2}: <http://example.org/{label2}/> \
                 PREFIX {label1}: <http://example.org/{label1}/> \
                 SELECT ?x WHERE {{ }}"
            );

            let r1 = Query::from_sparql(&order1);
            let r2 = Query::from_sparql(&order2);

            // Only check invariant when both succeed
            if r1.is_ok() && r2.is_ok() {
                let q1 = r1.unwrap();
                let q2 = r2.unwrap();
                let mut keys1: Vec<_> = q1.prefixes.keys().cloned().collect();
                let mut keys2: Vec<_> = q2.prefixes.keys().cloned().collect();
                keys1.sort();
                keys2.sort();
                prop_assert_eq!(
                    keys1,
                    keys2,
                    "prefix key sets must be identical regardless of declaration order"
                );
            }
        }

        /// A string starting with "SELECT " followed by a variable and `WHERE { }`
        /// must parse as `QueryType::Select`.
        #[test]
        fn query_type_classification_select(var in "[a-z][a-z0-9_]{0,8}") {
            let query = format!("SELECT ?{var} WHERE {{ }}");
            let result = Query::from_sparql(&query);
            prop_assert!(result.is_ok(), "simple SELECT must parse: {:?}", result);

            let q = result.unwrap();
            prop_assert_eq!(
                q.query_type,
                oxirouter::core::query::QueryType::Select,
                "query_type must be Select for a SELECT query"
            );
        }

        /// A query with a triple pattern containing a full-IRI predicate must
        /// produce a non-empty `predicates` set.
        #[test]
        fn valid_predicates_non_empty_on_triple(local_name in "[a-zA-Z][a-zA-Z0-9_]{0,16}") {
            let query = format!(
                "SELECT ?s WHERE {{ ?s <http://example.org/{local_name}> ?o }}"
            );
            let result = Query::from_sparql(&query);
            prop_assert!(result.is_ok(), "triple query must parse: {:?}", result);

            let q = result.unwrap();
            prop_assert!(
                !q.predicates.is_empty(),
                "predicates must be non-empty when query has an IRI predicate"
            );
        }
    }
}

// ── Turtle/VoID parser properties ────────────────────────────────────────────

/// Property tests for the VoID/Turtle parser (`Router::register_from_void_ttl`).
///
/// Gated on the `void` feature.
#[cfg(feature = "void")]
mod turtle_parser_props {
    use super::*;
    use oxirouter::Router;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 256,
            ..Default::default()
        })]

        /// Passing an empty string to `register_from_void_ttl` must return `Ok`.
        ///
        /// An empty document has no `void:Dataset` entries, so the router
        /// acquires zero new sources.
        #[test]
        fn empty_input_ok(_dummy in 0u8..1u8) {
            let mut router = Router::new();
            let result = router.register_from_void_ttl("");
            prop_assert!(
                result.is_ok(),
                "empty TTL must return Ok, got: {:?}", result
            );
            prop_assert_eq!(
                router.source_count(),
                0,
                "no sources should be registered from empty input"
            );
        }

        /// A document consisting only of comment lines must return `Ok`.
        ///
        /// SPARQL/Turtle comments (`# ...`) are whitespace-equivalent; a file
        /// containing only comments is a valid (empty) document.
        #[test]
        fn comment_only_ok(n_lines in 1usize..8usize, suffix in "[a-zA-Z0-9 ]{0,32}") {
            let comment_block = (0..n_lines).fold(String::new(), |mut acc, i| {
                use std::fmt::Write;
                let _ = writeln!(acc, "# comment line {i}: {suffix}");
                acc
            });

            let mut router = Router::new();
            let result = router.register_from_void_ttl(&comment_block);
            prop_assert!(
                result.is_ok(),
                "comment-only TTL must return Ok, got: {:?}", result
            );
        }

        /// Feeding random short ASCII strings must not cause a panic.
        ///
        /// The parser may return `Ok` or `Err`; the only invariant is that it
        /// does not panic or abort.
        #[test]
        fn random_short_no_panic(s in "[ -~]{0,128}") {
            let mut router = Router::new();
            let _ = router.register_from_void_ttl(&s);
        }
    }
}

// ── Property-path parser properties ─────────────────────────────────────────

/// Property tests for property-path expressions.
///
/// `parse_property_path` is `pub(crate)` and not accessible from integration
/// tests.  All tests exercise it indirectly through `Query::from_sparql` with
/// a property-path expression embedded in a WHERE clause.
///
/// Gated on the `sparql` feature.
#[cfg(feature = "sparql")]
mod property_path_parser_props {
    use super::*;
    use oxirouter::Query;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 256,
            ..Default::default()
        })]

        /// Any random short string embedded as the predicate position must not
        /// cause a panic when the full query is fed to `from_sparql`.
        ///
        /// The parser may return `Ok` or `Err`; panics are the only failure.
        #[test]
        fn path_parser_no_panic(path_expr in "[ -~]{0,64}") {
            // Embed the random token as a predicate in an otherwise-valid query.
            let query = format!("SELECT ?s WHERE {{ ?s {path_expr} ?o }}");
            let _ = Query::from_sparql(&query);
        }

        /// A simple query with a full-IRI predicate must parse successfully and
        /// the IRI must appear in `query.predicates`.
        #[test]
        fn simple_iri_path_parses_in_query(local in "[a-zA-Z][a-zA-Z0-9]{0,12}") {
            let iri = format!("http://example.org/{local}");
            let query = format!("SELECT ?s WHERE {{ ?s <{iri}> ?o }}");

            let result = Query::from_sparql(&query);
            prop_assert!(result.is_ok(), "simple IRI path must parse: {:?}", result);

            let q = result.unwrap();
            prop_assert!(
                q.predicates.contains(&iri),
                "predicates must contain '{iri}', got: {:?}", q.predicates
            );
        }

        /// A deeply nested path expression must not cause a stack overflow.
        ///
        /// We wrap a simple IRI in 10 levels of parentheses, which corresponds
        /// to 10 recursive descent calls in the property-path parser.
        #[test]
        fn nesting_depth_safe(depth in 1usize..=10usize) {
            let base = "<http://example.org/p>";
            let nested = format!(
                "{}{base}{}",
                "(".repeat(depth),
                ")".repeat(depth),
            );
            let query = format!("SELECT ?s WHERE {{ ?s {nested} ?o }}");
            // Must not panic; Ok or Err are both acceptable
            let _ = Query::from_sparql(&query);
        }
    }
}

// ── RouterState wire format properties ───────────────────────────────────────

/// Property tests for the `RouterState` wire format (save/load round-trip).
///
/// Gated on the `std` feature (the binary wire format uses `serde_json` which
/// is available in std environments; `alloc` is also always enabled in std).
#[cfg(feature = "std")]
mod router_state_wire_props {
    use oxirouter::{DataSource, Router};
    #[cfg(not(feature = "sparql"))]
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 128,
            ..Default::default()
        })]

        /// A minimal router can be serialized and deserialized without error.
        ///
        /// The source count before and after the round-trip must match.
        #[test]
        fn state_roundtrip(n_sources in 0usize..=4usize) {
            let mut router = Router::new();
            for i in 0..n_sources {
                router.add_source(
                    DataSource::new(
                        format!("src-{i}"),
                        format!("https://example.org/sparql/{i}"),
                    )
                );
            }

            let bytes = router.save_state();
            prop_assert!(bytes.is_ok(), "save_state must succeed: {:?}", bytes);

            let bytes = bytes.unwrap();
            let mut fresh = Router::new();
            let load_result = fresh.load_state(&bytes);
            prop_assert!(
                load_result.is_ok(),
                "load_state must succeed after save_state: {:?}", load_result
            );
            prop_assert_eq!(
                fresh.source_count(),
                n_sources,
                "source count must be preserved across round-trip"
            );
        }

        /// Truncating valid state bytes by at least 1 byte must return `Err`,
        /// not panic.
        ///
        /// We take valid state bytes, lop off 1..N bytes from the end, and
        /// verify that `load_state` returns an error.
        #[test]
        fn state_truncation_handled(n_cut in 1usize..=16usize) {
            let mut router = Router::new();
            router.add_source(DataSource::new("src", "https://example.org/sparql"));

            let bytes = router.save_state().expect("save_state must not fail");

            // Guard: can only truncate up to bytes.len() - 1 bytes
            prop_assume!(n_cut < bytes.len());

            let truncated = &bytes[..bytes.len() - n_cut];
            let mut fresh = Router::new();
            let result = fresh.load_state(truncated);
            prop_assert!(
                result.is_err(),
                "load_state on truncated bytes must return Err, got Ok"
            );
        }

        /// Passing empty bytes to `load_state` must return `Err`, not panic.
        #[test]
        fn state_empty_bytes_err(_dummy in 0u8..1u8) {
            let mut router = Router::new();
            let result = router.load_state(&[]);
            prop_assert!(
                result.is_err(),
                "load_state(&[]) must return Err, got Ok"
            );
        }
    }
}
