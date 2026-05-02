//! Tests for capability-aware scoring components in `compute_source_components`.
//!
//! Verifies that `capability_federation_match`, `capability_aggregation_match`,
//! `capability_property_paths_match`, `capability_subqueries_match`, and
//! `result_density` components are produced correctly by `Router::explain`.

use oxirouter::RoutingExplanation;
use oxirouter::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Capability flags for test source construction.
///
/// Wraps the 4 bool flags to avoid the `fn_params_excessive_bools` lint.
#[derive(Clone, Copy, Default)]
struct TestCaps {
    federation: bool,
    aggregation: bool,
    property_paths: bool,
    subqueries: bool,
}

impl TestCaps {
    const fn with_federation() -> Self {
        Self {
            federation: true,
            aggregation: false,
            property_paths: false,
            subqueries: false,
        }
    }

    const fn with_aggregation() -> Self {
        Self {
            federation: false,
            aggregation: true,
            property_paths: false,
            subqueries: false,
        }
    }

    const fn with_property_paths() -> Self {
        Self {
            federation: false,
            aggregation: false,
            property_paths: true,
            subqueries: false,
        }
    }

    const fn with_subqueries() -> Self {
        Self {
            federation: false,
            aggregation: false,
            property_paths: false,
            subqueries: true,
        }
    }
}

/// Build a `DataSource` with explicit capability flags.
///
/// All sources get `sparql_1_1: true` so that the hard capability gate
/// (`requires_sparql_1_1`) does not drop them from `explain()` output
/// even when the query uses SPARQL 1.1 features (SERVICE, GROUP BY, etc.).
fn make_source(id: &str, caps: TestCaps) -> DataSource {
    DataSource::new(id, format!("https://{id}.example.org/sparql")).with_capabilities(
        SourceCapabilities {
            sparql_1_1: true,
            federation: caps.federation,
            aggregation: caps.aggregation,
            property_paths: caps.property_paths,
            subqueries: caps.subqueries,
            ..SourceCapabilities::default()
        },
    )
}

/// Build a two-source router: one "capable" (cap under test is true), one "incapable"
/// (only `sparql_1_1: true`, all specific caps false).
fn make_router_pair(capable_caps: TestCaps) -> Router {
    let mut router = Router::new();
    router.add_source(make_source("capable", capable_caps));
    // Incapable source: sparql_1_1 true but all specific caps are false.
    router.add_source(make_source("incapable", TestCaps::default()));
    router
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: find a named component in an explanation
// ─────────────────────────────────────────────────────────────────────────────

fn find_component<'a>(
    explanations: &'a [RoutingExplanation],
    source_id: &str,
    component_name: &str,
) -> Option<&'a oxirouter::ScoreComponent> {
    explanations
        .iter()
        .find(|e| e.source_id == source_id)
        .and_then(|e| e.components.iter().find(|c| c.name == component_name))
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: SERVICE query gives positive federation match for capable source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_service_clause_increases_federation_capable_score() {
    let router = make_router_pair(TestCaps::with_federation());

    let query = Query::parse(
        "SELECT ?s WHERE { SERVICE <https://remote.example.org/sparql> { ?s ?p ?o } }",
    )
    .expect("parse SERVICE query");

    let explanations = router.explain(&query).expect("explain succeeded");

    // Capable source: positive contribution
    let comp = find_component(&explanations, "capable", "capability_federation_match")
        .expect("capable source must have capability_federation_match component");
    assert!(
        comp.contribution > 0.0,
        "capable source federation contribution must be positive; got {}",
        comp.contribution
    );
    assert!(
        (comp.raw_value - 1.0_f32).abs() < f32::EPSILON,
        "capable source federation raw_value must be 1.0; got {}",
        comp.raw_value
    );

    // Incapable source: negative contribution
    let comp_neg = find_component(&explanations, "incapable", "capability_federation_match")
        .expect("incapable source must have capability_federation_match component");
    assert!(
        comp_neg.contribution < 0.0,
        "incapable source federation contribution must be negative; got {}",
        comp_neg.contribution
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: GROUP BY routes to aggregation-capable source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_aggregation_query_routes_to_aggregation_capable() {
    let router = make_router_pair(TestCaps::with_aggregation());

    let query =
        Query::parse("SELECT ?type (COUNT(?s) AS ?count) WHERE { ?s a ?type } GROUP BY ?type")
            .expect("parse GROUP BY query");

    let explanations = router.explain(&query).expect("explain succeeded");

    let comp_pos = find_component(&explanations, "capable", "capability_aggregation_match")
        .expect("capable source must have capability_aggregation_match");
    assert!(
        comp_pos.contribution > 0.0,
        "capable source aggregation contribution must be positive; got {}",
        comp_pos.contribution
    );

    let comp_neg = find_component(&explanations, "incapable", "capability_aggregation_match")
        .expect("incapable source must have capability_aggregation_match");
    assert!(
        comp_neg.contribution < 0.0,
        "incapable source aggregation contribution must be negative; got {}",
        comp_neg.contribution
    );

    // The capable source should rank higher via route()
    let ranking = router.route(&query).expect("route succeeded");
    assert_eq!(
        ranking.best().map(|s| s.source_id.as_str()),
        Some("capable"),
        "aggregation-capable source must rank first"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: property-path query routes to path-capable source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_property_path_query_routes_to_path_capable() {
    let router = make_router_pair(TestCaps::with_property_paths());

    // Use a query with an inverse property-path (^), which the heuristic
    // parser detects via `query.contains('^')`.
    let query = Query::parse("SELECT ?s WHERE { ?s ^<http://ex.org/parent> ?o }")
        .expect("parse path query");

    // Verify the parser actually flagged it — fail loudly if the parser changes.
    assert!(
        query.has_property_paths,
        "query with '^' must have has_property_paths=true; heuristic parser may have changed"
    );

    let explanations = router.explain(&query).expect("explain succeeded");

    let comp_pos = find_component(&explanations, "capable", "capability_property_paths_match")
        .expect("capable source must have capability_property_paths_match");
    assert!(
        comp_pos.contribution > 0.0,
        "capable source property_paths contribution must be positive; got {}",
        comp_pos.contribution
    );

    let comp_neg = find_component(
        &explanations,
        "incapable",
        "capability_property_paths_match",
    )
    .expect("incapable source must have capability_property_paths_match");
    assert!(
        comp_neg.contribution < 0.0,
        "incapable source property_paths contribution must be negative; got {}",
        comp_neg.contribution
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: nested SELECT routes to subquery-capable source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_subquery_routes_to_subquery_capable() {
    let router = make_router_pair(TestCaps::with_subqueries());

    let query = Query::parse("SELECT ?s WHERE { { SELECT ?s WHERE { ?s ?p ?o } LIMIT 10 } }")
        .expect("parse subquery");

    let explanations = router.explain(&query).expect("explain succeeded");

    let comp_pos = find_component(&explanations, "capable", "capability_subqueries_match")
        .expect("capable source must have capability_subqueries_match");
    assert!(
        comp_pos.contribution > 0.0,
        "capable source subqueries contribution must be positive; got {}",
        comp_pos.contribution
    );

    let comp_neg = find_component(&explanations, "incapable", "capability_subqueries_match")
        .expect("incapable source must have capability_subqueries_match");
    assert!(
        comp_neg.contribution < 0.0,
        "incapable source subqueries contribution must be negative; got {}",
        comp_neg.contribution
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: avg_results_per_query produces result_density component
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_total_results_density_component() {
    let mut router = Router::new();

    // Source with query history: 10 queries, 500 results → avg 50.0
    let mut source = DataSource::new("dense", "https://dense.example.org/sparql");
    source.stats.total_queries = 10;
    source.stats.total_results = 500;
    source.stats.successful_queries = 10;
    source.stats.success_rate = 1.0;
    router.add_source(source);

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse simple query");
    let explanations = router.explain(&query).expect("explain succeeded");

    let comp = find_component(&explanations, "dense", "result_density")
        .expect("result_density component must be present");

    assert!(
        comp.raw_value > 0.0,
        "result_density raw_value must be > 0.0 for source with avg 50 results; got {}",
        comp.raw_value
    );

    // avg 50.0 / 100.0 = 0.5
    let expected = (50.0_f32 / 100.0_f32).min(1.0_f32);
    let diff = (comp.raw_value - expected).abs();
    assert!(
        diff < 1e-4,
        "result_density raw_value expected {expected:.4}, got {:.4}",
        comp.raw_value
    );
    assert!(
        comp.contribution > 0.0,
        "result_density contribution must be positive; got {}",
        comp.contribution
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: capability mismatch produces negative contribution
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_capability_mismatch_negative_contribution() {
    let mut router = Router::new();

    // Source that does NOT support federation, but sparql_1_1 is true so it
    // passes the hard gate and appears as a regularly scored entry in explain().
    router.add_source(
        DataSource::new("no-federation", "https://no-fed.example.org/sparql").with_capabilities(
            SourceCapabilities {
                sparql_1_1: true,
                federation: false,
                ..SourceCapabilities::default()
            },
        ),
    );

    let query =
        Query::parse("SELECT ?s WHERE { SERVICE <https://other.example.org/sparql> { ?s ?p ?o } }")
            .expect("parse SERVICE query");

    let explanations = router.explain(&query).expect("explain succeeded");

    let comp = find_component(
        &explanations,
        "no-federation",
        "capability_federation_match",
    )
    .expect("capability_federation_match must be present for SERVICE query");

    assert!(
        comp.contribution < 0.0,
        "federation mismatch must produce negative contribution; got {}",
        comp.contribution
    );
    assert!(
        (comp.raw_value - (-1.0_f32)).abs() < f32::EPSILON,
        "federation mismatch raw_value must be -1.0; got {}",
        comp.raw_value
    );
}
