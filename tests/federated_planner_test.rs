//! Integration tests for the federated query planner (Block Z).
//!
//! These tests validate BGP decomposition, fallback behaviour, confidence
//! sorting, and error paths of [`DefaultPlanner`].

use oxirouter::core::term::{StructuredTriple, Term};
use oxirouter::federation::planner::{DefaultPlanner, FederatedPlanner};
use oxirouter::{DataSource, Query};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_foaf_source() -> DataSource {
    DataSource::new("foaf-source", "http://localhost:9999/sparql")
        .with_vocabulary("http://xmlns.com/foaf/0.1/")
}

fn make_dcterms_source() -> DataSource {
    DataSource::new("dcterms-source", "http://localhost:9999/sparql")
        .with_vocabulary("http://purl.org/dc/terms/")
}

/// Build a foaf:name triple: `?s foaf:name ?name`
fn foaf_name_triple() -> StructuredTriple {
    StructuredTriple {
        subject: Term::Variable("s".to_string()),
        predicate: Term::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
        object: Term::Variable("name".to_string()),
    }
}

/// Build a dcterms:title triple: `?s dcterms:title ?title`
fn dcterms_title_triple() -> StructuredTriple {
    StructuredTriple {
        subject: Term::Variable("s".to_string()),
        predicate: Term::Iri("http://purl.org/dc/terms/title".to_string()),
        object: Term::Variable("title".to_string()),
    }
}

/// Inject `triples` into a base query (replaces any prior structured_triples).
fn query_with_triples(triples: Vec<StructuredTriple>) -> Query {
    let mut q = Query::parse("SELECT * WHERE { ?s ?p ?o }").expect("parse");
    q.structured_triples = triples;
    q
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 – single triple routes to the best source
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_single_triple_routes_to_best_source() {
    let planner = DefaultPlanner::default();
    let q = query_with_triples(vec![foaf_name_triple()]);
    let sources = vec![make_foaf_source(), make_dcterms_source()];

    let plan = planner.plan(&q, &sources).expect("plan should succeed");

    assert!(
        !plan.fallback_used,
        "BGP should be decomposed, not fall back"
    );
    assert_eq!(plan.sub_plans.len(), 1, "one sub-plan for foaf predicate");
    assert_eq!(
        plan.sub_plans[0].source_id, "foaf-source",
        "foaf predicate must route to foaf-source"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 – multi-triple decomposition across two sources
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multi_triple_decomposition() {
    let planner = DefaultPlanner::default();
    let q = query_with_triples(vec![foaf_name_triple(), dcterms_title_triple()]);
    let sources = vec![make_foaf_source(), make_dcterms_source()];

    let plan = planner.plan(&q, &sources).expect("plan should succeed");

    assert!(!plan.fallback_used);
    assert_eq!(
        plan.sub_plans.len(),
        2,
        "each predicate namespace should produce a separate sub-plan"
    );

    let ids: Vec<&str> = plan
        .sub_plans
        .iter()
        .map(|sp| sp.source_id.as_str())
        .collect();
    assert!(
        ids.contains(&"foaf-source"),
        "foaf-source should be in sub-plans"
    );
    assert!(
        ids.contains(&"dcterms-source"),
        "dcterms-source should be in sub-plans"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 – multiple triples from the same namespace → one sub-plan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_decomposition_groups_same_source() {
    let planner = DefaultPlanner::default();

    let triple1 = foaf_name_triple();
    let triple2 = StructuredTriple {
        subject: Term::Variable("s".to_string()),
        predicate: Term::Iri("http://xmlns.com/foaf/0.1/mbox".to_string()),
        object: Term::Variable("mbox".to_string()),
    };
    let triple3 = StructuredTriple {
        subject: Term::Variable("s".to_string()),
        predicate: Term::Iri("http://xmlns.com/foaf/0.1/knows".to_string()),
        object: Term::Variable("friend".to_string()),
    };

    let q = query_with_triples(vec![triple1, triple2, triple3]);
    let sources = vec![make_foaf_source()];

    let plan = planner.plan(&q, &sources).expect("plan should succeed");

    assert!(!plan.fallback_used);
    assert_eq!(
        plan.sub_plans.len(),
        1,
        "all three foaf triples should be grouped into one sub-plan"
    );
    assert_eq!(plan.sub_plans[0].source_id, "foaf-source");
    assert_eq!(
        plan.sub_plans[0].triples.len(),
        3,
        "sub-plan should carry all 3 triples"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 – unmatched predicate returns NoSources error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_unmatched_triple_returns_error() {
    let planner = DefaultPlanner::default();

    // Predicate from a completely unregistered namespace.
    let unknown_triple = StructuredTriple {
        subject: Term::Variable("s".to_string()),
        predicate: Term::Iri("http://unknown.example.org/onto/property".to_string()),
        object: Term::Variable("o".to_string()),
    };

    let q = query_with_triples(vec![unknown_triple]);
    // Sources with success_rate == 0 (new DataSource has success_rate 0.0).
    let sources = vec![
        DataSource::new("src1", "http://localhost:9999/sparql"),
        DataSource::new("src2", "http://localhost:9998/sparql"),
    ];

    let result = planner.plan(&q, &sources);
    assert!(
        result.is_err(),
        "unmatched predicate with low-reliability sources should fail"
    );

    let err = result.expect_err("expected error");
    match err {
        oxirouter::OxiRouterError::NoSources { .. } => {}
        other => panic!("expected NoSources, got: {:?}", other),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 – fallback on empty structured_triples
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_fallback_on_empty_structured_triples() {
    let planner = DefaultPlanner::default();
    // Query.parse() produces structured_triples == [] (heuristic path).
    let q = Query::parse("SELECT * WHERE { ?s ?p ?o }").expect("parse");
    assert!(
        q.structured_triples.is_empty(),
        "heuristic parse must not populate structured_triples"
    );

    let sources = vec![make_foaf_source(), make_dcterms_source()];
    let plan = planner.plan(&q, &sources).expect("plan should succeed");

    assert!(
        plan.fallback_used,
        "planner must use fallback for empty BGP"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 – sub_query.raw is valid SPARQL SELECT
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sub_query_raw_valid_sparql() {
    let planner = DefaultPlanner::default();
    let q = query_with_triples(vec![foaf_name_triple()]);
    let sources = vec![make_foaf_source()];

    let plan = planner.plan(&q, &sources).expect("plan should succeed");
    assert_eq!(plan.sub_plans.len(), 1);

    let raw = &plan.sub_plans[0].sub_query.raw;
    assert!(
        raw.starts_with("SELECT * WHERE {"),
        "sub_query.raw must start with 'SELECT * WHERE {{', got: {raw}"
    );
    // Variable subject should appear as ?s.
    assert!(raw.contains("?s"), "raw SPARQL should contain ?s");
    // IRI predicate should be angle-bracketed.
    assert!(
        raw.contains("<http://xmlns.com/foaf/0.1/name>"),
        "raw SPARQL should contain the foaf:name IRI, got: {raw}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7 – variable predicate routes by reliability (success_rate)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_variable_predicate_routes_by_reliability() {
    let planner = DefaultPlanner::default();

    // Variable predicate — no specific namespace match possible.
    let var_triple = StructuredTriple {
        subject: Term::Variable("s".to_string()),
        predicate: Term::Variable("p".to_string()),
        object: Term::Variable("o".to_string()),
    };

    let mut reliable_source = DataSource::new("reliable", "http://localhost:9999/sparql");
    // success_rate * 0.5 = 0.9 * 0.5 = 0.45  >=  min_triple_confidence (0.3)
    reliable_source.stats.success_rate = 0.9;

    let mut unreliable_source = DataSource::new("unreliable", "http://localhost:9998/sparql");
    unreliable_source.stats.success_rate = 0.7;

    let q = query_with_triples(vec![var_triple]);
    let sources = vec![reliable_source, unreliable_source];

    let plan = planner.plan(&q, &sources).expect("plan should succeed");

    assert!(!plan.fallback_used);
    assert_eq!(plan.sub_plans.len(), 1);
    assert_eq!(
        plan.sub_plans[0].source_id, "reliable",
        "variable predicate should route to the source with highest success_rate"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8 – sub-plans sorted by confidence descending
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_planner_confidence_sort_order() {
    let planner = DefaultPlanner::default();

    let q = query_with_triples(vec![foaf_name_triple(), dcterms_title_triple()]);
    let sources = vec![make_foaf_source(), make_dcterms_source()];

    let plan = planner.plan(&q, &sources).expect("plan should succeed");

    assert_eq!(plan.sub_plans.len(), 2);

    // All vocabulary-matched sub-plans have confidence == 1.0 (same score),
    // so we just verify sorting is non-decreasing from front to back.
    let confidences: Vec<f32> = plan.sub_plans.iter().map(|sp| sp.confidence).collect();
    for pair in confidences.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "sub-plans must be sorted confidence descending, got {:?}",
            confidences
        );
    }
}
