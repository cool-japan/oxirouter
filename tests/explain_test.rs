//! Tests for `Router::explain()` and `RouterAgent::dispatch("oxirouter.explain", ...)`.

use oxirouter::OxiRouterError;
use oxirouter::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_two_source_router() -> Router {
    let mut router = Router::new();
    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/entity/")
            .with_region("US"),
    );
    router
}

fn make_foaf_router() -> Router {
    let mut router = Router::new();
    router.add_source(
        DataSource::new("foaf-src", "https://foaf.example.org/sparql")
            .with_vocabulary("http://xmlns.com/foaf/0.1/"),
    );
    router
}

fn simple_query() -> Query {
    Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse simple query")
}

fn foaf_query() -> Query {
    Query::parse(
        "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name WHERE { ?s foaf:name ?name }",
    )
    .expect("parse foaf query")
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: sum(component.contribution) ≈ total_score (within 1e-4)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_explain_components_sum_to_total() {
    let router = make_two_source_router();
    let query = simple_query();
    let explanations = router.explain(&query).expect("explain succeeded");

    assert!(
        !explanations.is_empty(),
        "expected at least one explanation"
    );

    for exp in &explanations {
        let component_sum: f32 = exp.components.iter().map(|c| c.contribution).sum();
        let diff = (component_sum - exp.total_score).abs();
        assert!(
            diff < 1e-4,
            "source '{}': component sum {component_sum} differs from total_score {} by {diff}",
            exp.source_id,
            exp.total_score,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: vocabulary component exists and has raw_value > 0.0 for foaf match
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_explain_vocab_component_present() {
    let router = make_foaf_router();
    let query = foaf_query();
    let explanations = router.explain(&query).expect("explain succeeded");

    assert_eq!(explanations.len(), 1, "expected exactly one explanation");

    let exp = &explanations[0];
    let vocab_comp = exp
        .components
        .iter()
        .find(|c| c.name == "vocabulary")
        .expect("vocabulary component must be present");

    assert!(
        vocab_comp.raw_value > 0.0,
        "vocabulary raw_value must be > 0.0 for matching foaf source; got {}",
        vocab_comp.raw_value
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: empty router returns NoSources error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_explain_no_sources_error() {
    let router = Router::new();
    let query = simple_query();
    let result = router.explain(&query);

    assert!(
        result.is_err(),
        "explain on empty router must return an error"
    );
    assert!(
        matches!(result.unwrap_err(), OxiRouterError::NoSources { .. }),
        "expected OxiRouterError::NoSources"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: all 3 sources are returned by explain
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_explain_all_sources_returned() {
    let mut router = Router::new();
    router.add_source(DataSource::new("src-a", "https://a.example.org/sparql"));
    router.add_source(DataSource::new("src-b", "https://b.example.org/sparql"));
    router.add_source(DataSource::new("src-c", "https://c.example.org/sparql"));

    let query = simple_query();
    let explanations = router.explain(&query).expect("explain succeeded");

    assert_eq!(
        explanations.len(),
        3,
        "explain must return all 3 registered sources"
    );

    let mut ids: Vec<&str> = explanations.iter().map(|e| e.source_id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["src-a", "src-b", "src-c"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: RouterAgent::dispatch("oxirouter.explain", …) JSON contains "components"
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "agent")]
#[test]
fn test_agent_explain_contains_components() {
    use oxirouter::RouterAgent;

    let mut agent = RouterAgent::new(make_two_source_router());
    let json_out = agent
        .dispatch(
            "oxirouter.explain",
            r#"{"query": "SELECT ?s WHERE { ?s ?p ?o }"}"#,
        )
        .expect("dispatch succeeded");

    let parsed: serde_json::Value = serde_json::from_str(&json_out).expect("output is valid JSON");

    assert!(
        parsed.get("components").is_some(),
        "JSON output must contain a 'components' key; got: {json_out}"
    );

    let components = parsed["components"]
        .as_array()
        .expect("'components' must be a JSON array");

    assert!(
        !components.is_empty(),
        "'components' array must not be empty"
    );

    // Each entry must have source_id, total_score, and a non-empty components array.
    for entry in components {
        assert!(
            entry.get("source_id").is_some(),
            "each component entry must have 'source_id'"
        );
        assert!(
            entry.get("total_score").is_some(),
            "each component entry must have 'total_score'"
        );
        let inner = entry["components"]
            .as_array()
            .expect("each entry's 'components' must be a JSON array");
        assert!(
            !inner.is_empty(),
            "inner components array must not be empty"
        );
    }
}
