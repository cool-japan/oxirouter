//! Integration tests for the `agent` feature.
//!
//! All tests are gated behind `#[cfg(feature = "agent")]` so that they are
//! compiled and run only when the feature is enabled.

#![cfg(feature = "agent")]

use oxirouter::agent::{
    EXPLAIN_INPUT_SCHEMA, ExplainOutput, LEARN_INPUT_SCHEMA, LearnOutput, ROUTE_INPUT_SCHEMA,
    RouteOutput, RouteOutputSource,
};
use oxirouter::context::DefaultContextProvider;
use oxirouter::{DataSource, Router, RouterAgent};

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

fn make_router() -> Router<DefaultContextProvider> {
    let mut router = Router::new();
    router.add_source(DataSource::new("dbpedia", "https://dbpedia.org/sparql"));
    router.add_source(DataSource::new(
        "wikidata",
        "https://query.wikidata.org/sparql",
    ));
    router
}

/// Convenience: dispatch "oxirouter.route" and return the parsed `RouteOutput`.
fn route(agent: &mut RouterAgent<DefaultContextProvider>, query: &str) -> RouteOutput {
    let input = serde_json::json!({ "query": query }).to_string();
    let json = agent
        .dispatch("oxirouter.route", &input)
        .expect("route should succeed");
    serde_json::from_str(&json).expect("route output should be valid JSON")
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: route action returns at least one source with valid confidence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_route_action() {
    let mut agent = RouterAgent::new(make_router());
    let output = route(
        &mut agent,
        "SELECT ?s WHERE { ?s a <http://schema.org/Person> }",
    );

    assert!(
        !output.sources.is_empty(),
        "expected at least one source, got none"
    );
    for src in &output.sources {
        assert!(
            (0.0..=1.0).contains(&src.confidence),
            "confidence {} out of range for source {}",
            src.confidence,
            src.id,
        );
        assert!(!src.reason.is_empty(), "reason should not be empty");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: learn action is recorded with total_routed >= 1
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_learn_action() {
    let mut agent = RouterAgent::new(make_router());

    // First route so that there is at least one log entry for the source.
    let route_output = route(
        &mut agent,
        "SELECT ?s WHERE { ?s a <http://schema.org/Person> }",
    );
    let source_id = route_output
        .sources
        .first()
        .expect("at least one source")
        .id
        .clone();

    let learn_input = serde_json::json!({
        "query_id": 1_u64,
        "source_id": source_id,
        "success": true,
        "latency_ms": 120_u32,
        "result_count": 42_u32,
    })
    .to_string();

    let json = agent
        .dispatch("oxirouter.learn", &learn_input)
        .expect("learn should succeed");
    let output: LearnOutput =
        serde_json::from_str(&json).expect("learn output should be valid JSON");

    assert!(output.recorded, "recorded should be true");
    assert!(
        output.source_total_routed >= 1,
        "source_total_routed should be at least 1, got {}",
        output.source_total_routed,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: explain action returns a non-empty explanation containing a source ID
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_explain_action() {
    let mut agent = RouterAgent::new(make_router());
    let input = serde_json::json!({
        "query": "SELECT ?s WHERE { ?s a <http://schema.org/Person> }",
    })
    .to_string();

    let json = agent
        .dispatch("oxirouter.explain", &input)
        .expect("explain should succeed");
    let output: ExplainOutput =
        serde_json::from_str(&json).expect("explain output should be valid JSON");

    assert!(
        !output.explanation.is_empty(),
        "explanation should not be empty"
    );
    assert!(
        !output.ranked_sources.is_empty(),
        "ranked_sources should not be empty"
    );

    // The explanation should mention at least one of the source IDs.
    let any_id_present = output
        .ranked_sources
        .iter()
        .any(|src| output.explanation.contains(&src.id));
    assert!(
        any_id_present,
        "explanation should contain at least one source ID"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: unknown action returns an error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_unknown_action() {
    let mut agent = RouterAgent::new(make_router());
    let result = agent.dispatch("nonexistent.action", r#"{"foo": "bar"}"#);
    assert!(result.is_err(), "unknown action should return Err");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: invalid JSON input returns an error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_invalid_input_json() {
    let mut agent = RouterAgent::new(make_router());
    let result = agent.dispatch("oxirouter.route", "not valid json {{{{");
    assert!(result.is_err(), "invalid JSON should return Err");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: missing required field returns an error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_missing_required_field() {
    let mut agent = RouterAgent::new(make_router());
    // `query` field is required but absent.
    let result = agent.dispatch("oxirouter.route", "{}");
    assert!(result.is_err(), "missing required field should return Err");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: list_actions returns exactly 3 well-formed entries
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_router_agent_list_actions() {
    let actions = RouterAgent::<DefaultContextProvider>::list_actions();
    assert_eq!(actions.len(), 3, "should have exactly 3 actions");
    for action in &actions {
        assert!(
            action.name.starts_with("oxirouter."),
            "action name '{}' should start with 'oxirouter.'",
            action.name,
        );
        assert!(
            !action.description.is_empty(),
            "description should not be empty"
        );
        assert!(
            !action.input_schema.is_empty(),
            "input_schema should not be empty"
        );
        assert!(
            !action.output_schema.is_empty(),
            "output_schema should not be empty"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: schema constants are valid JSON objects with expected structure
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_route_input_schemas_are_valid_json() {
    for (name, schema) in [
        ("ROUTE_INPUT_SCHEMA", ROUTE_INPUT_SCHEMA),
        ("LEARN_INPUT_SCHEMA", LEARN_INPUT_SCHEMA),
        ("EXPLAIN_INPUT_SCHEMA", EXPLAIN_INPUT_SCHEMA),
    ] {
        let value: serde_json::Value = serde_json::from_str(schema)
            .unwrap_or_else(|e| panic!("{name} is not valid JSON: {e}"));

        assert_eq!(
            value["type"], "object",
            "{name} should have \"type\": \"object\""
        );
        assert!(
            value.get("properties").is_some(),
            "{name} should have a \"properties\" key"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: RouteOutput round-trip serialization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_route_output_round_trip() {
    let original = RouteOutput {
        sources: vec![RouteOutputSource {
            id: "test".to_string(),
            endpoint: "http://test".to_string(),
            confidence: 0.85,
            reason: "vocabulary_match".to_string(),
        }],
        total_evaluated: 1,
    };

    let json = serde_json::to_string(&original).expect("serialization should succeed");
    let reconstructed: RouteOutput =
        serde_json::from_str(&json).expect("deserialization should succeed");

    assert_eq!(reconstructed.total_evaluated, original.total_evaluated);
    assert_eq!(reconstructed.sources.len(), original.sources.len());

    let orig_src = &original.sources[0];
    let recon_src = &reconstructed.sources[0];
    assert_eq!(recon_src.id, orig_src.id);
    assert_eq!(recon_src.endpoint, orig_src.endpoint);
    assert!(
        (recon_src.confidence - orig_src.confidence).abs() < f32::EPSILON,
        "confidence mismatch: {} vs {}",
        recon_src.confidence,
        orig_src.confidence,
    );
    assert_eq!(recon_src.reason, orig_src.reason);
}
