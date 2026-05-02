//! Agent actions example — requires the `agent` and `sparql` features.
//!
//! Demonstrates the `RouterAgent` dispatch interface: listing available actions,
//! routing a query through the JSON-based `oxirouter.route` action, and recording
//! feedback via `oxirouter.learn`.
//!
//! Run with:
//!   cargo run --example agent_actions --features agent,sparql

use oxirouter::agent::{ExplainOutput, LearnOutput, RouteOutput, RouterAgent};
use oxirouter::{DataSource, OxiRouterError, Router};

fn main() -> Result<(), OxiRouterError> {
    // ── 1. Build a router ─────────────────────────────────────────────────────
    let mut router = Router::new();
    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/prop/direct/")
            .with_region("EU"),
    );

    // ── 2. Wrap in a RouterAgent ──────────────────────────────────────────────
    let mut agent = RouterAgent::new(router);

    // ── 3. List available actions (for LLM tool registry) ─────────────────────
    println!("Available agent actions:");
    for meta in RouterAgent::<oxirouter::context::DefaultContextProvider>::list_actions() {
        println!("  {} — {}", meta.name, meta.description);
    }
    println!();

    // ── 4. Route a query via the dispatch interface ───────────────────────────
    let route_input = r#"{
        "query": "PREFIX dbo: <http://dbpedia.org/ontology/> SELECT ?p WHERE { ?p a dbo:Person } LIMIT 10",
        "max_sources": 3
    }"#;

    let route_json = agent.dispatch("oxirouter.route", route_input)?;
    let route_output: RouteOutput = serde_json::from_str(&route_json)
        .map_err(|e| OxiRouterError::ExecutionError(format!("deserialize route output: {e}")))?;

    println!(
        "Route result ({} sources evaluated):",
        route_output.total_evaluated
    );
    for src in &route_output.sources {
        println!(
            "  {}: confidence={:.3}  endpoint={}  reason={}",
            src.id, src.confidence, src.endpoint, src.reason
        );
    }
    println!();

    // ── 5. Explain the routing decision ───────────────────────────────────────
    let explain_input = r#"{
        "query": "PREFIX dbo: <http://dbpedia.org/ontology/> SELECT ?p WHERE { ?p a dbo:Person } LIMIT 10",
        "max_sources": 5
    }"#;

    let explain_json = agent.dispatch("oxirouter.explain", explain_input)?;
    let explain_output: ExplainOutput = serde_json::from_str(&explain_json)
        .map_err(|e| OxiRouterError::ExecutionError(format!("deserialize explain output: {e}")))?;

    println!("Explanation:\n{}", explain_output.explanation);

    // ── 6. Record feedback (closing the learning loop) ────────────────────────
    if let Some(first) = route_output.sources.first() {
        let learn_input = format!(
            r#"{{
                "query_id": 0,
                "source_id": "{}",
                "success": true,
                "latency_ms": 312,
                "result_count": 10
            }}"#,
            first.id
        );

        let learn_json = agent.dispatch("oxirouter.learn", &learn_input)?;
        let learn_output: LearnOutput = serde_json::from_str(&learn_json).map_err(|e| {
            OxiRouterError::ExecutionError(format!("deserialize learn output: {e}"))
        })?;

        println!(
            "Feedback recorded for '{}': success_rate={:.2}  avg_latency={:.0}ms",
            first.id, learn_output.source_success_rate, learn_output.source_avg_latency_ms
        );
    }

    Ok(())
}
