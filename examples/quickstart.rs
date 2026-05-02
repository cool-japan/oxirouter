//! Quickstart example — no extra feature flags required.
//!
//! Demonstrates basic OxiRouter usage: registering data sources, parsing a SPARQL query,
//! and routing it to the best source using vocabulary-based heuristics.
//!
//! Run with:
//!   cargo run --example quickstart

use oxirouter::{DataSource, OxiRouterError, Query, Router};

fn main() -> Result<(), OxiRouterError> {
    // ── 1. Create a router ────────────────────────────────────────────────────
    let mut router = Router::new();

    // ── 2. Register data sources ──────────────────────────────────────────────
    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/")
            .with_region("EU"),
    );

    println!("Registered {} sources.", router.source_count());

    // ── 3. Parse a SPARQL query ───────────────────────────────────────────────
    let sparql = "PREFIX dbo: <http://dbpedia.org/ontology/> \
                  SELECT ?name WHERE { \
                    ?p a dbo:Person ; dbo:name ?name \
                  } LIMIT 10";

    let query = Query::parse(sparql)?;
    println!("Query type:  {:?}", query.query_type);
    println!("Complexity:  {:.3}", query.complexity);

    // ── 4. Route the query ────────────────────────────────────────────────────
    let ranking = router.route(&query)?;

    if let Some(best) = ranking.best() {
        println!(
            "Best source: {} (confidence: {:.2})",
            best.source_id, best.confidence
        );
    }

    println!("Full ranking ({} sources):", ranking.len());
    for sel in ranking.sources.iter() {
        println!(
            "  {}: confidence={:.3}  latency_est={}ms  reason={:?}",
            sel.source_id, sel.confidence, sel.estimated_latency_ms, sel.reason
        );
    }

    Ok(())
}
