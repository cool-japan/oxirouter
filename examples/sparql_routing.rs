//! SPARQL routing example — requires the `sparql` feature.
//!
//! Demonstrates accurate PREFIX expansion and SELECT-variable extraction using
//! `Query::from_sparql`, and routes the enriched query via `Router::route_sparql`.
//!
//! Run with:
//!   cargo run --example sparql_routing --features sparql

use oxirouter::{DataSource, OxiRouterError, Router};

fn main() -> Result<(), OxiRouterError> {
    // ── 1. Build a router with several sources ────────────────────────────────
    let mut router = Router::new();

    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/prop/direct/")
            .with_vocabulary("http://www.wikidata.org/entity/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("schema-org", "https://schema.org/sparql")
            .with_vocabulary("http://schema.org/")
            .with_region("US"),
    );

    // ── 2. Use a SPARQL string with PREFIX declarations ───────────────────────
    //    `route_sparql` calls `Query::from_sparql` internally, which expands
    //    the prefixes and extracts projection variables accurately.
    let sparql = r#"
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?person ?label WHERE {
            ?person a dbo:Person ;
                    rdfs:label ?label .
            FILTER(LANG(?label) = "en")
        }
        ORDER BY ?label
        LIMIT 25
    "#;

    println!("Routing SPARQL query with `route_sparql`...");
    let ranking = router.route_sparql(sparql)?;

    println!(
        "Processing time: {}μs  (ml_used={}, context_used={})",
        ranking.processing_time_us, ranking.ml_used, ranking.context_used
    );

    if ranking.is_empty() {
        println!("No sources matched the query.");
        return Ok(());
    }

    println!("Ranked sources ({}):", ranking.len());
    for (i, sel) in ranking.sources.iter().enumerate() {
        println!(
            "  {}. {} — confidence={:.3}  reason={:?}",
            i + 1,
            sel.source_id,
            sel.confidence,
            sel.reason
        );
    }

    // ── 3. Also demonstrate Query::from_sparql directly ──────────────────────
    let query = oxirouter::Query::from_sparql(sparql)?;
    println!("\nQuery details from `from_sparql`:");
    println!("  type:             {:?}", query.query_type);
    println!("  projection_vars:  {:?}", query.projection_vars);
    println!("  predicates count: {}", query.predicates.len());
    println!("  has_filter:       {}", query.has_filter);

    Ok(())
}
