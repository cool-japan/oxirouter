//! VoID descriptor example — requires the `void` feature.
//!
//! Shows how to import source capabilities from an inline VoID/Turtle descriptor
//! using `Router::register_from_void_ttl`, then route a query over the loaded sources.
//!
//! Run with:
//!   cargo run --example void_descriptor --features void

use oxirouter::{OxiRouterError, Query, Router};

/// An inline VoID/Turtle descriptor that describes DBpedia as a SPARQL endpoint.
const VOID_TTL: &str = r#"
@prefix void:      <http://rdfs.org/ns/void#> .
@prefix dcterms:   <http://purl.org/dc/terms/> .
@prefix oxirouter: <https://oxirouter.dev/ns#> .
@prefix dbo:       <http://dbpedia.org/ontology/> .

<https://dbpedia.org/sparql> a void:Dataset ;
    void:sparqlEndpoint <https://dbpedia.org/sparql> ;
    void:vocabulary     <http://dbpedia.org/ontology/> ;
    dcterms:spatial     "EU" .
"#;

fn main() -> Result<(), OxiRouterError> {
    // ── 1. Create a router and import sources from VoID TTL ───────────────────
    let mut router = Router::new();
    router.register_from_void_ttl(VOID_TTL)?;

    println!(
        "Loaded {} source(s) from VoID descriptor.",
        router.source_count()
    );

    // ── 2. Also add a second source manually for comparison ───────────────────
    router.add_source(
        oxirouter::DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/")
            .with_region("EU"),
    );

    // ── 3. Route a query that references the DBpedia ontology ─────────────────
    let query = Query::parse(
        "PREFIX dbo: <http://dbpedia.org/ontology/> \
         SELECT ?city ?pop WHERE { \
           ?city a dbo:City ; dbo:populationTotal ?pop \
         } ORDER BY DESC(?pop) LIMIT 20",
    )?;

    let ranking = router.route(&query)?;

    println!("Routing result for DBpedia city query:");
    if let Some(best) = ranking.best() {
        println!(
            "  Best:  {} (confidence: {:.2})",
            best.source_id, best.confidence
        );
    }
    for sel in ranking.sources.iter() {
        println!(
            "  {}: confidence={:.3}  reason={:?}",
            sel.source_id, sel.confidence, sel.reason
        );
    }

    Ok(())
}
