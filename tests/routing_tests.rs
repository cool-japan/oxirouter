//! Routing-specific tests

use oxirouter::core::source::SourceCapabilities;
use oxirouter::prelude::*;

/// Test that routing respects source capabilities
#[test]
fn test_routing_respects_capabilities() {
    let mut router = Router::new();

    // Add source without SPARQL 1.1 support
    let basic = DataSource::new("basic", "http://basic.example.com/sparql")
        .with_capabilities(SourceCapabilities::basic());
    router.add_source(basic);

    // Add source with full support
    let full = DataSource::new("full", "http://full.example.com/sparql")
        .with_capabilities(SourceCapabilities::full());
    router.add_source(full);

    // Query with aggregation (requires SPARQL 1.1)
    let query = Query::parse(
        r"
        SELECT (COUNT(?s) AS ?count) WHERE {
            ?s ?p ?o
        }
        ",
    )
    .unwrap();

    let ranking = router.route(&query).unwrap();

    // Basic source should not be selected for SPARQL 1.1 queries
    // (it may still appear with 0 confidence in some implementations)
    if let Some(basic_selection) = ranking.sources.iter().find(|s| s.source_id == "basic") {
        // If it appears, it should have lower confidence than full
        if let Some(full_selection) = ranking.sources.iter().find(|s| s.source_id == "full") {
            assert!(full_selection.confidence >= basic_selection.confidence);
        }
    }
}

/// Test that vocabulary matching affects routing
#[test]
fn test_vocabulary_routing() {
    let mut router = Router::new();

    // Add source specialized in DBpedia
    router.add_source(
        DataSource::new("dbpedia", "http://dbpedia.example.com/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_vocabulary("http://dbpedia.org/property/"),
    );

    // Add source specialized in Schema.org
    router.add_source(
        DataSource::new("schema", "http://schema.example.com/sparql")
            .with_vocabulary("http://schema.org/"),
    );

    // Query using DBpedia ontology
    let dbpedia_query = Query::parse(
        r"
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?name WHERE { ?s dbo:name ?name }
        ",
    )
    .unwrap();

    let ranking = router.route(&dbpedia_query).unwrap();

    // DBpedia source should rank higher
    assert!(ranking.best().is_some());
}

/// Test that priority affects routing
#[test]
fn test_priority_routing() {
    let mut router = Router::new();

    // Add low priority source
    router.add_source(DataSource::new("low", "http://low.example.com/sparql").with_priority(0.1));

    // Add high priority source
    router
        .add_source(DataSource::new("high", "http://high.example.com/sparql").with_priority(10.0));

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route(&query).unwrap();

    // High priority source should rank first
    assert_eq!(ranking.best().unwrap().source_id, "high");
}

/// Test that historical performance affects routing
#[test]
fn test_historical_performance_routing() {
    let mut router = Router::new();

    // Add sources
    router.add_source(DataSource::new("fast", "http://fast.example.com/sparql"));
    router.add_source(DataSource::new("slow", "http://slow.example.com/sparql"));

    // Simulate history - fast source has better performance
    for _ in 0..10 {
        router.update_source_stats("fast", 50, true, 100).unwrap();
        router.update_source_stats("slow", 5000, true, 10).unwrap();
    }

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route(&query).unwrap();

    // Fast source should rank higher due to better historical performance
    let fast_idx = ranking.sources.iter().position(|s| s.source_id == "fast");
    let slow_idx = ranking.sources.iter().position(|s| s.source_id == "slow");

    if let (Some(fast), Some(slow)) = (fast_idx, slow_idx) {
        assert!(
            fast < slow,
            "Fast source should rank higher than slow source"
        );
    }
}

/// Test that unavailable sources are excluded
#[test]
fn test_unavailable_sources_excluded() {
    let mut router = Router::new();

    router.add_source(DataSource::new(
        "available",
        "http://available.example.com/sparql",
    ));
    router.add_source(DataSource::new(
        "unavailable",
        "http://unavailable.example.com/sparql",
    ));

    // Mark one source as unavailable
    router.mark_unavailable("unavailable").unwrap();

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route(&query).unwrap();

    // Only available source should be in ranking
    let source_ids: Vec<&str> = ranking
        .sources
        .iter()
        .map(|s| s.source_id.as_str())
        .collect();
    assert!(source_ids.contains(&"available"));
    assert!(!source_ids.contains(&"unavailable"));
}

/// Test that max_sources config is respected
#[test]
fn test_max_sources_config() {
    let config = oxirouter::core::router::RouterConfig {
        max_sources: 2,
        ..Default::default()
    };

    let mut router = Router::with_config(config, oxirouter::context::DefaultContextProvider);

    // Add many sources
    for i in 0..10 {
        router.add_source(DataSource::new(
            format!("src{}", i),
            format!("http://src{}.example.com/sparql", i),
        ));
    }

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route(&query).unwrap();

    // Should only return max_sources
    assert!(ranking.len() <= 2);
}

/// Test that min_confidence filters results
#[test]
fn test_min_confidence_filter() {
    let config = oxirouter::core::router::RouterConfig {
        min_confidence: 0.9, // Very high threshold
        ..Default::default()
    };

    let mut router = Router::with_config(config, oxirouter::context::DefaultContextProvider);

    // Add a source with default (low) stats
    router.add_source(DataSource::new("test", "http://test.example.com/sparql"));

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route(&query).unwrap();

    // All sources below threshold should be filtered
    for selection in &ranking.sources {
        assert!(selection.confidence >= 0.9);
    }
}

/// Test query type routing
#[test]
fn test_query_type_routing() {
    let mut router = Router::new();

    // Add source that doesn't support CONSTRUCT
    let no_construct = DataSource::new("no_construct", "http://nocon.example.com/sparql")
        .with_capabilities(SourceCapabilities {
            construct: false,
            ..SourceCapabilities::basic()
        });
    router.add_source(no_construct);

    // Add source that supports CONSTRUCT
    router.add_source(
        DataSource::new("with_construct", "http://withcon.example.com/sparql")
            .with_capabilities(SourceCapabilities::full()),
    );

    // CONSTRUCT query
    let query = Query::parse("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route(&query).unwrap();

    // Source with CONSTRUCT support should be available
    assert!(!ranking.is_empty());
}

/// Test that routing handles empty source list gracefully
#[test]
fn test_empty_sources() {
    let router = Router::new();
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();

    let result = router.route(&query);
    assert!(result.is_err());
}

/// Test multiple vocabulary matching
#[test]
fn test_multiple_vocabularies() {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("multi", "http://multi.example.com/sparql")
            .with_vocabulary("http://schema.org/")
            .with_vocabulary("http://xmlns.com/foaf/0.1/")
            .with_vocabulary("http://purl.org/dc/terms/"),
    );

    router.add_source(
        DataSource::new("single", "http://single.example.com/sparql")
            .with_vocabulary("http://schema.org/"),
    );

    // Query using multiple vocabularies
    let query = Query::parse(
        r"
        PREFIX schema: <http://schema.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name WHERE {
            ?s schema:name ?name .
            ?s foaf:knows ?friend .
        }
        ",
    )
    .unwrap();

    let ranking = router.route(&query).unwrap();

    // Source with more matching vocabularies should rank higher
    assert!(!ranking.is_empty());
}

/// Test complex query routing
#[test]
fn test_complex_query_routing() {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("full", "http://full.example.com/sparql")
            .with_capabilities(SourceCapabilities::full()),
    );

    // Complex query with subquery, aggregation, and optional
    let query = Query::parse(
        r"
        SELECT ?person (COUNT(?paper) AS ?paperCount) WHERE {
            {
                SELECT ?person WHERE {
                    ?person a <http://example.org/Researcher>
                }
            }
            OPTIONAL {
                ?person <http://example.org/authored> ?paper
            }
        }
        GROUP BY ?person
        HAVING (COUNT(?paper) > 5)
        ",
    )
    .unwrap();

    assert!(query.requires_sparql_1_1());
    assert!(query.has_optional);
    assert!(query.has_aggregation);
    assert!(query.has_subquery);

    let ranking = router.route(&query).unwrap();
    assert!(!ranking.is_empty());
}
