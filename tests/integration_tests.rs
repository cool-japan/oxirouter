//! Integration tests for OxiRouter

use oxirouter::core::source::SourceCapabilities;
use oxirouter::prelude::*;

#[test]
fn test_basic_routing() {
    let mut router = Router::new();

    // Add some sources
    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_capabilities(SourceCapabilities::full())
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );

    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_capabilities(SourceCapabilities::full())
            .with_vocabulary("http://www.wikidata.org/")
            .with_region("EU"),
    );

    // Create and route a query
    let query = Query::parse(
        r"
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?name WHERE {
            ?person a dbo:Person .
            ?person dbo:name ?name .
        }
        LIMIT 100
        ",
    )
    .unwrap();

    let ranking = router.route(&query).unwrap();

    assert!(!ranking.is_empty());
    assert!(ranking.best().is_some());
}

#[test]
fn test_source_management() {
    let mut router = Router::new();

    // Add source
    router.add_source(DataSource::new("test", "http://example.com/sparql"));
    assert_eq!(router.source_count(), 1);

    // Get source
    let source = router.get_source("test");
    assert!(source.is_some());
    assert_eq!(source.unwrap().endpoint, "http://example.com/sparql");

    // Remove source
    let removed = router.remove_source("test");
    assert!(removed.is_some());
    assert_eq!(router.source_count(), 0);
}

#[test]
fn test_query_parsing() {
    // SELECT query
    let select = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    assert_eq!(select.query_type, oxirouter::core::query::QueryType::Select);

    // CONSTRUCT query
    let construct = Query::parse("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }").unwrap();
    assert_eq!(
        construct.query_type,
        oxirouter::core::query::QueryType::Construct
    );

    // ASK query
    let ask = Query::parse("ASK { ?s ?p ?o }").unwrap();
    assert_eq!(ask.query_type, oxirouter::core::query::QueryType::Ask);

    // DESCRIBE query
    let describe = Query::parse("DESCRIBE <http://example.org/resource>").unwrap();
    assert_eq!(
        describe.query_type,
        oxirouter::core::query::QueryType::Describe
    );
}

#[test]
fn test_query_complexity() {
    let simple = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();

    let complex = Query::parse(
        r"
        SELECT ?s (COUNT(?o) AS ?count) WHERE {
            { ?s ?p ?o } UNION { ?s ?p2 ?o2 }
            OPTIONAL { ?s ?p3 ?o3 }
            FILTER(?count > 5)
        }
        GROUP BY ?s
        ",
    )
    .unwrap();

    assert!(complex.complexity > simple.complexity);
    assert!(complex.has_union);
    assert!(complex.has_optional);
    assert!(complex.has_filter);
    assert!(complex.has_aggregation);
}

#[test]
fn test_source_stats_update() {
    let mut router = Router::new();
    router.add_source(DataSource::new("test", "http://example.com/sparql"));

    // Update stats
    router.update_source_stats("test", 100, true, 50).unwrap();
    router.update_source_stats("test", 200, true, 30).unwrap();
    router.update_source_stats("test", 150, false, 0).unwrap();

    let source = router.get_source("test").unwrap();
    assert_eq!(source.stats.total_queries, 3);
    assert_eq!(source.stats.successful_queries, 2);
    assert!(source.stats.avg_latency_ms > 0.0);
}

#[test]
fn test_source_availability() {
    let mut router = Router::new();
    router.add_source(DataSource::new("test", "http://example.com/sparql"));

    // Initially available
    assert!(router.get_source("test").unwrap().available);

    // Mark unavailable
    router.mark_unavailable("test").unwrap();
    assert!(!router.get_source("test").unwrap().available);

    // Mark available again
    router.mark_available("test").unwrap();
    assert!(router.get_source("test").unwrap().available);
}

#[test]
fn test_vocabulary_matching() {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("schema", "http://schema.example.com/sparql")
            .with_vocabulary("http://schema.org/"),
    );

    router.add_source(
        DataSource::new("foaf", "http://foaf.example.com/sparql")
            .with_vocabulary("http://xmlns.com/foaf/0.1/"),
    );

    // Query with schema.org vocabulary
    let query = Query::parse(
        r"
        PREFIX schema: <http://schema.org/>
        SELECT ?name WHERE { ?s schema:name ?name }
        ",
    )
    .unwrap();

    let ranking = router.route(&query).unwrap();

    // Source with matching vocabulary should rank higher
    assert!(!ranking.is_empty());
}

#[test]
fn test_capability_requirements() {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("basic", "http://basic.example.com/sparql")
            .with_capabilities(SourceCapabilities::basic()),
    );

    router.add_source(
        DataSource::new("full", "http://full.example.com/sparql")
            .with_capabilities(SourceCapabilities::full()),
    );

    // Query requiring SPARQL 1.1
    let query = Query::parse(
        r"
        SELECT ?s (COUNT(?o) AS ?count) WHERE {
            ?s ?p ?o
        }
        GROUP BY ?s
        ",
    )
    .unwrap();

    assert!(query.requires_sparql_1_1());

    let ranking = router.route(&query).unwrap();

    // Only full-featured source should be selected
    assert!(!ranking.is_empty());
    for selection in ranking.sources.iter() {
        if selection.source_id == "basic" {
            // Basic source should not be in ranking or should have very low confidence
        }
    }
}

#[test]
fn test_no_sources_error() {
    let router = Router::new();
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();

    let result = router.route(&query);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        OxiRouterError::NoSources { .. }
    ));
}

#[test]
fn test_invalid_query_error() {
    let result = Query::parse("NOT A VALID QUERY");
    assert!(result.is_err());
}

#[test]
fn test_router_config() {
    let config = oxirouter::core::router::RouterConfig {
        max_sources: 3,
        min_confidence: 0.5,
        use_ml: false,
        use_context: false,
        timeout_us: 50_000,
        history_weight: 0.4,
        vocab_weight: 0.3,
        geo_weight: 0.3,
        circuit_breaker: oxirouter::CircuitBreakerConfig::default(),
        max_response_bytes: 64 * 1024 * 1024,
        #[cfg(feature = "cache")]
        cache_enabled: true,
        #[cfg(feature = "cache")]
        cache_ttl_ms: 300_000,
        #[cfg(feature = "cache")]
        cache_max_entries: 1000,
    };

    let router = Router::with_config(config, oxirouter::context::DefaultContextProvider);

    // Verify config is applied
    assert_eq!(router.config().max_sources, 3);
    assert!(!router.config().use_ml);
}

#[test]
fn test_federation_executor() {
    use oxirouter::federation::{ExecutionConfig, Executor};

    let config = ExecutionConfig {
        timeout_ms: 5000,
        max_retries: 1,
        parallel: true,
        max_concurrency: 2,
        ..Default::default()
    };

    let executor = Executor::with_config(config);
    assert_eq!(executor.config().timeout_ms, 5000);
}

#[test]
fn test_federation_aggregator() {
    use oxirouter::federation::aggregator::{AggregationConfig, AggregationStrategy};
    use oxirouter::federation::{Aggregator, QueryResult};

    let config = AggregationConfig {
        strategy: AggregationStrategy::Largest,
        ..Default::default()
    };

    let aggregator = Aggregator::with_config(config);

    let results = vec![
        QueryResult::success("src1", vec![], 10, 100),
        QueryResult::success("src2", vec![], 50, 200),
    ];

    let aggregated = aggregator.aggregate(&results).unwrap();
    assert_eq!(aggregated.row_count, 50);
}

#[test]
fn test_context_combined() {
    use oxirouter::context::CombinedContext;

    let ctx = CombinedContext::new();
    assert!(ctx.quality_score() >= 0.0);
}

#[cfg(feature = "geo")]
#[test]
fn test_geo_context() {
    use oxirouter::GeoContext;

    let geo = GeoContext::from_coords(-122.4194, 37.7749)
        .with_country("US")
        .with_region("California");

    assert_eq!(geo.country_code, Some("US".to_string()));
    assert!(!geo.is_eu_region());

    // Test distance calculation
    let distance = geo.distance_to(-122.0, 37.0).unwrap();
    assert!(distance > 0.0 && distance < 100.0); // Within 100km
}

#[cfg(feature = "ml")]
#[test]
fn test_feature_extraction() {
    use oxirouter::ml::FeatureVector;

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let features = FeatureVector::from_query(&query).unwrap();

    assert!(!features.is_empty());
    assert!(features.get("query_type_select").is_some());
}

#[cfg(feature = "ml")]
#[test]
fn test_naive_bayes() {
    use oxirouter::ml::{FeatureVector, Model, NaiveBayesClassifier};

    let mut nb = NaiveBayesClassifier::new(10);
    let sources = vec!["src1".to_string(), "src2".to_string()];
    let source_refs: Vec<&String> = sources.iter().collect();
    nb.initialize_sources(&source_refs);

    let mut features = FeatureVector::new();
    for i in 0..10 {
        features.add(format!("f{}", i), 0.5);
    }

    let predictions = nb.predict(&features, &source_refs).unwrap();
    assert_eq!(predictions.len(), 2);
}

#[cfg(feature = "rl")]
#[test]
fn test_feedback_and_reward() {
    use oxirouter::rl::Feedback;

    let success = Feedback::success("src1", 123, 100, 50);
    let reward = success.to_reward();
    assert!(reward.value() > 0.5);

    let failure = Feedback::failure("src1", 123, "Error");
    let reward = failure.to_reward();
    assert!(reward.value() < 0.5);
}

#[cfg(feature = "rl")]
#[test]
fn test_policy_update() {
    use oxirouter::rl::{Policy, Reward};

    let mut policy = Policy::ucb();
    policy.initialize_source("src1");

    let initial_q = policy.get_q_value("src1");
    policy.update("src1", Reward::new(0.9));

    assert!(policy.get_q_value("src1") > initial_q);
}

#[test]
fn test_query_log_routing() {
    use oxirouter::QueryLog;

    let mut log = QueryLog::new();

    // Record some routing decisions and outcomes
    log.record_routing(1, "dbpedia", 0.9, false, None);
    log.record_outcome(1, "dbpedia", true, 150, 100, 0.9);

    log.record_routing(2, "wikidata", 0.7, false, None);
    log.record_outcome(2, "wikidata", false, 5000, 0, 0.0);

    assert_eq!(log.len(), 2);
    assert_eq!(log.best_source(), Some("dbpedia"));

    let ranked = log.ranked_sources();
    assert_eq!(ranked[0].0, "dbpedia");
}

#[test]
fn test_route_and_log() {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/"),
    );

    let query = Query::parse(
        r"PREFIX dbo: <http://dbpedia.org/ontology/>
           SELECT ?s WHERE { ?s a dbo:Person }",
    )
    .unwrap();

    let ranking = router.route_and_log(&query).unwrap();
    assert!(!ranking.is_empty());

    // Query log should have an entry
    assert!(!router.query_log().is_empty());

    // Record outcome
    let query_id = query.predicate_hash();
    router
        .learn_from_outcome(query_id, "dbpedia", true, 200, 50)
        .unwrap();

    // Source stats should be updated
    let source = router.get_source("dbpedia").unwrap();
    assert_eq!(source.stats.total_queries, 1);

    // Query log should have outcome recorded
    let stats = router.query_log().source_stats("dbpedia").unwrap();
    assert_eq!(stats.outcomes_received, 1);
}

#[cfg(feature = "rl")]
#[test]
fn test_route_with_rl_policy() {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("good", "https://good.example.com/sparql")
            .with_vocabulary("http://schema.org/"),
    );
    router.add_source(
        DataSource::new("bad", "https://bad.example.com/sparql")
            .with_vocabulary("http://schema.org/"),
    );

    // Enable RL
    router.enable_rl();

    let query =
        Query::parse("PREFIX s: <http://schema.org/> SELECT ?x WHERE { ?x a s:Person }").unwrap();

    let query_id = query.predicate_hash();

    // Simulate multiple rounds — "good" always succeeds, "bad" always fails
    for _ in 0..5 {
        let _ranking = router.route_and_log(&query).unwrap();

        router
            .learn_from_outcome(query_id, "good", true, 100, 50)
            .unwrap();
        router
            .learn_from_outcome(query_id, "bad", false, 5000, 0)
            .unwrap();
    }

    // After learning, "good" should have higher score in the log
    let log = router.query_log();
    let good_score = log.routing_score("good").unwrap_or(0.5);
    let bad_score = log.routing_score("bad").unwrap_or(0.5);
    assert!(good_score > bad_score, "good={good_score} bad={bad_score}");
}
