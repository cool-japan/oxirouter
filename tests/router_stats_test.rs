//! Tests for Router query-log stat delegators and model_type() discriminant.

use oxirouter::{DataSource, Query, Router};

#[test]
fn test_router_source_stats_after_routing() {
    let mut router = Router::new();
    router.add_source(
        DataSource::new("src1", "https://example.org/sparql")
            .with_vocabulary("http://example.org/"),
    );
    let query = Query::parse("SELECT ?s WHERE { ?s <http://example.org/p> ?o }").unwrap();
    // route_and_log a few times to build up log entries
    for _ in 0..3 {
        let _ = router.route_and_log(&query);
    }
    let stats = router.source_stats("src1");
    assert!(
        stats.is_some(),
        "source_stats should return Some after routing"
    );
    let stats = stats.unwrap();
    assert!(
        stats.total_routed >= 1,
        "total_routed should be >= 1 after routing"
    );
}

#[test]
fn test_router_ranked_sources_from_log() {
    let mut router = Router::new();
    router.add_source(DataSource::new("a", "https://a.example/sparql"));
    router.add_source(DataSource::new("b", "https://b.example/sparql"));
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route_and_log(&query).unwrap();

    // Record an outcome so at least one source has log history with outcomes
    if let Some(top) = ranking.sources.first() {
        let query_id = query.predicate_hash();
        let _ = router.learn_from_outcome(query_id, &top.source_id, true, 50, 5);
    }

    let ranked = router.ranked_sources_from_log();
    // ranked_sources only returns sources with outcomes; at least one should appear
    assert!(
        !ranked.is_empty(),
        "should have at least one ranked source after outcome"
    );
}

#[test]
fn test_router_best_source_from_log_and_len() {
    let mut router = Router::new();
    router.add_source(DataSource::new("x", "https://x.example/sparql"));
    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    let ranking = router.route_and_log(&query).unwrap();

    // query_log_len should be > 0 now
    assert!(
        router.query_log_len() > 0,
        "query_log_len should be positive after routing"
    );

    // Record outcome so best_source_from_log can return Some
    if let Some(top) = ranking.sources.first() {
        let query_id = query.predicate_hash();
        let _ = router.learn_from_outcome(query_id, &top.source_id, true, 30, 10);
    }

    let best = router.best_source_from_log();
    assert!(
        best.is_some(),
        "best_source_from_log should be Some after an outcome is recorded"
    );
}

#[cfg(feature = "ml")]
#[test]
fn test_model_type_discriminants() {
    use oxirouter::{
        EnsembleClassifier, Model, ModelConfig, ModelType, NaiveBayesClassifier, NeuralNetwork,
    };

    let nb = NaiveBayesClassifier::new(48);
    assert_eq!(nb.model_type(), "naive_bayes");

    let nn = NeuralNetwork::from_config(&ModelConfig {
        model_type: ModelType::NeuralNetwork,
        ..ModelConfig::default()
    });
    assert_eq!(nn.model_type(), "neural");

    let ens = EnsembleClassifier::new(48);
    assert_eq!(ens.model_type(), "ensemble");
}
