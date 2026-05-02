//! Benchmarks for OxiRouter inference performance

// criterion_group! macro generates undocumented entry-point functions
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxirouter::core::source::SourceCapabilities;
use oxirouter::prelude::*;
use std::hint::black_box;

fn create_router_with_sources(source_count: usize) -> Router {
    let mut router = Router::new();

    for i in 0..source_count {
        let source = DataSource::new(
            format!("source_{}", i),
            format!("http://source{}.example.com/sparql", i),
        )
        .with_capabilities(SourceCapabilities::full())
        .with_vocabulary(format!("http://vocab{}.example.org/", i % 5))
        .with_region(match i % 4 {
            0 => "US",
            1 => "EU",
            2 => "JP",
            _ => "AU",
        });

        router.add_source(source);
    }

    router
}

fn create_simple_query() -> Query {
    Query::parse("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100").unwrap()
}

fn create_complex_query() -> Query {
    Query::parse(
        r#"
        PREFIX schema: <http://schema.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name (COUNT(?friend) AS ?friendCount) WHERE {
            ?person a schema:Person .
            ?person schema:name ?name .
            OPTIONAL {
                ?person foaf:knows ?friend .
            }
            FILTER (LANG(?name) = "en")
        }
        GROUP BY ?name
        HAVING (COUNT(?friend) > 5)
        ORDER BY DESC(?friendCount)
        LIMIT 100
        "#,
    )
    .unwrap()
}

fn bench_query_parsing(c: &mut Criterion) {
    let simple_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let complex_query = r#"
        PREFIX schema: <http://schema.org/>
        SELECT ?name (COUNT(?o) AS ?count) WHERE {
            { ?s schema:name ?name } UNION { ?s schema:title ?name }
            OPTIONAL { ?s ?p ?o }
            FILTER(LANG(?name) = "en")
        }
        GROUP BY ?name
    "#;

    let mut group = c.benchmark_group("query_parsing");

    group.bench_function("simple_query", |b| {
        b.iter(|| Query::parse(black_box(simple_query)))
    });

    group.bench_function("complex_query", |b| {
        b.iter(|| Query::parse(black_box(complex_query)))
    });

    group.finish();
}

fn bench_routing_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_latency");

    for source_count in [5, 10, 50, 100].iter() {
        let router = create_router_with_sources(*source_count);
        let simple = create_simple_query();
        let complex = create_complex_query();

        group.bench_with_input(
            BenchmarkId::new("simple_query", source_count),
            source_count,
            |b, _| b.iter(|| router.route(black_box(&simple))),
        );

        group.bench_with_input(
            BenchmarkId::new("complex_query", source_count),
            source_count,
            |b, _| b.iter(|| router.route(black_box(&complex))),
        );
    }

    group.finish();
}

fn bench_source_stats_update(c: &mut Criterion) {
    let mut router = create_router_with_sources(10);

    c.bench_function("source_stats_update", |b| {
        let mut i = 0;
        b.iter(|| {
            let source_id = format!("source_{}", i % 10);
            router
                .update_source_stats(&source_id, 100, true, 50)
                .unwrap();
            i += 1;
        })
    });
}

#[cfg(feature = "ml")]
fn bench_feature_extraction(c: &mut Criterion) {
    use oxirouter::ml::FeatureVector;

    let simple = create_simple_query();
    let complex = create_complex_query();

    let mut group = c.benchmark_group("feature_extraction");

    group.bench_function("simple_query", |b| {
        b.iter(|| FeatureVector::from_query(black_box(&simple)))
    });

    group.bench_function("complex_query", |b| {
        b.iter(|| FeatureVector::from_query(black_box(&complex)))
    });

    group.finish();
}

#[cfg(feature = "ml")]
fn bench_naive_bayes_inference(c: &mut Criterion) {
    use oxirouter::ml::{FeatureVector, Model, NaiveBayesClassifier};

    let mut nb = NaiveBayesClassifier::new(24);
    let sources: Vec<String> = (0..10).map(|i| format!("source_{}", i)).collect();
    let source_refs: Vec<&String> = sources.iter().collect();
    nb.initialize_sources(&source_refs);

    let query = create_simple_query();
    let features = FeatureVector::from_query(&query).unwrap();

    c.bench_function("naive_bayes_inference", |b| {
        b.iter(|| nb.predict(black_box(&features), black_box(&source_refs)))
    });
}

#[cfg(feature = "ml")]
fn bench_neural_network_inference(c: &mut Criterion) {
    use oxirouter::ml::{FeatureVector, Model, NeuralNetwork};

    let nn = NeuralNetwork::new(24, &[16, 8], 10);
    let sources: Vec<String> = (0..10).map(|i| format!("source_{}", i)).collect();
    let source_refs: Vec<&String> = sources.iter().collect();

    let query = create_simple_query();
    let features = FeatureVector::from_query(&query).unwrap();

    c.bench_function("neural_network_inference", |b| {
        b.iter(|| nn.predict(black_box(&features), black_box(&source_refs)))
    });
}

#[cfg(feature = "rl")]
fn bench_policy_update(c: &mut Criterion) {
    use oxirouter::rl::{Policy, Reward};

    let mut policy = Policy::ucb();
    for i in 0..10 {
        policy.initialize_source(format!("source_{}", i));
    }

    c.bench_function("policy_update", |b| {
        let mut i = 0;
        b.iter(|| {
            let source_id = format!("source_{}", i % 10);
            policy.update(&source_id, Reward::new(0.7));
            i += 1;
        })
    });
}

fn bench_aggregation(c: &mut Criterion) {
    use oxirouter::federation::aggregator::{AggregationConfig, AggregationStrategy};
    use oxirouter::federation::{Aggregator, QueryResult};

    let results: Vec<QueryResult> = (0..10)
        .map(|i| QueryResult::success(format!("source_{}", i), vec![0; 1000], i * 10, 100))
        .collect();

    let mut group = c.benchmark_group("aggregation");

    for strategy in [
        AggregationStrategy::First,
        AggregationStrategy::Union,
        AggregationStrategy::Largest,
        AggregationStrategy::Fastest,
    ]
    .iter()
    {
        let config = AggregationConfig {
            strategy: *strategy,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        group.bench_with_input(
            BenchmarkId::new("strategy", format!("{:?}", strategy)),
            strategy,
            |b, _| b.iter(|| aggregator.aggregate(black_box(&results))),
        );
    }

    group.finish();
}

// Define criterion groups
#[cfg(feature = "ml")]
criterion_group!(
    ml_benches,
    bench_feature_extraction,
    bench_naive_bayes_inference,
    bench_neural_network_inference,
);

#[cfg(feature = "rl")]
criterion_group!(rl_benches, bench_policy_update);

criterion_group!(
    core_benches,
    bench_query_parsing,
    bench_routing_latency,
    bench_source_stats_update,
    bench_aggregation,
);

// Main criterion entry point
#[cfg(all(feature = "ml", feature = "rl"))]
criterion_main!(core_benches, ml_benches, rl_benches);

#[cfg(all(feature = "ml", not(feature = "rl")))]
criterion_main!(core_benches, ml_benches);

#[cfg(all(not(feature = "ml"), feature = "rl"))]
criterion_main!(core_benches, rl_benches);

#[cfg(all(not(feature = "ml"), not(feature = "rl")))]
criterion_main!(core_benches);
