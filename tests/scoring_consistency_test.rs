//! Scoring consistency tests verifying `route()` and `explain()` agree.
//!
//! These tests enforce the invariant:
//! For any source that passes routing filters,
//! `route().sources[i].confidence == sum(explain()[i].components[*].contribution)`
//! within `f32::EPSILON * 4`.

#[cfg(feature = "rl")]
use oxirouter::ScoreComponent;
#[cfg(all(feature = "p2p", any(feature = "device", feature = "std")))]
use oxirouter::SourceKind;
#[cfg(all(feature = "p2p", any(feature = "device", feature = "std")))]
use oxirouter::context::{CombinedContext, ContextProvider};
use oxirouter::core::query::Query;
use oxirouter::prelude::SourceCapabilities;
use oxirouter::{DataSource, core::router::Router};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "p2p", any(feature = "device", feature = "std")))]
struct FixedContextProvider {
    ctx: CombinedContext,
}

#[cfg(all(feature = "p2p", any(feature = "device", feature = "std")))]
impl ContextProvider for FixedContextProvider {
    fn get_combined_context(&self) -> CombinedContext {
        self.ctx.clone()
    }
}

/// Build a simple SELECT query with no SPARQL 1.1 features.
fn simple_query() -> Query {
    Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse simple query")
}

/// Build a query that requires SPARQL 1.1 (uses aggregation → GROUP BY).
fn sparql_1_1_query() -> Query {
    Query::parse("SELECT (COUNT(?s) AS ?count) WHERE { ?s ?p ?o }").expect("parse sparql 1.1 query")
}

/// Build a simple router with 3–5 diverse sources.
fn make_diverse_router() -> Router {
    let mut router = Router::new();
    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU")
            .with_priority(1.0)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/entity/")
            .with_region("US")
            .with_priority(0.8)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("schema-org", "https://schema.example.org/sparql")
            .with_vocabulary("http://schema.org/")
            .with_priority(1.2)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("foaf-src", "https://foaf.example.org/sparql")
            .with_vocabulary("http://xmlns.com/foaf/0.1/")
            .with_priority(0.5)
            .with_capabilities(SourceCapabilities::basic()),
    );
    router
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: route() and explain() agree on confidence for every matched source.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_route_and_explain_agree() {
    let router = make_diverse_router();

    // Try multiple distinct queries.
    let queries = [
        "SELECT ?s WHERE { ?s ?p ?o }",
        "PREFIX dbo: <http://dbpedia.org/ontology/> SELECT ?name WHERE { ?s dbo:name ?name }",
        "PREFIX schema: <http://schema.org/> SELECT ?s WHERE { ?s a schema:Person }",
    ];

    for raw_query in &queries {
        let query = Query::parse(raw_query).expect("parse query");

        let route_result = router.route(&query).expect("route should succeed");
        let explain_result = router.explain(&query).expect("explain should succeed");

        for routed in &route_result.sources {
            // Find matching explanation by source_id (not by index).
            let explanation = explain_result
                .iter()
                .find(|e| e.source_id == routed.source_id);

            let explanation = match explanation {
                Some(e) => e,
                None => panic!(
                    "source '{}' in route() but missing in explain(); query: {raw_query}",
                    routed.source_id,
                ),
            };

            // Skip capability-gate failures (different type of entry).
            if explanation
                .components
                .iter()
                .any(|c| c.name == "capability_gate_failed")
            {
                continue;
            }

            let component_sum: f32 = explanation.components.iter().map(|c| c.contribution).sum();

            let delta = (routed.confidence - component_sum).abs();
            assert!(
                delta <= f32::EPSILON * 4.0,
                "source '{}': route confidence {:.8} != sum of explain contributions \
                 {:.8} (delta {:.2e}); query: {raw_query}",
                routed.source_id,
                routed.confidence,
                component_sum,
                delta,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: sum(contributions) == total_score for every explain() output.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_components_sum_to_total_score() {
    let router = make_diverse_router();
    let query = simple_query();

    let explanations = router.explain(&query).expect("explain succeeded");
    assert!(
        !explanations.is_empty(),
        "expected at least one explanation"
    );

    for exp in &explanations {
        // Skip diagnostic gate entries — their total_score is -1.0 by convention.
        if exp
            .components
            .iter()
            .any(|c| c.name == "capability_gate_failed")
        {
            // Even for gate entries the invariant holds (single component, contribution == total_score).
            let sum: f32 = exp.components.iter().map(|c| c.contribution).sum();
            assert_eq!(
                sum, exp.total_score,
                "capability_gate entry '{}': component sum {sum} != total_score {}",
                exp.source_id, exp.total_score,
            );
            continue;
        }

        let component_sum: f32 = exp.components.iter().map(|c| c.contribution).sum();
        let diff = (component_sum - exp.total_score).abs();
        assert!(
            diff < 1e-5,
            "source '{}': component sum {component_sum:.8} != total_score {:.8} (diff {diff:.2e})",
            exp.source_id,
            exp.total_score,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: rl_q_blend component appears with non-zero contribution when RL
// is enabled and a source has been visited (non-default Q-value from reward).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "rl")]
#[test]
fn test_rl_blend_appears_as_component() {
    use oxirouter::{Policy, Reward};

    let mut router = Router::new();
    router.add_source(
        DataSource::new("src-a", "https://a.example.org/sparql")
            .with_priority(1.0)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("src-b", "https://b.example.org/sparql")
            .with_priority(0.8)
            .with_capabilities(SourceCapabilities::full()),
    );

    // Enable RL with a deterministic UCB policy.
    let mut policy = Policy::ucb();
    policy.initialize_source("src-a");
    policy.initialize_source("src-b");
    // Apply positive rewards so Q-value rises well above 0.0.
    for _ in 0..5 {
        policy.update("src-a", Reward::new(1.0));
        policy.update("src-b", Reward::new(0.8));
    }
    router.set_policy(policy);

    let query = simple_query();
    let explanations = router.explain(&query).expect("explain succeeded");

    for exp in &explanations {
        // Skip any diagnostic entries.
        if exp
            .components
            .iter()
            .any(|c| c.name == "capability_gate_failed")
        {
            continue;
        }

        let rl_component: Option<&ScoreComponent> =
            exp.components.iter().find(|c| c.name == "rl_q_blend");

        assert!(
            rl_component.is_some(),
            "source '{}' missing 'rl_q_blend' component; components: {:?}",
            exp.source_id,
            exp.components,
        );

        let rl = rl_component.expect("just checked is_some");
        assert!(
            rl.contribution.abs() > f32::EPSILON,
            "source '{}': rl_q_blend contribution should be non-zero; got {}",
            exp.source_id,
            rl.contribution,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: p2p_kind_boost component appears with non-zero contribution for a
// P2P source when device is offline (gated on p2p + device/std features).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "p2p", any(feature = "device", feature = "std")))]
#[test]
fn test_p2p_boost_appears_as_component() {
    use oxirouter::context::{DeviceContext, NetworkType};

    // Offline device → multiplier 1.25 → p2p_kind_boost should appear.
    let offline_ctx = CombinedContext::new()
        .with_device(DeviceContext::new().with_network(NetworkType::Offline, 0));
    let provider = FixedContextProvider { ctx: offline_ctx };

    let mut router = Router::with_context_provider(provider);
    router.add_source(
        DataSource::new("sparql-src", "https://sparql.example.org/sparql")
            .with_priority(1.0)
            .with_kind(SourceKind::Sparql)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("ipfs-src", "/ip4/127.0.0.1/tcp/4001/p2p/QmTest")
            .with_priority(1.0)
            .with_kind(SourceKind::P2pIpfs {
                multiaddr: "/ip4/127.0.0.1/tcp/4001/p2p/QmTest".into(),
            })
            .with_capabilities(SourceCapabilities::full()),
    );

    let query = simple_query();
    let explanations = router.explain(&query).expect("explain succeeded");

    let ipfs_exp = explanations
        .iter()
        .find(|e| e.source_id == "ipfs-src")
        .expect("ipfs-src must appear in explain()");

    let p2p_component = ipfs_exp
        .components
        .iter()
        .find(|c| c.name == "p2p_kind_boost");

    assert!(
        p2p_component.is_some(),
        "P2P source should have a 'p2p_kind_boost' component; components: {:?}",
        ipfs_exp.components,
    );

    let p2p = p2p_component.expect("just checked is_some");
    assert!(
        p2p.contribution.abs() > f32::EPSILON,
        "p2p_kind_boost contribution should be non-zero for offline device; got {}",
        p2p.contribution,
    );
    // The weight field must be 1.0 per spec.
    assert_eq!(
        p2p.weight, 1.0,
        "p2p_kind_boost weight must be 1.0 per spec; got {}",
        p2p.weight,
    );
    // For offline device, multiplier is 1.25 → raw_value should be 1.25.
    assert!(
        (p2p.raw_value - 1.25).abs() < 1e-6,
        "expected raw_value=1.25 for offline P2P boost; got {}",
        p2p.raw_value,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: capability-filtered source appears in explain() but not in route().
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_capability_filter_visible() {
    let mut router = Router::new();

    // This source does NOT support SPARQL 1.1 → will be filtered by capability gate.
    router.add_source(
        DataSource::new("basic-src", "https://basic.example.org/sparql")
            .with_priority(1.0)
            .with_capabilities(SourceCapabilities::basic()),
    );

    // This source DOES support SPARQL 1.1 → will be routed normally.
    router.add_source(
        DataSource::new("full-src", "https://full.example.org/sparql")
            .with_priority(1.0)
            .with_capabilities(SourceCapabilities::full()),
    );

    // Query that requires SPARQL 1.1 (has aggregation).
    let query = sparql_1_1_query();

    // Verify the query actually triggers the capability gate.
    assert!(
        query.requires_sparql_1_1(),
        "test fixture query must require SPARQL 1.1 for this test to be meaningful"
    );

    let route_result = router.route(&query).expect("route should succeed");
    let explain_result = router.explain(&query).expect("explain should succeed");

    // "basic-src" must NOT appear in route() results.
    assert!(
        !route_result
            .sources
            .iter()
            .any(|s| s.source_id == "basic-src"),
        "capability-filtered source 'basic-src' must not appear in route() results"
    );

    // "basic-src" MUST appear in explain() with a capability_gate_failed component.
    let basic_exp = explain_result
        .iter()
        .find(|e| e.source_id == "basic-src")
        .expect("capability-filtered source 'basic-src' must appear in explain()");

    assert!(
        basic_exp
            .components
            .iter()
            .any(|c| c.name == "capability_gate_failed"),
        "basic-src in explain() must have a 'capability_gate_failed' component; \
         components: {:?}",
        basic_exp.components,
    );

    // The gate entry must have raw_value = -1.0 and contribution = -1.0.
    let gate = basic_exp
        .components
        .iter()
        .find(|c| c.name == "capability_gate_failed")
        .expect("already checked above");
    assert_eq!(
        gate.raw_value, -1.0,
        "capability_gate_failed raw_value must be -1.0; got {}",
        gate.raw_value
    );
    assert_eq!(
        gate.contribution, -1.0,
        "capability_gate_failed contribution must be -1.0; got {}",
        gate.contribution
    );

    // "full-src" must appear in route() results.
    assert!(
        route_result
            .sources
            .iter()
            .any(|s| s.source_id == "full-src"),
        "'full-src' with SPARQL 1.1 support must appear in route() results"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: regression gate — ordering and confidences must match pre-refactor
// values within 1e-3 for a fixed input set.
// ─────────────────────────────────────────────────────────────────────────────
//
// How these values were captured:
//   - Ran the router with the exact fixture below (no RL, no ML, no context).
//   - Collected route().sources in confidence order.
//   - Hardcoded the (source_id, confidence) pairs.
// If routing decisions change unexpectedly, this test fails as a regression gate.

#[test]
fn test_regression_scoring() {
    // Fixed, no-context router for deterministic output.
    // All priorities must yield confidence >= min_confidence (default 0.1).
    // confidence = priority * 0.1, so priority must be >= 1.0.
    let mut router = Router::new();
    router.add_source(
        DataSource::new("src-alpha", "https://alpha.example.org/sparql")
            .with_priority(3.0)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("src-beta", "https://beta.example.org/sparql")
            .with_priority(2.0)
            .with_capabilities(SourceCapabilities::full()),
    );
    router.add_source(
        DataSource::new("src-gamma", "https://gamma.example.org/sparql")
            .with_priority(1.0)
            .with_capabilities(SourceCapabilities::full()),
    );

    let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("parse regression query");

    let ranking = router.route(&query).expect("route should succeed");

    // Must have exactly 3 sources.
    assert_eq!(
        ranking.sources.len(),
        3,
        "expected 3 sources in regression ranking, got {}; sources: {:?}",
        ranking.sources.len(),
        ranking
            .sources
            .iter()
            .map(|s| &s.source_id)
            .collect::<Vec<_>>(),
    );

    // Verify descending confidence order.
    let confidences: Vec<f32> = ranking.sources.iter().map(|s| s.confidence).collect();
    for window in confidences.windows(2) {
        assert!(
            window[0] >= window[1],
            "ranking is not sorted by descending confidence: {:?}",
            confidences,
        );
    }

    // Verify the ordering: higher priority => higher confidence.
    let source_order: Vec<&str> = ranking
        .sources
        .iter()
        .map(|s| s.source_id.as_str())
        .collect();
    assert_eq!(
        source_order[0], "src-alpha",
        "src-alpha (priority=3.0) must rank first; order: {source_order:?}",
    );
    assert_eq!(
        source_order[1], "src-beta",
        "src-beta (priority=2.0) must rank second; order: {source_order:?}",
    );
    assert_eq!(
        source_order[2], "src-gamma",
        "src-gamma (priority=1.0) must rank third; order: {source_order:?}",
    );

    // Hardcoded confidence values captured from canonical scoring.
    // priority contribution = priority * 0.1; no vocab, no perf, no geo, no
    // availability delta, circuit_breaker = 0, no rl, no p2p.
    // Clamped to [0.0, 1.0]:
    // src-alpha: 3.0 * 0.1 = 0.3  (no clamp needed)
    // src-beta:  2.0 * 0.1 = 0.2
    // src-gamma: 1.0 * 0.1 = 0.1
    let expected = [
        ("src-alpha", 0.3_f32),
        ("src-beta", 0.2_f32),
        ("src-gamma", 0.1_f32),
    ];

    for (src, &(expected_id, expected_conf)) in ranking.sources.iter().zip(expected.iter()) {
        assert_eq!(
            src.source_id, expected_id,
            "regression: expected source '{expected_id}' at this rank, got '{}'",
            src.source_id,
        );
        let delta = (src.confidence - expected_conf).abs();
        assert!(
            delta < 1e-3,
            "regression: source '{}' confidence {:.6} differs from expected {:.6} by {:.2e}",
            src.source_id,
            src.confidence,
            expected_conf,
            delta,
        );
    }
}
