//! Integration tests for P2P / IPFS source kinds and offline-aware routing.
//!
//! These tests cover:
//! - P2P sources rank first when the device is offline
//! - SPARQL sources are not outranked on good connectivity
//! - `DataSource` serde roundtrip with `SourceKind::P2pIpfs`
//! - Backward-compat: v1 JSON without `kind` deserialises to `SourceKind::Sparql`
//! - `SourceKind::is_p2p()` / `is_cache()` predicate correctness

// The connectivity-routing tests require both `p2p` (scoring adjustments) and
// device context (any(device, std) gating in router.rs).
// Serde / predicate tests run unconditionally.

use oxirouter::{DataSource, SourceKind};

// ---------------------------------------------------------------------------
// Unconditional predicate tests
// ---------------------------------------------------------------------------

/// `SourceKind::is_p2p()` returns true for P2P variants only.
#[test]
fn test_source_kind_is_p2p() {
    assert!(
        SourceKind::P2pIpfs {
            multiaddr: "/ip4/0.0.0.0/tcp/0".into()
        }
        .is_p2p(),
        "P2pIpfs must be P2P"
    );
    assert!(
        SourceKind::P2pLibp2p {
            peer_id: "QmFakePeerId".into()
        }
        .is_p2p(),
        "P2pLibp2p must be P2P"
    );
    assert!(!SourceKind::Sparql.is_p2p(), "Sparql must not be P2P");
    assert!(!SourceKind::Cache.is_p2p(), "Cache must not be P2P");
    assert!(
        !SourceKind::Custom("custom".into()).is_p2p(),
        "Custom must not be P2P"
    );
}

/// `SourceKind::is_cache()` returns true only for the Cache variant.
#[test]
fn test_source_kind_is_cache() {
    assert!(SourceKind::Cache.is_cache(), "Cache must report is_cache()");
    assert!(!SourceKind::Sparql.is_cache(), "Sparql must not be Cache");
    assert!(
        !SourceKind::P2pIpfs {
            multiaddr: "/ip4/0.0.0.0/tcp/0".into()
        }
        .is_cache(),
        "P2pIpfs must not be Cache"
    );
    assert!(
        !SourceKind::P2pLibp2p {
            peer_id: "QmFakePeerId".into()
        }
        .is_cache(),
        "P2pLibp2p must not be Cache"
    );
    assert!(
        !SourceKind::Custom("c".into()).is_cache(),
        "Custom must not be Cache"
    );
}

/// `DataSource` serde roundtrip preserves `SourceKind::P2pIpfs`.
#[test]
fn test_datasource_serde_roundtrip_p2p_ipfs() {
    let multiaddr = "/ip4/127.0.0.1/tcp/4001/p2p/QmTest";
    let source = DataSource::new("ipfs-node", multiaddr).with_kind(SourceKind::P2pIpfs {
        multiaddr: multiaddr.into(),
    });

    let json = serde_json::to_string(&source).expect("serialisation must succeed");
    let restored: DataSource = serde_json::from_str(&json).expect("deserialisation must succeed");

    assert_eq!(restored.id, "ipfs-node");
    assert!(
        restored.kind.is_p2p(),
        "restored SourceKind should still be P2P"
    );

    if let SourceKind::P2pIpfs { multiaddr: ref ma } = restored.kind {
        assert_eq!(ma, multiaddr, "multiaddr must survive roundtrip");
    } else {
        panic!("expected SourceKind::P2pIpfs after roundtrip");
    }
}

/// Deserialising a v1 `DataSource` JSON that lacks a `kind` field must succeed
/// and produce `SourceKind::Sparql` (the serde default).
#[test]
fn test_backward_compat_no_kind_field() {
    let v1_json = r#"{
        "id": "legacy-source",
        "endpoint": "http://legacy.example.com/sparql",
        "capabilities": {
            "sparql_1_1": false,
            "federation": false,
            "construct": false,
            "ask": false,
            "describe": false,
            "property_paths": false,
            "aggregation": false,
            "subqueries": false,
            "bind": false,
            "values": false,
            "max_results": 0,
            "full_text_search": false,
            "geospatial": false
        },
        "stats": {
            "total_queries": 0,
            "successful_queries": 0,
            "total_results": 0,
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
            "last_query_time": 0
        },
        "regions": [],
        "vocabularies": [],
        "available": true,
        "priority": 1.0
    }"#;

    let source: DataSource =
        serde_json::from_str(v1_json).expect("v1 JSON must deserialise successfully");

    assert_eq!(source.id, "legacy-source");
    assert_eq!(
        source.kind,
        SourceKind::Sparql,
        "missing `kind` field must default to SourceKind::Sparql"
    );
}

// ---------------------------------------------------------------------------
// Connectivity-routing tests (require p2p + device/std features)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "p2p", any(feature = "device", feature = "std")))]
mod connectivity_routing {
    use oxirouter::{
        DataSource, SourceKind,
        context::ContextProvider,
        context::{CombinedContext, DeviceContext, NetworkType},
        core::{query::Query, router::Router},
    };

    // A minimal `ContextProvider` that always returns a preset `CombinedContext`.
    struct FixedContextProvider {
        ctx: CombinedContext,
    }

    impl ContextProvider for FixedContextProvider {
        fn get_combined_context(&self) -> CombinedContext {
            self.ctx.clone()
        }
    }

    fn sparql_source(id: &str, priority: f32) -> DataSource {
        DataSource::new(id, "http://example.com/sparql")
            .with_priority(priority)
            .with_kind(SourceKind::Sparql)
    }

    fn ipfs_source(id: &str, priority: f32, multiaddr: &str) -> DataSource {
        DataSource::new(id, multiaddr)
            .with_priority(priority)
            .with_kind(SourceKind::P2pIpfs {
                multiaddr: multiaddr.to_string(),
            })
    }

    fn simple_query() -> Query {
        Query::parse("SELECT ?s WHERE { ?s ?p ?o }").expect("valid query")
    }

    fn offline_device() -> DeviceContext {
        DeviceContext::new().with_network(NetworkType::Offline, 0)
    }

    fn good_device() -> DeviceContext {
        // Wifi with 100 Mbps → network_quality() > 0.3
        DeviceContext::new().with_network(NetworkType::Wifi, 100_000)
    }

    /// P2P source must rank first when the device is offline.
    #[test]
    fn test_p2p_ranks_first_when_offline() {
        let ctx = CombinedContext::new().with_device(offline_device());
        let provider = FixedContextProvider { ctx };

        let mut router = Router::with_context_provider(provider);

        // Equal priority so P2P boost alone determines the winner.
        router.add_source(sparql_source("sparql-a", 1.0));
        router.add_source(ipfs_source(
            "ipfs-b",
            1.0,
            "/ip4/127.0.0.1/tcp/4001/p2p/QmTest",
        ));

        let query = simple_query();
        let ranking = router.route(&query).expect("routing should succeed");

        assert!(
            ranking.sources.len() >= 2,
            "expected at least 2 sources, got {}",
            ranking.sources.len()
        );

        let best = ranking.best().expect("ranking must have a best source");
        assert_eq!(
            best.source_id, "ipfs-b",
            "P2P source should be ranked first when device is offline"
        );
    }

    /// SPARQL should not be outranked by P2P on good connectivity.
    ///
    /// With the 0.95 P2P penalty on good links, the IPFS source's confidence may
    /// drop below `min_confidence` (0.1) and be filtered out, leaving only SPARQL.
    /// Either outcome (SPARQL ranked first, or SPARQL is the only source) means
    /// SPARQL has not been outranked by P2P.
    #[test]
    fn test_sparql_not_outranked_on_good_connectivity() {
        let ctx = CombinedContext::new().with_device(good_device());
        let provider = FixedContextProvider { ctx };

        let mut router = Router::with_context_provider(provider);

        // Equal priority: the 0.95 P2P penalty must not let IPFS beat SPARQL.
        router.add_source(sparql_source("sparql-a", 1.0));
        router.add_source(ipfs_source(
            "ipfs-b",
            1.0,
            "/ip4/127.0.0.1/tcp/4001/p2p/QmTest",
        ));

        let query = simple_query();
        let ranking = router.route(&query).expect("routing should succeed");

        assert!(
            !ranking.is_empty(),
            "ranking must contain at least one source"
        );

        // SPARQL must appear in ranking.
        assert!(
            ranking.sources.iter().any(|s| s.source_id == "sparql-a"),
            "sparql-a must appear in ranking"
        );

        // If both sources appear, SPARQL must rank >= IPFS.
        let sparql_conf = ranking
            .sources
            .iter()
            .find(|s| s.source_id == "sparql-a")
            .map(|s| s.confidence)
            .expect("sparql-a must appear in ranking");

        if let Some(ipfs_conf) = ranking
            .sources
            .iter()
            .find(|s| s.source_id == "ipfs-b")
            .map(|s| s.confidence)
        {
            assert!(
                sparql_conf >= ipfs_conf,
                "SPARQL ({sparql_conf}) should not lose to P2P ({ipfs_conf}) on good connectivity"
            );
        }
        // If ipfs-b was filtered out (confidence below min_confidence) that's
        // also a valid "SPARQL wins" outcome — no assertion needed.
    }
}
