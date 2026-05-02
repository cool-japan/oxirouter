//! E2E tests that property-path predicates steer routing to the right source.
//!
//! Both test sources are configured with `SourceCapabilities::full()` because:
//!
//! 1. The `^p` inverse-path syntax sets `has_property_paths = true` in the
//!    heuristic parser (via the `query.contains('^')` check).
//! 2. `has_property_paths = true` causes `requires_sparql_1_1()` to return
//!    `true`, which triggers the hard capability gate in `route_heuristic`:
//!    sources without `sparql_1_1 = true` are skipped entirely.
//! 3. A real SPARQL endpoint that can answer property-path queries must
//!    declare SPARQL 1.1 support — so using `SourceCapabilities::full()` is
//!    semantically correct for these tests.

#[cfg(all(test, feature = "sparql"))]
mod tests {
    use oxirouter::prelude::{DataSource, Query, Router, SourceCapabilities};

    fn foaf_source() -> DataSource {
        DataSource::new("foaf-source", "http://localhost:9999/sparql")
            .with_vocabulary("http://xmlns.com/foaf/0.1/")
            .with_capabilities(SourceCapabilities::full())
    }

    fn schema_source() -> DataSource {
        DataSource::new("schema-source", "http://localhost:9999/sparql")
            .with_vocabulary("http://schema.org/")
            .with_capabilities(SourceCapabilities::full())
    }

    fn make_router() -> Router {
        let mut r = Router::new();
        r.add_source(foaf_source());
        r.add_source(schema_source());
        r
    }

    fn foaf_rank(q: &str) -> (usize, usize) {
        let router = make_router();
        let query = Query::from_sparql(q).expect("parse failed");
        let ranking = router.route(&query).expect("route failed");
        let foaf_idx = ranking
            .sources
            .iter()
            .position(|s| s.source_id == "foaf-source")
            .expect("foaf-source not in ranking");
        let schema_idx = ranking
            .sources
            .iter()
            .position(|s| s.source_id == "schema-source")
            .expect("schema-source not in ranking");
        (foaf_idx, schema_idx)
    }

    #[test]
    fn property_path_plus_routes_to_foaf() {
        // foaf:knows+ — foaf source must rank above schema source
        let (foaf, schema) = foaf_rank(
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/>
             SELECT ?o WHERE { <#alice> foaf:knows+ ?o }",
        );
        assert!(
            foaf < schema,
            "foaf source (idx {foaf}) must outrank schema (idx {schema}) for foaf:knows+"
        );
    }

    #[test]
    fn property_path_sequence_routes_to_foaf() {
        // foaf:knows/foaf:name — sequence path, both leaves are foaf
        let (foaf, schema) = foaf_rank(
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/>
             SELECT ?n WHERE { <#alice> foaf:knows/foaf:name ?n }",
        );
        assert!(
            foaf < schema,
            "foaf source must outrank schema for foaf:knows/foaf:name"
        );
    }

    #[test]
    fn property_path_inverse_routes_to_foaf() {
        // ^foaf:knows — inverse path; leaf IRI is still foaf:knows
        let (foaf, schema) = foaf_rank(
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/>
             SELECT ?s WHERE { ?s ^foaf:knows <#alice> }",
        );
        assert!(
            foaf < schema,
            "foaf source must outrank schema for ^foaf:knows"
        );
    }

    #[test]
    fn property_path_alternation_routes_to_foaf() {
        // foaf:knows|foaf:made — alternation; both leaves are foaf
        let (foaf, schema) = foaf_rank(
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/>
             SELECT ?o WHERE { <#alice> foaf:knows|foaf:made ?o }",
        );
        assert!(
            foaf < schema,
            "foaf source must outrank schema for foaf:knows|foaf:made"
        );
    }
}
