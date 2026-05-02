//! Integration tests for the `sparql` feature — pure-Rust SPARQL prefix expansion
//! and projection-variable extraction via `src/core/sparql.rs`.

#[cfg(feature = "sparql")]
mod sparql_tests {
    use oxirouter::{DataSource, Query, Router};

    /// Test 1: real PREFIX declarations expand to full URIs in predicates
    #[test]
    fn test_from_sparql_expands_prefixes() {
        let sparql = "\
            PREFIX ex: <http://example.org/> \
            SELECT ?s WHERE { ?s ex:foo ?o }";
        let query = Query::from_sparql(sparql).unwrap();
        // ex:foo should be expanded to http://example.org/foo
        assert!(
            query.predicates.contains("http://example.org/foo"),
            "predicates should contain expanded URI; got: {:?}",
            query.predicates
        );
    }

    /// Test 2: SELECT projection variables are extracted (without '?')
    #[test]
    fn test_from_sparql_projection_vars() {
        let sparql = "SELECT ?a ?b ?c WHERE { ?a ?b ?c }";
        let query = Query::from_sparql(sparql).unwrap();
        assert_eq!(
            query.projection_vars.len(),
            3,
            "expected 3 projection vars; got: {:?}",
            query.projection_vars
        );
        let names: Vec<&str> = query
            .projection_vars
            .iter()
            .map(|v| v.trim_start_matches('?'))
            .collect();
        assert!(
            names.contains(&"a") && names.contains(&"b") && names.contains(&"c"),
            "expected a, b, c; got: {:?}",
            names
        );
    }

    /// Test 3: non-SELECT query has empty projection_vars
    #[test]
    fn test_from_sparql_non_select_query() {
        // ASK query — projection_vars must be empty
        let sparql = "ASK { <http://example.org/s> ?p ?o }";
        let query = Query::from_sparql(sparql).unwrap();
        assert!(
            query.projection_vars.is_empty(),
            "ASK query should have empty projection_vars; got: {:?}",
            query.projection_vars
        );
    }

    /// Test 4: Query::parse has empty projection_vars (heuristic path unchanged)
    #[test]
    fn test_parse_has_no_projection_vars() {
        let sparql = "SELECT ?a ?b WHERE { ?a ?b ?c }";
        let query = Query::parse(sparql).unwrap();
        assert!(
            query.projection_vars.is_empty(),
            "heuristic Query::parse should NOT populate projection_vars; got: {:?}",
            query.projection_vars
        );
    }

    /// Test 5: Router::route_sparql convenience method
    #[test]
    fn test_route_sparql_method() {
        let mut router = Router::new();
        router.add_source(DataSource::new("dbpedia", "https://dbpedia.org/sparql"));
        router.add_source(DataSource::new(
            "wikidata",
            "https://query.wikidata.org/sparql",
        ));
        let sparql = "SELECT ?s WHERE { ?s a <http://schema.org/Person> }";
        let ranking = router.route_sparql(sparql).unwrap();
        assert!(
            !ranking.sources.is_empty(),
            "route_sparql should return at least one source"
        );
    }

    /// Test 6: agent route action uses sparql path when feature enabled
    #[cfg(feature = "agent")]
    #[test]
    fn test_agent_route_uses_sparql_when_enabled() {
        use oxirouter::RouterAgent;
        let mut router = Router::new();
        router.add_source(DataSource::new("dbpedia", "https://dbpedia.org/sparql"));
        router.add_source(DataSource::new(
            "wikidata",
            "https://query.wikidata.org/sparql",
        ));
        let mut agent = RouterAgent::new(router);
        let input = r#"{"query": "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?p WHERE { ?p a foaf:Person }"}"#;
        let result = agent.dispatch("oxirouter.route", input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["sources"]
                .as_array()
                .map(|a| !a.is_empty())
                .unwrap_or(false),
            "agent route should return at least one source; got: {}",
            result
        );
    }
}
