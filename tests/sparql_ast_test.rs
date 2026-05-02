//! Integration tests for Block C: SPARQL AST structural parser + 10 ML feature dimensions.

#[cfg(feature = "sparql")]
mod sparql_ast_tests {
    use oxirouter::{DataSource, Query, Router};

    // ─────────────────────────────────────────────────────────────────────────
    // AST feature tests (via Query::from_sparql → query.ast_features)
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 1: OPTIONAL detected → optional_count > 0.0
    #[test]
    fn test_ast_features_optional() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s a <http://example.org/T> } }";
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.optional_count > 0.0,
            "expected optional_count > 0, got {}",
            af.optional_count
        );
    }

    /// Test 2: FILTER detected → filter_count > 0.0
    #[test]
    fn test_ast_features_filter() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o . FILTER(?o > 5) }";
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.filter_count > 0.0,
            "expected filter_count > 0, got {}",
            af.filter_count
        );
    }

    /// Test 3: UNION detected → union_branch_count > 0.0
    #[test]
    fn test_ast_features_union() {
        let sparql =
            "SELECT ?s WHERE { { ?s <http://a.org/p> ?o } UNION { ?s <http://b.org/p> ?o } }";
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.union_branch_count > 0.0,
            "expected union_branch_count > 0, got {}",
            af.union_branch_count
        );
    }

    /// Test 4: SELECT DISTINCT → has_distinct == 1.0
    #[test]
    fn test_ast_features_distinct() {
        let sparql = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert_eq!(
            af.has_distinct, 1.0,
            "expected has_distinct = 1.0, got {}",
            af.has_distinct
        );
    }

    /// Test 5: Subquery (nested SELECT ... WHERE {...}) → subquery_count > 0.0
    #[test]
    fn test_ast_features_subquery() {
        let sparql = r#"
            SELECT ?s ?count WHERE {
                ?s ?p ?o .
                { SELECT ?s (COUNT(?o) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?s }
            }
        "#;
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.subquery_count > 0.0,
            "expected subquery_count > 0, got {}",
            af.subquery_count
        );
    }

    /// Test 6: FeatureVector length with sparql feature = base + 10 AST dims.
    ///
    /// `from_query` adds 22 base dims; with `sparql` enabled, 10 SPARQL AST dims
    /// are appended → total 32 (no context features added by `from_query`).
    #[cfg(feature = "ml")]
    #[test]
    fn test_feature_vector_length_with_sparql() {
        use oxirouter::FeatureVector;
        let sparql = "SELECT ?s WHERE { ?s ?p ?o }";
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let fv = FeatureVector::from_query(&query).expect("from_query failed");
        // 22 base + 10 SPARQL AST = 32 (no context features)
        assert_eq!(
            fv.values.len(),
            32,
            "expected 32 features (22 base + 10 SPARQL AST), got {}. \
             Feature names: {:?}",
            fv.values.len(),
            fv.names
        );
    }

    /// Test 7: Router::route_sparql returns Ok with non-empty ranking.
    #[test]
    fn test_routing_with_ast_features() {
        let mut router = Router::new();
        router.add_source(DataSource::new("dbpedia", "https://dbpedia.org/sparql"));
        router.add_source(DataSource::new(
            "wikidata",
            "https://query.wikidata.org/sparql",
        ));
        let sparql = "SELECT ?s WHERE { ?s a <http://schema.org/Person> }";
        let ranking = router.route_sparql(sparql).expect("route_sparql failed");
        assert!(
            !ranking.sources.is_empty(),
            "route_sparql should return at least one source"
        );
    }

    /// Test 8: Query::parse (heuristic) → ast_features is None.
    #[test]
    fn test_ast_features_default_on_parse_query() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o }";
        let query = Query::parse(sparql).expect("parse failed");
        assert!(
            query.ast_features.is_none(),
            "heuristic parse should leave ast_features as None"
        );
    }

    /// Test 9: Literal in triple pattern → literal_count > 0.0
    #[test]
    fn test_ast_features_literal_count() {
        let sparql = r#"SELECT ?s WHERE { ?s <http://schema.org/name> "hello" }"#;
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.literal_count > 0.0,
            "expected literal_count > 0, got {}",
            af.literal_count
        );
    }

    /// Test 10: HAVING clause detected → has_having == 1.0
    #[test]
    fn test_ast_features_having() {
        let sparql = r#"
            SELECT ?s (COUNT(?o) AS ?count) WHERE { ?s ?p ?o }
            GROUP BY ?s HAVING (COUNT(?o) > 5)
        "#;
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert_eq!(
            af.has_having, 1.0,
            "expected has_having = 1.0, got {}",
            af.has_having
        );
    }

    /// Test 11: All AST feature values stay in [0.0, 1.0].
    #[test]
    fn test_ast_features_range() {
        let sparqls = [
            "SELECT ?s WHERE { ?s ?p ?o }",
            "SELECT DISTINCT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s a <http://T.org/> } FILTER(?p > 0) }",
            "SELECT ?s WHERE { { ?s <http://a/p> ?o } UNION { ?s <http://b/p> ?o } }",
        ];
        for sparql in &sparqls {
            let query = Query::from_sparql(sparql).expect("from_sparql failed");
            let af = query.ast_features.expect("ast_features should be Some");
            let fields = [
                af.join_depth,
                af.optional_count,
                af.filter_count,
                af.union_branch_count,
                af.has_distinct,
                af.has_having,
                af.subquery_count,
                af.path_expr_count,
                af.literal_count,
                af.blank_node_count,
            ];
            for &v in &fields {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "feature out of [0,1] range: {v} in query: {sparql}"
                );
            }
        }
    }

    /// Test 12: Nested OPTIONAL (depth ≥ 2) → join_depth > 0.0 and optional_count ≥ 2 dims.
    #[test]
    fn test_ast_features_nested_optional_depth() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s ?q ?r . OPTIONAL { ?r ?t ?u } } }";
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.join_depth > 0.0,
            "expected join_depth > 0 for nested OPTIONAL, got {}",
            af.join_depth
        );
        assert!(
            af.optional_count > 0.0,
            "expected optional_count > 0, got {}",
            af.optional_count
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Block H: Property-path integration tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Test 13 (Block H): OneOrMore path `foaf:knows+` → predicates contains
    /// the expanded `http://xmlns.com/foaf/0.1/knows`.
    #[test]
    fn test_property_path_vocab_decomposition() {
        let sparql = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?o WHERE { ?s foaf:knows+ ?o }
        "#;
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        assert!(
            query.predicates.contains("http://xmlns.com/foaf/0.1/knows"),
            "expected http://xmlns.com/foaf/0.1/knows in predicates, got {:?}",
            query.predicates
        );
    }

    /// Test 14 (Block H): Sequence `foaf:knows/foaf:name` → predicates contains
    /// both expanded IRIs.
    #[test]
    fn test_property_path_sequence_vocab() {
        let sparql = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?n WHERE { ?s foaf:knows/foaf:name ?n }
        "#;
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        assert!(
            query.predicates.contains("http://xmlns.com/foaf/0.1/knows"),
            "expected foaf:knows IRI in predicates, got {:?}",
            query.predicates
        );
        assert!(
            query.predicates.contains("http://xmlns.com/foaf/0.1/name"),
            "expected foaf:name IRI in predicates, got {:?}",
            query.predicates
        );
    }

    /// Test 15 (Block H): Query with 2 path triples and 1 plain triple →
    /// `path_expr_count` in AST features is greater than 0.0.
    #[test]
    fn test_path_expr_count_with_paths() {
        let sparql = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?o WHERE {
                ?s foaf:knows+ ?o .
                ?o foaf:knows/foaf:name ?n .
                ?s foaf:name ?label .
            }
        "#;
        let query = Query::from_sparql(sparql).expect("from_sparql failed");
        let af = query.ast_features.expect("ast_features should be Some");
        assert!(
            af.path_expr_count > 0.0,
            "expected path_expr_count > 0 for query with path triples, got {}",
            af.path_expr_count
        );
    }
}
