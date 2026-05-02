//! Structural SPARQL AST parser for ML feature extraction.
//!
//! Provides a best-effort, infallible parser that extracts structural
//! characteristics of a SPARQL query (nesting depth, OPTIONAL/FILTER/UNION
//! counts, etc.) for use as ML feature dimensions.
//!
//! The parser never returns errors — it returns a partial or empty AST on
//! any parse difficulty.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

use serde::{Deserialize, Serialize};

mod parser;
mod property_path;
mod scanner;

pub(crate) use parser::compute_ast_features;
pub(crate) use property_path::parse_property_path;

// ─────────────────────────────────────────────────────────────────────────────
// AST Types
// ─────────────────────────────────────────────────────────────────────────────

/// 10 SPARQL AST-derived features for ML routing.
///
/// All fields are normalized to `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SparqlAstFeatures {
    /// Normalized max nesting depth of BGP patterns (0.0–1.0, capped at depth 10).
    pub join_depth: f32,
    /// Normalized count of OPTIONAL clauses (count / 10.0, capped at 1.0).
    pub optional_count: f32,
    /// Normalized count of FILTER expressions (count / 10.0).
    pub filter_count: f32,
    /// Normalized total UNION branches (total branches / 10.0).
    pub union_branch_count: f32,
    /// 1.0 if SELECT DISTINCT or SELECT REDUCED, 0.0 otherwise.
    pub has_distinct: f32,
    /// 1.0 if HAVING clause present, 0.0 otherwise.
    pub has_having: f32,
    /// Normalized count of sub-SELECT queries (count / 5.0).
    pub subquery_count: f32,
    /// Normalized count of property path expressions (count / 10.0).
    pub path_expr_count: f32,
    /// Normalized count of literal objects in triple patterns (count / 20.0).
    pub literal_count: f32,
    /// Normalized count of blank node occurrences (count / 10.0).
    pub blank_node_count: f32,
}

impl SparqlAstFeatures {
    /// Clamp all fields to `[0.0, 1.0]`.
    #[must_use]
    pub fn clamp(self) -> Self {
        Self {
            join_depth: self.join_depth.clamp(0.0_f32, 1.0_f32),
            optional_count: self.optional_count.clamp(0.0_f32, 1.0_f32),
            filter_count: self.filter_count.clamp(0.0_f32, 1.0_f32),
            union_branch_count: self.union_branch_count.clamp(0.0_f32, 1.0_f32),
            has_distinct: self.has_distinct.clamp(0.0_f32, 1.0_f32),
            has_having: self.has_having.clamp(0.0_f32, 1.0_f32),
            subquery_count: self.subquery_count.clamp(0.0_f32, 1.0_f32),
            path_expr_count: self.path_expr_count.clamp(0.0_f32, 1.0_f32),
            literal_count: self.literal_count.clamp(0.0_f32, 1.0_f32),
            blank_node_count: self.blank_node_count.clamp(0.0_f32, 1.0_f32),
        }
    }
}

/// A simplified SPARQL graph pattern (best-effort, not full spec conformance).
#[derive(Debug, Clone)]
pub(crate) enum GraphPattern {
    /// Basic Graph Pattern with triple/literal/blank counts.
    Bgp {
        triples: u32,
        literals: u32,
        blank_nodes: u32,
    },
    /// OPTIONAL { ... }
    Optional(Vec<GraphPattern>),
    /// { ... } UNION { ... } — each branch is a Vec<GraphPattern>
    Union(Vec<Vec<GraphPattern>>),
    /// FILTER (...)
    Filter,
    /// GROUP BY clause marker
    GroupBy,
    /// HAVING clause marker
    Having,
    /// Nested SELECT subquery
    Subquery(Box<SparqlAst>),
    /// SERVICE <iri> { ... }
    Service(Vec<GraphPattern>),
    /// BIND (expr AS ?var)
    Bind,
    /// VALUES clause
    Values,
}

/// Best-effort structural SPARQL AST.
#[derive(Debug, Clone, Default)]
pub(crate) struct SparqlAst {
    /// True if SELECT DISTINCT or SELECT REDUCED was found.
    pub has_distinct: bool,
    /// True if SELECT REDUCED was found.
    pub has_reduced: bool,
    /// Top-level graph patterns.
    pub patterns: Vec<GraphPattern>,
    /// True if a top-level HAVING clause was detected.
    pub has_having: bool,
    /// Count of GROUP BY occurrences.
    pub group_by_count: u32,
    /// Count of ORDER BY occurrences.
    pub order_by_count: u32,
    /// True if LIMIT was detected.
    pub has_limit: bool,
    /// Count of detected property path expressions.
    pub path_count: u32,
}

#[cfg(test)]
mod tests {
    use super::property_path::parse_property_path;
    use super::*;

    /// Test 1: Empty/trivial query → all features near zero.
    #[test]
    fn test_empty_query_features() {
        let features = compute_ast_features("");
        assert_eq!(features.optional_count, 0.0);
        assert_eq!(features.filter_count, 0.0);
        assert_eq!(features.union_branch_count, 0.0);
        assert_eq!(features.has_distinct, 0.0);
        assert_eq!(features.has_having, 0.0);
        assert_eq!(features.subquery_count, 0.0);
    }

    /// Test 2: OPTIONAL detected.
    #[test]
    fn test_optional_detected() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s a <http://example.org/T> } }";
        let features = compute_ast_features(sparql);
        assert!(
            features.optional_count > 0.0,
            "expected optional_count > 0, got {}",
            features.optional_count
        );
    }

    /// Test 3: FILTER detected.
    #[test]
    fn test_filter_detected() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o . FILTER(?o > 5) }";
        let features = compute_ast_features(sparql);
        assert!(
            features.filter_count > 0.0,
            "expected filter_count > 0, got {}",
            features.filter_count
        );
    }

    /// Test 4: UNION detected.
    #[test]
    fn test_union_detected() {
        let sparql =
            "SELECT ?s WHERE { { ?s <http://a.org/p> ?o } UNION { ?s <http://b.org/p> ?o } }";
        let features = compute_ast_features(sparql);
        assert!(
            features.union_branch_count > 0.0,
            "expected union_branch_count > 0, got {}",
            features.union_branch_count
        );
    }

    /// Test 5: SELECT DISTINCT → has_distinct == 1.0.
    #[test]
    fn test_distinct_detected() {
        let sparql = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
        let features = compute_ast_features(sparql);
        assert_eq!(
            features.has_distinct, 1.0,
            "expected has_distinct = 1.0, got {}",
            features.has_distinct
        );
    }

    /// Test 6: Nested OPTIONAL → join_depth > 0.0.
    #[test]
    fn test_nested_optional_depth() {
        let sparql = "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s ?q ?r . OPTIONAL { ?r ?t ?u } } }";
        let features = compute_ast_features(sparql);
        assert!(
            features.join_depth > 0.0,
            "expected join_depth > 0, got {}",
            features.join_depth
        );
        assert!(
            features.optional_count > 0.0,
            "expected optional_count > 0, got {}",
            features.optional_count
        );
    }

    /// Test 7: Subquery detected.
    #[test]
    fn test_subquery_detected() {
        let sparql = r#"
            SELECT ?s ?count WHERE {
                ?s ?p ?o .
                { SELECT ?s (COUNT(?o) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?s }
            }
        "#;
        let features = compute_ast_features(sparql);
        assert!(
            features.subquery_count > 0.0,
            "expected subquery_count > 0, got {}",
            features.subquery_count
        );
    }

    /// Test 8: Literal in triple pattern → literal_count > 0.0.
    #[test]
    fn test_literal_count() {
        let sparql = r#"SELECT ?s WHERE { ?s <http://schema.org/name> "hello" }"#;
        let features = compute_ast_features(sparql);
        assert!(
            features.literal_count > 0.0,
            "expected literal_count > 0, got {}",
            features.literal_count
        );
    }

    /// Test 9: HAVING detected.
    #[test]
    fn test_having_detected() {
        let sparql = r#"
            SELECT ?s (COUNT(?o) AS ?count) WHERE { ?s ?p ?o }
            GROUP BY ?s HAVING (COUNT(?o) > 5)
        "#;
        let features = compute_ast_features(sparql);
        assert_eq!(
            features.has_having, 1.0,
            "expected has_having = 1.0, got {}",
            features.has_having
        );
    }

    /// Test 10: All features clamped to [0.0, 1.0].
    #[test]
    fn test_clamp() {
        let f = SparqlAstFeatures {
            join_depth: 2.0,
            optional_count: -1.0,
            filter_count: 0.5,
            union_branch_count: 1.5,
            has_distinct: 0.0,
            has_having: 1.0,
            subquery_count: 0.0,
            path_expr_count: 0.3,
            literal_count: 0.9,
            blank_node_count: -0.1,
        };
        let clamped = f.clamp();
        assert_eq!(clamped.join_depth, 1.0);
        assert_eq!(clamped.optional_count, 0.0);
        assert_eq!(clamped.union_branch_count, 1.0);
        assert_eq!(clamped.blank_node_count, 0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PropertyPath unit tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Test PP-1: Inverse path `^foaf:knows` → base_iris contains `foaf:knows`.
    #[test]
    fn test_path_inverse_base_iris() {
        let path = parse_property_path("^foaf:knows");
        let iris = path.base_iris();
        assert!(
            iris.contains(&"foaf:knows".to_owned()),
            "expected 'foaf:knows' in {:?}",
            iris
        );
    }

    /// Test PP-2: Sequence `foaf:knows/foaf:name` → base_iris contains both.
    #[test]
    fn test_path_sequence_base_iris() {
        let path = parse_property_path("foaf:knows/foaf:name");
        let iris = path.base_iris();
        assert!(
            iris.contains(&"foaf:knows".to_owned()) && iris.contains(&"foaf:name".to_owned()),
            "expected both foaf:knows and foaf:name in {:?}",
            iris
        );
    }

    /// Test PP-3: Alternative `foaf:knows|foaf:friend` → base_iris contains both.
    #[test]
    fn test_path_alternative_base_iris() {
        let path = parse_property_path("foaf:knows|foaf:friend");
        let iris = path.base_iris();
        assert!(
            iris.contains(&"foaf:knows".to_owned()) && iris.contains(&"foaf:friend".to_owned()),
            "expected both foaf:knows and foaf:friend in {:?}",
            iris
        );
    }

    /// Test PP-4: ZeroOrMore `foaf:knows*` → base_iris contains `foaf:knows`.
    #[test]
    fn test_path_zero_or_more_base_iris() {
        let path = parse_property_path("foaf:knows*");
        let iris = path.base_iris();
        assert!(
            iris.contains(&"foaf:knows".to_owned()),
            "expected 'foaf:knows' in {:?}",
            iris
        );
    }

    /// Test PP-5: Nested `(foaf:knows/foaf:name)+` → base_iris contains both.
    #[test]
    fn test_path_nested_quantifier_base_iris() {
        let path = parse_property_path("(foaf:knows/foaf:name)+");
        let iris = path.base_iris();
        assert!(
            iris.contains(&"foaf:knows".to_owned()) && iris.contains(&"foaf:name".to_owned()),
            "expected both foaf:knows and foaf:name in {:?}",
            iris
        );
    }

    /// Test PP-6: Negated property set `!(foaf:knows|rdfs:type)` → base_iris contains both.
    #[test]
    fn test_path_negated_set_base_iris() {
        let path = parse_property_path("!(foaf:knows|rdfs:type)");
        let iris = path.base_iris();
        assert!(
            iris.contains(&"foaf:knows".to_owned()) && iris.contains(&"rdfs:type".to_owned()),
            "expected both foaf:knows and rdfs:type in {:?}",
            iris
        );
    }
}
