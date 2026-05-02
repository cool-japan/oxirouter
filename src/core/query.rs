//! SPARQL query parsing and feature extraction

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::error::{OxiRouterError, Result};
use super::term::StructuredTriple;

/// SPARQL query type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// SELECT query
    Select,
    /// CONSTRUCT query
    Construct,
    /// ASK query
    Ask,
    /// DESCRIBE query
    Describe,
}

impl Default for QueryType {
    fn default() -> Self {
        Self::Select
    }
}

/// A parsed SPARQL query with extracted features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Original query string
    pub raw: String,
    /// Query type (SELECT, CONSTRUCT, ASK, DESCRIBE)
    pub query_type: QueryType,
    /// Extracted triple patterns
    pub triple_patterns: Vec<TriplePattern>,
    /// Predicates used in the query (URIs)
    pub predicates: HashSet<String>,
    /// Types/classes referenced (rdf:type objects)
    pub types: HashSet<String>,
    /// Named graphs referenced
    pub graphs: SmallVec<[String; 2]>,
    /// Whether the query uses OPTIONAL
    pub has_optional: bool,
    /// Whether the query uses UNION
    pub has_union: bool,
    /// Whether the query uses FILTER
    pub has_filter: bool,
    /// Whether the query uses aggregation
    pub has_aggregation: bool,
    /// Whether the query uses property paths
    pub has_property_paths: bool,
    /// Whether the query uses subqueries
    pub has_subquery: bool,
    /// Whether the query uses SERVICE (federation)
    pub has_service: bool,
    /// LIMIT value (if specified)
    pub limit: Option<u32>,
    /// OFFSET value (if specified)
    pub offset: Option<u32>,
    /// Complexity score (0.0 - 1.0)
    pub complexity: f32,
    /// Variables named in the SELECT projection (e.g. `["a", "b"]` for `SELECT ?a ?b`).
    /// `["*"]` for `SELECT *`. Empty for ASK, CONSTRUCT, DESCRIBE, or when constructed
    /// via the heuristic [`Query::parse`] (kept for backwards compatibility).
    #[serde(default)]
    pub projection_vars: SmallVec<[String; 4]>,
    /// SPARQL AST-derived features extracted when `sparql` feature is enabled.
    /// Populated by [`Query::from_sparql`]. `None` when using heuristic [`Query::parse`].
    #[cfg(feature = "sparql")]
    #[serde(default)]
    pub ast_features: Option<crate::core::sparql_ast::SparqlAstFeatures>,
    /// Per-triple structured terms from BGP triples (populated by [`Query::from_sparql`]).
    /// Empty when using heuristic [`Query::parse`].
    #[serde(default)]
    pub structured_triples: Vec<StructuredTriple>,
    /// Prefix declarations from the SPARQL source (populated by [`Query::from_sparql`]).
    /// Keys are bare prefix labels (e.g., `"foaf"`), values are namespace IRIs.
    /// Empty when using heuristic [`Query::parse`].
    #[serde(default)]
    pub prefixes: HashMap<String, String>,
}

impl Query {
    /// Parse a SPARQL query string
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be parsed
    pub fn parse(query: &str) -> Result<Self> {
        let raw = query.to_string();
        let query_upper = query.to_uppercase();

        // Determine query type
        let query_type = if query_upper.contains("SELECT") {
            QueryType::Select
        } else if query_upper.contains("CONSTRUCT") {
            QueryType::Construct
        } else if query_upper.contains("ASK") {
            QueryType::Ask
        } else if query_upper.contains("DESCRIBE") {
            QueryType::Describe
        } else {
            return Err(OxiRouterError::QueryParse("Unknown query type".to_string()));
        };

        // Extract predicates and types
        let mut predicates = HashSet::new();
        let mut types = HashSet::new();
        let mut graphs = SmallVec::new();

        // Simple regex-free parsing for predicates (URIs in angle brackets)
        let mut in_uri = false;
        let mut uri_start = 0;
        let chars: Vec<char> = query.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            if c == '<' && !in_uri {
                in_uri = true;
                uri_start = i + 1;
            } else if c == '>' && in_uri {
                in_uri = false;
                let uri: String = chars[uri_start..i].iter().collect();
                if !uri.is_empty() && uri.contains("://") {
                    predicates.insert(uri.clone());

                    // Check for rdf:type patterns
                    if uri.ends_with("#type") || uri.ends_with("/type") {
                        // Look ahead for the type object
                        if let Some(type_uri) = Self::extract_next_uri(&chars[i + 1..]) {
                            types.insert(type_uri);
                        }
                    }
                }
            }
        }

        // Extract prefixed names (e.g., schema:Person)
        Self::extract_prefixed_names(query, &mut predicates, &mut types);

        // Extract triple patterns (simplified)
        let triple_patterns = Self::extract_triple_patterns(query);

        // Extract named graphs
        Self::extract_graphs(query, &mut graphs);

        // Check for features
        let has_optional = query_upper.contains("OPTIONAL");
        let has_union = query_upper.contains("UNION");
        let has_filter = query_upper.contains("FILTER");
        let has_aggregation = query_upper.contains("GROUP BY")
            || query_upper.contains("COUNT(")
            || query_upper.contains("SUM(")
            || query_upper.contains("AVG(")
            || query_upper.contains("MIN(")
            || query_upper.contains("MAX(");
        let has_property_paths = query.contains('/')
            && (query.contains("*/") || query.contains("+/") || query.contains("?/"))
            || query.contains('^');
        let has_subquery = query_upper.matches("SELECT").count() > 1;
        let has_service = query_upper.contains("SERVICE");

        // Extract LIMIT
        let limit = Self::extract_clause_value(&query_upper, "LIMIT");
        let offset = Self::extract_clause_value(&query_upper, "OFFSET");

        // Calculate complexity
        let complexity = Self::calculate_complexity(
            triple_patterns.len(),
            has_optional,
            has_union,
            has_filter,
            has_aggregation,
            has_property_paths,
            has_subquery,
        );

        Ok(Self {
            raw,
            query_type,
            triple_patterns,
            predicates,
            types,
            graphs,
            has_optional,
            has_union,
            has_filter,
            has_aggregation,
            has_property_paths,
            has_subquery,
            has_service,
            limit,
            offset,
            complexity,
            projection_vars: SmallVec::new(),
            #[cfg(feature = "sparql")]
            ast_features: None,
            structured_triples: Vec::new(),
            prefixes: HashMap::new(),
        })
    }

    /// Extract the next URI from a character slice
    fn extract_next_uri(chars: &[char]) -> Option<String> {
        let mut in_uri = false;
        let mut uri_start = 0;

        for (i, &c) in chars.iter().enumerate() {
            if c == '<' && !in_uri {
                in_uri = true;
                uri_start = i + 1;
            } else if c == '>' && in_uri {
                let uri: String = chars[uri_start..i].iter().collect();
                if !uri.is_empty() && uri.contains("://") {
                    return Some(uri);
                }
                break;
            }
        }
        None
    }

    /// Extract prefixed names from query
    fn extract_prefixed_names(
        query: &str,
        predicates: &mut HashSet<String>,
        types: &mut HashSet<String>,
    ) {
        // Common prefixes (simplified mapping)
        let common_prefixes = [
            ("rdf:", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("rdfs:", "http://www.w3.org/2000/01/rdf-schema#"),
            ("owl:", "http://www.w3.org/2002/07/owl#"),
            ("xsd:", "http://www.w3.org/2001/XMLSchema#"),
            ("schema:", "http://schema.org/"),
            ("foaf:", "http://xmlns.com/foaf/0.1/"),
            ("dc:", "http://purl.org/dc/elements/1.1/"),
            ("dct:", "http://purl.org/dc/terms/"),
            ("skos:", "http://www.w3.org/2004/02/skos/core#"),
            ("dbo:", "http://dbpedia.org/ontology/"),
            ("dbr:", "http://dbpedia.org/resource/"),
            ("wdt:", "http://www.wikidata.org/prop/direct/"),
            ("wd:", "http://www.wikidata.org/entity/"),
        ];

        for (prefix, base) in &common_prefixes {
            let prefix_len = prefix.len();
            let mut search_start = 0;

            while let Some(pos) = query[search_start..].find(prefix) {
                let abs_pos = search_start + pos + prefix_len;
                if abs_pos < query.len() {
                    // Extract local name
                    let remaining = &query[abs_pos..];
                    let local_name: String = remaining
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                        .collect();

                    if !local_name.is_empty() {
                        let full_uri = format!("{base}{local_name}");
                        predicates.insert(full_uri.clone());

                        // Check if this is a type declaration
                        if *prefix == "rdf:" && local_name == "type" {
                            // Mark that we found a type predicate
                        } else if query[search_start..abs_pos].contains("rdf:type")
                            || query[search_start..abs_pos].contains("a ")
                        {
                            types.insert(full_uri);
                        }
                    }
                }
                search_start = abs_pos;
            }
        }
    }

    /// Extract triple patterns from query
    fn extract_triple_patterns(query: &str) -> Vec<TriplePattern> {
        let mut patterns = Vec::new();

        // Find WHERE clause content
        let query_upper = query.to_uppercase();
        if let Some(where_pos) = query_upper.find("WHERE") {
            let after_where = &query[where_pos + 5..];

            // Find opening brace
            if let Some(brace_pos) = after_where.find('{') {
                let pattern_section = &after_where[brace_pos + 1..];

                // Split by periods and semicolons (simplified)
                for segment in pattern_section.split(['.', ';']) {
                    let trimmed = segment.trim();
                    if trimmed.is_empty() || trimmed.starts_with('}') {
                        continue;
                    }

                    // Count tokens (very simplified)
                    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                    if tokens.len() >= 3 {
                        let subject = Self::token_type(tokens[0]);
                        let predicate = Self::token_type(tokens[1]);
                        let object = Self::token_type(tokens[2]);

                        patterns.push(TriplePattern {
                            subject,
                            predicate,
                            object,
                        });
                    }
                }
            }
        }

        patterns
    }

    /// Determine the type of a SPARQL token
    fn token_type(token: &str) -> TermType {
        if token.starts_with('?') || token.starts_with('$') {
            TermType::Variable
        } else if token.starts_with('<') && token.ends_with('>') {
            TermType::Uri
        } else if token.starts_with('"') || token.starts_with('\'') {
            TermType::Literal
        } else if token.contains(':') {
            TermType::PrefixedName
        } else if token == "a" {
            TermType::Uri // rdf:type shorthand
        } else {
            TermType::Unknown
        }
    }

    /// Extract named graphs from query
    fn extract_graphs(query: &str, graphs: &mut SmallVec<[String; 2]>) {
        let query_upper = query.to_uppercase();
        let mut search_start = 0;

        while let Some(pos) = query_upper[search_start..].find("GRAPH") {
            let abs_pos = search_start + pos + 5;
            if abs_pos < query.len() {
                let remaining = &query[abs_pos..];
                // Look for URI after GRAPH keyword
                if let Some(uri_start) = remaining.find('<') {
                    if let Some(uri_end) = remaining[uri_start..].find('>') {
                        let uri = remaining[uri_start + 1..uri_start + uri_end].to_string();
                        if !uri.is_empty() {
                            graphs.push(uri);
                        }
                    }
                }
            }
            search_start = abs_pos;
        }
    }

    /// Extract numeric value from a clause (e.g., LIMIT 100)
    fn extract_clause_value(query: &str, clause: &str) -> Option<u32> {
        if let Some(pos) = query.find(clause) {
            let after = &query[pos + clause.len()..];
            let value_str: String = after
                .trim()
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            value_str.parse().ok()
        } else {
            None
        }
    }

    /// Calculate query complexity score
    #[allow(clippy::fn_params_excessive_bools)]
    fn calculate_complexity(
        pattern_count: usize,
        has_optional: bool,
        has_union: bool,
        has_filter: bool,
        has_aggregation: bool,
        has_property_paths: bool,
        has_subquery: bool,
    ) -> f32 {
        let mut score = 0.0;

        // Base complexity from pattern count (0-0.3)
        score += (pattern_count as f32 / 20.0).min(0.3);

        // Feature contributions
        if has_optional {
            score += 0.1;
        }
        if has_union {
            score += 0.15;
        }
        if has_filter {
            score += 0.1;
        }
        if has_aggregation {
            score += 0.15;
        }
        if has_property_paths {
            score += 0.1;
        }
        if has_subquery {
            score += 0.2;
        }

        score.min(1.0)
    }

    /// Get vocabulary namespaces used in this query
    #[must_use]
    pub fn vocabularies(&self) -> HashSet<String> {
        let mut vocabs = HashSet::new();

        for predicate in &self.predicates {
            // Extract namespace (everything before the last # or /)
            if let Some(pos) = predicate.rfind('#') {
                vocabs.insert(predicate[..=pos].to_string());
            } else if let Some(pos) = predicate.rfind('/') {
                vocabs.insert(predicate[..=pos].to_string());
            }
        }

        vocabs
    }

    /// Hash the predicates for feature extraction (FNV-1a)
    #[must_use]
    pub fn predicate_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis

        for predicate in &self.predicates {
            for byte in predicate.bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3); // FNV prime
            }
        }

        hash
    }

    /// Check if query requires a specific capability
    #[must_use]
    pub const fn requires_sparql_1_1(&self) -> bool {
        self.has_aggregation || self.has_property_paths || self.has_subquery || self.has_service
    }
}

/// A triple pattern in a SPARQL query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriplePattern {
    /// Subject term type
    pub subject: TermType,
    /// Predicate term type
    pub predicate: TermType,
    /// Object term type
    pub object: TermType,
}

impl TriplePattern {
    /// Check if all terms are bound (no variables)
    #[must_use]
    pub const fn is_fully_bound(&self) -> bool {
        !matches!(self.subject, TermType::Variable)
            && !matches!(self.predicate, TermType::Variable)
            && !matches!(self.object, TermType::Variable)
    }

    /// Count the number of variables
    #[must_use]
    pub const fn variable_count(&self) -> u8 {
        let mut count = 0;
        if matches!(self.subject, TermType::Variable) {
            count += 1;
        }
        if matches!(self.predicate, TermType::Variable) {
            count += 1;
        }
        if matches!(self.object, TermType::Variable) {
            count += 1;
        }
        count
    }

    /// Get the selectivity estimate (lower = more selective)
    #[must_use]
    pub const fn selectivity(&self) -> f32 {
        match self.variable_count() {
            0 => 0.001, // Fully bound - very selective
            1 => 0.01,  // One variable - selective
            2 => 0.1,   // Two variables - less selective
            3 => 1.0,   // All variables - not selective
            _ => 1.0,
        }
    }
}

/// Type of a term in a triple pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TermType {
    /// Variable (?x or $x)
    Variable,
    /// Full URI (<http://...>)
    Uri,
    /// Prefixed name (prefix:local)
    PrefixedName,
    /// Literal ("value" or "value"@lang or "value"^^type)
    Literal,
    /// Blank node (_:b0)
    BlankNode,
    /// Unknown term type
    Unknown,
}

#[cfg(feature = "sparql")]
impl Query {
    /// Parse a SPARQL query string with real prefix expansion and SELECT-variable extraction.
    ///
    /// Calls [`Query::parse`] for the structural heuristics (query type, triple patterns,
    /// boolean flags, LIMIT/OFFSET, complexity), then overlays accurate prefix-expanded
    /// predicate and type URIs using `extract_prefix_map`, and
    /// populates [`Query::projection_vars`] via `extract_select_variables`.
    ///
    /// Also populates [`Query::structured_triples`] (structured BGP triples with actual
    /// `Term` values) and [`Query::prefixes`] (prefix declarations with bare label keys).
    ///
    /// Prefer this over [`Query::parse`] when the `sparql` feature is enabled and the
    /// input is a real SPARQL query with `PREFIX` declarations or complex projections.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying heuristic parser fails.
    pub fn from_sparql(sparql: &str) -> crate::core::error::Result<Self> {
        let mut q = Self::parse(sparql)?;
        let prefix_map = crate::core::sparql::extract_prefix_map(sparql);
        if !prefix_map.is_empty() {
            // Expand prefixed names already in predicates/types
            q.predicates = q
                .predicates
                .iter()
                .map(|p| {
                    crate::core::sparql::expand_prefixed_name(p, &prefix_map)
                        .unwrap_or_else(|| p.clone())
                })
                .collect();
            q.types = q
                .types
                .iter()
                .map(|t| {
                    crate::core::sparql::expand_prefixed_name(t, &prefix_map)
                        .unwrap_or_else(|| t.clone())
                })
                .collect();
            // Also extract and insert any prefixed names that heuristic parse missed
            for uri in crate::core::sparql::extract_expanded_prefixed_uris(sparql, &prefix_map) {
                q.predicates.insert(uri);
            }
        }

        // Extract base IRIs from property-path expressions.
        //
        // Strategy: scan whitespace-separated tokens in the WHERE body.  Any
        // token that contains a path operator (`* + ? ^ / | ! ( )`) is
        // treated as a property-path expression and decomposed via
        // `parse_property_path`.  The raw leaf IRIs are then expanded using
        // the prefix map and inserted into `q.predicates`.
        //
        // This runs regardless of whether a prefix map was found — full-IRI
        // paths like `<http://ex.org/p>+` are handled correctly too.
        Self::extract_path_predicates(sparql, &prefix_map, &mut q.predicates);

        q.projection_vars = crate::core::sparql::extract_select_variables(sparql)
            .into_iter()
            .collect();
        q.ast_features = Some(crate::core::sparql_ast::compute_ast_features(sparql));

        // Populate bare-label prefix map (keys without trailing colon).
        // `extract_prefix_map` returns "foaf:" → IRI; we strip the trailing ':'
        // so that `Term::resolve` can look up "foaf" directly.
        q.prefixes = prefix_map
            .iter()
            .map(|(k, v)| {
                let bare = k.trim_end_matches(':').to_string();
                (bare, v.clone())
            })
            .collect();

        // Populate structured triples from the BGP body (delegated to term module).
        q.structured_triples = crate::core::term::extract_structured_triples(sparql);

        // For property-path predicates, augment structured_triples with leaf
        // IRI entries (same subject/object, leaf as predicate).
        crate::core::term::augment_path_structured_triples(sparql, &mut q.structured_triples);

        Ok(q)
    }

    /// Scan the WHERE body of `sparql` for property-path expression tokens,
    /// decompose each one into base IRIs, expand them, and insert into `dest`.
    fn extract_path_predicates(
        sparql: &str,
        prefix_map: &hashbrown::HashMap<String, String>,
        dest: &mut HashSet<String>,
    ) {
        // Delegate WHERE-body extraction to term module to avoid duplication.
        let where_body = crate::core::term::find_where_body(sparql).unwrap_or(sparql);

        for token in where_body.split_ascii_whitespace() {
            if crate::core::term::predicate_token_has_path_operator(token) {
                let path = crate::core::sparql_ast::parse_property_path(token);
                for raw_iri in path.base_iris() {
                    let expanded = crate::core::sparql::expand_prefixed_name(&raw_iri, prefix_map)
                        .unwrap_or_else(|| {
                            if raw_iri.contains("://") {
                                raw_iri.clone()
                            } else {
                                String::new()
                            }
                        });
                    if !expanded.is_empty() {
                        dest.insert(expanded);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_query() {
        let query = Query::parse(
            r"
            PREFIX schema: <http://schema.org/>
            SELECT ?name WHERE {
                ?s a schema:Person .
                ?s schema:name ?name .
            }
            ",
        )
        .unwrap();

        assert_eq!(query.query_type, QueryType::Select);
        assert!(!query.predicates.is_empty());
        assert!(query.complexity > 0.0);
    }

    #[test]
    fn test_parse_construct_query() {
        let query = Query::parse(
            r"
            CONSTRUCT { ?s ?p ?o }
            WHERE { ?s ?p ?o }
            LIMIT 100
            ",
        )
        .unwrap();

        assert_eq!(query.query_type, QueryType::Construct);
        assert_eq!(query.limit, Some(100));
    }

    #[test]
    fn test_complexity_calculation() {
        let simple = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let complex = Query::parse(
            r"
            SELECT ?s (COUNT(?o) as ?count) WHERE {
                { ?s ?p ?o } UNION { ?s ?p2 ?o2 }
                OPTIONAL { ?s ?p3 ?o3 }
                FILTER(?count > 5)
            }
            GROUP BY ?s
            ",
        )
        .unwrap();

        assert!(complex.complexity > simple.complexity);
    }

    #[test]
    fn test_predicate_hash() {
        let query1 = Query::parse("SELECT ?s WHERE { ?s <http://a.com/p> ?o }").unwrap();
        let query2 = Query::parse("SELECT ?x WHERE { ?x <http://a.com/p> ?y }").unwrap();
        let query3 = Query::parse("SELECT ?s WHERE { ?s <http://b.com/p> ?o }").unwrap();

        // Same predicate should have same hash
        assert_eq!(query1.predicate_hash(), query2.predicate_hash());
        // Different predicate should have different hash
        assert_ne!(query1.predicate_hash(), query3.predicate_hash());
    }
}
