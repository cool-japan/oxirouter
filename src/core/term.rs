//! SPARQL term types for structured triple representation.
//!
//! Provides [`Term`] (a SPARQL term with resolved values) and [`StructuredTriple`]
//! (a Basic Graph Pattern triple with actual term values). These types are the
//! foundation for BGP decomposition in Block Z.

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
};

#[cfg(all(feature = "alloc", feature = "sparql"))]
use alloc::vec::Vec;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// A SPARQL term with resolved values.
///
/// Unlike [`crate::core::query::TermType`] which only carries the kind, this
/// enum carries the actual string content of the term.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    /// Variable name without the leading `?`.
    Variable(String),
    /// Fully-qualified IRI.
    Iri(String),
    /// Prefixed name: (prefix_label, local_part).
    PrefixedName(String, String),
    /// Literal lexical form, raw (including `"..."@en` or `"..."^^xsd:integer`).
    Literal(String),
    /// Blank node label without the leading `_:`.
    BlankNode(String),
}

impl Term {
    /// Resolve `PrefixedName(p, l)` → `Iri(prefix_map[p] + l)` if `p` is in
    /// the map; otherwise return `self` unchanged.
    ///
    /// Note: `prefix_map` keys are bare prefix labels (e.g., `"foaf"`), not
    /// the colon-terminated form.
    pub fn resolve(&self, prefix_map: &HashMap<String, String>) -> Self {
        match self {
            Term::PrefixedName(prefix, local) => {
                if let Some(base) = prefix_map.get(prefix.as_str()) {
                    Term::Iri(format!("{}{}", base, local))
                } else {
                    self.clone()
                }
            }
            _ => self.clone(),
        }
    }

    /// Parse a raw SPARQL token into a [`Term`].
    ///
    /// - `?name` or `$name` → `Variable("name")`
    /// - `<iri>` → `Iri("iri")`
    /// - `_:label` → `BlankNode("label")`
    /// - `"..."` / `'...'` or starting with a digit / sign → `Literal(raw)`
    /// - `prefix:local` → `PrefixedName("prefix", "local")`
    /// - `a` (rdf:type shorthand) → `PrefixedName("rdf", "type")`
    /// - anything else → `Iri(token)` (best-effort)
    #[must_use]
    pub fn from_token(token: &str) -> Self {
        let t = token.trim();
        if t.is_empty() {
            return Term::Iri(String::new());
        }
        // Variable
        if t.starts_with('?') || t.starts_with('$') {
            return Term::Variable(t[1..].to_string());
        }
        // Full IRI
        if t.starts_with('<') && t.ends_with('>') {
            return Term::Iri(t[1..t.len() - 1].to_string());
        }
        // Blank node
        if let Some(label) = t.strip_prefix("_:") {
            return Term::BlankNode(label.to_string());
        }
        // Literal (double-quoted, single-quoted, or numeric)
        if t.starts_with('"') || t.starts_with('\'') {
            return Term::Literal(t.to_string());
        }
        // Numeric literals
        if t.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '+') {
            return Term::Literal(t.to_string());
        }
        // rdf:type shorthand `a`
        if t == "a" {
            return Term::PrefixedName("rdf".to_string(), "type".to_string());
        }
        // Prefixed name  prefix:local
        if let Some(colon_pos) = t.find(':') {
            // Exclude tokens that are full IRIs (contain "://")
            if !t.contains("://") {
                let prefix = t[..colon_pos].to_string();
                let local = t[colon_pos + 1..].to_string();
                return Term::PrefixedName(prefix, local);
            }
        }
        // Full IRI without angle brackets (fallback)
        Term::Iri(t.to_string())
    }
}

/// A SPARQL Basic Graph Pattern triple with resolved term values.
///
/// Each field carries a [`Term`] rather than just a [`crate::core::query::TermType`],
/// enabling BGP-level query decomposition in Block Z.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StructuredTriple {
    /// Subject term.
    pub subject: Term,
    /// Predicate term.
    pub predicate: Term,
    /// Object term.
    pub object: Term,
}

// ─────────────────────────────────────────────────────────────────────────────
// BGP extraction helpers (sparql feature only)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract structured triples from the WHERE body of a SPARQL query.
///
/// Produces one [`StructuredTriple`] per token-triple found in the WHERE
/// clause BGP (best-effort). Segments whose predicate token contains a path
/// operator are skipped — use [`augment_path_structured_triples`] to handle
/// those separately.
#[cfg(feature = "sparql")]
pub(crate) fn extract_structured_triples(sparql: &str) -> Vec<StructuredTriple> {
    let mut triples = Vec::new();
    let where_body = match find_where_body(sparql) {
        Some(body) => body,
        None => return triples,
    };

    for segment in where_body.split(['.', ';']) {
        let trimmed = segment.trim();
        if trimmed.is_empty() || trimmed.starts_with('}') || trimmed.starts_with('{') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < 3 {
            continue;
        }

        // Skip SPARQL keyword segments
        if is_sparql_keyword(tokens[0]) || is_sparql_keyword(tokens[1]) {
            continue;
        }

        let pred_token = tokens[1];

        // Skip path-operator predicates here; leaf extraction is done separately.
        if predicate_token_has_path_operator(pred_token) {
            continue;
        }

        let subject = Term::from_token(tokens[0]);
        let predicate = Term::from_token(pred_token);
        // Object may span several tokens for literals with spaces; join the rest.
        let object_raw = tokens[2..].join(" ");
        let object = Term::from_token(&object_raw);

        triples.push(StructuredTriple {
            subject,
            predicate,
            object,
        });
    }

    triples
}

/// Augment `dest` with additional [`StructuredTriple`] entries for each leaf
/// IRI extracted from property-path predicates in the WHERE body.
///
/// For each segment whose predicate token contains a path operator, the leaf
/// IRIs are parsed from the path expression and each leaf yields a new
/// `StructuredTriple` preserving the same subject and object.
#[cfg(feature = "sparql")]
pub(crate) fn augment_path_structured_triples(sparql: &str, dest: &mut Vec<StructuredTriple>) {
    let where_body = match find_where_body(sparql) {
        Some(body) => body,
        None => return,
    };

    for segment in where_body.split(['.', ';']) {
        let trimmed = segment.trim();
        if trimmed.is_empty() || trimmed.starts_with('}') || trimmed.starts_with('{') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < 3 {
            continue;
        }
        let pred_token = tokens[1];
        if !predicate_token_has_path_operator(pred_token) {
            continue;
        }
        // Skip SPARQL keywords
        if is_sparql_keyword(tokens[0]) || is_sparql_keyword(tokens[1]) {
            continue;
        }

        let subject = Term::from_token(tokens[0]);
        let object_raw = tokens[2..].join(" ");
        let object = Term::from_token(&object_raw);

        let path = crate::core::sparql_ast::parse_property_path(pred_token);
        for raw_iri in path.base_iris() {
            if raw_iri.is_empty() {
                continue;
            }
            let predicate = Term::from_token(&raw_iri);
            dest.push(StructuredTriple {
                subject: subject.clone(),
                predicate,
                object: object.clone(),
            });
        }
    }
}

/// Return `true` for SPARQL keywords that should not be treated as triple
/// subject or predicate tokens.
#[cfg(feature = "sparql")]
fn is_sparql_keyword(token: &str) -> bool {
    matches!(
        token.to_ascii_uppercase().as_str(),
        "OPTIONAL"
            | "UNION"
            | "FILTER"
            | "GRAPH"
            | "SERVICE"
            | "BIND"
            | "VALUES"
            | "MINUS"
            | "SELECT"
            | "WHERE"
            | "CONSTRUCT"
            | "DESCRIBE"
            | "ASK"
            | "FROM"
            | "NAMED"
            | "GROUP"
            | "BY"
            | "HAVING"
            | "ORDER"
            | "LIMIT"
            | "OFFSET"
            | "DISTINCT"
            | "REDUCED"
            | "NOT"
            | "EXISTS"
            | "LET"
    )
}

/// Return `true` when a predicate token contains a SPARQL property-path
/// operator character *outside* a full-IRI `<...>` region.
#[cfg(feature = "sparql")]
pub(crate) fn predicate_token_has_path_operator(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut in_iri = false;
    for &b in bytes {
        if b == b'<' {
            in_iri = true;
            continue;
        }
        if b == b'>' {
            in_iri = false;
            continue;
        }
        if !in_iri
            && matches!(
                b,
                b'*' | b'+' | b'?' | b'^' | b'/' | b'|' | b'!' | b'(' | b')'
            )
        {
            return true;
        }
    }
    false
}

/// Locate the content inside the outermost `WHERE { ... }` block.
///
/// Returns `None` if the query has no WHERE keyword or no `{`.
#[cfg(feature = "sparql")]
pub(crate) fn find_where_body(sparql: &str) -> Option<&str> {
    let upper = sparql.to_ascii_uppercase();
    let where_pos = upper.find("WHERE")?;
    let after = &sparql[where_pos + 5..];
    let brace = after.find('{')?;
    let body_start = where_pos + 5 + brace + 1;
    let mut depth: u32 = 1;
    let mut end = body_start;
    for (i, b) in sparql[body_start..].bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = body_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        Some(&sparql[body_start..end])
    } else {
        None
    }
}
