//! Pure-Rust SPARQL parsing utilities for the `sparql` feature.
//!
//! Provides prefix-declaration extraction and SELECT variable enumeration.
//! These are simplified but correct implementations tuned for OxiRouter's
//! routing use-case: accurate enough to improve ML feature extraction without
//! requiring a full grammar or external parsing crates.

#![cfg(feature = "sparql")]

#[cfg(feature = "alloc")]
use alloc::{
    borrow::ToOwned,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use hashbrown::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Internal scanner state machine
// ─────────────────────────────────────────────────────────────────────────────

/// Lexical-scanner state for navigating SPARQL text.
///
/// Shared by all parsing helpers to ensure consistent skip behaviour for
/// string literals, URI brackets, and line comments.
#[derive(Clone, Copy, PartialEq)]
enum ScanState {
    Normal,
    InDoubleQuote,
    InSingleQuote,
    InUri,
    InComment,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Scan `sparql` for `PREFIX name: <uri>` declarations and return a map
/// from `"name:"` (lowercase prefix name + colon) to the namespace URI string.
/// The default namespace (`PREFIX : <uri>`) maps the empty-prefix key `":"`.
///
/// Occurrences inside string literals (`"..."`, `'...'`), URI brackets (`<...>`),
/// and line comments (`# ... \n`) are skipped.
pub(crate) fn extract_prefix_map(sparql: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let bytes = sparql.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;

    while i < len {
        match state {
            ScanState::Normal => {
                match bytes[i] {
                    b'"' => {
                        state = ScanState::InDoubleQuote;
                        i += 1;
                    }
                    b'\'' => {
                        state = ScanState::InSingleQuote;
                        i += 1;
                    }
                    b'<' => {
                        state = ScanState::InUri;
                        i += 1;
                    }
                    b'#' => {
                        state = ScanState::InComment;
                        i += 1;
                    }
                    b'P' | b'p' if is_whole_word_keyword(bytes, i, "PREFIX") => {
                        // Consume "PREFIX"
                        i += 6;
                        // Skip whitespace
                        while i < len && bytes[i].is_ascii_whitespace() {
                            i += 1;
                        }
                        // Read prefix name (alphanumeric, _, -)
                        let name_start = i;
                        while i < len
                            && (bytes[i].is_ascii_alphanumeric()
                                || bytes[i] == b'_'
                                || bytes[i] == b'-')
                        {
                            i += 1;
                        }
                        let name_end = i;
                        // Expect ':'
                        if i < len && bytes[i] == b':' {
                            i += 1; // consume ':'
                            // Skip whitespace
                            while i < len && bytes[i].is_ascii_whitespace() {
                                i += 1;
                            }
                            // Expect '<'
                            if i < len && bytes[i] == b'<' {
                                i += 1; // consume '<'
                                let uri_start = i;
                                while i < len && bytes[i] != b'>' {
                                    i += 1;
                                }
                                let uri_end = i;
                                if i < len {
                                    i += 1;
                                } // consume '>'
                                let prefix_name = &sparql[name_start..name_end];
                                let namespace = &sparql[uri_start..uri_end];
                                let key = format!("{}:", prefix_name.to_lowercase());
                                map.insert(key, namespace.to_string());
                            }
                        }
                    }
                    b'P' | b'p' => {
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            ScanState::InDoubleQuote => {
                match bytes[i] {
                    b'\\' => {
                        i += 2;
                    } // skip escape
                    b'"' => {
                        state = ScanState::Normal;
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            ScanState::InSingleQuote => {
                match bytes[i] {
                    b'\\' => {
                        i += 2;
                    } // skip escape
                    b'\'' => {
                        state = ScanState::Normal;
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            ScanState::InUri => {
                if bytes[i] == b'>' {
                    state = ScanState::Normal;
                }
                i += 1;
            }
            ScanState::InComment => {
                if bytes[i] == b'\n' {
                    state = ScanState::Normal;
                }
                i += 1;
            }
        }
    }

    map
}

/// Expand a prefixed name token (e.g. `"foaf:Person"`) to a full URI
/// (e.g. `"http://xmlns.com/foaf/0.1/Person"`) using a previously computed
/// prefix map. Returns `None` if `token` contains no `:`, if the prefix is
/// not in the map, or if the token already looks like a full URI (`"http://..."`).
pub(crate) fn expand_prefixed_name(token: &str, map: &HashMap<String, String>) -> Option<String> {
    // Already a full URI — don't attempt expansion
    if token.contains("://") {
        return None;
    }
    let colon = token.find(':')?;
    let prefix_key = token[..=colon].to_lowercase(); // includes the ':'
    let local = &token[colon + 1..];
    let namespace = map.get(&prefix_key)?;
    Some(format!("{}{}", namespace, local))
}

/// Extract and expand all prefixed name tokens from the SPARQL body using the
/// provided prefix map.
///
/// Scans the entire query (skipping literals, URIs, comments) for tokens of
/// the form `prefix:local` where `prefix:` is present in `map`. Returns the
/// expanded URIs. PREFIX declaration lines themselves are skipped because
/// after the `:` comes `<`, not an alphanumeric local name, so they are
/// filtered out naturally.
pub(crate) fn extract_expanded_prefixed_uris(
    sparql: &str,
    map: &HashMap<String, String>,
) -> Vec<String> {
    let mut results = Vec::new();
    if map.is_empty() {
        return results;
    }

    let bytes = sparql.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;

    while i < len {
        match state {
            ScanState::Normal => {
                match bytes[i] {
                    b'"' => {
                        state = ScanState::InDoubleQuote;
                        i += 1;
                    }
                    b'\'' => {
                        state = ScanState::InSingleQuote;
                        i += 1;
                    }
                    b'<' => {
                        state = ScanState::InUri;
                        i += 1;
                    }
                    b'#' => {
                        state = ScanState::InComment;
                        i += 1;
                    }
                    c if c.is_ascii_alphabetic() || c == b'_' => {
                        // Read a potential prefixed-name token
                        let tok_start = i;
                        while i < len
                            && (bytes[i].is_ascii_alphanumeric()
                                || bytes[i] == b'_'
                                || bytes[i] == b'-')
                        {
                            i += 1;
                        }
                        // If followed by ':' and then a non-'<' alphanumeric char,
                        // it's a prefixed name (not a PREFIX declaration).
                        if i < len && bytes[i] == b':' {
                            let prefix_end = i; // index of ':'
                            let colon_pos = i;
                            i += 1; // consume ':'
                            if i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                                // Read local name
                                let local_start = i;
                                while i < len
                                    && (bytes[i].is_ascii_alphanumeric()
                                        || bytes[i] == b'_'
                                        || bytes[i] == b'-')
                                {
                                    i += 1;
                                }
                                let local = &sparql[local_start..i];
                                if !local.is_empty() {
                                    let prefix_str = sparql[tok_start..prefix_end].to_lowercase();
                                    let key = format!("{}:", prefix_str);
                                    if let Some(ns) = map.get(&key) {
                                        results.push(format!("{}{}", ns, local));
                                    }
                                }
                            } else {
                                // Not a prefixed local name (e.g., PREFIX decl or bare colon)
                                // back up to colon_pos+1 — we already consumed ':', that's fine
                                let _ = colon_pos; // just to suppress unused variable
                            }
                        }
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            ScanState::InDoubleQuote => match bytes[i] {
                b'\\' => {
                    i += 2;
                }
                b'"' => {
                    state = ScanState::Normal;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            },
            ScanState::InSingleQuote => match bytes[i] {
                b'\\' => {
                    i += 2;
                }
                b'\'' => {
                    state = ScanState::Normal;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            },
            ScanState::InUri => {
                if bytes[i] == b'>' {
                    state = ScanState::Normal;
                }
                i += 1;
            }
            ScanState::InComment => {
                if bytes[i] == b'\n' {
                    state = ScanState::Normal;
                }
                i += 1;
            }
        }
    }

    results
}

/// Extract SELECT projection variable names from a SPARQL query.
/// Returns names WITHOUT the leading `?` or `$` sigil.
/// Returns `["*"]` for `SELECT *`.
/// Returns `Vec::new()` for non-SELECT queries (ASK, CONSTRUCT, DESCRIBE).
///
/// This function is infallible — it returns an empty Vec on any ambiguity.
pub(crate) fn extract_select_variables(sparql: &str) -> Vec<String> {
    let select_pos = match find_keyword(sparql, "SELECT") {
        Some(pos) => pos,
        None => return Vec::new(),
    };

    let bytes = sparql.as_bytes();
    let len = bytes.len();
    let mut i = select_pos + 6; // skip "SELECT"

    // Skip whitespace
    while i < len && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // Skip optional DISTINCT or REDUCED keyword
    for kw in &["DISTINCT", "REDUCED"] {
        if i + kw.len() <= len {
            let chunk = &sparql[i..i + kw.len()];
            if chunk.eq_ignore_ascii_case(kw) {
                let after = i + kw.len();
                if after >= len || !bytes[after].is_ascii_alphanumeric() {
                    i = after;
                    // Skip whitespace
                    while i < len && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    break;
                }
            }
        }
    }

    // Check for SELECT *
    if i < len && bytes[i] == b'*' {
        return vec!["*".to_owned()];
    }

    // Collect variable names (tokens starting with '?' or '$')
    let mut vars = Vec::new();
    while i < len {
        match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'?' | b'$' => {
                i += 1; // skip sigil
                let var_start = i;
                while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let name = &sparql[var_start..i];
                if !name.is_empty() {
                    vars.push(name.to_owned());
                }
            }
            _ => {
                // Anything other than '?', '$', or whitespace terminates the projection list
                break;
            }
        }
    }

    vars
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the byte offset of a whole-word occurrence of `keyword` (case-insensitive)
/// in `text`, skipping string literals and line comments.
///
/// "Whole word" means the character before is non-alphanumeric (or start-of-string),
/// and the character after is non-alphanumeric (or end-of-string).
fn find_keyword(text: &str, keyword: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let klen = keyword.len();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;

    while i < len {
        match state {
            ScanState::Normal => {
                match bytes[i] {
                    b'"' => {
                        state = ScanState::InDoubleQuote;
                        i += 1;
                    }
                    b'\'' => {
                        state = ScanState::InSingleQuote;
                        i += 1;
                    }
                    b'<' => {
                        state = ScanState::InUri;
                        i += 1;
                    }
                    b'#' => {
                        state = ScanState::InComment;
                        i += 1;
                    }
                    _ => {
                        if i + klen <= len {
                            let chunk = &text[i..i + klen];
                            if chunk.eq_ignore_ascii_case(keyword) {
                                // Whole-word check: char before
                                let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                                // Whole-word check: char after
                                let after_ok =
                                    (i + klen) >= len || !bytes[i + klen].is_ascii_alphanumeric();
                                if before_ok && after_ok {
                                    return Some(i);
                                }
                            }
                        }
                        i += 1;
                    }
                }
            }
            ScanState::InDoubleQuote => match bytes[i] {
                b'\\' => {
                    i += 2;
                }
                b'"' => {
                    state = ScanState::Normal;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            },
            ScanState::InSingleQuote => match bytes[i] {
                b'\\' => {
                    i += 2;
                }
                b'\'' => {
                    state = ScanState::Normal;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            },
            ScanState::InUri => {
                if bytes[i] == b'>' {
                    state = ScanState::Normal;
                }
                i += 1;
            }
            ScanState::InComment => {
                if bytes[i] == b'\n' {
                    state = ScanState::Normal;
                }
                i += 1;
            }
        }
    }

    None
}

/// Return true if `bytes[pos..]` starts a whole-word occurrence of `keyword`
/// (case-insensitive). The character at `pos - 1` (if any) must be
/// non-alphanumeric; the character at `pos + keyword.len()` (if any) must be
/// non-alphanumeric.
fn is_whole_word_keyword(bytes: &[u8], pos: usize, keyword: &str) -> bool {
    let klen = keyword.len();
    if pos + klen > bytes.len() {
        return false;
    }
    // Case-insensitive prefix comparison
    let chunk = match core::str::from_utf8(&bytes[pos..pos + klen]) {
        Ok(s) => s,
        Err(_) => return false,
    };
    if !chunk.eq_ignore_ascii_case(keyword) {
        return false;
    }
    // Whole-word: char before must be non-alphanumeric or start-of-string
    if pos > 0 && bytes[pos - 1].is_ascii_alphanumeric() {
        return false;
    }
    // Whole-word: char after must be non-alphanumeric or end-of-string
    let after_pos = pos + klen;
    if after_pos < bytes.len() && bytes[after_pos].is_ascii_alphanumeric() {
        return false;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_map_basic() {
        let sparql =
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?s WHERE { ?s foaf:name ?n }";
        let map = extract_prefix_map(sparql);
        assert_eq!(
            map.get("foaf:").map(String::as_str),
            Some("http://xmlns.com/foaf/0.1/")
        );
    }

    #[test]
    fn prefix_map_multiple() {
        let sparql = "PREFIX ex: <http://example.org/> PREFIX dbo: <http://dbpedia.org/ontology/> SELECT ?x WHERE { ?x ex:role dbo:Person }";
        let map = extract_prefix_map(sparql);
        assert_eq!(
            map.get("ex:").map(String::as_str),
            Some("http://example.org/")
        );
        assert_eq!(
            map.get("dbo:").map(String::as_str),
            Some("http://dbpedia.org/ontology/")
        );
    }

    #[test]
    fn prefix_map_default_namespace() {
        let sparql = "PREFIX : <http://example.org/default/> SELECT ?s WHERE { ?s :foo ?o }";
        let map = extract_prefix_map(sparql);
        assert_eq!(
            map.get(":").map(String::as_str),
            Some("http://example.org/default/")
        );
    }

    #[test]
    fn prefix_map_skips_literals() {
        // "PREFIX foo: <bar>" inside a string should NOT be parsed as a declaration
        let sparql = r#"SELECT ?s WHERE { ?s <http://ex.org/p> "PREFIX foo: <http://notreal/>" }"#;
        let map = extract_prefix_map(sparql);
        assert!(
            map.is_empty(),
            "should not parse PREFIX inside string literal"
        );
    }

    #[test]
    fn prefix_map_skips_comments() {
        let sparql = "# PREFIX foo: <http://notreal/>\nSELECT ?s WHERE { ?s ?p ?o }";
        let map = extract_prefix_map(sparql);
        assert!(map.is_empty(), "should not parse PREFIX inside comment");
    }

    #[test]
    fn expand_prefixed_name_basic() {
        let mut map = HashMap::new();
        map.insert("foaf:".to_owned(), "http://xmlns.com/foaf/0.1/".to_owned());
        let result = expand_prefixed_name("foaf:Person", &map);
        assert_eq!(result.as_deref(), Some("http://xmlns.com/foaf/0.1/Person"));
    }

    #[test]
    fn expand_prefixed_name_full_uri_unchanged() {
        let map = HashMap::new();
        let result = expand_prefixed_name("http://example.org/foo", &map);
        assert!(result.is_none(), "full URIs should not be expanded");
    }

    #[test]
    fn expand_prefixed_name_unknown_prefix() {
        let map = HashMap::new();
        let result = expand_prefixed_name("unknown:bar", &map);
        assert!(result.is_none());
    }

    #[test]
    fn extract_expanded_uris_basic() {
        let sparql = "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:foo ?o }";
        let map = extract_prefix_map(sparql);
        let uris = extract_expanded_prefixed_uris(sparql, &map);
        assert!(
            uris.contains(&"http://example.org/foo".to_owned()),
            "should contain expanded uri; got: {:?}",
            uris
        );
    }

    #[test]
    fn extract_expanded_uris_skips_prefix_decl() {
        // The PREFIX declaration itself (ex: <http://example.org/>) should not produce a URI
        let sparql = "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:name ?o }";
        let map = extract_prefix_map(sparql);
        let uris = extract_expanded_prefixed_uris(sparql, &map);
        // Only ex:name should be expanded, not the declaration
        assert_eq!(
            uris.iter()
                .filter(|u| u.as_str() == "http://example.org/name")
                .count(),
            1
        );
    }

    #[test]
    fn select_vars_basic() {
        let sparql = "SELECT ?a ?b ?c WHERE { ?a ?b ?c }";
        let vars = extract_select_variables(sparql);
        assert_eq!(vars, vec!["a", "b", "c"]);
    }

    #[test]
    fn select_vars_distinct() {
        let sparql = "SELECT DISTINCT ?x ?y WHERE { ?x ?p ?y }";
        let vars = extract_select_variables(sparql);
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn select_vars_star() {
        let sparql = "SELECT * WHERE { ?s ?p ?o }";
        let vars = extract_select_variables(sparql);
        assert_eq!(vars, vec!["*"]);
    }

    #[test]
    fn select_vars_non_select() {
        let sparql = "ASK { ?s <http://example.org/p> ?o }";
        let vars = extract_select_variables(sparql);
        assert!(vars.is_empty());
    }

    #[test]
    fn select_vars_construct() {
        let sparql = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
        let vars = extract_select_variables(sparql);
        assert!(vars.is_empty());
    }
}
