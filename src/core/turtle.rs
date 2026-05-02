//! Pure-Rust Turtle subset scanner for the `void` feature.
//!
//! Implements a character-level state-machine tokenizer and a recursive-descent
//! parser covering the Turtle subset needed to read VoID/OxiRouter descriptor
//! files.  No external dependencies — mirrors the style of `src/core/sparql.rs`.

#![cfg(feature = "void")]

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use hashbrown::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Scanner state machine
// ─────────────────────────────────────────────────────────────────────────────

/// Lexical-scanner state used by the Turtle byte scanner.
#[derive(Clone, Copy, PartialEq)]
enum ScanState {
    Normal,
    InComment,            // after '#' until newline
    InIriRef,             // inside '<' '>'
    InStringDouble,       // inside '"'
    InStringTripleDouble, // inside '"""'
    InStringSingle,       // inside "'"
    InStringTripleSingle, // inside "'''"
}

// ─────────────────────────────────────────────────────────────────────────────
// Token type
// ─────────────────────────────────────────────────────────────────────────────

/// All tokens produced by the Turtle tokenizer.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// `<...>` — raw IRI reference content (not yet expanded against base)
    IriRef(String),
    /// `pfx:local` — expand later once prefix map is built
    PrefixedName(String),
    /// `"..."` / `'...'` / `"""..."""` / `'''...'''` — unescaped content
    StringLit(String),
    /// `_:xxx` — blank node label
    BlankNodeLabel(String),
    /// `.`
    Dot,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `[`
    BracketOpen,
    /// `]`
    BracketClose,
    /// `@prefix`
    AtPrefix,
    /// `PREFIX` (SPARQL-style, case-insensitive)
    KwPrefix,
    /// `a`, unknown identifiers
    Keyword(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// Walk `ttl` byte-by-byte, producing a flat token stream.
///
/// All state transitions are governed by `ScanState`; string/IRI/comment
/// regions are correctly handled without regex or nom.
fn tokenize(ttl: &str) -> Vec<Tok> {
    let mut tokens: Vec<Tok> = Vec::new();
    let bytes = ttl.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;

    // Reusable helper: peek 1 byte ahead safely
    macro_rules! peek1 {
        () => {
            if i + 1 < len { bytes[i + 1] } else { 0 }
        };
    }
    macro_rules! peek2 {
        () => {
            if i + 2 < len { bytes[i + 2] } else { 0 }
        };
    }

    while i < len {
        match state {
            ScanState::Normal => {
                let b = bytes[i];
                match b {
                    // Skip whitespace
                    b' ' | b'\t' | b'\r' | b'\n' => {
                        i += 1;
                    }

                    // Comment
                    b'#' => {
                        state = ScanState::InComment;
                        i += 1;
                    }

                    // IRI reference
                    b'<' => {
                        state = ScanState::InIriRef;
                        i += 1;
                    }

                    // String literals — check for triple-quote first
                    b'"' => {
                        if peek1!() == b'"' && peek2!() == b'"' {
                            i += 3;
                            state = ScanState::InStringTripleDouble;
                        } else {
                            i += 1;
                            state = ScanState::InStringDouble;
                        }
                    }
                    b'\'' => {
                        if peek1!() == b'\'' && peek2!() == b'\'' {
                            i += 3;
                            state = ScanState::InStringTripleSingle;
                        } else {
                            i += 1;
                            state = ScanState::InStringSingle;
                        }
                    }

                    // Blank node label: _:xxx
                    b'_' if peek1!() == b':' => {
                        i += 2; // skip '_:'
                        let start = i;
                        while i < len && is_bnode_char(bytes[i]) {
                            i += 1;
                        }
                        let label = ttl[start..i].to_string();
                        tokens.push(Tok::BlankNodeLabel(label));
                    }

                    // Punctuation
                    b'.' => {
                        tokens.push(Tok::Dot);
                        i += 1;
                    }
                    b';' => {
                        tokens.push(Tok::Semicolon);
                        i += 1;
                    }
                    b',' => {
                        tokens.push(Tok::Comma);
                        i += 1;
                    }
                    b'[' => {
                        tokens.push(Tok::BracketOpen);
                        i += 1;
                    }
                    b']' => {
                        tokens.push(Tok::BracketClose);
                        i += 1;
                    }

                    // @-keywords
                    b'@' => {
                        i += 1;
                        let start = i;
                        while i < len && bytes[i].is_ascii_alphabetic() {
                            i += 1;
                        }
                        let kw = ttl[start..i].to_string();
                        if kw.eq_ignore_ascii_case("prefix") {
                            tokens.push(Tok::AtPrefix);
                        } else {
                            // Other @-directives (e.g. @base) — emit as keyword
                            tokens.push(Tok::Keyword(format!("@{kw}")));
                        }
                    }

                    // Identifier-like tokens (PREFIX, a, prefixed-names, bare keywords)
                    c if c.is_ascii_alphabetic() => {
                        let start = i;
                        // Scan identifier chars including `:`, `_`, `-`, `.`
                        while i < len && is_ident_char(bytes[i]) {
                            i += 1;
                        }
                        // If trailing dots that are not part of the ident, trim them
                        while i > start && bytes[i - 1] == b'.' {
                            // Only trim if the char before the dot is non-alphanumeric
                            // (handles `foaf:Person.` → `foaf:Person` + Dot)
                            if i >= 2 && is_ident_inner(bytes[i - 2]) {
                                break;
                            }
                            i -= 1;
                        }
                        let tok_str = &ttl[start..i];
                        let tok = classify_ident_token(tok_str);
                        tokens.push(tok);
                    }

                    // Digits (e.g. numeric literals in property values — emit as keyword/string)
                    c if c.is_ascii_digit() || c == b'+' || c == b'-' => {
                        let start = i;
                        i += 1;
                        while i < len
                            && (bytes[i].is_ascii_digit()
                                || bytes[i] == b'.'
                                || bytes[i] == b'e'
                                || bytes[i] == b'E'
                                || bytes[i] == b'+'
                                || bytes[i] == b'-')
                        {
                            i += 1;
                        }
                        let num_str = ttl[start..i].to_string();
                        tokens.push(Tok::Keyword(num_str));
                    }

                    _ => {
                        i += 1;
                    } // skip unknown bytes
                }
            }

            ScanState::InComment => {
                if bytes[i] == b'\n' {
                    state = ScanState::Normal;
                }
                i += 1;
            }

            ScanState::InIriRef => {
                let start = i;
                while i < len && bytes[i] != b'>' {
                    i += 1;
                }
                if i < len {
                    let content = ttl[start..i].to_string();
                    tokens.push(Tok::IriRef(content));
                    i += 1; // consume '>'
                } else {
                    // Unclosed IRI — push a sentinel that the parser will reject
                    tokens.push(Tok::IriRef(ttl[start..i].to_string()));
                }
                state = ScanState::Normal;
            }

            ScanState::InStringDouble => {
                let (content, new_i) = scan_quoted_string(ttl, bytes, i, len, b'"');
                tokens.push(Tok::StringLit(content));
                i = new_i;
                state = ScanState::Normal;
            }

            ScanState::InStringSingle => {
                let (content, new_i) = scan_quoted_string(ttl, bytes, i, len, b'\'');
                tokens.push(Tok::StringLit(content));
                i = new_i;
                state = ScanState::Normal;
            }

            ScanState::InStringTripleDouble => {
                let start = i;
                while i + 2 < len {
                    if bytes[i] == b'"' && bytes[i + 1] == b'"' && bytes[i + 2] == b'"' {
                        let content = ttl[start..i].to_string();
                        tokens.push(Tok::StringLit(content));
                        i += 3;
                        state = ScanState::Normal;
                        break;
                    }
                    i += 1;
                }
                if state == ScanState::InStringTripleDouble {
                    // Unclosed
                    let content = ttl[start..i].to_string();
                    tokens.push(Tok::StringLit(content));
                    state = ScanState::Normal;
                    i = len;
                }
            }

            ScanState::InStringTripleSingle => {
                let start = i;
                while i + 2 < len {
                    if bytes[i] == b'\'' && bytes[i + 1] == b'\'' && bytes[i + 2] == b'\'' {
                        let content = ttl[start..i].to_string();
                        tokens.push(Tok::StringLit(content));
                        i += 3;
                        state = ScanState::Normal;
                        break;
                    }
                    i += 1;
                }
                if state == ScanState::InStringTripleSingle {
                    let content = ttl[start..i].to_string();
                    tokens.push(Tok::StringLit(content));
                    state = ScanState::Normal;
                    i = len;
                }
            }
        }
    }

    tokens
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Scan a single-quoted string (with escape handling) starting at byte position `i`.
///
/// On entry `i` points to the first byte AFTER the opening delimiter.
/// Returns `(unescaped_content, new_i)` where `new_i` is one past the closing delimiter.
fn scan_quoted_string(
    ttl: &str,
    bytes: &[u8],
    mut i: usize,
    len: usize,
    delim: u8,
) -> (String, usize) {
    let mut buf = String::new();
    while i < len {
        let b = bytes[i];
        if b == b'\\' {
            i += 1;
            if i < len {
                buf.push(unescape_char(bytes[i]));
                i += 1;
            }
        } else if b == delim {
            i += 1; // consume closing delimiter
            return (buf, i);
        } else {
            // Push the character (handle multi-byte UTF-8 safely via char boundary)
            let char_end = next_char_boundary(ttl, i);
            buf.push_str(&ttl[i..char_end]);
            i = char_end;
        }
    }
    // Unclosed string — return what was accumulated
    (buf, i)
}

/// Return the byte index of the start of the next UTF-8 character after position `i`.
#[inline]
fn next_char_boundary(s: &str, i: usize) -> usize {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut j = i + 1;
    while j < len && !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

/// Characters valid inside an identifier/prefixed-name token.
/// Includes `:` for `pfx:local`, `-` and `_` for local names, `.` for compound names.
#[inline]
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b':' || b == b'_' || b == b'-' || b == b'.'
}

/// Characters valid in the interior of an identifier (not period-termination logic).
#[inline]
fn is_ident_inner(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// Characters valid in blank-node labels: alphanumeric, `_`, `-`, `.`
#[inline]
fn is_bnode_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

/// Map an escape character (the byte after `\`) to the actual character.
#[inline]
fn unescape_char(b: u8) -> char {
    match b {
        b'n' => '\n',
        b't' => '\t',
        b'r' => '\r',
        b'"' => '"',
        b'\'' => '\'',
        b'\\' => '\\',
        other => other as char,
    }
}

/// Classify a scanned identifier/keyword string into the appropriate `Tok`.
fn classify_ident_token(tok_str: &str) -> Tok {
    if tok_str.eq_ignore_ascii_case("prefix") {
        return Tok::KwPrefix;
    }
    if tok_str == "a" {
        return Tok::Keyword("a".to_string());
    }
    if tok_str.contains(':') {
        return Tok::PrefixedName(tok_str.to_string());
    }
    Tok::Keyword(tok_str.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// IRI expansion
// ─────────────────────────────────────────────────────────────────────────────

/// Expand a raw IRI string or prefixed name to an absolute IRI.
///
/// - If it looks like an absolute IRI (contains `://` or starts with `urn:`), return as-is.
/// - If it contains `:`, split at the first `:`, look up the prefix key `"pfx:"` (lowercase)
///   in `prefixes`, and concatenate.  If the prefix is unknown, return as-is.
/// - Otherwise return as-is.
pub(crate) fn expand_iri(iri_or_prefixed: &str, prefixes: &HashMap<String, String>) -> String {
    // Already absolute
    if iri_or_prefixed.contains("://") || iri_or_prefixed.starts_with("urn:") {
        return iri_or_prefixed.to_string();
    }
    if let Some(colon_pos) = iri_or_prefixed.find(':') {
        let pfx_raw = &iri_or_prefixed[..colon_pos];
        let local = &iri_or_prefixed[colon_pos + 1..];
        let key = format!("{}:", pfx_raw.to_lowercase());
        if let Some(ns) = prefixes.get(&key) {
            return format!("{ns}{local}");
        }
        // Default prefix (empty prefix name)
        if pfx_raw.is_empty() {
            if let Some(ns) = prefixes.get(":") {
                return format!("{ns}{local}");
            }
        }
    }
    iri_or_prefixed.to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public output types
// ─────────────────────────────────────────────────────────────────────────────

/// A single RDF triple produced by `parse_turtle`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TurtleTriple {
    /// Subject IRI / blank node IRI
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// Object IRI / blank node IRI / string literal (prefixed with `"`)
    pub object: String,
}

/// The output of `parse_turtle`: RDF triples with all prefixes already expanded.
pub(crate) struct TurtleDoc {
    /// All triples extracted from the Turtle document (IRIs fully expanded).
    pub triples: Vec<TurtleTriple>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a Turtle document into a `TurtleDoc` (prefix map + triples).
///
/// The parser handles:
/// - `@prefix`/`PREFIX` directives
/// - Subject-predicate-object triples with `;` and `,` continuations
/// - Anonymous blank nodes via `[ predicate object ; ... ]`
/// - The `a` keyword as shorthand for `rdf:type`
///
/// This function is infallible: malformed tokens are silently skipped so that
/// partially-valid descriptors still yield the usable triples.
pub(crate) fn parse_turtle(ttl: &str) -> TurtleDoc {
    let tokens = tokenize(ttl);

    // ── Pass 1: collect prefix declarations ──────────────────────────────────
    let mut prefixes: HashMap<String, String> = HashMap::new();
    {
        let mut idx = 0;
        while idx < tokens.len() {
            match &tokens[idx] {
                Tok::AtPrefix | Tok::KwPrefix => {
                    idx += 1;
                    // Next token: the prefix name as a PrefixedName (e.g. `void:`)
                    // or Keyword for the default prefix (just `:`)
                    if idx >= tokens.len() {
                        break;
                    }
                    let pfx_key = match &tokens[idx] {
                        Tok::PrefixedName(s) => {
                            // s is like "void:" — normalise to lowercase
                            s.to_lowercase()
                        }
                        Tok::Keyword(s) if s == ":" => ":".to_string(),
                        // Default prefix: the token could be just ":"
                        // Some tokenizers produce it differently; handle Keyword("")
                        Tok::Keyword(s) if s.is_empty() => ":".to_string(),
                        // Could be a Keyword that is just the prefix name without ":"
                        // (when the "@prefix void :" pattern splits differently)
                        _ => {
                            idx += 1;
                            continue;
                        }
                    };
                    // Normalise: ensure key ends with ":"
                    let pfx_key = if pfx_key.ends_with(':') {
                        pfx_key
                    } else {
                        format!("{pfx_key}:")
                    };
                    idx += 1;
                    if idx >= tokens.len() {
                        break;
                    }
                    if let Tok::IriRef(uri) = &tokens[idx] {
                        prefixes.insert(pfx_key, uri.clone());
                    }
                    idx += 1;
                }
                _ => {
                    idx += 1;
                }
            }
        }
    }

    // ── Pass 2: parse triples ────────────────────────────────────────────────
    let mut triples: Vec<TurtleTriple> = Vec::new();
    let mut bnode_counter: u64 = 0;
    let mut idx = 0;

    while idx < tokens.len() {
        match &tokens[idx] {
            // Skip prefix directives
            Tok::AtPrefix | Tok::KwPrefix => {
                idx += 1;
                // skip prefix-name token
                if idx < tokens.len() {
                    idx += 1;
                }
                // skip IriRef token
                if idx < tokens.len() {
                    idx += 1;
                }
                // skip optional trailing dot
                if idx < tokens.len() {
                    if let Tok::Dot = tokens[idx] {
                        idx += 1;
                    }
                }
            }

            // Top-level blank node: [ ... ] possibly followed by more pred-obj or just '.'
            Tok::BracketOpen => {
                let (bn_iri, new_idx) = parse_blank_node_prop_list(
                    &tokens,
                    idx,
                    &prefixes,
                    &mut triples,
                    &mut bnode_counter,
                );
                idx = new_idx;
                // After the blank node block there may be more pred-obj pairs or just '.'
                idx = parse_predicate_object_list(
                    &tokens,
                    idx,
                    &bn_iri,
                    &prefixes,
                    &mut triples,
                    &mut bnode_counter,
                );
                // Consume trailing dot
                if idx < tokens.len() {
                    if let Tok::Dot = tokens[idx] {
                        idx += 1;
                    }
                }
            }

            _ => {
                // Try to read subject
                match extract_node(&tokens, idx, &prefixes) {
                    Some((subject_iri, new_idx)) => {
                        idx = new_idx;
                        idx = parse_predicate_object_list(
                            &tokens,
                            idx,
                            &subject_iri,
                            &prefixes,
                            &mut triples,
                            &mut bnode_counter,
                        );
                        // Consume trailing dot
                        if idx < tokens.len() {
                            if let Tok::Dot = tokens[idx] {
                                idx += 1;
                            }
                        }
                    }
                    None => {
                        // Skip unrecognised token
                        idx += 1;
                    }
                }
            }
        }
    }

    let _ = prefixes; // consumed during parsing for IRI expansion
    TurtleDoc { triples }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Try to extract an IRI/blank-node node from `tokens[idx]`.
/// Returns `(expanded_iri, new_idx)` or `None` if the token is not a node.
fn extract_node(
    tokens: &[Tok],
    idx: usize,
    prefixes: &HashMap<String, String>,
) -> Option<(String, usize)> {
    if idx >= tokens.len() {
        return None;
    }
    match &tokens[idx] {
        Tok::IriRef(s) => Some((s.clone(), idx + 1)),
        Tok::PrefixedName(s) => Some((expand_iri(s, prefixes), idx + 1)),
        Tok::BlankNodeLabel(s) => Some((format!("_:{s}"), idx + 1)),
        _ => None,
    }
}

/// Extract the predicate IRI from `tokens[idx]`.
/// Handles `a` → `rdf:type`, IriRef, PrefixedName.
fn extract_predicate(
    tokens: &[Tok],
    idx: usize,
    prefixes: &HashMap<String, String>,
) -> Option<(String, usize)> {
    if idx >= tokens.len() {
        return None;
    }
    match &tokens[idx] {
        Tok::Keyword(s) if s == "a" => Some((
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            idx + 1,
        )),
        Tok::IriRef(s) => Some((s.clone(), idx + 1)),
        Tok::PrefixedName(s) => Some((expand_iri(s, prefixes), idx + 1)),
        _ => None,
    }
}

/// Parse a `[ predPredObj ; ... ]` anonymous blank node property list.
///
/// On entry `idx` points at the `BracketOpen` token.
/// Returns `(blank_node_iri, new_idx_after_BracketClose)`.
fn parse_blank_node_prop_list(
    tokens: &[Tok],
    idx: usize,
    prefixes: &HashMap<String, String>,
    triples: &mut Vec<TurtleTriple>,
    counter: &mut u64,
) -> (String, usize) {
    // Consume '['
    debug_assert!(matches!(tokens.get(idx), Some(Tok::BracketOpen)));
    let mut idx = idx + 1;

    *counter += 1;
    let bn_iri = format!("_:b{}", *counter);

    // Parse inner predicate-object pairs
    loop {
        if idx >= tokens.len() {
            break;
        }
        if let Tok::BracketClose = tokens[idx] {
            idx += 1; // consume ']'
            break;
        }
        if let Tok::Dot = tokens[idx] {
            break; // malformed, but recover
        }

        // Try to read a predicate
        match extract_predicate(tokens, idx, prefixes) {
            None => {
                idx += 1;
            }
            Some((pred, new_idx)) => {
                idx = new_idx;
                // Read object list
                idx = parse_object_list_inner(
                    tokens, idx, &bn_iri, &pred, prefixes, triples, counter,
                );
                // Optional ';'
                while idx < tokens.len() {
                    if let Tok::Semicolon = tokens[idx] {
                        idx += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    (bn_iri, idx)
}

/// Parse `verbObjectList (';' verbObjectList)*` after `subject`.
/// Returns new `idx` after consuming everything up to (but not including) the terminal `.`.
fn parse_predicate_object_list(
    tokens: &[Tok],
    mut idx: usize,
    subject: &str,
    prefixes: &HashMap<String, String>,
    triples: &mut Vec<TurtleTriple>,
    counter: &mut u64,
) -> usize {
    loop {
        if idx >= tokens.len() {
            break;
        }
        // Stop at statement terminators
        if let Tok::Dot = tokens[idx] {
            break;
        }
        if let Tok::BracketClose = tokens[idx] {
            break;
        }

        // Try to read predicate
        match extract_predicate(tokens, idx, prefixes) {
            None => {
                idx += 1;
            }
            Some((pred, new_idx)) => {
                idx = new_idx;
                idx = parse_object_list_inner(
                    tokens, idx, subject, &pred, prefixes, triples, counter,
                );
                // Consume optional semicolons (possibly multiple)
                while idx < tokens.len() {
                    if let Tok::Semicolon = tokens[idx] {
                        idx += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }
    idx
}

/// Parse `object (',' object)*` for one predicate and emit triples.
/// Returns new `idx`.
fn parse_object_list_inner(
    tokens: &[Tok],
    mut idx: usize,
    subject: &str,
    predicate: &str,
    prefixes: &HashMap<String, String>,
    triples: &mut Vec<TurtleTriple>,
    counter: &mut u64,
) -> usize {
    loop {
        if idx >= tokens.len() {
            break;
        }
        match &tokens[idx] {
            Tok::Dot | Tok::Semicolon | Tok::BracketClose => {
                break;
            }
            Tok::Comma => {
                idx += 1;
            }
            Tok::BracketOpen => {
                // Anonymous blank node as object
                let (bn_iri, new_idx) =
                    parse_blank_node_prop_list(tokens, idx, prefixes, triples, counter);
                triples.push(TurtleTriple {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: bn_iri,
                });
                idx = new_idx;
            }
            Tok::IriRef(s) => {
                triples.push(TurtleTriple {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: s.clone(),
                });
                idx += 1;
            }
            Tok::PrefixedName(s) => {
                triples.push(TurtleTriple {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: expand_iri(s, prefixes),
                });
                idx += 1;
            }
            Tok::BlankNodeLabel(s) => {
                triples.push(TurtleTriple {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: format!("_:{s}"),
                });
                idx += 1;
            }
            Tok::StringLit(s) => {
                // Store string literals with surrounding quotes stripped
                // (they are already unescaped by the tokenizer)
                triples.push(TurtleTriple {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: s.clone(),
                });
                idx += 1;
                // Skip optional language tag (@en) or datatype (^^xsd:string)
                if idx < tokens.len() {
                    match &tokens[idx] {
                        Tok::Keyword(kw) if kw.starts_with('@') => {
                            idx += 1;
                        }
                        _ => {}
                    }
                }
                if idx + 1 < tokens.len() {
                    if let Tok::Keyword(kw) = &tokens[idx] {
                        if kw == "^^" {
                            idx += 2; // skip '^^' and datatype
                        }
                    }
                }
            }
            Tok::Keyword(kw) if kw == "^^" => {
                // Datatype annotation on previous literal — skip two tokens
                idx += 2;
            }
            Tok::Keyword(_) => {
                // Numeric literals or other keyword objects
                idx += 1;
            }
            _ => {
                idx += 1;
            }
        }
    }
    idx
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_iri_ref() {
        let toks = tokenize("<http://example.org/>");
        assert_eq!(toks, vec![Tok::IriRef("http://example.org/".to_string())]);
    }

    #[test]
    fn tokenize_prefix_directive() {
        let toks = tokenize("@prefix void: <http://rdfs.org/ns/void#> .");
        assert!(toks.contains(&Tok::AtPrefix));
        assert!(toks.contains(&Tok::PrefixedName("void:".to_string())));
        assert!(toks.contains(&Tok::IriRef("http://rdfs.org/ns/void#".to_string())));
        assert!(toks.contains(&Tok::Dot));
    }

    #[test]
    fn tokenize_string_literal() {
        let toks = tokenize(r#""hello world""#);
        assert_eq!(toks, vec![Tok::StringLit("hello world".to_string())]);
    }

    #[test]
    fn tokenize_triple_double_string() {
        let toks = tokenize(r#""""multi line""" "#);
        assert_eq!(toks, vec![Tok::StringLit("multi line".to_string())]);
    }

    #[test]
    fn tokenize_blank_node() {
        let toks = tokenize("_:myNode");
        assert_eq!(toks, vec![Tok::BlankNodeLabel("myNode".to_string())]);
    }

    #[test]
    fn tokenize_semicolon_comma_dot() {
        let toks = tokenize("; , .");
        assert_eq!(toks, vec![Tok::Semicolon, Tok::Comma, Tok::Dot]);
    }

    #[test]
    fn tokenize_keyword_a() {
        let toks = tokenize("a");
        assert_eq!(toks, vec![Tok::Keyword("a".to_string())]);
    }

    #[test]
    fn tokenize_comment_skipped() {
        let toks = tokenize("# this is a comment\n<http://example.org/>");
        assert_eq!(toks, vec![Tok::IriRef("http://example.org/".to_string())]);
    }

    #[test]
    fn expand_iri_absolute() {
        let map: HashMap<String, String> = HashMap::new();
        assert_eq!(
            expand_iri("http://example.org/foo", &map),
            "http://example.org/foo"
        );
    }

    #[test]
    fn expand_iri_prefixed() {
        let mut map: HashMap<String, String> = HashMap::new();
        map.insert("void:".to_string(), "http://rdfs.org/ns/void#".to_string());
        assert_eq!(
            expand_iri("void:Dataset", &map),
            "http://rdfs.org/ns/void#Dataset"
        );
    }

    #[test]
    fn parse_turtle_prefix_and_triple() {
        let ttl = r#"
@prefix void: <http://rdfs.org/ns/void#> .
<http://example.org/ds1> a void:Dataset .
"#;
        let doc = parse_turtle(ttl);
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(
            doc.triples[0].predicate,
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
        assert_eq!(doc.triples[0].object, "http://rdfs.org/ns/void#Dataset");
    }
}
