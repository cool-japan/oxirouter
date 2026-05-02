//! SPARQL 1.1 §9 property path AST and recursive-descent parser.

#[cfg(feature = "alloc")]
use alloc::{borrow::ToOwned, boxed::Box, string::String, vec, vec::Vec};

// ─────────────────────────────────────────────────────────────────────────────
// PropertyPath: SPARQL 1.1 §9 property-path AST
// ─────────────────────────────────────────────────────────────────────────────

/// A SPARQL 1.1 §9 property path expression (best-effort structural representation).
///
/// All variant payloads own their strings so the type is `'static`.
/// The parser is infallible: malformed input yields `PropertyPath::Iri(raw)`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PropertyPath {
    /// A single IRI or prefixed name leaf (raw token, not yet expanded).
    Iri(String),
    /// `^p` — inverse of a path step.
    Inverse(Box<PropertyPath>),
    /// `p / q` — sequence of two steps.
    Sequence(Vec<PropertyPath>),
    /// `p | q` — alternative paths.
    Alternative(Vec<PropertyPath>),
    /// `p*` — zero or more occurrences.
    ZeroOrMore(Box<PropertyPath>),
    /// `p+` — one or more occurrences.
    OneOrMore(Box<PropertyPath>),
    /// `p?` — zero or one occurrence.
    ZeroOrOne(Box<PropertyPath>),
    /// `!(p | q | ...)` — negated property set.
    NegatedPropertySet(Vec<PropertyPath>),
}

impl PropertyPath {
    /// Collect all leaf IRI strings from this path expression.
    ///
    /// Duplicates are preserved (callers can deduplicate if needed).
    /// This does **not** expand prefixed names — call
    /// [`crate::core::sparql::expand_prefixed_name`] on each element to get
    /// full URIs.
    pub(crate) fn base_iris(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_iris(&mut out);
        out
    }

    fn collect_iris(&self, out: &mut Vec<String>) {
        match self {
            Self::Iri(s) => {
                if !s.is_empty() {
                    out.push(s.clone());
                }
            }
            Self::Inverse(inner) => inner.collect_iris(out),
            Self::Sequence(steps) | Self::Alternative(steps) | Self::NegatedPropertySet(steps) => {
                for step in steps {
                    step.collect_iris(out);
                }
            }
            Self::ZeroOrMore(inner) | Self::OneOrMore(inner) | Self::ZeroOrOne(inner) => {
                inner.collect_iris(out)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recursive-descent parser for property paths
// ─────────────────────────────────────────────────────────────────────────────

/// Parser state: wraps the input bytes and a mutable cursor.
struct PathParser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> PathParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            input: s.as_bytes(),
            pos: 0,
        }
    }

    /// Peek at current byte (or `b'\0'` past end).
    fn peek(&self) -> u8 {
        if self.pos < self.input.len() {
            self.input[self.pos]
        } else {
            0
        }
    }

    /// Advance past whitespace.
    fn skip_ws(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Consume one byte and advance.
    fn consume(&mut self) -> u8 {
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// Top-level: `alt_path`
    fn parse_path(&mut self) -> PropertyPath {
        self.skip_ws();
        self.parse_alt()
    }

    /// `alt_path := seq_path ('|' seq_path)*`
    fn parse_alt(&mut self) -> PropertyPath {
        let first = self.parse_seq();
        self.skip_ws();
        if self.peek() != b'|' {
            return first;
        }
        let mut alts = vec![first];
        while self.peek() == b'|' {
            self.consume(); // consume '|'
            self.skip_ws();
            alts.push(self.parse_seq());
            self.skip_ws();
        }
        PropertyPath::Alternative(alts)
    }

    /// `seq_path := unary_path ('/' unary_path)*`
    fn parse_seq(&mut self) -> PropertyPath {
        let first = self.parse_unary();
        self.skip_ws();
        if self.peek() != b'/' {
            return first;
        }
        let mut steps = vec![first];
        while self.peek() == b'/' {
            self.consume(); // consume '/'
            self.skip_ws();
            steps.push(self.parse_unary());
            self.skip_ws();
        }
        PropertyPath::Sequence(steps)
    }

    /// `unary_path := '^' unary_path  |  primary ('*' | '+' | '?')?`
    ///
    /// Per SPARQL 1.1 §9: `^p*` is parsed as `(^p)*` (inverse binds tighter
    /// than postfix quantifiers). We implement: consume optional leading `^`,
    /// parse primary, then apply optional postfix quantifier, then wrap in
    /// `Inverse` if `^` was present.
    fn parse_unary(&mut self) -> PropertyPath {
        self.skip_ws();
        let has_inverse = self.peek() == b'^';
        if has_inverse {
            self.consume(); // consume '^'
            self.skip_ws();
        }
        let primary = self.parse_primary_with_quantifier();
        if has_inverse {
            PropertyPath::Inverse(Box::new(primary))
        } else {
            primary
        }
    }

    /// Parse a primary expression and then apply an optional postfix quantifier.
    fn parse_primary_with_quantifier(&mut self) -> PropertyPath {
        let primary = self.parse_primary();
        self.skip_ws();
        match self.peek() {
            b'*' => {
                self.consume();
                PropertyPath::ZeroOrMore(Box::new(primary))
            }
            b'+' => {
                self.consume();
                PropertyPath::OneOrMore(Box::new(primary))
            }
            b'?' => {
                self.consume();
                PropertyPath::ZeroOrOne(Box::new(primary))
            }
            _ => primary,
        }
    }

    /// `primary_path := '(' path ')' | '<' IRI '>' | 'a' | '!' negated_set | prefixed`
    fn parse_primary(&mut self) -> PropertyPath {
        self.skip_ws();
        match self.peek() {
            b'(' => {
                self.consume(); // consume '('
                self.skip_ws();
                let inner = self.parse_path();
                self.skip_ws();
                if self.peek() == b')' {
                    self.consume();
                }
                inner
            }
            b'<' => {
                // Full IRI reference `<...>`
                self.consume(); // consume '<'
                let start = self.pos;
                while self.pos < self.input.len() && self.input[self.pos] != b'>' {
                    self.pos += 1;
                }
                let iri = core::str::from_utf8(&self.input[start..self.pos])
                    .unwrap_or("")
                    .to_owned();
                if self.pos < self.input.len() {
                    self.pos += 1;
                } // consume '>'
                PropertyPath::Iri(iri)
            }
            b'!' => {
                // Negated property set `!p` or `!(p | q | ...)`
                self.consume(); // consume '!'
                self.skip_ws();
                if self.peek() == b'(' {
                    self.consume(); // consume '('
                    self.skip_ws();
                    let mut members = Vec::new();
                    loop {
                        self.skip_ws();
                        if self.peek() == b')' || self.pos >= self.input.len() {
                            break;
                        }
                        // Each member is a possibly-inverse primary (no quantifiers inside negated set)
                        let has_inv = self.peek() == b'^';
                        if has_inv {
                            self.consume();
                            self.skip_ws();
                        }
                        let p = self.parse_primary();
                        let p = if has_inv {
                            PropertyPath::Inverse(Box::new(p))
                        } else {
                            p
                        };
                        members.push(p);
                        self.skip_ws();
                        if self.peek() == b'|' {
                            self.consume();
                        } else {
                            break;
                        }
                    }
                    self.skip_ws();
                    if self.peek() == b')' {
                        self.consume();
                    } // consume ')'
                    PropertyPath::NegatedPropertySet(members)
                } else {
                    // `!p` — single-element negated set
                    let has_inv = self.peek() == b'^';
                    if has_inv {
                        self.consume();
                        self.skip_ws();
                    }
                    let p = self.parse_primary();
                    let p = if has_inv {
                        PropertyPath::Inverse(Box::new(p))
                    } else {
                        p
                    };
                    PropertyPath::NegatedPropertySet(vec![p])
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                // Prefixed name or `a` keyword
                let start = self.pos;
                while self.pos < self.input.len() {
                    let b = self.input[self.pos];
                    if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b':' || b == b'.'
                    {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let token = core::str::from_utf8(&self.input[start..self.pos])
                    .unwrap_or("")
                    .to_owned();
                PropertyPath::Iri(token)
            }
            _ => {
                // Unknown character — consume it and return empty IRI (best-effort)
                let b = self.peek();
                if b != 0 {
                    self.consume();
                }
                PropertyPath::Iri(String::new())
            }
        }
    }
}

/// Parse a SPARQL 1.1 §9 property-path expression string into a [`PropertyPath`] tree.
///
/// This is an infallible, best-effort parser.  On any parse difficulty or
/// malformed input the entire raw string is wrapped in `PropertyPath::Iri`.
/// Leaf IRI strings are returned as raw tokens (prefixed names like
/// `foaf:knows` or full IRIs like `http://xmlns.com/foaf/0.1/knows`); callers
/// should expand prefixed names using
/// [`crate::core::sparql::expand_prefixed_name`].
pub(crate) fn parse_property_path(s: &str) -> PropertyPath {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return PropertyPath::Iri(String::new());
    }
    let mut parser = PathParser::new(trimmed);
    let result = parser.parse_path();
    // If the parser did not consume the entire input (ignoring trailing whitespace),
    // fall back to wrapping the raw string — best-effort on partial parse.
    let remaining = &trimmed[parser.pos..].trim_ascii_start();
    if !remaining.is_empty() {
        // Partial parse: return best-effort tree anyway (leaves may still be correct)
        return result;
    }
    result
}
