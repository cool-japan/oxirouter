//! SPARQL AST pattern parser and feature extraction.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use super::scanner::{
    ScanState, count_blank_nodes, count_keyword_occurrences, count_literals,
    count_path_expressions, find_balanced_end, find_balanced_paren_end, find_keyword_ast,
    has_keyword, heuristic_triple_count,
};
use super::{GraphPattern, SparqlAst, SparqlAstFeatures};

// ─────────────────────────────────────────────────────────────────────────────
// Recursive pattern parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse graph patterns from a WHERE-block content string.
///
/// `depth` is the current nesting level (for depth limiting to avoid stack
/// overflow on pathological inputs). We limit to 32 levels.
fn parse_patterns(content: &str, depth: u32) -> Vec<GraphPattern> {
    const MAX_DEPTH: u32 = 32;
    if depth > MAX_DEPTH {
        return Vec::new();
    }

    let mut patterns: Vec<GraphPattern> = Vec::new();
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    // We scan the content and collect non-nested "flat" regions between blocks.
    // For each recognized keyword we emit the appropriate pattern.
    let mut flat_start = 0usize;

    while i < len {
        // Skip over string literals, URIs, comments to find structural tokens
        match bytes[i] {
            b'"' => {
                // Skip string literal
                i += 1;
                while i < len {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            b'\'' => {
                i += 1;
                while i < len {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'\'' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            b'<' => {
                i += 1;
                while i < len {
                    if bytes[i] == b'>' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            b'#' => {
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'{' => {
                // A nested block — we determine what kind based on the preceding keyword
                let brace_pos = i;
                let block_end = match find_balanced_end(content, brace_pos, '{', '}') {
                    Some(e) => e,
                    None => {
                        i += 1;
                        continue;
                    }
                };
                let inner_content = &content[brace_pos + 1..block_end];

                // Emit flat BGP for any content before this block
                let flat_segment = content[flat_start..brace_pos].trim();
                if !flat_segment.is_empty() {
                    let lits = count_literals(flat_segment);
                    let blanks = count_blank_nodes(flat_segment);
                    let trips = heuristic_triple_count(flat_segment);
                    if trips > 0 || lits > 0 || blanks > 0 {
                        patterns.push(GraphPattern::Bgp {
                            triples: trips,
                            literals: lits,
                            blank_nodes: blanks,
                        });
                    }
                }

                // Determine keyword preceding this block
                let preceding = &content[flat_start..brace_pos];
                let preceding_upper = preceding.to_uppercase();
                let keyword = last_structural_keyword(preceding_upper.trim());

                match keyword {
                    BlockKeyword::Optional => {
                        let inner_patterns = parse_patterns(inner_content, depth + 1);
                        patterns.push(GraphPattern::Optional(inner_patterns));
                    }
                    BlockKeyword::Union => {
                        let right_patterns = parse_patterns(inner_content, depth + 1);
                        let left_patterns = extract_left_union_branch(preceding, depth);
                        patterns.push(GraphPattern::Union(vec![left_patterns, right_patterns]));
                    }
                    BlockKeyword::Service => {
                        let inner_patterns = parse_patterns(inner_content, depth + 1);
                        patterns.push(GraphPattern::Service(inner_patterns));
                    }
                    BlockKeyword::Select => {
                        // Subquery: parse the entire inner block as a sub-AST.
                        let sub_ast = parse_sparql_ast(inner_content);
                        patterns.push(GraphPattern::Subquery(alloc::boxed::Box::new(sub_ast)));
                    }
                    BlockKeyword::Filter => {
                        patterns.push(GraphPattern::Filter);
                    }
                    BlockKeyword::None => {
                        let inner_patterns = parse_patterns(inner_content, depth + 1);
                        patterns.extend(inner_patterns);
                    }
                }

                flat_start = block_end + 1;
                i = flat_start;
            }
            b'(' => {
                // FILTER(...) paren block — skip it entirely
                let paren_end = match find_balanced_paren_end(content, i) {
                    Some(e) => e,
                    None => {
                        i += 1;
                        continue;
                    }
                };

                // Check if preceded by FILTER keyword
                let preceding = content[flat_start..i].trim_end();
                let preceding_upper = preceding.to_ascii_uppercase();
                if preceding_upper.ends_with("FILTER") || preceding_upper.ends_with("NOT EXISTS") {
                    // Emit flat BGP for any content before FILTER keyword
                    let before_filter = strip_trailing_keyword(preceding, "FILTER");
                    if !before_filter.trim().is_empty() {
                        let lits = count_literals(before_filter);
                        let blanks = count_blank_nodes(before_filter);
                        let trips = heuristic_triple_count(before_filter);
                        if trips > 0 || lits > 0 || blanks > 0 {
                            patterns.push(GraphPattern::Bgp {
                                triples: trips,
                                literals: lits,
                                blank_nodes: blanks,
                            });
                        }
                    }
                    patterns.push(GraphPattern::Filter);
                    flat_start = paren_end + 1;
                }
                i = paren_end + 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Final flat segment after all blocks
    let tail = content[flat_start..].trim();
    if !tail.is_empty() {
        if has_keyword(tail, "HAVING") {
            patterns.push(GraphPattern::Having);
        }
        if has_keyword(tail, "GROUP") {
            patterns.push(GraphPattern::GroupBy);
        }
        if has_keyword(tail, "BIND") {
            patterns.push(GraphPattern::Bind);
        }
        if has_keyword(tail, "VALUES") {
            patterns.push(GraphPattern::Values);
        }

        let lits = count_literals(tail);
        let blanks = count_blank_nodes(tail);
        let trips = heuristic_triple_count(tail);
        if trips > 0 || lits > 0 || blanks > 0 {
            patterns.push(GraphPattern::Bgp {
                triples: trips,
                literals: lits,
                blank_nodes: blanks,
            });
        }
    }

    patterns
}

/// The recognized structural keyword types that precede a `{...}` block.
#[derive(PartialEq)]
enum BlockKeyword {
    Optional,
    Union,
    Service,
    Select,
    Filter,
    None,
}

/// Given the text *preceding* a `{` block (already uppercased), determine what
/// structural keyword it ends with.
fn last_structural_keyword(preceding_upper: &str) -> BlockKeyword {
    let trimmed = preceding_upper.trim_end();
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if let Some(&last) = words.last() {
        match last {
            "OPTIONAL" => return BlockKeyword::Optional,
            "FILTER" => return BlockKeyword::Filter,
            _ => {}
        }
    }
    if let [.., second_last, last] = words.as_slice() {
        if *second_last == "UNION" || *last == "UNION" {
            return BlockKeyword::Union;
        }
        if (*second_last == "SERVICE" || (*last).starts_with('<'))
            && words.iter().rev().skip(1).any(|w| *w == "SERVICE")
        {
            return BlockKeyword::Service;
        }
        if *last == "WHERE" && words.contains(&"SELECT") {
            return BlockKeyword::Select;
        }
    }
    if trimmed.ends_with("UNION") {
        return BlockKeyword::Union;
    }
    if has_keyword(preceding_upper, "SERVICE") {
        return BlockKeyword::Service;
    }
    BlockKeyword::None
}

/// Extract the left branch of a UNION by finding the last `{...}` block in `preceding`.
fn extract_left_union_branch(preceding: &str, depth: u32) -> Vec<GraphPattern> {
    let bytes = preceding.as_bytes();
    let len = bytes.len();

    let mut last_brace_start: Option<usize> = None;
    let mut i = 0;
    let mut state = ScanState::Normal;

    while i < len {
        match state {
            ScanState::Normal => match bytes[i] {
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
                b'{' => {
                    last_brace_start = Some(i);
                    if let Some(end) = find_balanced_end(preceding, i, '{', '}') {
                        i = end + 1;
                    } else {
                        i += 1;
                    }
                }
                _ => {
                    i += 1;
                }
            },
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

    if let Some(brace_start) = last_brace_start {
        if let Some(brace_end) = find_balanced_end(preceding, brace_start, '{', '}') {
            let inner = &preceding[brace_start + 1..brace_end];
            return parse_patterns(inner, depth + 1);
        }
    }
    Vec::new()
}

/// Strip a trailing keyword (case-insensitive) from a string,
/// returning the part before it.
fn strip_trailing_keyword<'a>(text: &'a str, keyword: &str) -> &'a str {
    let upper = text.to_ascii_uppercase();
    if let Some(pos) = upper.rfind(&keyword.to_ascii_uppercase()) {
        &text[..pos]
    } else {
        text
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API: AST parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a SPARQL query string into a best-effort [`SparqlAst`].
///
/// This function is **infallible** — it never returns an error.  On any parse
/// difficulty it returns a partial or default AST.
pub(crate) fn parse_sparql_ast(sparql: &str) -> SparqlAst {
    let mut ast = SparqlAst::default();

    // 1. Detect DISTINCT / REDUCED in SELECT clause
    if let Some(sel_pos) = find_keyword_ast(sparql, "SELECT", 0) {
        let after_select = &sparql[sel_pos + 6..];
        let trimmed = after_select.trim_start();
        let trimmed_upper = trimmed.to_ascii_uppercase();
        if trimmed_upper.starts_with("DISTINCT") {
            let after = trimmed_upper.trim_start_matches("DISTINCT");
            if after.is_empty()
                || !after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
            {
                ast.has_distinct = true;
            }
        } else if trimmed_upper.starts_with("REDUCED") {
            let after = trimmed_upper.trim_start_matches("REDUCED");
            if after.is_empty()
                || !after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
            {
                ast.has_distinct = true;
                ast.has_reduced = true;
            }
        }
    }

    // 2. Find the WHERE clause body
    let where_content: Option<&str> = find_where_content(sparql);

    // 3. Parse graph patterns from WHERE content
    if let Some(content) = where_content {
        ast.patterns = parse_patterns(content, 0);
    }

    // 4. Detect HAVING in the full query
    ast.has_having = has_keyword(sparql, "HAVING");

    // 5. COUNT GROUP BY / ORDER BY
    ast.group_by_count = count_keyword_occurrences(sparql, "GROUP");
    ast.order_by_count = count_keyword_occurrences(sparql, "ORDER");

    // 6. Detect LIMIT
    ast.has_limit = has_keyword(sparql, "LIMIT");

    // 7. Count property paths in the full query
    if let Some(content) = where_content {
        ast.path_count = count_path_expressions(content);
    }

    ast
}

/// Find the WHERE clause body content (the text inside the outermost `{...}`
/// following the WHERE keyword).  Returns `None` if not found.
fn find_where_content(sparql: &str) -> Option<&str> {
    if let Some(where_pos) = find_keyword_ast(sparql, "WHERE", 0) {
        let after_where = &sparql[where_pos + 5..];
        let brace_offset = after_where.find('{')?;
        let abs_brace = where_pos + 5 + brace_offset;
        let close = find_balanced_end(sparql, abs_brace, '{', '}')?;
        return Some(&sparql[abs_brace + 1..close]);
    }
    let first_brace = sparql.find('{')?;
    let close = find_balanced_end(sparql, first_brace, '{', '}')?;
    Some(&sparql[first_brace + 1..close])
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────────────

struct Counter {
    max_depth: u32,
    optional_count: u32,
    filter_count: u32,
    union_branches: u32,
    subquery_count: u32,
    triple_count: u32,
    literal_count: u32,
    blank_count: u32,
}

impl Counter {
    fn new() -> Self {
        Self {
            max_depth: 0,
            optional_count: 0,
            filter_count: 0,
            union_branches: 0,
            subquery_count: 0,
            triple_count: 0,
            literal_count: 0,
            blank_count: 0,
        }
    }
}

fn count_patterns(patterns: &[GraphPattern], depth: u32, c: &mut Counter) {
    c.max_depth = c.max_depth.max(depth);
    for pat in patterns {
        match pat {
            GraphPattern::Bgp {
                triples,
                literals,
                blank_nodes,
            } => {
                c.triple_count += triples;
                c.literal_count += literals;
                c.blank_count += blank_nodes;
            }
            GraphPattern::Optional(inner) => {
                c.optional_count += 1;
                count_patterns(inner, depth + 1, c);
            }
            GraphPattern::Union(branches) => {
                c.union_branches += branches.len() as u32;
                for b in branches {
                    count_patterns(b, depth + 1, c);
                }
            }
            GraphPattern::Filter => {
                c.filter_count += 1;
            }
            GraphPattern::Subquery(_sub) => {
                c.subquery_count += 1;
            }
            GraphPattern::Service(inner) => {
                count_patterns(inner, depth, c);
            }
            GraphPattern::GroupBy
            | GraphPattern::Having
            | GraphPattern::Bind
            | GraphPattern::Values => {}
        }
    }
}

/// Extract [`SparqlAstFeatures`] from a parsed [`SparqlAst`].
pub(crate) fn extract_ast_features(ast: &SparqlAst) -> SparqlAstFeatures {
    let mut c = Counter::new();
    count_patterns(&ast.patterns, 0, &mut c);

    SparqlAstFeatures {
        join_depth: (c.max_depth as f32 / 10.0).min(1.0),
        optional_count: (c.optional_count as f32 / 10.0).min(1.0),
        filter_count: (c.filter_count as f32 / 10.0).min(1.0),
        union_branch_count: (c.union_branches as f32 / 10.0).min(1.0),
        has_distinct: if ast.has_distinct || ast.has_reduced {
            1.0
        } else {
            0.0
        },
        has_having: if ast.has_having { 1.0 } else { 0.0 },
        subquery_count: (c.subquery_count as f32 / 5.0).min(1.0),
        path_expr_count: (ast.path_count as f32 / 10.0).min(1.0),
        literal_count: (c.literal_count as f32 / 20.0).min(1.0),
        blank_node_count: (c.blank_count as f32 / 10.0).min(1.0),
    }
}

/// Parse and extract AST features in one step.
///
/// This is the primary entry point for the ML feature pipeline.
pub(crate) fn compute_ast_features(sparql: &str) -> SparqlAstFeatures {
    extract_ast_features(&parse_sparql_ast(sparql))
}
