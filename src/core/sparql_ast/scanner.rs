//! Low-level byte-scanning helpers for SPARQL text analysis.
//!
//! All functions skip string literals, URI references, and line comments
//! to avoid false-positive matches inside quoted content.

// ─────────────────────────────────────────────────────────────────────────────
// Lexical scanner state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ScanState {
    Normal,
    InDoubleQuote,
    InSingleQuote,
    InUri,
    InComment,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scanning helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the byte offset of a whole-word occurrence of `keyword` (case-insensitive)
/// in `text`, starting at `from`, skipping string literals and line comments.
///
/// Returns `Some(abs_offset)` or `None`.
pub(super) fn find_keyword_ast(text: &str, keyword: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let klen = keyword.len();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = from;

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
                _ => {
                    if i + klen <= len {
                        let chunk = &text[i..i + klen];
                        if chunk.eq_ignore_ascii_case(keyword) {
                            let before_ok = i == 0
                                || !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_';
                            let after_pos = i + klen;
                            let after_ok = after_pos >= len
                                || (!bytes[after_pos].is_ascii_alphanumeric()
                                    && bytes[after_pos] != b'_');
                            if before_ok && after_ok {
                                return Some(i);
                            }
                        }
                    }
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

    None
}

/// Convenience: search from position 0.
pub(super) fn has_keyword(text: &str, keyword: &str) -> bool {
    find_keyword_ast(text, keyword, 0).is_some()
}

/// Find the matching close char for an opening char at `start` in `s`.
/// Returns the index of the closing char if balanced, otherwise `None`.
///
/// Handles nesting, and skips string literals and comments for `{`/`}`.
pub(super) fn find_balanced_end(s: &str, start: usize, open: char, close: char) -> Option<usize> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if start >= len || bytes[start] != open as u8 {
        return None;
    }
    let mut depth: u32 = 1;
    let mut state = ScanState::Normal;
    let mut i = start + 1;

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
                c => {
                    if c == open as u8 {
                        depth += 1;
                    } else if c == close as u8 {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i);
                        }
                    }
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

    None
}

/// Find the matching close paren for an opening paren.
pub(super) fn find_balanced_paren_end(s: &str, start: usize) -> Option<usize> {
    find_balanced_end(s, start, '(', ')')
}

/// Count occurrences of a keyword in `text` from position `from`,
/// skipping literals/comments/URIs.
pub(super) fn count_keyword_occurrences(text: &str, keyword: &str) -> u32 {
    let mut count = 0u32;
    let mut search_from = 0;
    while let Some(pos) = find_keyword_ast(text, keyword, search_from) {
        count += 1;
        search_from = pos + keyword.len();
    }
    count
}

/// Count property path tokens: sequences like `*/`, `+/`, `?/`, `^`, or `|`
/// used as path operators in the query text (outside strings/URIs).
pub(super) fn count_path_expressions(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;
    let mut count = 0u32;

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
                    b'|' | b'^' => {
                        count += 1;
                        i += 1;
                    }
                    b'/' => {
                        count += 1;
                        i += 1;
                    }
                    b'*' | b'+' | b'?' => {
                        // Path quantifiers when preceded by '>' or identifier char or ')'
                        if i > 0 {
                            let prev = bytes[i - 1];
                            if prev == b'>'
                                || prev == b')'
                                || prev.is_ascii_alphanumeric()
                                || prev == b'_'
                            {
                                count += 1;
                            }
                        }
                        i += 1;
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

    count
}

/// Count literal objects in a text segment (strings starting with `"` or `'`).
pub(super) fn count_literals(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;
    let mut count = 0u32;

    while i < len {
        match state {
            ScanState::Normal => match bytes[i] {
                b'"' => {
                    count += 1;
                    state = ScanState::InDoubleQuote;
                    i += 1;
                }
                b'\'' => {
                    count += 1;
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

    count
}

/// Count blank node tokens (`_:xxx` or `[]`) in a text segment.
pub(super) fn count_blank_nodes(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;
    let mut count = 0u32;

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
                b'[' if i + 1 < len && bytes[i + 1] == b']' => {
                    count += 1;
                    i += 2;
                }
                b'[' => {
                    i += 1;
                }
                b'_' if i + 1 < len && bytes[i + 1] == b':' => {
                    count += 1;
                    i += 2;
                    while i < len
                        && (bytes[i].is_ascii_alphanumeric()
                            || bytes[i] == b'_'
                            || bytes[i] == b'-')
                    {
                        i += 1;
                    }
                }
                b'_' => {
                    i += 1;
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

    count
}

/// Rough triple count heuristic: count `.` separators not inside
/// string literals/URIs/comments. Returns 0 for empty content.
pub(super) fn heuristic_triple_count(content: &str) -> u32 {
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut state = ScanState::Normal;
    let mut i = 0;
    let mut dots = 0u32;

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
                b'.' => {
                    dots += 1;
                    i += 1;
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

    if dots == 0 && bytes.iter().any(|b| !b.is_ascii_whitespace()) {
        1
    } else {
        dots
    }
}
