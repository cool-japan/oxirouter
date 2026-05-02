//! Federated query planner: BGP decomposition into per-source sub-queries.
//!
//! The [`DefaultPlanner`] assigns each BGP triple to the best-matching source
//! based on vocabulary namespace coverage. Sources without a vocabulary match
//! are scored by reliability (`success_rate`). When `structured_triples` is
//! empty the planner falls back to dispatching the full query to the top-N
//! sources ordered by success rate.

#[cfg(feature = "alloc")]
use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use hashbrown::HashMap;

use crate::core::error::{OxiRouterError, Result};
use crate::core::query::{Query, QueryType};
use crate::core::source::DataSource;
use crate::core::term::{StructuredTriple, Term};

/// A plan for executing a federated query: one sub-query per source.
#[derive(Debug)]
pub struct FederatedPlan {
    /// Ordered sub-plans (highest confidence first).
    pub sub_plans: Vec<SubPlan>,
    /// `true` when BGP decomposition was skipped because `structured_triples`
    /// was empty and the full query was dispatched to top-N sources instead.
    pub fallback_used: bool,
}

/// A sub-query assigned to a specific source.
#[derive(Debug)]
pub struct SubPlan {
    /// Identifier of the target source.
    pub source_id: String,
    /// BGP triples assigned to this source (empty in fallback mode).
    pub triples: Vec<StructuredTriple>,
    /// Reconstructed SPARQL sub-query covering only `triples`.
    pub sub_query: Query,
    /// Average confidence score for this sub-plan (0.0–1.0).
    pub confidence: f32,
}

/// Trait for federated query planners.
pub trait FederatedPlanner: Send + Sync {
    /// Decompose `q` across `sources` into a [`FederatedPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiRouterError::NoSources`] when a BGP triple's predicate
    /// cannot be matched to any source with confidence ≥ `min_triple_confidence`.
    fn plan(&self, q: &Query, sources: &[DataSource]) -> Result<FederatedPlan>;
}

// ─────────────────────────────────────────────────────────────────────────────
// DefaultPlanner
// ─────────────────────────────────────────────────────────────────────────────

/// Default greedy per-triple planner.
///
/// Each BGP triple is independently assigned to the source whose vocabulary
/// namespace best matches the triple's predicate IRI. All triples routed to
/// the same source are then grouped into a single [`SubPlan`].
pub struct DefaultPlanner {
    /// Minimum triple-to-source confidence score (0.0–1.0).
    ///
    /// Triples for which no source reaches this threshold cause a
    /// [`OxiRouterError::NoSources`] error.
    pub min_triple_confidence: f32,
    /// Top-N sources to use in fallback mode (when `structured_triples` is empty).
    pub fallback_top_n: usize,
}

impl Default for DefaultPlanner {
    fn default() -> Self {
        Self {
            min_triple_confidence: 0.3,
            fallback_top_n: 3,
        }
    }
}

impl FederatedPlanner for DefaultPlanner {
    fn plan(&self, q: &Query, sources: &[DataSource]) -> Result<FederatedPlan> {
        if sources.is_empty() {
            return Err(OxiRouterError::NoSources {
                reason: "No sources registered".to_string(),
                missing_vocabularies: vec![],
            });
        }

        if q.structured_triples.is_empty() {
            return Ok(self.plan_fallback(q, sources));
        }

        self.plan_bgp_decomposition(q, sources)
    }
}

impl DefaultPlanner {
    // ── Fallback path ──────────────────────────────────────────────────────

    /// Dispatch the full query to the top-N sources ordered by success rate.
    fn plan_fallback(&self, q: &Query, sources: &[DataSource]) -> FederatedPlan {
        let mut ranked: Vec<&DataSource> = sources.iter().collect();
        ranked.sort_by(|a, b| {
            b.stats
                .success_rate
                .partial_cmp(&a.stats.success_rate)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let sub_plans: Vec<SubPlan> = ranked
            .iter()
            .take(self.fallback_top_n)
            .map(|s| SubPlan {
                source_id: s.id.clone(),
                triples: Vec::new(),
                sub_query: q.clone(),
                confidence: s.stats.success_rate,
            })
            .collect();

        FederatedPlan {
            sub_plans,
            fallback_used: true,
        }
    }

    // ── BGP decomposition path ─────────────────────────────────────────────

    /// Greedy per-triple assignment: each triple goes to the best-matching source.
    fn plan_bgp_decomposition(&self, q: &Query, sources: &[DataSource]) -> Result<FederatedPlan> {
        // BTreeMap gives deterministic sub_plan ordering within the same source.
        let mut triple_assignments: BTreeMap<String, Vec<StructuredTriple>> = BTreeMap::new();
        let mut source_confidences: BTreeMap<String, Vec<f32>> = BTreeMap::new();

        for triple in &q.structured_triples {
            let (best_source_id, best_score) = self.score_triple(triple, sources, q);

            if best_score < self.min_triple_confidence {
                let pred_label = predicate_display_name(&triple.predicate);
                return Err(OxiRouterError::NoSources {
                    reason: format!("No source covers predicate {pred_label}"),
                    missing_vocabularies: vec![pred_label],
                });
            }

            triple_assignments
                .entry(best_source_id.clone())
                .or_default()
                .push(triple.clone());

            source_confidences
                .entry(best_source_id)
                .or_default()
                .push(best_score);
        }

        let mut sub_plans: Vec<SubPlan> = triple_assignments
            .into_iter()
            .map(|(source_id, triples)| {
                let confidence = source_confidences
                    .get(&source_id)
                    .map(|scores| scores.iter().copied().sum::<f32>() / scores.len() as f32)
                    .unwrap_or(0.0_f32);

                let sub_query = reconstruct_query(q, &triples);

                SubPlan {
                    source_id,
                    triples,
                    sub_query,
                    confidence,
                }
            })
            .collect();

        // Highest confidence first.
        sub_plans.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        Ok(FederatedPlan {
            sub_plans,
            fallback_used: false,
        })
    }

    // ── Scoring ────────────────────────────────────────────────────────────

    /// Compute `(best_source_id, best_score)` for a single BGP triple.
    ///
    /// Score 1.0  → source vocabulary namespace matches the predicate IRI.
    /// Score 0.0  → predicate IRI present but no vocabulary match.
    /// Fallback   → variable/unknown predicate: `success_rate × 0.5`.
    fn score_triple(
        &self,
        triple: &StructuredTriple,
        sources: &[DataSource],
        q: &Query,
    ) -> (String, f32) {
        let pred_iri = resolve_predicate_iri(&triple.predicate, &q.prefixes);

        let mut best_id = String::new();
        let mut best_score = 0.0_f32;

        for source in sources {
            let score = match &pred_iri {
                Some(iri) => {
                    // Vocabulary namespace match: full 1.0 if any registered
                    // vocabulary is a prefix of the predicate IRI.
                    let namespace = namespace_of(iri);
                    if source
                        .vocabularies
                        .iter()
                        .any(|v| v == namespace || iri.starts_with(v.as_str()))
                    {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                }
                None => {
                    // Variable / unknown predicate: rank by historical reliability.
                    source.stats.success_rate * 0.5_f32
                }
            };

            // Tie-break: prefer the source that already has the highest score.
            // When no source scores > 0 and there is only one source, it still wins.
            if score > best_score || (best_id.is_empty() && sources.len() == 1) {
                best_score = score;
                best_id.clone_from(&source.id);
            }
        }

        // Secondary fallback: if no source reached threshold, pick the most
        // reliable one and score it by success_rate * 0.5 (variable-predicate logic).
        if best_score < self.min_triple_confidence {
            if let Some(most_reliable) = sources.iter().max_by(|a, b| {
                a.stats
                    .success_rate
                    .partial_cmp(&b.stats.success_rate)
                    .unwrap_or(core::cmp::Ordering::Equal)
            }) {
                let reliability_score = most_reliable.stats.success_rate * 0.5_f32;
                if reliability_score >= self.min_triple_confidence {
                    return (most_reliable.id.clone(), reliability_score);
                }
                // Still below threshold — return best we have (caller will reject).
                if best_id.is_empty() {
                    return (most_reliable.id.clone(), reliability_score);
                }
            }
        }

        (best_id, best_score)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Query reconstruction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct a minimal `SELECT * WHERE { ... }` query covering `triples`.
fn reconstruct_query(original: &Query, triples: &[StructuredTriple]) -> Query {
    let triple_strs: Vec<String> = triples
        .iter()
        .map(|t| {
            format!(
                "{} {} {} .",
                term_to_sparql(&t.subject),
                term_to_sparql(&t.predicate),
                term_to_sparql(&t.object),
            )
        })
        .collect();

    let raw = format!("SELECT * WHERE {{ {} }}", triple_strs.join(" "));

    // Build a predicate set from the reconstructed triples.
    let mut predicates = original.predicates.clone();
    for triple in triples {
        if let Some(iri) = resolve_predicate_iri(&triple.predicate, &original.prefixes) {
            predicates.insert(iri);
        }
    }

    Query {
        raw,
        query_type: QueryType::Select,
        triple_patterns: Vec::new(),
        structured_triples: triples.to_vec(),
        predicates,
        prefixes: original.prefixes.clone(),
        // Carry over all other fields from the original so that capability
        // checks (sparql_1_1, etc.) still work correctly.
        ..original.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Term utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Render a [`Term`] as a SPARQL token.
fn term_to_sparql(term: &Term) -> String {
    match term {
        Term::Variable(v) => format!("?{v}"),
        Term::Iri(iri) => format!("<{iri}>"),
        Term::PrefixedName(p, l) => format!("{p}:{l}"),
        Term::Literal(raw) => raw.clone(),
        Term::BlankNode(label) => format!("_:{label}"),
    }
}

/// Attempt to resolve a predicate [`Term`] to a fully-qualified IRI string.
///
/// Returns `None` for variables, blank nodes, and bare literals
/// (which cannot be used as predicate IRIs in SPARQL).
fn resolve_predicate_iri(term: &Term, prefixes: &HashMap<String, String>) -> Option<String> {
    match term {
        Term::Iri(iri) => Some(iri.clone()),
        Term::PrefixedName(p, l) => {
            if let Some(base) = prefixes.get(p.as_str()) {
                Some(format!("{base}{l}"))
            } else {
                // Unknown prefix — return the raw `p:l` form so callers can
                // still attempt namespace matching.
                Some(format!("{p}:{l}"))
            }
        }
        // Variable, Literal, BlankNode — not resolvable to a concrete IRI.
        Term::Variable(_) | Term::Literal(_) | Term::BlankNode(_) => None,
    }
}

/// Extract the namespace portion of an IRI (up to and including the last `#` or `/`).
fn namespace_of(iri: &str) -> &str {
    iri.rfind('#')
        .or_else(|| iri.rfind('/'))
        .map(|i| &iri[..=i])
        .unwrap_or(iri)
}

/// Return a human-readable display name for a predicate term (used in error messages).
fn predicate_display_name(term: &Term) -> String {
    match term {
        Term::Iri(iri) => iri.clone(),
        Term::PrefixedName(p, l) => format!("{p}:{l}"),
        Term::Variable(v) => format!("?{v}"),
        Term::Literal(raw) => raw.clone(),
        Term::BlankNode(label) => format!("_:{label}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::source::DataSource;

    fn foaf_source() -> DataSource {
        DataSource::new("foaf-source", "http://localhost:9999/sparql")
            .with_vocabulary("http://xmlns.com/foaf/0.1/")
    }

    fn make_query_with_triples(triples: Vec<StructuredTriple>) -> Query {
        let mut q = Query::parse("SELECT * WHERE { ?s ?p ?o }").expect("parse");
        q.structured_triples = triples;
        q
    }

    #[test]
    fn test_term_to_sparql_variable() {
        assert_eq!(term_to_sparql(&Term::Variable("s".to_string())), "?s");
    }

    #[test]
    fn test_term_to_sparql_iri() {
        assert_eq!(
            term_to_sparql(&Term::Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            "<http://xmlns.com/foaf/0.1/name>"
        );
    }

    #[test]
    fn test_term_to_sparql_prefixed() {
        assert_eq!(
            term_to_sparql(&Term::PrefixedName("foaf".to_string(), "name".to_string())),
            "foaf:name"
        );
    }

    #[test]
    fn test_namespace_of_hash() {
        assert_eq!(
            namespace_of("http://xmlns.com/foaf/0.1/name"),
            "http://xmlns.com/foaf/0.1/"
        );
    }

    #[test]
    fn test_namespace_of_slash() {
        assert_eq!(
            namespace_of("http://purl.org/dc/terms/title"),
            "http://purl.org/dc/terms/"
        );
    }

    #[test]
    fn test_fallback_empty_triples() {
        let planner = DefaultPlanner::default();
        let q = Query::parse("SELECT * WHERE { ?s ?p ?o }").expect("parse");
        let sources = vec![foaf_source()];
        let plan = planner.plan(&q, &sources).expect("plan");
        assert!(plan.fallback_used);
        assert_eq!(plan.sub_plans.len(), 1);
    }

    #[test]
    fn test_bgp_routes_to_matching_source() {
        let planner = DefaultPlanner::default();
        let triple = StructuredTriple {
            subject: Term::Variable("s".to_string()),
            predicate: Term::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: Term::Variable("name".to_string()),
        };
        let q = make_query_with_triples(vec![triple]);
        let sources = vec![foaf_source()];
        let plan = planner.plan(&q, &sources).expect("plan");
        assert!(!plan.fallback_used);
        assert_eq!(plan.sub_plans.len(), 1);
        assert_eq!(plan.sub_plans[0].source_id, "foaf-source");
    }

    #[test]
    fn test_reconstruct_query_raw_starts_with_select() {
        let triples = vec![StructuredTriple {
            subject: Term::Variable("s".to_string()),
            predicate: Term::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: Term::Variable("name".to_string()),
        }];
        let original = Query::parse("SELECT * WHERE { ?s ?p ?o }").expect("parse");
        let reconstructed = reconstruct_query(&original, &triples);
        assert!(reconstructed.raw.starts_with("SELECT * WHERE {"));
        assert!(reconstructed.raw.contains("?s"));
    }
}
