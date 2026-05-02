//! Agent action primitives for OxiRouter.
//!
//! Exposes the routing engine as self-describing actions consumable by any
//! agent runtime (oxi-agent, LangChain FFI, OpenAI tool-call protocol, etc.).
//!
//! Enable with the `agent` Cargo feature.

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use crate::context::ContextProvider;
use crate::core::error::{OxiRouterError, Result};
use crate::core::query::Query;
use crate::core::router::Router;

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A self-describing agent action primitive consumable by any agent runtime
/// (oxi-agent, LangChain FFI, custom embedding, OpenAI tool-call protocol).
pub trait AgentAction: Send + Sync {
    /// Returns the action's metadata (name, description, JSON schemas).
    fn meta(&self) -> AgentActionMeta;
    /// Executes the action given a JSON-encoded input; returns a JSON-encoded output.
    fn execute(&mut self, input_json: &str) -> Result<String>;
}

/// Metadata describing an agent action — compatible with OpenAI/Anthropic tool-call protocols.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentActionMeta {
    /// Action name, namespaced: e.g. "oxirouter.route"
    pub name: String,
    /// Human-readable description for LLM consumption.
    pub description: String,
    /// JSON Schema string describing the input.
    pub input_schema: String,
    /// JSON Schema string describing the output.
    pub output_schema: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Action input / output types
// ─────────────────────────────────────────────────────────────────────────────

/// Input for the `oxirouter.route` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInput {
    /// SPARQL query string to route.
    pub query: String,
    /// Maximum number of sources to return (default: all).
    #[serde(default)]
    pub max_sources: Option<usize>,
    /// Optional domain hint (e.g. "biology", "geography").
    #[serde(default)]
    pub domain: Option<String>,
    /// Estimated result count hint.
    #[serde(default)]
    pub expected_results: Option<usize>,
}

/// Output for the `oxirouter.route` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteOutput {
    /// Ranked source selections.
    pub sources: Vec<RouteOutputSource>,
    /// Total number of sources evaluated before `max_sources` truncation.
    pub total_evaluated: usize,
}

/// A single source in the routing output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteOutputSource {
    /// Source identifier.
    pub id: String,
    /// SPARQL endpoint URL.
    pub endpoint: String,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Human-readable reason string (see `reason` constants).
    pub reason: String,
}

/// Input for the `oxirouter.learn` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnInput {
    /// Query identifier from a prior route call (use `Query::predicate_hash()`).
    pub query_id: u64,
    /// Source ID that was used.
    pub source_id: String,
    /// Whether the query executed successfully.
    pub success: bool,
    /// Observed latency in milliseconds.
    pub latency_ms: u32,
    /// Number of results returned.
    pub result_count: u32,
}

/// Output for the `oxirouter.learn` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnOutput {
    /// Whether the outcome was successfully recorded.
    pub recorded: bool,
    /// Total routing decisions sent to this source in the log.
    pub source_total_routed: u64,
    /// Success rate for this source (0.0 – 1.0).
    pub source_success_rate: f64,
    /// Average latency for this source in milliseconds.
    pub source_avg_latency_ms: f64,
}

/// Input for the `oxirouter.explain` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainInput {
    /// SPARQL query to explain routing for.
    pub query: String,
    /// Maximum number of sources to include in the explanation (default: 5).
    #[serde(default)]
    pub max_sources: Option<usize>,
}

/// Output for the `oxirouter.explain` action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainOutput {
    /// Human-readable routing explanation.
    pub explanation: String,
    /// Ranked sources (same as route output, limited by `max_sources`).
    pub ranked_sources: Vec<RouteOutputSource>,
    /// Per-source score component breakdowns (one entry per source).
    pub components: Vec<crate::core::router::RoutingExplanation>,
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON Schema constants
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal-but-valid JSON Schema for `RouteInput`.
pub const ROUTE_INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "SPARQL query string to route"},
    "max_sources": {"type": "integer", "minimum": 1, "description": "Maximum sources to return (default: all)"},
    "domain": {"type": "string", "description": "Optional domain hint (e.g. 'biology', 'geography')"},
    "expected_results": {"type": "integer", "minimum": 0, "description": "Estimated result count hint"}
  },
  "required": ["query"]
}"#;

/// Minimal-but-valid JSON Schema for `RouteOutput`.
pub const ROUTE_OUTPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "endpoint": {"type": "string"},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "reason": {"type": "string"}
        },
        "required": ["id", "endpoint", "confidence", "reason"]
      }
    },
    "total_evaluated": {"type": "integer"}
  },
  "required": ["sources", "total_evaluated"]
}"#;

/// Minimal-but-valid JSON Schema for `LearnInput`.
pub const LEARN_INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "query_id": {"type": "integer", "description": "Query identifier from route output"},
    "source_id": {"type": "string", "description": "Source ID that was used"},
    "success": {"type": "boolean", "description": "Whether the query executed successfully"},
    "latency_ms": {"type": "integer", "minimum": 0, "description": "Observed latency in milliseconds"},
    "result_count": {"type": "integer", "minimum": 0, "description": "Number of results returned"}
  },
  "required": ["query_id", "source_id", "success", "latency_ms", "result_count"]
}"#;

/// Minimal-but-valid JSON Schema for `LearnOutput`.
pub const LEARN_OUTPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "recorded": {"type": "boolean"},
    "source_total_routed": {"type": "integer"},
    "source_success_rate": {"type": "number"},
    "source_avg_latency_ms": {"type": "number"}
  },
  "required": ["recorded", "source_total_routed", "source_success_rate", "source_avg_latency_ms"]
}"#;

/// Minimal-but-valid JSON Schema for `ExplainInput`.
pub const EXPLAIN_INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "SPARQL query to explain routing for"},
    "max_sources": {"type": "integer", "minimum": 1}
  },
  "required": ["query"]
}"#;

/// Minimal-but-valid JSON Schema for `ExplainOutput`.
pub const EXPLAIN_OUTPUT_SCHEMA: &str = r##"{
  "type": "object",
  "properties": {
    "explanation": {"type": "string", "description": "Human-readable routing explanation"},
    "ranked_sources": {
      "type": "array",
      "items": {"$ref": "#/definitions/RouteOutputSource"}
    },
    "components": {
      "type": "array",
      "description": "Per-source score component breakdowns",
      "items": {
        "type": "object",
        "properties": {
          "source_id": {"type": "string"},
          "total_score": {"type": "number"},
          "components": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "weight": {"type": "number"},
                "raw_value": {"type": "number"},
                "contribution": {"type": "number"}
              },
              "required": ["name", "weight", "raw_value", "contribution"]
            }
          }
        },
        "required": ["source_id", "total_score", "components"]
      }
    }
  },
  "required": ["explanation", "ranked_sources", "components"]
}"##;

// ─────────────────────────────────────────────────────────────────────────────
// Reason constants
// ─────────────────────────────────────────────────────────────────────────────

/// Stable string constants for the `reason` field in routing output.
///
/// These map from internal [`SelectionReason`][crate::core::source::SelectionReason]
/// variants to a stable wire-format string. Forward-compatible placeholders are
/// included for future variants.
pub mod reason {
    /// Selected because the source's vocabularies match the query predicates.
    pub const VOCABULARY_MATCH: &str = "vocabulary_match";
    /// Selected by the ML model prediction.
    pub const MODEL_PREDICTION: &str = "model_prediction";
    /// Selected because the source is geographically close to the user.
    pub const GEO_PROXIMITY: &str = "geo_proximity";
    /// Selected based on good historical performance.
    pub const HISTORICAL_PERFORMANCE: &str = "historical_performance";
    /// Selected as a last-resort fallback (no better option available).
    pub const FALLBACK: &str = "fallback";
    /// Selected due to an explicit user preference.
    pub const USER_PREFERENCE: &str = "user_preference";
    /// Selected because compliance/legal constraints require this source.
    pub const COMPLIANCE_REQUIRED: &str = "compliance_required";
    /// Forward-compat: load-aware selection (future feature).
    pub const LOAD_AWARE: &str = "load_aware";
    /// Forward-compat: reinforcement-learning policy (future feature).
    pub const RL_POLICY: &str = "rl_policy";
    /// Forward-compat: P2P source chosen because device is offline (future feature).
    pub const P2P_OFFLINE: &str = "p2p_offline";
    /// Forward-compat: P2P source chosen on low connectivity (future feature).
    pub const P2P_LOW_CONNECTIVITY: &str = "p2p_low_connectivity";
    /// Heuristic default when no stronger signal is available.
    pub const HEURISTIC: &str = "heuristic";
    /// Generic default for unrecognised reasons.
    pub const DEFAULT: &str = "default";
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Maps a [`SelectionReason`][crate::core::source::SelectionReason] to a stable wire string.
fn map_reason(r: &crate::core::source::SelectionReason) -> &'static str {
    use crate::core::source::SelectionReason;
    match r {
        SelectionReason::ModelPrediction => reason::MODEL_PREDICTION,
        SelectionReason::VocabularyMatch => reason::VOCABULARY_MATCH,
        SelectionReason::GeographicProximity => reason::GEO_PROXIMITY,
        SelectionReason::HistoricalPerformance => reason::HISTORICAL_PERFORMANCE,
        SelectionReason::Fallback => reason::FALLBACK,
        SelectionReason::UserPreference => reason::USER_PREFERENCE,
        SelectionReason::ComplianceRequired => reason::COMPLIANCE_REQUIRED,
    }
}

/// Truncates a query string to `max_len` bytes for display purposes.
///
/// # Panics (never)
///
/// Slicing at a non-char boundary would panic, so we use
/// `char_indices` to find the last valid boundary.
fn truncate_query(q: &str, max_len: usize) -> &str {
    if q.len() <= max_len {
        return q;
    }
    // Find the largest char boundary ≤ max_len
    match q.char_indices().take_while(|(i, _)| *i < max_len).last() {
        Some((i, c)) => &q[..i + c.len_utf8()],
        None => "",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RouterAgent
// ─────────────────────────────────────────────────────────────────────────────

/// Agent host that wraps a `Router<C>` and exposes its capabilities as
/// self-describing action primitives.
///
/// Use [`dispatch`][RouterAgent::dispatch] to invoke actions by name.
/// Use [`list_actions`][RouterAgent::list_actions] to enumerate all
/// registered actions for LLM tool registration.
///
/// ## Example
///
/// ```rust,no_run
/// use oxirouter::{Router, RouterAgent};
/// use oxirouter::context::DefaultContextProvider;
///
/// let mut agent: RouterAgent<DefaultContextProvider> = RouterAgent::new(Router::new());
/// let result = agent.dispatch(
///     "oxirouter.route",
///     r#"{"query": "SELECT ?s WHERE { ?s a <http://schema.org/Person> }"}"#,
/// );
/// ```
pub struct RouterAgent<C: ContextProvider> {
    inner: Router<C>,
}

impl<C: ContextProvider> RouterAgent<C> {
    /// Wraps an existing `Router`.
    pub fn new(router: Router<C>) -> Self {
        Self { inner: router }
    }

    /// Consumes the `RouterAgent` and returns the inner `Router`.
    pub fn into_inner(self) -> Router<C> {
        self.inner
    }

    /// Returns a mutable reference to the inner `Router`.
    pub fn router_mut(&mut self) -> &mut Router<C> {
        &mut self.inner
    }

    /// Returns a shared reference to the inner `Router`.
    pub fn router(&self) -> &Router<C> {
        &self.inner
    }

    /// Returns metadata for all registered actions.
    ///
    /// Use this to populate an LLM tool registry.
    pub fn list_actions() -> Vec<AgentActionMeta> {
        vec![
            AgentActionMeta {
                name: "oxirouter.route".to_string(),
                description: "Routes a SPARQL query to the best data source(s) based on context \
                              (vocabulary, geo, device, load, legal, ML model)."
                    .to_string(),
                input_schema: ROUTE_INPUT_SCHEMA.to_string(),
                output_schema: ROUTE_OUTPUT_SCHEMA.to_string(),
            },
            AgentActionMeta {
                name: "oxirouter.learn".to_string(),
                description: "Records the outcome of a routing decision to update the ML model \
                              and RL policy."
                    .to_string(),
                input_schema: LEARN_INPUT_SCHEMA.to_string(),
                output_schema: LEARN_OUTPUT_SCHEMA.to_string(),
            },
            AgentActionMeta {
                name: "oxirouter.explain".to_string(),
                description: "Explains why the router would select certain sources for a given \
                              SPARQL query."
                    .to_string(),
                input_schema: EXPLAIN_INPUT_SCHEMA.to_string(),
                output_schema: EXPLAIN_OUTPUT_SCHEMA.to_string(),
            },
        ]
    }

    /// Dispatches an action by name.
    ///
    /// Returns `Err` when the action name is unknown.
    pub fn dispatch(&mut self, action_name: &str, input_json: &str) -> Result<String> {
        match action_name {
            "oxirouter.route" => self.execute_route(input_json),
            "oxirouter.learn" => self.execute_learn(input_json),
            "oxirouter.explain" => self.execute_explain(input_json),
            other => Err(OxiRouterError::ExecutionError(format!(
                "unknown action: {}",
                other
            ))),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private action implementations
    // ─────────────────────────────────────────────────────────────────────────

    fn execute_route(&mut self, input_json: &str) -> Result<String> {
        let input: RouteInput = serde_json::from_str(input_json)
            .map_err(|e| OxiRouterError::ExecutionError(format!("invalid route input: {}", e)))?;

        let query = {
            #[cfg(feature = "sparql")]
            {
                Query::from_sparql(&input.query).or_else(|_| Query::parse(&input.query))?
            }
            #[cfg(not(feature = "sparql"))]
            {
                Query::parse(&input.query)?
            }
        };
        let max_sources = input.max_sources.unwrap_or(usize::MAX);

        // route_and_log records the decision in the QueryLog so that a
        // subsequent `oxirouter.learn` call can close the feedback loop.
        let ranking = self.inner.route_and_log(&query)?;

        let total_evaluated = ranking.sources.len();
        let sources: Vec<RouteOutputSource> = ranking
            .sources
            .into_iter()
            .take(max_sources)
            .map(|sel| {
                let endpoint = self
                    .inner
                    .get_source(&sel.source_id)
                    .map(|s| s.endpoint.clone())
                    .unwrap_or_default();
                RouteOutputSource {
                    id: sel.source_id,
                    endpoint,
                    confidence: sel.confidence,
                    reason: map_reason(&sel.reason).to_string(),
                }
            })
            .collect();

        let output = RouteOutput {
            sources,
            total_evaluated,
        };
        serde_json::to_string(&output)
            .map_err(|e| OxiRouterError::ExecutionError(format!("serialization error: {}", e)))
    }

    fn execute_learn(&mut self, input_json: &str) -> Result<String> {
        let input: LearnInput = serde_json::from_str(input_json)
            .map_err(|e| OxiRouterError::ExecutionError(format!("invalid learn input: {}", e)))?;

        self.inner.learn_from_outcome(
            input.query_id,
            &input.source_id,
            input.success,
            input.latency_ms,
            input.result_count,
        )?;

        // Read updated stats from the query log.
        let (total_routed, success_rate, avg_latency_ms) = self
            .inner
            .query_log()
            .source_stats(&input.source_id)
            .map(|s| (s.total_routed, s.success_rate(), s.avg_latency_ms()))
            .unwrap_or((0, 0.0, 0.0));

        let output = LearnOutput {
            recorded: true,
            source_total_routed: total_routed,
            source_success_rate: success_rate,
            source_avg_latency_ms: avg_latency_ms,
        };
        serde_json::to_string(&output)
            .map_err(|e| OxiRouterError::ExecutionError(format!("serialization error: {}", e)))
    }

    fn execute_explain(&mut self, input_json: &str) -> Result<String> {
        let input: ExplainInput = serde_json::from_str(input_json)
            .map_err(|e| OxiRouterError::ExecutionError(format!("invalid explain input: {}", e)))?;

        let query = {
            #[cfg(feature = "sparql")]
            {
                Query::from_sparql(&input.query).or_else(|_| Query::parse(&input.query))?
            }
            #[cfg(not(feature = "sparql"))]
            {
                Query::parse(&input.query)?
            }
        };
        let max_sources = input.max_sources.unwrap_or(5);

        // Obtain per-feature explanations for all eligible sources.
        let mut all_explanations = self.inner.explain(&query)?;

        // Sort descending by total_score so ranked_sources mirrors route() order.
        all_explanations.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Build ranked_sources from explanations (capped at max_sources).
        let sources: Vec<RouteOutputSource> = all_explanations
            .iter()
            .take(max_sources)
            .map(|exp| {
                let endpoint = self
                    .inner
                    .get_source(&exp.source_id)
                    .map(|s| s.endpoint.clone())
                    .unwrap_or_default();
                RouteOutputSource {
                    id: exp.source_id.clone(),
                    endpoint,
                    confidence: exp.total_score,
                    reason: crate::agent::reason::HEURISTIC.to_string(),
                }
            })
            .collect();

        // Build a human-readable explanation.
        let source_count = all_explanations.len();
        let truncated_query = truncate_query(&input.query, 80);
        let mut explanation = format!(
            "Ranked {} source(s) for query '{}':\n",
            source_count, truncated_query,
        );
        for (i, src) in sources.iter().enumerate() {
            explanation.push_str(&format!(
                "  {}. {} (score: {:.4})\n",
                i + 1,
                src.id,
                src.confidence,
            ));
        }
        if sources.is_empty() {
            explanation.push_str("  No sources available.");
        }

        let output = ExplainOutput {
            explanation,
            ranked_sources: sources,
            components: all_explanations,
        };
        serde_json::to_string(&output)
            .map_err(|e| OxiRouterError::ExecutionError(format!("serialization error: {}", e)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DefaultContextProvider;
    use crate::core::source::DataSource;

    fn make_router() -> Router<DefaultContextProvider> {
        let mut router = Router::new();
        router.add_source(DataSource::new("dbpedia", "https://dbpedia.org/sparql"));
        router.add_source(DataSource::new(
            "wikidata",
            "https://query.wikidata.org/sparql",
        ));
        router
    }

    #[test]
    fn test_map_reason_all_variants() {
        use crate::core::source::SelectionReason;
        assert_eq!(
            map_reason(&SelectionReason::ModelPrediction),
            "model_prediction"
        );
        assert_eq!(
            map_reason(&SelectionReason::VocabularyMatch),
            "vocabulary_match"
        );
        assert_eq!(
            map_reason(&SelectionReason::GeographicProximity),
            "geo_proximity"
        );
        assert_eq!(
            map_reason(&SelectionReason::HistoricalPerformance),
            "historical_performance"
        );
        assert_eq!(map_reason(&SelectionReason::Fallback), "fallback");
        assert_eq!(
            map_reason(&SelectionReason::UserPreference),
            "user_preference"
        );
        assert_eq!(
            map_reason(&SelectionReason::ComplianceRequired),
            "compliance_required"
        );
    }

    #[test]
    fn test_truncate_query() {
        assert_eq!(truncate_query("hello", 10), "hello");
        assert_eq!(truncate_query("hello world", 5), "hello");
        assert_eq!(truncate_query("", 5), "");
    }

    #[test]
    fn test_list_actions() {
        let actions = RouterAgent::<DefaultContextProvider>::list_actions();
        assert_eq!(actions.len(), 3);
        for a in &actions {
            assert!(a.name.starts_with("oxirouter."));
            assert!(!a.description.is_empty());
            assert!(!a.input_schema.is_empty());
            assert!(!a.output_schema.is_empty());
        }
    }

    #[test]
    fn test_dispatch_unknown_action() {
        let mut agent = RouterAgent::new(make_router());
        let result = agent.dispatch("nonexistent.action", "{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_route_action() {
        let mut agent = RouterAgent::new(make_router());
        let result = agent.dispatch(
            "oxirouter.route",
            r#"{"query": "SELECT ?s WHERE { ?s a <http://schema.org/Person> }"}"#,
        );
        assert!(result.is_ok(), "route failed: {:?}", result);
        let json = result.unwrap();
        let output: RouteOutput = serde_json::from_str(&json).unwrap();
        assert!(!output.sources.is_empty());
        for src in &output.sources {
            assert!(src.confidence >= 0.0 && src.confidence <= 1.0);
            assert!(!src.reason.is_empty());
        }
    }
}
