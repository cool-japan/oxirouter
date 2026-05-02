//! Main routing engine for source selection

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(all(feature = "alloc", feature = "std"))]
use alloc::string::ToString;

use hashbrown::HashMap;

#[cfg(feature = "std")]
use super::error::{OxiRouterError, Result};
use super::query_log::QueryLog;
use super::source::DataSource;
use crate::context::ContextProvider;

#[cfg(feature = "ml")]
use crate::ml::Model;

#[cfg(feature = "rl")]
use crate::rl::Policy;

#[cfg(feature = "cache")]
use crate::cache::CacheManager;

mod cache;
mod federation;
mod operations;
mod routing;

// Re-export public surface from sub-modules
// (All methods are implemented on Router<C> in sub-modules as inherent impls)

/// A single scoring component contributing to a source's total routing score.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreComponent {
    /// Human-readable name (e.g. "vocabulary", "region", "performance").
    pub name: String,
    /// Relative weight of this component in the final score.
    pub weight: f32,
    /// Raw score value before weighting (0.0–1.0).
    pub raw_value: f32,
    /// Contribution = weight × raw_value.
    pub contribution: f32,
}

/// Per-source routing explanation with decomposed score components.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutingExplanation {
    /// Source identifier.
    pub source_id: String,
    /// Final combined score (sum of contributions).
    pub total_score: f32,
    /// Individual score components (one per scoring dimension).
    pub components: Vec<ScoreComponent>,
}

/// Returns default value for `cache_enabled` serde default.
#[cfg(feature = "cache")]
const fn default_cache_enabled() -> bool {
    true
}

/// Returns default value for `cache_ttl_ms` serde default.
#[cfg(feature = "cache")]
const fn default_cache_ttl_ms() -> u64 {
    300_000
}

/// Returns default value for `cache_max_entries` serde default.
#[cfg(feature = "cache")]
const fn default_cache_max_entries() -> usize {
    1000
}

/// Returns default value for `max_response_bytes` serde default.
const fn default_max_response_bytes() -> u64 {
    64 * 1024 * 1024 // 64 MiB
}

/// Returns the default `now_ms` clock function for `CircuitBreakerConfig`.
///
/// Used as the `serde(default)` deserializer target so that JSON blobs
/// written without a `now_ms` field still get a working clock on load.
#[allow(clippy::unnecessary_wraps)]
fn default_now_ms() -> Option<fn() -> u64> {
    #[cfg(feature = "std")]
    {
        Some(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// Configuration for the router-side circuit breaker.
///
/// Sources that fail consecutively `failure_threshold` times are auto-skipped
/// for `cooldown_ms` milliseconds, then automatically probed again.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before a source is tripped.
    /// Set to `0` to disable the circuit breaker entirely.
    pub failure_threshold: u32,
    /// How long (milliseconds) a tripped source stays skipped.
    pub cooldown_ms: u64,
    /// Monotonic clock returning milliseconds since an arbitrary epoch.
    /// `None` means the circuit breaker is disabled (no tripping occurs).
    ///
    /// Function pointers are not serializable; this field is skipped during
    /// JSON encoding and restored from `default_now_ms` on decode.
    #[serde(skip, default = "default_now_ms")]
    pub now_ms: Option<fn() -> u64>,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            cooldown_ms: 30_000,
            #[cfg(feature = "std")]
            now_ms: Some(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0)
            }),
            #[cfg(not(feature = "std"))]
            now_ms: None,
        }
    }
}

/// Configuration for the router
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RouterConfig {
    /// Maximum number of sources to return
    pub max_sources: usize,
    /// Minimum confidence threshold for selection
    pub min_confidence: f32,
    /// Whether to use ML model for routing
    pub use_ml: bool,
    /// Whether to consider context in routing
    pub use_context: bool,
    /// Timeout for source selection (microseconds)
    pub timeout_us: u64,
    /// Weight for historical performance
    pub history_weight: f32,
    /// Weight for vocabulary match
    pub vocab_weight: f32,
    /// Weight for geographic proximity
    pub geo_weight: f32,
    /// Circuit breaker configuration for automatic source failure isolation.
    pub circuit_breaker: CircuitBreakerConfig,
    /// Maximum HTTP response body size in bytes for federated requests.
    /// Default: 64 MiB.
    #[serde(default = "default_max_response_bytes")]
    pub max_response_bytes: u64,
    /// Whether to enable caching (only used when cache feature is enabled)
    #[cfg(feature = "cache")]
    #[serde(default = "default_cache_enabled")]
    pub cache_enabled: bool,
    /// Cache TTL for query results in milliseconds
    #[cfg(feature = "cache")]
    #[serde(default = "default_cache_ttl_ms")]
    pub cache_ttl_ms: u64,
    /// Maximum cache entries
    #[cfg(feature = "cache")]
    #[serde(default = "default_cache_max_entries")]
    pub cache_max_entries: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_sources: 5,
            min_confidence: 0.1,
            use_ml: true,
            use_context: true,
            timeout_us: 100_000, // 100μs target
            history_weight: 0.3,
            vocab_weight: 0.4,
            geo_weight: 0.3,
            circuit_breaker: CircuitBreakerConfig::default(),
            max_response_bytes: default_max_response_bytes(),
            #[cfg(feature = "cache")]
            cache_enabled: true,
            #[cfg(feature = "cache")]
            cache_ttl_ms: 300_000, // 5 minutes
            #[cfg(feature = "cache")]
            cache_max_entries: 1000,
        }
    }
}

#[cfg(feature = "std")]
impl RouterConfig {
    /// Load a `RouterConfig` from a JSON file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`OxiRouterError::InvalidSource`] if the file cannot be read
    /// or if the JSON does not match the `RouterConfig` schema.
    pub fn from_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<RouterConfig> {
        let bytes =
            std::fs::read(path).map_err(|e| OxiRouterError::InvalidSource(e.to_string()))?;
        serde_json::from_slice(&bytes).map_err(|e| OxiRouterError::InvalidSource(e.to_string()))
    }
}

/// The main routing engine for SPARQL federation
pub struct Router<C: ContextProvider = crate::context::DefaultContextProvider> {
    /// Registered data sources
    pub(super) sources: HashMap<String, DataSource>,
    /// Router configuration
    pub(super) config: RouterConfig,
    /// Context provider
    pub(super) context_provider: C,
    /// ML model for source selection
    #[cfg(feature = "ml")]
    pub(super) model: Option<Box<dyn Model>>,
    /// Cached bytes of the loaded model (set by `load_model_from_bytes`, cleared by online training)
    #[cfg(feature = "ml")]
    pub(super) model_bytes_cache: Option<Vec<u8>>,
    /// Whether online training updates are enabled
    #[cfg(feature = "ml")]
    pub(super) online_training_enabled: bool,
    /// RL policy for adaptive source selection
    #[cfg(feature = "rl")]
    pub(super) policy: Option<Policy>,
    /// Query log for statistics-based routing and analytics
    pub(super) query_log: QueryLog,
    /// Cache manager for query results, context, and source info
    #[cfg(feature = "cache")]
    pub(super) cache: CacheManager,
    /// Federated query planner: decomposes BGP into per-source sub-queries.
    #[cfg(feature = "http")]
    pub(super) planner: Box<dyn crate::federation::planner::FederatedPlanner>,
}

impl Router<crate::context::DefaultContextProvider> {
    /// Create a new router with default context provider
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            config: RouterConfig::default(),
            context_provider: crate::context::DefaultContextProvider,
            #[cfg(feature = "ml")]
            model: None,
            #[cfg(feature = "ml")]
            model_bytes_cache: None,
            #[cfg(feature = "ml")]
            online_training_enabled: true,
            #[cfg(feature = "rl")]
            policy: None,
            query_log: QueryLog::new(),
            #[cfg(feature = "cache")]
            cache: CacheManager::new(),
            #[cfg(feature = "http")]
            planner: Box::new(crate::federation::planner::DefaultPlanner::default()),
        }
    }

    /// Create a router by reading a JSON config file from disk.
    ///
    /// Deserializes the file as a [`RouterConfig`] and calls
    /// [`Router::with_config`] with the result.
    ///
    /// # Errors
    ///
    /// Returns [`OxiRouterError::InvalidSource`] if the file cannot be read
    /// or if its contents are not valid `RouterConfig` JSON.
    #[cfg(feature = "std")]
    pub fn with_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let cfg = RouterConfig::from_config_file(path)?;
        Ok(Self::with_config(
            cfg,
            crate::context::DefaultContextProvider,
        ))
    }
}

impl Default for Router<crate::context::DefaultContextProvider> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: ContextProvider> Router<C> {
    /// Create a new router with a custom context provider
    #[must_use]
    pub fn with_context_provider(context_provider: C) -> Self {
        Self {
            sources: HashMap::new(),
            config: RouterConfig::default(),
            context_provider,
            #[cfg(feature = "ml")]
            model: None,
            #[cfg(feature = "ml")]
            model_bytes_cache: None,
            #[cfg(feature = "ml")]
            online_training_enabled: true,
            #[cfg(feature = "rl")]
            policy: None,
            query_log: QueryLog::new(),
            #[cfg(feature = "cache")]
            cache: CacheManager::new(),
            #[cfg(feature = "http")]
            planner: Box::new(crate::federation::planner::DefaultPlanner::default()),
        }
    }

    /// Create a new router with custom configuration
    #[must_use]
    pub fn with_config(config: RouterConfig, context_provider: C) -> Self {
        #[cfg(feature = "cache")]
        let cache = CacheManager::with_config(
            config.cache_max_entries,
            config.cache_ttl_ms,
            crate::cache::DEFAULT_CONTEXT_TTL_MS,
            crate::cache::DEFAULT_SOURCE_TTL_MS,
        );

        Self {
            sources: HashMap::new(),
            config,
            context_provider,
            #[cfg(feature = "ml")]
            model: None,
            #[cfg(feature = "ml")]
            model_bytes_cache: None,
            #[cfg(feature = "ml")]
            online_training_enabled: true,
            #[cfg(feature = "rl")]
            policy: None,
            query_log: QueryLog::new(),
            #[cfg(feature = "cache")]
            cache,
            #[cfg(feature = "http")]
            planner: Box::new(crate::federation::planner::DefaultPlanner::default()),
        }
    }

    /// Add a data source
    pub fn add_source(&mut self, source: DataSource) {
        self.sources.insert(source.id.clone(), source);
    }

    /// Remove a data source
    pub fn remove_source(&mut self, id: &str) -> Option<DataSource> {
        self.sources.remove(id)
    }

    /// Get a data source by ID
    #[must_use]
    pub fn get_source(&self, id: &str) -> Option<&DataSource> {
        self.sources.get(id)
    }

    /// Get a mutable reference to a data source
    #[must_use]
    pub fn get_source_mut(&mut self, id: &str) -> Option<&mut DataSource> {
        self.sources.get_mut(id)
    }

    /// Get all registered sources
    pub fn sources(&self) -> impl Iterator<Item = &DataSource> {
        self.sources.values()
    }

    /// Get the number of registered sources
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::OxiRouterError;
    use crate::core::query::Query;

    #[test]
    fn test_router_creation() {
        let router = Router::new();
        assert_eq!(router.source_count(), 0);
    }

    #[test]
    fn test_add_source() {
        let mut router = Router::new();
        router.add_source(DataSource::new("test", "http://example.com/sparql"));
        assert_eq!(router.source_count(), 1);
        assert!(router.get_source("test").is_some());
    }

    #[test]
    fn test_route_no_sources() {
        let router = Router::new();
        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let result = router.route(&query);
        assert!(matches!(result, Err(OxiRouterError::NoSources { .. })));
    }

    #[test]
    fn test_route_with_sources() {
        let mut router = Router::new();
        router.add_source(
            DataSource::new("dbpedia", "http://dbpedia.org/sparql")
                .with_vocabulary("http://schema.org/"),
        );
        router.add_source(
            DataSource::new("wikidata", "http://wikidata.org/sparql")
                .with_vocabulary("http://www.wikidata.org/"),
        );

        let query = Query::parse(
            "PREFIX schema: <http://schema.org/> SELECT ?s WHERE { ?s a schema:Person }",
        )
        .unwrap();

        let ranking = router.route(&query).unwrap();
        assert!(!ranking.is_empty());
        assert!(ranking.best().is_some());
    }

    #[test]
    fn test_update_stats() {
        let mut router = Router::new();
        router.add_source(DataSource::new("test", "http://example.com/sparql"));

        router.update_source_stats("test", 100, true, 50).unwrap();

        let source = router.get_source("test").unwrap();
        assert_eq!(source.stats.total_queries, 1);
        assert_eq!(source.stats.successful_queries, 1);
    }

    // -------------------------------------------------------------------------
    // Cache Tests (only available with cache feature)
    // -------------------------------------------------------------------------

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_result() {
        let mut router = Router::new();

        // Cache a result
        router.cache_result(12345, "test result".to_string(), "source1".to_string());

        // Retrieve it
        let entry = router.get_cached_result(12345).unwrap();
        assert_eq!(entry.result, "test result");
        assert_eq!(entry.source_id, "source1");
    }

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_miss() {
        let mut router = Router::new();
        assert!(router.get_cached_result(99999).is_none());
    }

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_stats() {
        let mut router = Router::new();

        // Miss
        let _ = router.get_cached_result(1);

        // Hit after caching
        router.cache_result(1, "result".to_string(), "s1".to_string());
        let _ = router.get_cached_result(1);

        let stats = router.query_cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_enable_disable() {
        let mut router = Router::new();

        assert!(router.is_cache_enabled());

        router.disable_cache();
        assert!(!router.is_cache_enabled());

        // Caching should be disabled
        router.cache_result(1, "result".to_string(), "s1".to_string());
        assert!(!router.is_result_cached(1));

        router.enable_cache();
        router.cache_result(1, "result".to_string(), "s1".to_string());
        assert!(router.is_result_cached(1));
    }

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_clear() {
        let mut router = Router::new();

        router.cache_result(1, "r1".to_string(), "s1".to_string());
        router.cache_result(2, "r2".to_string(), "s2".to_string());
        assert_eq!(router.cached_query_count(), 2);

        router.clear_query_cache();
        assert_eq!(router.cached_query_count(), 0);
    }

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_invalidate() {
        let mut router = Router::new();

        router.cache_result(1, "r1".to_string(), "s1".to_string());
        assert!(router.is_result_cached(1));

        router.invalidate_cached_result(1);
        assert!(!router.is_result_cached(1));
    }

    #[cfg(feature = "cache")]
    #[test]
    fn test_cache_context() {
        let mut router = Router::new();
        let ctx = crate::context::CombinedContext::new().with_timestamp(12345);

        router.cache_context("provider1", ctx);

        let cached = router.get_cached_context("provider1").unwrap();
        assert_eq!(cached.timestamp, 12345);
    }
}
