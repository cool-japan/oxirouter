//! Data source definitions and management

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Discriminates the transport/protocol kind of a data source.
///
/// The `kind` field drives P2P-aware heuristic scoring when the `p2p`
/// feature is enabled: offline or poor-connectivity contexts boost P2P
/// sources over conventional SPARQL endpoints.
///
/// # Backward compatibility
///
/// Older serialised `DataSource` values without a `kind` field deserialise to
/// [`SourceKind::Sparql`] via `#[serde(default)]` on the containing struct.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum SourceKind {
    /// A conventional SPARQL endpoint (default).
    #[default]
    Sparql,
    /// A local or remote result cache.
    Cache,
    /// An IPFS node identified by a multiaddr string.
    ///
    /// Example multiaddr: `/ip4/127.0.0.1/tcp/4001/p2p/QmTest`.
    P2pIpfs {
        /// IPFS multiaddr of the target node.
        multiaddr: String,
    },
    /// A libp2p peer identified by its peer-id string.
    P2pLibp2p {
        /// libp2p peer identifier (multihash / CIDv0/CIDv1 string).
        peer_id: String,
    },
    /// Application-defined source kind.
    Custom(String),
}

impl SourceKind {
    /// Returns `true` for any P2P source kind ([`P2pIpfs`][SourceKind::P2pIpfs]
    /// or [`P2pLibp2p`][SourceKind::P2pLibp2p]).
    #[must_use]
    pub fn is_p2p(&self) -> bool {
        matches!(self, Self::P2pIpfs { .. } | Self::P2pLibp2p { .. })
    }

    /// Returns `true` only for the [`Cache`][SourceKind::Cache] variant.
    #[must_use]
    pub fn is_cache(&self) -> bool {
        matches!(self, Self::Cache)
    }
}

/// A SPARQL data source endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Unique identifier for this source
    pub id: String,
    /// SPARQL endpoint URL
    pub endpoint: String,
    /// Source capabilities
    pub capabilities: SourceCapabilities,
    /// Performance statistics
    pub stats: SourceStats,
    /// Geographic region (ISO 3166-1 alpha-2 country codes)
    pub regions: SmallVec<[String; 4]>,
    /// Supported vocabularies/ontologies
    pub vocabularies: HashSet<String>,
    /// Whether the source is currently available
    pub available: bool,
    /// Priority weight (higher = preferred)
    pub priority: f32,
    /// Transport/protocol kind of this source.
    ///
    /// Absent in v1 serialised data; deserialises to [`SourceKind::Sparql`]
    /// via the `#[serde(default)]` attribute.
    #[serde(default)]
    pub kind: SourceKind,
}

impl DataSource {
    /// Create a new data source with default settings
    #[must_use]
    pub fn new(id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            capabilities: SourceCapabilities::default(),
            stats: SourceStats::default(),
            regions: SmallVec::new(),
            vocabularies: HashSet::new(),
            available: true,
            priority: 1.0,
            kind: SourceKind::default(),
        }
    }

    /// Builder method to set the source kind (transport/protocol).
    ///
    /// # Example
    ///
    /// ```
    /// use oxirouter::{DataSource, SourceKind};
    ///
    /// let source = DataSource::new("ipfs-node", "/ip4/127.0.0.1/tcp/4001/p2p/QmTest")
    ///     .with_kind(SourceKind::P2pIpfs {
    ///         multiaddr: "/ip4/127.0.0.1/tcp/4001/p2p/QmTest".into(),
    ///     });
    /// assert!(source.kind.is_p2p());
    /// ```
    #[must_use]
    pub fn with_kind(mut self, kind: SourceKind) -> Self {
        self.kind = kind;
        self
    }

    /// Builder method to add a region
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.regions.push(region.into());
        self
    }

    /// Builder method to add a vocabulary
    #[must_use]
    pub fn with_vocabulary(mut self, vocab: impl Into<String>) -> Self {
        self.vocabularies.insert(vocab.into());
        self
    }

    /// Builder method to set capabilities
    #[must_use]
    pub const fn with_capabilities(mut self, capabilities: SourceCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Builder method to set priority
    #[must_use]
    pub const fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority;
        self
    }

    /// Update statistics after a query execution
    pub fn update_stats(&mut self, latency_ms: u32, success: bool, result_count: u32) {
        self.stats.total_queries += 1;
        if success {
            self.stats.successful_queries += 1;
        }
        self.stats.total_results += u64::from(result_count);

        // Update running average latency
        let n = self.stats.total_queries as f32;
        self.stats.avg_latency_ms = ((n - 1.0) * self.stats.avg_latency_ms + latency_ms as f32) / n;

        // Update success rate
        self.stats.success_rate =
            self.stats.successful_queries as f32 / self.stats.total_queries as f32;
    }

    /// Check if source supports a given vocabulary
    #[must_use]
    pub fn supports_vocabulary(&self, vocab: &str) -> bool {
        self.vocabularies.contains(vocab)
    }

    /// Check if source is in a given region
    #[must_use]
    pub fn in_region(&self, region: &str) -> bool {
        self.regions.iter().any(|r| r == region)
    }
}

/// Capabilities of a SPARQL data source
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct SourceCapabilities {
    /// Supports SPARQL 1.1
    pub sparql_1_1: bool,
    /// Supports federated queries (SERVICE keyword)
    pub federation: bool,
    /// Supports CONSTRUCT queries
    pub construct: bool,
    /// Supports ASK queries
    pub ask: bool,
    /// Supports DESCRIBE queries
    pub describe: bool,
    /// Supports property paths
    pub property_paths: bool,
    /// Supports aggregation (GROUP BY, COUNT, etc.)
    pub aggregation: bool,
    /// Supports subqueries
    pub subqueries: bool,
    /// Supports BIND
    pub bind: bool,
    /// Supports VALUES
    pub values: bool,
    /// Maximum results per query (0 = unlimited)
    pub max_results: u32,
    /// Supports full-text search
    pub full_text_search: bool,
    /// Supports geospatial queries
    pub geospatial: bool,
}

impl SourceCapabilities {
    /// Create capabilities for a basic SPARQL 1.0 endpoint
    #[must_use]
    pub const fn basic() -> Self {
        Self {
            sparql_1_1: false,
            federation: false,
            construct: true,
            ask: true,
            describe: true,
            property_paths: false,
            aggregation: false,
            subqueries: false,
            bind: false,
            values: false,
            max_results: 10000,
            full_text_search: false,
            geospatial: false,
        }
    }

    /// Create capabilities for a full SPARQL 1.1 endpoint
    #[must_use]
    pub const fn full() -> Self {
        Self {
            sparql_1_1: true,
            federation: true,
            construct: true,
            ask: true,
            describe: true,
            property_paths: true,
            aggregation: true,
            subqueries: true,
            bind: true,
            values: true,
            max_results: 0,
            full_text_search: false,
            geospatial: false,
        }
    }
}

/// Performance statistics for a data source
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct SourceStats {
    /// Total number of queries executed
    pub total_queries: u32,
    /// Number of successful queries
    pub successful_queries: u32,
    /// Total results returned across all queries
    pub total_results: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
    /// Last query timestamp (Unix epoch seconds)
    pub last_query_time: u64,
    /// Number of consecutive failures since the last success.
    #[serde(default)]
    pub consecutive_failures: u32,
    /// Millisecond timestamp after which this source is no longer circuit-tripped.
    /// `None` means the source is not tripped.
    #[serde(default)]
    pub tripped_until_ms: Option<u64>,
}

impl SourceStats {
    /// Check if the source has any historical data
    #[must_use]
    pub const fn has_history(&self) -> bool {
        self.total_queries > 0
    }

    /// Get the average results per query
    #[must_use]
    pub fn avg_results_per_query(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.total_results as f64 / self.total_queries as f64
        }
    }
}

/// Selection result for a source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelection {
    /// Source ID
    pub source_id: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: u32,
    /// Reason for selection
    pub reason: SelectionReason,
}

/// Reason why a source was selected
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SelectionReason {
    /// Selected by ML model inference
    ModelPrediction,
    /// Selected based on vocabulary coverage
    VocabularyMatch,
    /// Selected based on geographic proximity
    GeographicProximity,
    /// Selected based on historical performance
    HistoricalPerformance,
    /// Selected as fallback (no better option)
    Fallback,
    /// Selected by explicit user preference
    UserPreference,
    /// Selected due to legal/compliance requirements
    ComplianceRequired,
}

/// A ranked list of source selections
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRanking {
    /// Ranked sources (highest confidence first)
    pub sources: Vec<SourceSelection>,
    /// Total processing time in microseconds
    pub processing_time_us: u64,
    /// Whether ML model was used
    pub ml_used: bool,
    /// Whether context was considered
    pub context_used: bool,
}

impl SourceRanking {
    /// Create a new empty ranking
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sources: Vec::new(),
            processing_time_us: 0,
            ml_used: false,
            context_used: false,
        }
    }

    /// Add a source selection
    pub fn add(&mut self, selection: SourceSelection) {
        self.sources.push(selection);
    }

    /// Sort by confidence (descending)
    pub fn sort_by_confidence(&mut self) {
        self.sources.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }

    /// Get the top N sources
    #[must_use]
    pub fn top(&self, n: usize) -> &[SourceSelection] {
        let end = n.min(self.sources.len());
        &self.sources[..end]
    }

    /// Get the best source (if any)
    #[must_use]
    pub fn best(&self) -> Option<&SourceSelection> {
        self.sources.first()
    }

    /// Check if any sources were selected
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Get the number of selected sources
    #[must_use]
    pub fn len(&self) -> usize {
        self.sources.len()
    }
}
