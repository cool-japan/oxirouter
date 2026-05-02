//! # OxiRouter - Autonomous Semantic Federation Engine for the Edge
//!
//! OxiRouter is a Pure Rust + WASM library implementing learned source selection
//! for SPARQL federated queries with context-awareness (geo, network, device, legal).
//!
//! ## Features
//!
//! - **Learned Source Selection**: ML-based routing using Naive Bayes and neural networks
//! - **Context-Aware**: Integrates with 4 "brains" (physical, body, situation, social)
//! - **Edge-Ready**: Optimized for constrained environments with no_std support
//! - **WASM Compatible**: Runs in browsers and edge runtimes
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirouter::{Router, DataSource, Query};
//!
//! // Create a router
//! let mut router = Router::new();
//!
//! // Register data sources
//! router.add_source(DataSource::new("dbpedia", "https://dbpedia.org/sparql"));
//! router.add_source(DataSource::new("wikidata", "https://query.wikidata.org/sparql"));
//!
//! // Route a query
//! let query = Query::parse("SELECT ?s WHERE { ?s a <http://schema.org/Person> }").unwrap();
//! let sources = router.route(&query);
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default) - Standard library support
//! - `alloc` (default) - Heap allocation
//! - `no_std` - Embedded/bare-metal environments
//! - `wasm` - WebAssembly bindings
//! - `ml` - Machine learning inference
//! - `rl` - Reinforcement learning feedback
//! - `cache` - Query result and context caching with LRU eviction
//! - `ecosystem` - Full ecosystem integration (4 brains)
//! - `agent` - Agent action primitives for LLM/agent runtime integration

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod context;
pub mod core;
pub mod federation;

#[cfg(feature = "cache")]
pub mod cache;

#[cfg(feature = "ml")]
pub mod ml;

#[cfg(feature = "rl")]
pub mod rl;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "agent")]
pub mod agent;

// Re-export core types
pub use crate::core::error::{OxiRouterError, Result};
pub use crate::core::query::Query;
pub use crate::core::query_log::{QueryLog, RoutingLogEntry, RoutingOutcome, SourceLogStats};
pub use crate::core::router::{
    CircuitBreakerConfig, Router, RouterConfig, RoutingExplanation, ScoreComponent,
};
pub use crate::core::source::{DataSource, SourceKind};

#[cfg(feature = "alloc")]
pub use crate::core::state::RouterState;

// Re-export context types
pub use context::{CombinedContext, ContextProvider, DefaultContextProvider};

#[cfg(feature = "geo")]
pub use context::GeoContext;

#[cfg(any(feature = "device", feature = "std"))]
pub use context::{DeviceContext, DeviceType, NetworkType};

#[cfg(any(feature = "load", feature = "std"))]
pub use context::{CircuitState, LoadContext};

#[cfg(any(feature = "legal", feature = "std"))]
pub use context::LegalContext;

#[cfg(feature = "ecosystem")]
pub use context::EcosystemContextProvider;

// Expose the sensor module so users can write `use oxirouter::sensor::GeoSensor`.
pub use context::sensor;

// Re-export sensor traits and null implementations at crate root for convenience
#[cfg(feature = "geo")]
pub use context::{GeoSensor, NullGeoSensor};

#[cfg(any(feature = "device", feature = "std"))]
pub use context::{DeviceSensor, NullDeviceSensor};

#[cfg(any(feature = "load", feature = "std"))]
pub use context::{LoadSensor, NullLoadSensor};

#[cfg(any(feature = "legal", feature = "std"))]
pub use context::{NullPolicyEngine, PolicyEngine};

// Re-export concrete sensor types
#[cfg(feature = "geo")]
pub use context::{DynamicOxigdalGeoSensor, StaticOxigdalGeoSensor};

#[cfg(feature = "device")]
pub use context::MielinDeviceSensor;

#[cfg(feature = "load")]
pub use context::CelersLoadSensor;

#[cfg(feature = "legal")]
pub use context::LegalisPolicyEngine;

// Re-export federation types
pub use federation::planner::{DefaultPlanner, FederatedPlan, FederatedPlanner, SubPlan};
pub use federation::{AggregatedResult, AggregationStrategy, Aggregator, Executor, QueryResult};

// Re-export ML types when enabled
#[cfg(feature = "ml")]
pub use ml::{
    EnsembleClassifier, FeatureVector, MergeStrategy, Model, ModelConfig, ModelPersistence,
    ModelState, ModelType, NaiveBayesClassifier, NeuralNetwork, merge_states,
};

// Re-export VoID parser when enabled
#[cfg(feature = "void")]
pub use crate::core::void::parse_oxirouter_ttl;

// Re-export SPARQL AST feature type when enabled
#[cfg(feature = "sparql")]
pub use crate::core::sparql_ast::SparqlAstFeatures;

// Re-export structured triple types
pub use crate::core::term::{StructuredTriple, Term};

// Re-export RL types when enabled
#[cfg(feature = "rl")]
pub use rl::{Feedback, Policy, PolicyType, Reward};

// Re-export agent types when enabled
#[cfg(feature = "agent")]
pub use agent::{AgentAction, AgentActionMeta, RouterAgent};

// Re-export cache types when enabled
#[cfg(feature = "cache")]
pub use cache::{
    CacheEntry, CacheManager, CacheStats, CombinedCacheStats, ContextCache, ContextCacheEntry,
    QueryCache, SourceCache, SourceCacheEntry,
};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Prelude module for convenient imports - use `use oxirouter::prelude::*`
pub mod prelude {

    pub use crate::context::{CombinedContext, ContextProvider};
    pub use crate::core::error::{OxiRouterError, Result};
    pub use crate::core::query::Query;
    pub use crate::core::router::{CircuitBreakerConfig, Router, RouterConfig};
    pub use crate::core::source::{DataSource, SourceCapabilities, SourceKind, SourceStats};
    pub use crate::core::term::{StructuredTriple, Term};
    pub use crate::federation::{
        AggregatedResult, AggregationStrategy, Aggregator, Executor, QueryResult,
    };

    #[cfg(feature = "ml")]
    pub use crate::ml::{FeatureVector, Model};

    pub use crate::core::query_log::{QueryLog, SourceLogStats};

    #[cfg(feature = "rl")]
    pub use crate::rl::{Feedback, Policy, PolicyType, Reward};

    #[cfg(feature = "cache")]
    pub use crate::cache::{
        CacheEntry, CacheManager, CacheStats, ContextCache, QueryCache, SourceCache,
    };
}
