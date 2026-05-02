//! Error types for OxiRouter

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use core::fmt;

/// Result type alias for OxiRouter operations
pub type Result<T> = core::result::Result<T, OxiRouterError>;

/// Error type for OxiRouter operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OxiRouterError {
    /// Query parsing error
    QueryParse(String),
    /// Invalid source configuration
    InvalidSource(String),
    /// No sources available for routing
    NoSources {
        /// Human-readable explanation of why no sources are available.
        reason: String,
        /// Vocabulary IRIs that were required but not matched by any source.
        missing_vocabularies: Vec<String>,
    },
    /// Source not found
    SourceNotFound(String),
    /// Context provider error
    ContextError(String),
    /// ML model error
    ModelError(String),
    /// Federation execution error
    ExecutionError(String),
    /// Feature extraction error
    FeatureError(String),
    /// Timeout error with full context for operator diagnostics.
    Timeout {
        /// The data source (or router-level) identifier where the timeout occurred.
        source_id: String,
        /// Name of the operation that timed out (e.g. `"route"`, `"execute"`, `"model_predict"`).
        operation: String,
        /// Milliseconds elapsed when the timeout was detected.
        elapsed_ms: u64,
        /// Configured deadline in milliseconds.
        deadline_ms: u64,
    },
    /// HTTP response body exceeded the configured size limit.
    ResponseTooLarge {
        /// Source that returned the oversized response.
        source_id: String,
        /// Approximate bytes observed before aborting.
        observed_bytes: u64,
        /// Configured limit in bytes.
        limit_bytes: u64,
    },
    /// Legal/compliance constraint violation
    ComplianceViolation(String),
    /// Model weight incompatibility during federated merge
    IncompatibleModel {
        /// Reason for incompatibility
        reason: String,
    },
    /// Feature vector dimension does not match the model's expected dimension.
    FeatureDimMismatch {
        /// Expected feature dimension
        expected: usize,
        /// Actual feature dimension supplied
        found: usize,
    },
}

impl fmt::Display for OxiRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueryParse(msg) => write!(f, "Query parse error: {msg}"),
            Self::InvalidSource(msg) => write!(f, "Invalid source: {msg}"),
            Self::NoSources {
                reason,
                missing_vocabularies,
            } => {
                write!(f, "No sources available for routing: {reason}")?;
                if !missing_vocabularies.is_empty() {
                    write!(
                        f,
                        " (missing vocabularies: {})",
                        missing_vocabularies.join(", ")
                    )?;
                }
                Ok(())
            }
            Self::SourceNotFound(id) => write!(f, "Source not found: {id}"),
            Self::ContextError(msg) => write!(f, "Context error: {msg}"),
            Self::ModelError(msg) => write!(f, "Model error: {msg}"),
            Self::ExecutionError(msg) => write!(f, "Execution error: {msg}"),
            Self::FeatureError(msg) => write!(f, "Feature extraction error: {msg}"),
            Self::Timeout {
                source_id,
                operation,
                elapsed_ms,
                deadline_ms,
            } => write!(
                f,
                "Operation {operation} on source '{source_id}' timed out: \
                 {elapsed_ms}ms elapsed, {deadline_ms}ms deadline"
            ),
            Self::ResponseTooLarge {
                source_id,
                observed_bytes,
                limit_bytes,
            } => write!(
                f,
                "Response from '{source_id}' too large: {observed_bytes} bytes observed, \
                 limit is {limit_bytes} bytes"
            ),
            Self::ComplianceViolation(msg) => write!(f, "Compliance violation: {msg}"),
            Self::IncompatibleModel { reason } => write!(f, "Incompatible model: {reason}"),
            Self::FeatureDimMismatch { expected, found } => write!(
                f,
                "Feature dimension mismatch: model expects {expected}, got {found}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OxiRouterError {}
