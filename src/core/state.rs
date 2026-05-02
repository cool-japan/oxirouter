//! Router state snapshot — enables save/load across process restarts.
//!
//! Wire format (v2):
//!   [0..4]  magic = b"OXIR"
//!   [4..8]  version (u32 little-endian); v1 = 1, v2 = 2
//!   [8..]   JSON-encoded RouterStateBody
//!
//! Backwards compatibility: v1 blobs are accepted by `from_bytes` because
//! the `config` field carries `#[serde(default)]` and is simply absent from
//! v1 JSON.

use alloc::{format, string::ToString, vec::Vec};

use serde::{Deserialize, Serialize};

use crate::core::error::{OxiRouterError, Result};
use crate::core::query_log::QueryLog;
use crate::core::router::RouterConfig;
use crate::core::source::DataSource;

/// Magic bytes for the `RouterState` wire format.
pub(crate) const STATE_MAGIC: [u8; 4] = *b"OXIR";

/// Previous wire-format version, still accepted on load.
pub(crate) const STATE_VERSION_V1: u32 = 1;

/// Current version of the `RouterState` wire format.
pub(crate) const STATE_VERSION: u32 = 2;

/// Serializable snapshot of a [`Router`](crate::core::router::Router)'s full learnable state.
///
/// The snapshot captures all registered data sources, trained ML model bytes,
/// RL policy state, the query log, and (since v2) the router configuration.
/// Use [`RouterState::to_bytes`] to encode and [`RouterState::from_bytes`] to restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterState {
    /// Wire-format version written into this snapshot.
    pub(crate) version: u32,
    /// All registered data sources.
    pub(crate) sources: Vec<DataSource>,
    /// Serialized ML model bytes (NaiveBayes + NeuralNetwork + Ensemble), if trained.
    pub(crate) model_bytes: Option<Vec<u8>>,
    /// Serialized RL policy state, if enabled.
    pub(crate) rl_bytes: Option<Vec<u8>>,
    /// Query log snapshot.
    pub(crate) query_log: QueryLog,
    /// Router configuration (v2+).  `None` when loaded from a v1 blob.
    #[serde(default)]
    pub(crate) config: Option<RouterConfig>,
}

impl RouterState {
    /// Return the router configuration stored in this snapshot, if any.
    ///
    /// Returns `None` for states loaded from v1 blobs that predate the config field.
    #[must_use]
    pub fn config(&self) -> Option<&RouterConfig> {
        self.config.as_ref()
    }

    /// Encode `self` to the `OXIR`-prefixed wire format (v2).
    ///
    /// The output is:
    /// - 4 bytes magic (`OXIR`)
    /// - 4 bytes version (little-endian `u32`, always `STATE_VERSION`)
    /// - JSON body (UTF-8)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let body =
            serde_json::to_vec(self).map_err(|e| OxiRouterError::ModelError(e.to_string()))?;
        let mut out = Vec::with_capacity(8 + body.len());
        out.extend_from_slice(&STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Decode from `OXIR`-prefixed bytes.
    ///
    /// Accepts both v1 and v2 wire formats.  V1 blobs parse cleanly because
    /// the `config` field defaults to `None` via `#[serde(default)]`.
    ///
    /// Returns [`OxiRouterError::IncompatibleModel`] on magic mismatch or
    /// an unrecognised version, and [`OxiRouterError::ModelError`] on JSON
    /// parse failure.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(OxiRouterError::IncompatibleModel {
                reason: "state snapshot too short".into(),
            });
        }
        if bytes[0..4] != STATE_MAGIC {
            return Err(OxiRouterError::IncompatibleModel {
                reason: "invalid magic bytes (expected OXIR)".into(),
            });
        }
        let ver = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if ver != STATE_VERSION_V1 && ver != STATE_VERSION {
            return Err(OxiRouterError::IncompatibleModel {
                reason: format!(
                    "state version mismatch: expected {STATE_VERSION_V1} or {STATE_VERSION}, found {ver}"
                ),
            });
        }
        serde_json::from_slice(&bytes[8..]).map_err(|e| OxiRouterError::ModelError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    use alloc::vec;

    fn make_state() -> RouterState {
        RouterState {
            version: STATE_VERSION,
            sources: Vec::new(),
            model_bytes: None,
            rl_bytes: None,
            query_log: QueryLog::new(),
            config: None,
        }
    }

    #[test]
    fn round_trip_empty() {
        let state = make_state();
        let bytes = state.to_bytes().expect("encode failed");
        // Check magic
        assert_eq!(&bytes[0..4], b"OXIR");
        // Check version
        let ver = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(ver, STATE_VERSION);
        // Round-trip
        let restored = RouterState::from_bytes(&bytes).expect("decode failed");
        assert_eq!(restored.version, STATE_VERSION);
        assert!(restored.sources.is_empty());
        assert!(restored.model_bytes.is_none());
        assert!(restored.rl_bytes.is_none());
    }

    #[test]
    fn rejects_short_bytes() {
        let err = RouterState::from_bytes(&[0u8; 4]).unwrap_err();
        assert!(
            matches!(err, OxiRouterError::IncompatibleModel { .. }),
            "expected IncompatibleModel, got {err:?}"
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(b"BAAD");
        let err = RouterState::from_bytes(&bytes).unwrap_err();
        assert!(
            matches!(err, OxiRouterError::IncompatibleModel { .. }),
            "expected IncompatibleModel, got {err:?}"
        );
    }

    #[test]
    fn rejects_wrong_version() {
        let state = make_state();
        let mut bytes = state.to_bytes().expect("encode failed");
        // Overwrite version field with 99
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = RouterState::from_bytes(&bytes).unwrap_err();
        assert!(
            matches!(err, OxiRouterError::IncompatibleModel { .. }),
            "expected IncompatibleModel, got {err:?}"
        );
    }

    #[test]
    fn round_trip_with_data() {
        use crate::core::source::DataSource;

        let mut state = make_state();
        state
            .sources
            .push(DataSource::new("test-src", "https://example.org/sparql"));
        state.model_bytes = Some(vec![1, 2, 3, 4]);
        state.rl_bytes = Some(vec![5, 6, 7, 8]);

        let bytes = state.to_bytes().expect("encode failed");
        let restored = RouterState::from_bytes(&bytes).expect("decode failed");

        assert_eq!(restored.sources.len(), 1);
        assert_eq!(restored.sources[0].id, "test-src");
        assert_eq!(restored.model_bytes, Some(vec![1, 2, 3, 4]));
        assert_eq!(restored.rl_bytes, Some(vec![5, 6, 7, 8]));
    }
}
