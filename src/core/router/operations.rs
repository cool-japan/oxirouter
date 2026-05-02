//! Router operational methods: logging, learning, source management, state persistence.

#[cfg(feature = "alloc")]
use alloc::{string::ToString, vec::Vec};

#[cfg(all(feature = "alloc", feature = "ml"))]
use alloc::{boxed::Box, format};

use super::{CircuitBreakerConfig, Router, RouterConfig};
use crate::context::ContextProvider;
use crate::core::error::{OxiRouterError, Result};
use crate::core::query::Query;
use crate::core::query_log::QueryLog;
use crate::core::source::SourceRanking;

#[cfg(feature = "ml")]
use crate::ml::{FeatureVector, MergeStrategy, Model, ModelPersistence, ModelState};

#[cfg(feature = "rl")]
use crate::rl::{Policy, Reward};

impl<C: ContextProvider> Router<C> {
    /// Set the ML model for source selection.
    ///
    /// Clears the bytes cache; use [`Router::load_model_from_bytes`] if you need
    /// federated export/merge support.
    #[cfg(feature = "ml")]
    pub fn set_model(&mut self, model: Box<dyn Model>) {
        self.model_bytes_cache = None;
        self.model = Some(model);
    }

    /// Load a model from its serialized bytes.
    ///
    /// The bytes are stored internally so that [`Router::export_weights`] can return them
    /// and [`Router::merge_weights`] can merge with a remote model.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be deserialized as a known model type.
    #[cfg(feature = "ml")]
    pub fn load_model_from_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        use crate::ml::{
            EnsembleClassifier, ModelState, ModelType, NaiveBayesClassifier, NeuralNetwork,
        };

        // Fast path: inspect the ModelState header to dispatch to the correct type.
        // This avoids iterating through all decoders on every load.
        let model: Box<dyn Model> = if let Ok(state) = ModelState::from_bytes(bytes) {
            match state.config.model_type {
                ModelType::NeuralNetwork => NeuralNetwork::from_bytes(bytes)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        OxiRouterError::ModelError(format!("NeuralNetwork deserialization: {e}"))
                    })?,
                ModelType::NaiveBayes => NaiveBayesClassifier::from_bytes(bytes)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        OxiRouterError::ModelError(format!(
                            "NaiveBayesClassifier deserialization: {e}"
                        ))
                    })?,
                ModelType::Ensemble => EnsembleClassifier::from_bytes(bytes)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        OxiRouterError::ModelError(format!(
                            "EnsembleClassifier deserialization: {e}"
                        ))
                    })?,
            }
        } else {
            // Fallback for old blobs without a parseable ModelState header:
            // Try each decoder in turn.
            EnsembleClassifier::from_bytes(bytes)
                .map(|m| Box::new(m) as Box<dyn Model>)
                .or_else(|_| {
                    NaiveBayesClassifier::from_bytes(bytes).map(|m| Box::new(m) as Box<dyn Model>)
                })
                .or_else(|_| {
                    NeuralNetwork::from_bytes(bytes).map(|m| Box::new(m) as Box<dyn Model>)
                })
                .map_err(|e| OxiRouterError::ModelError(format!("load_model_from_bytes: {e}")))?
        };

        self.model = Some(model);
        self.model_bytes_cache = Some(bytes.to_vec());
        Ok(())
    }

    /// Export the current model weights as bytes for federated sharing.
    ///
    /// Only succeeds if the model was loaded via [`Router::load_model_from_bytes`].
    /// If online training has run since the last load, the returned bytes reflect the
    /// *original* loaded bytes (cache was invalidated and export is unavailable).
    ///
    /// # Errors
    ///
    /// Returns an error if no model has been loaded via `load_model_from_bytes`.
    #[cfg(feature = "ml")]
    pub fn export_weights(&self) -> Result<Vec<u8>> {
        match &self.model_bytes_cache {
            Some(bytes) => Ok(bytes.clone()),
            None => Err(OxiRouterError::ModelError(
                "No model bytes available for export. Load a model with load_model_from_bytes() first.".to_string(),
            )),
        }
    }

    /// Merge a remote model's weights into the local model.
    ///
    /// Deserializes the remote bytes into a [`ModelState`], merges into the local model's
    /// state using `strategy`, then reconstructs the model from the merged state.
    /// The bytes cache is invalidated after merge (to reflect updated weights).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No model is currently loaded.
    /// - The remote bytes cannot be deserialized.
    /// - The models are incompatible (type, feature dim, source IDs, weights length).
    #[cfg(feature = "ml")]
    pub fn merge_weights(&mut self, remote_bytes: &[u8], strategy: MergeStrategy) -> Result<()> {
        use crate::ml::{NaiveBayesClassifier, NeuralNetwork, merge_states};

        let remote_state = ModelState::from_bytes(remote_bytes).map_err(|e| {
            OxiRouterError::ModelError(format!("Remote model deserialization: {e}"))
        })?;

        // Reconstruct local model state from the current model
        // We need the model to implement ModelPersistence to get a state.
        // Since `model` is Box<dyn Model>, we do a round-trip through bytes if possible,
        // falling back to an error if not.
        let local_bytes = self.model_bytes_cache.clone().ok_or_else(|| {
            OxiRouterError::ModelError(
                "merge_weights requires a model loaded via load_model_from_bytes()".to_string(),
            )
        })?;

        let mut local_state = ModelState::from_bytes(&local_bytes)
            .map_err(|e| OxiRouterError::ModelError(format!("Local model deserialization: {e}")))?;

        merge_states(&mut local_state, &remote_state, strategy)?;

        // Reconstruct model from merged state
        let merged_bytes = local_state.to_bytes();
        let merged_model: Box<dyn Model> = NaiveBayesClassifier::from_bytes(&merged_bytes)
            .map(|m| Box::new(m) as Box<dyn Model>)
            .or_else(|_| {
                NeuralNetwork::from_bytes(&merged_bytes).map(|m| Box::new(m) as Box<dyn Model>)
            })
            .map_err(|e| OxiRouterError::ModelError(format!("Merged model reconstruction: {e}")))?;

        self.model = Some(merged_model);
        // Invalidate cache — bytes now reflect merged state
        self.model_bytes_cache = Some(merged_bytes);
        Ok(())
    }

    /// Enable or disable online training.
    ///
    /// When enabled (default), [`Router::learn_from_outcome`] will update the ML
    /// model weights using the feature vector stored at routing time.
    #[cfg(feature = "ml")]
    pub fn set_online_training(&mut self, enabled: bool) {
        self.online_training_enabled = enabled;
    }

    /// Whether online training is currently enabled.
    #[cfg(feature = "ml")]
    #[must_use]
    pub fn is_online_training_enabled(&self) -> bool {
        self.online_training_enabled
    }

    // Provide stub implementations when ml feature is disabled so callers using
    // the flag can still compile.
    /// Enable or disable online training (no-op without `ml` feature).
    #[cfg(not(feature = "ml"))]
    pub fn set_online_training(&mut self, _enabled: bool) {}

    /// Whether online training is enabled (always false without `ml` feature).
    #[cfg(not(feature = "ml"))]
    #[must_use]
    pub fn is_online_training_enabled(&self) -> bool {
        false
    }

    /// Set the RL policy for adaptive source selection
    #[cfg(feature = "rl")]
    pub fn set_policy(&mut self, policy: Policy) {
        // Initialize all current sources in the policy
        let source_ids: Vec<String> = self.sources.keys().cloned().collect();
        let mut policy = policy;
        for id in &source_ids {
            policy.initialize_source(id);
        }
        self.policy = Some(policy);
    }

    /// Enable RL with UCB policy (convenience method)
    #[cfg(feature = "rl")]
    pub fn enable_rl(&mut self) {
        let mut policy = Policy::ucb();
        for id in self.sources.keys() {
            policy.initialize_source(id);
        }
        self.policy = Some(policy);
    }

    /// Get a reference to the query log
    #[must_use]
    pub fn query_log(&self) -> &QueryLog {
        &self.query_log
    }

    /// Get a mutable reference to the query log
    pub fn query_log_mut(&mut self) -> &mut QueryLog {
        &mut self.query_log
    }

    /// Return historical stats for a specific source from the query log.
    #[must_use]
    pub fn source_stats(&self, source_id: &str) -> Option<&crate::core::query_log::SourceLogStats> {
        self.query_log.source_stats(source_id)
    }

    /// Return all sources ranked by their log-derived routing score.
    #[must_use]
    pub fn ranked_sources_from_log(&self) -> alloc::vec::Vec<(alloc::string::String, f32)> {
        self.query_log.ranked_sources()
    }

    /// Return the best source according to the query log, if any.
    #[must_use]
    pub fn best_source_from_log(&self) -> Option<alloc::string::String> {
        self.query_log
            .best_source()
            .map(alloc::string::String::from)
    }

    /// Return the total number of entries in the query log.
    #[must_use]
    pub fn query_log_len(&self) -> usize {
        self.query_log.len()
    }

    /// Route a query to the best sources, record in the query log, and apply RL feedback.
    ///
    /// Feature vectors are captured when `ml` is enabled and online training is active,
    /// stored in the query log for later use by [`Router::learn_from_outcome`].
    ///
    /// After execution, call [`Router::learn_from_outcome`] with the actual result to
    /// complete the feedback loop.
    ///
    /// # Errors
    ///
    /// Returns an error if no sources are available or routing fails.
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(skip(self, query), fields(sources_selected = tracing::field::Empty))
    )]
    pub fn route_and_log(&mut self, query: &Query) -> Result<SourceRanking> {
        let ranking = self.route(query)?;

        // Capture feature vector for online training (only when ML is active and enabled)
        #[cfg(feature = "ml")]
        let features: Option<Vec<f32>> = if ranking.ml_used && self.online_training_enabled {
            let context = if self.config.use_context {
                Some(self.context_provider.get_combined_context())
            } else {
                None
            };
            FeatureVector::from_query_and_context(query, context.as_ref())
                .ok()
                .map(|fv| fv.values)
        } else {
            None
        };
        #[cfg(not(feature = "ml"))]
        let features: Option<Vec<f32>> = None;

        // Record routing decisions in the log
        let ml_used = ranking.ml_used;
        for selection in &ranking.sources {
            self.query_log.record_routing(
                query.predicate_hash(),
                &selection.source_id,
                selection.confidence,
                ml_used,
                features.clone(),
            );
        }

        #[cfg(feature = "observability")]
        {
            let source_ids_str = ranking
                .sources
                .iter()
                .map(|s| s.source_id.as_str())
                .collect::<Vec<_>>()
                .join(",");
            tracing::Span::current().record("sources_selected", source_ids_str.as_str());
        }

        Ok(ranking)
    }

    /// Record the outcome of a routed query and update statistics + RL policy.
    ///
    /// When online training is enabled and the model supports it, this also updates
    /// the ML model weights using the feature vector stored at routing time.
    ///
    /// `query_id` should match the `predicate_hash()` of the query used in `route_and_log`.
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self),
            fields(source_id = source_id, success = success, latency_ms = latency_ms)
        )
    )]
    pub fn learn_from_outcome(
        &mut self,
        query_id: u64,
        source_id: &str,
        success: bool,
        latency_ms: u32,
        result_count: u32,
    ) -> Result<()> {
        // Update source stats
        self.update_source_stats(source_id, latency_ms, success, result_count)?;

        // Circuit breaker tracking — pull config values into locals first to
        // avoid simultaneous borrows of `self.sources` and `self.config`.
        let failure_threshold = self.config.circuit_breaker.failure_threshold;
        let cooldown_ms = self.config.circuit_breaker.cooldown_ms;
        let now_fn = self.config.circuit_breaker.now_ms;
        if let Some(source) = self.sources.get_mut(source_id) {
            if success {
                source.stats.consecutive_failures = 0;
                source.stats.tripped_until_ms = None;
            } else {
                source.stats.consecutive_failures =
                    source.stats.consecutive_failures.saturating_add(1);
                if failure_threshold > 0
                    && source.stats.consecutive_failures >= failure_threshold
                    && source.stats.tripped_until_ms.is_none()
                {
                    if let Some(now) = now_fn {
                        source.stats.tripped_until_ms = Some(now().saturating_add(cooldown_ms));
                        #[cfg(feature = "observability")]
                        {
                            metrics::counter!("oxirouter.circuit_breaker.tripped", "source" => source_id.to_string()).increment(1);
                        }
                    }
                }
            }
        }

        // Compute reward
        let reward = if success {
            let latency_penalty = (latency_ms as f32 / 10_000.0).min(1.0) * 0.5;
            (1.0 - latency_penalty).clamp(0.1, 1.0)
        } else {
            0.0
        };

        // Update query log
        self.query_log.record_outcome(
            query_id,
            source_id,
            success,
            latency_ms,
            result_count,
            reward,
        );

        // Update RL policy
        #[cfg(feature = "rl")]
        if let Some(ref mut policy) = self.policy {
            policy.update(source_id, Reward::new(reward));
        }

        // Online training: update ML model weights using stored feature vector
        #[cfg(feature = "ml")]
        if self.online_training_enabled {
            if let Some(ref mut model) = self.model {
                if let Some(features) = self.query_log.find_entry_features(query_id, source_id) {
                    let fv = FeatureVector {
                        values: features,
                        names: Vec::new(),
                    };
                    // Ignore update errors — model may not support online updates
                    let _ = model.update(&fv, source_id, reward);
                    // Invalidate the bytes cache since weights changed
                    self.model_bytes_cache = None;
                }
            }
        }

        Ok(())
    }

    /// Update source statistics after query execution
    pub fn update_source_stats(
        &mut self,
        source_id: &str,
        latency_ms: u32,
        success: bool,
        result_count: u32,
    ) -> Result<()> {
        let source = self
            .sources
            .get_mut(source_id)
            .ok_or_else(|| OxiRouterError::SourceNotFound(source_id.to_string()))?;

        source.update_stats(latency_ms, success, result_count);
        Ok(())
    }

    /// Mark a source as unavailable
    pub fn mark_unavailable(&mut self, source_id: &str) -> Result<()> {
        let source = self
            .sources
            .get_mut(source_id)
            .ok_or_else(|| OxiRouterError::SourceNotFound(source_id.to_string()))?;

        source.available = false;
        Ok(())
    }

    /// Mark a source as available
    pub fn mark_available(&mut self, source_id: &str) -> Result<()> {
        let source = self
            .sources
            .get_mut(source_id)
            .ok_or_else(|| OxiRouterError::SourceNotFound(source_id.to_string()))?;

        source.available = true;
        Ok(())
    }

    /// Get the router configuration
    #[must_use]
    pub const fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get a mutable reference to the router configuration
    pub fn config_mut(&mut self) -> &mut RouterConfig {
        &mut self.config
    }

    /// Replace the circuit breaker configuration.
    ///
    /// Useful for disabling (`failure_threshold = 0`) or tuning the breaker
    /// at runtime without rebuilding the whole router config.
    pub fn set_circuit_breaker_config(&mut self, cfg: CircuitBreakerConfig) {
        self.config.circuit_breaker = cfg;
    }

    /// Serialize the router's full learnable state to bytes (v2 format).
    ///
    /// Captures all registered sources, the trained ML model (if any),
    /// RL policy state (if any), the query log, and the current router
    /// configuration. Pass the resulting bytes to [`Router::load_state`] to
    /// restore the router after a process restart.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization of any component fails.
    #[cfg(feature = "alloc")]
    pub fn save_state(&self) -> Result<alloc::vec::Vec<u8>> {
        use crate::core::state::{RouterState, STATE_VERSION};

        #[cfg(feature = "ml")]
        let model_bytes = self.model.as_ref().map(|m| m.to_bytes());
        #[cfg(not(feature = "ml"))]
        let model_bytes: Option<alloc::vec::Vec<u8>> = None;

        #[cfg(feature = "rl")]
        let rl_bytes = if let Some(policy) = &self.policy {
            let b = serde_json::to_vec(policy)
                .map_err(|e| OxiRouterError::ModelError(e.to_string()))?;
            Some(b)
        } else {
            None
        };
        #[cfg(not(feature = "rl"))]
        let rl_bytes: Option<alloc::vec::Vec<u8>> = None;

        let state = RouterState {
            version: STATE_VERSION,
            sources: self.sources.values().cloned().collect(),
            model_bytes,
            rl_bytes,
            query_log: self.query_log.clone(),
            config: Some(self.config.clone()),
        };
        state.to_bytes()
    }

    /// Restore the router from bytes produced by [`Router::save_state`].
    ///
    /// Replaces sources, model, RL policy, query log, and (when present in
    /// the snapshot) the router configuration.  V1 snapshots that lack a
    /// `config` field leave the current configuration unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`OxiRouterError::IncompatibleModel`] on magic/version mismatch
    /// and [`OxiRouterError::ModelError`] if a component cannot be deserialized.
    #[cfg(feature = "alloc")]
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<()> {
        use crate::core::state::RouterState;

        let state = RouterState::from_bytes(bytes)?;

        // Restore sources
        self.sources.clear();
        for source in state.sources {
            self.add_source(source);
        }

        // Restore model
        #[cfg(feature = "ml")]
        {
            if let Some(mb) = state.model_bytes {
                self.load_model_from_bytes(&mb)?;
            } else {
                self.model = None;
                self.model_bytes_cache = None;
            }
        }

        // Restore RL policy
        #[cfg(feature = "rl")]
        {
            if let Some(rb) = state.rl_bytes {
                let policy: Policy = serde_json::from_slice(&rb)
                    .map_err(|e| OxiRouterError::ModelError(e.to_string()))?;
                self.policy = Some(policy);
            } else {
                self.policy = None;
            }
        }

        // Restore query log
        self.query_log = state.query_log;

        // Restore config when present (v2+); v1 blobs have None, keep current config
        if let Some(cfg) = state.config {
            self.config = cfg;
        }

        Ok(())
    }
}
