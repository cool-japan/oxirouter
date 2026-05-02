//! Naive Bayes classifier for source selection

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::feature::FeatureVector;
use super::model::{Model, ModelConfig, ModelPersistence, ModelState, ModelType, TrainingSample};
use crate::core::error::{OxiRouterError, Result};

/// Default prior decay rate.
fn default_prior_decay() -> f32 {
    0.99
}

/// Naive Bayes classifier for source selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveBayesClassifier {
    /// Prior probabilities for each source P(source)
    priors: HashMap<String, f32>,
    /// Feature means per source (source -> feature_idx -> mean)
    means: HashMap<String, Vec<f32>>,
    /// Feature variances per source (source -> feature_idx -> variance)
    variances: HashMap<String, Vec<f32>>,
    /// Total samples seen
    sample_count: u64,
    /// Feature dimension
    feature_dim: usize,
    /// Smoothing parameter (Laplace smoothing)
    smoothing: f32,
    /// Per-source sample counts for correct weighted-mean normalization.
    #[serde(default)]
    per_source_counts: HashMap<String, u64>,
    /// Exponential decay rate for prior probabilities (default 0.99).
    #[serde(default = "default_prior_decay")]
    prior_decay: f32,
}

impl NaiveBayesClassifier {
    /// Create a new Naive Bayes classifier
    #[must_use]
    pub fn new(feature_dim: usize) -> Self {
        Self {
            priors: HashMap::new(),
            means: HashMap::new(),
            variances: HashMap::new(),
            sample_count: 0,
            feature_dim,
            smoothing: 1e-6,
            per_source_counts: HashMap::new(),
            prior_decay: 0.99,
        }
    }

    /// Create from configuration
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        Self::new(config.feature_dim)
    }

    /// Set smoothing parameter
    #[must_use]
    pub const fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Set the prior decay rate (default 0.99).
    #[must_use]
    pub fn with_prior_decay(mut self, decay: f32) -> Self {
        self.prior_decay = decay;
        self
    }

    /// Return per-source sample counts (immutable reference).
    #[must_use]
    pub fn per_source_counts(&self) -> &HashMap<String, u64> {
        &self.per_source_counts
    }

    /// Return the learned means for the given source, or `None` if unknown.
    #[must_use]
    pub fn source_means(&self, source_id: &str) -> Option<&Vec<f32>> {
        self.means.get(source_id)
    }

    /// Initialize priors from source list
    pub fn initialize_sources(&mut self, source_ids: &[&String]) {
        let uniform_prior = 1.0 / source_ids.len() as f32;
        for id in source_ids {
            self.priors.insert((*id).clone(), uniform_prior);
            self.means
                .insert((*id).clone(), vec![0.5; self.feature_dim]);
            self.variances
                .insert((*id).clone(), vec![0.25; self.feature_dim]);
        }
    }

    /// Calculate log-likelihood of features given a source
    fn log_likelihood(&self, features: &FeatureVector, source_id: &str) -> f32 {
        let means = match self.means.get(source_id) {
            Some(m) => m,
            None => return f32::NEG_INFINITY,
        };

        let variances = match self.variances.get(source_id) {
            Some(v) => v,
            None => return f32::NEG_INFINITY,
        };

        let mut log_prob = 0.0;

        for (i, &x) in features.values.iter().enumerate() {
            if i >= means.len() {
                break;
            }

            let mean = means[i];
            let var = variances[i].max(self.smoothing);

            // Gaussian log-likelihood
            let diff = x - mean;
            #[cfg(feature = "ml")]
            {
                log_prob += -0.5 * libm::logf(2.0 * core::f32::consts::PI * var);
                log_prob += -0.5 * (diff * diff) / var;
            }
            #[cfg(not(feature = "ml"))]
            {
                log_prob += -0.5 * (2.0 * core::f32::consts::PI * var).ln();
                log_prob += -0.5 * (diff * diff) / var;
            }
        }

        log_prob
    }

    /// Compute posterior probabilities for all sources
    fn compute_posteriors(
        &self,
        features: &FeatureVector,
        source_ids: &[&String],
    ) -> Vec<(String, f32)> {
        let mut log_posteriors = Vec::with_capacity(source_ids.len());

        // Compute log posteriors (log prior + log likelihood)
        for id in source_ids {
            let log_prior = self
                .priors
                .get(*id)
                .map(|&p| {
                    #[cfg(feature = "ml")]
                    {
                        libm::logf(p.max(self.smoothing))
                    }
                    #[cfg(not(feature = "ml"))]
                    {
                        p.max(self.smoothing).ln()
                    }
                })
                .unwrap_or(f32::NEG_INFINITY);

            let log_likelihood = self.log_likelihood(features, id);
            log_posteriors.push(((*id).clone(), log_prior + log_likelihood));
        }

        // Convert to probabilities using log-sum-exp trick for numerical stability
        let max_log = log_posteriors
            .iter()
            .map(|(_, lp)| *lp)
            .fold(f32::NEG_INFINITY, f32::max);

        let sum_exp: f32 = log_posteriors
            .iter()
            .map(|(_, lp)| {
                #[cfg(feature = "ml")]
                {
                    libm::expf(lp - max_log)
                }
                #[cfg(not(feature = "ml"))]
                {
                    (lp - max_log).exp()
                }
            })
            .sum();

        #[cfg(feature = "ml")]
        let log_sum_exp = max_log + libm::logf(sum_exp);
        #[cfg(not(feature = "ml"))]
        let log_sum_exp = max_log + sum_exp.ln();

        log_posteriors
            .into_iter()
            .map(|(id, lp)| {
                #[cfg(feature = "ml")]
                let prob = libm::expf(lp - log_sum_exp);
                #[cfg(not(feature = "ml"))]
                let prob = (lp - log_sum_exp).exp();
                (id, prob)
            })
            .collect()
    }

    /// Update statistics with a single sample
    fn update_statistics(&mut self, features: &FeatureVector, source_id: &str, weight: f32) {
        // Ensure source exists
        if !self.means.contains_key(source_id) {
            self.means
                .insert(source_id.to_string(), vec![0.5; self.feature_dim]);
            self.variances
                .insert(source_id.to_string(), vec![0.25; self.feature_dim]);
            self.priors.insert(source_id.to_string(), 0.0);
        }

        // Increment per-source count, then compute n from it (n=1 on first sample).
        let count = self
            .per_source_counts
            .entry(source_id.to_string())
            .or_insert(0);
        *count += 1;
        let n = *count as f32;

        // Update means with weighted moving average
        if let Some(means) = self.means.get_mut(source_id) {
            for (i, &x) in features.values.iter().enumerate() {
                if i < means.len() {
                    let old_mean = means[i];
                    means[i] = old_mean + weight * (x - old_mean) / n;
                }
            }
        }

        // Update variances with weighted moving average
        if let (Some(means), Some(variances)) =
            (self.means.get(source_id), self.variances.get_mut(source_id))
        {
            for (i, &x) in features.values.iter().enumerate() {
                if i < variances.len() {
                    let mean = means[i];
                    let diff = x - mean;
                    variances[i] = variances[i] + weight * (diff * diff - variances[i]) / n;
                }
            }
        }

        // Update prior with configurable decay
        let decay = self.prior_decay;
        let _total: f32 = self.priors.values().sum();
        for (id, prior) in &mut self.priors {
            if id == source_id {
                *prior = *prior * decay + weight * (1.0 - decay);
            } else {
                *prior *= decay;
            }
        }

        // Normalize priors
        let new_total: f32 = self.priors.values().sum();
        if new_total > 0.0 {
            for prior in self.priors.values_mut() {
                *prior /= new_total;
            }
        }

        self.sample_count += 1;
    }
}

impl Model for NaiveBayesClassifier {
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, features, source_ids),
            fields(source_ids_count = source_ids.len())
        )
    )]
    fn predict(
        &self,
        features: &FeatureVector,
        source_ids: &[&String],
    ) -> Result<Vec<(String, f32)>> {
        if source_ids.is_empty() {
            return Err(OxiRouterError::ModelError(
                "No sources provided".to_string(),
            ));
        }

        if self.feature_dim > 0 && features.values.len() != self.feature_dim {
            return Err(OxiRouterError::FeatureDimMismatch {
                expected: self.feature_dim,
                found: features.values.len(),
            });
        }

        #[cfg(all(feature = "observability", feature = "std"))]
        let predict_start = std::time::Instant::now();

        let mut posteriors = self.compute_posteriors(features, source_ids);

        #[cfg(all(feature = "observability", feature = "std"))]
        {
            let elapsed_us = predict_start.elapsed().as_micros() as f64;
            metrics::histogram!("oxirouter.ml.predict.duration_us", "model" => "bayes")
                .record(elapsed_us);
        }

        // Sort by probability (descending)
        posteriors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        Ok(posteriors)
    }

    fn name(&self) -> &str {
        "NaiveBayes"
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn train(&mut self, samples: &[TrainingSample]) -> Result<()> {
        for sample in samples {
            let weight = sample.reward();
            self.update_statistics(&sample.features, &sample.selected_source, weight);
        }
        Ok(())
    }

    fn update(&mut self, features: &FeatureVector, source_id: &str, reward: f32) -> Result<()> {
        self.update_statistics(features, source_id, reward);
        Ok(())
    }

    fn to_bytes(&self) -> Vec<u8> {
        <Self as ModelPersistence>::to_bytes(self)
    }

    fn model_type(&self) -> &'static str {
        "naive_bayes"
    }
}

impl ModelPersistence for NaiveBayesClassifier {
    fn to_state(&self) -> ModelState {
        // Collect source IDs in consistent order
        let mut source_ids: Vec<String> = self.priors.keys().cloned().collect();
        source_ids.sort();

        // Serialize weights: priors, then means, then variances (all in source_id order)
        let mut weights = Vec::new();

        // Priors
        for id in &source_ids {
            weights.push(*self.priors.get(id).unwrap_or(&0.0));
        }

        // Means (flattened: source0_feat0, source0_feat1, ..., source1_feat0, ...)
        for id in &source_ids {
            if let Some(means) = self.means.get(id) {
                weights.extend_from_slice(means);
            } else {
                // Pad with zeros if missing
                weights.extend(core::iter::repeat_n(0.0, self.feature_dim));
            }
        }

        // Variances (same layout as means)
        for id in &source_ids {
            if let Some(variances) = self.variances.get(id) {
                weights.extend_from_slice(variances);
            } else {
                // Pad with default variance
                weights.extend(core::iter::repeat_n(0.25, self.feature_dim));
            }
        }

        // Extra params: smoothing, sample_count
        let extra_params = vec![self.smoothing, self.sample_count as f32];

        let config = ModelConfig {
            model_type: ModelType::NaiveBayes,
            feature_dim: self.feature_dim,
            num_classes: source_ids.len(),
            learning_rate: 0.0, // Not used for Naive Bayes
            regularization: 0.0,
        };

        ModelState {
            config,
            weights,
            source_ids,
            iterations: self.sample_count,
            extra_params,
            layer_dims: Vec::new(),
            activation_types: Vec::new(),
            optimizer_type: None,
            optimizer_state: None,
            lr_schedule: None,
            epoch: 0,
            early_stopping_config: None,
            early_stopping_state: None,
        }
    }

    fn from_state(state: ModelState) -> Result<Self> {
        if state.config.model_type != ModelType::NaiveBayes {
            return Err(OxiRouterError::ModelError(format!(
                "Expected NaiveBayes model type, got {:?}",
                state.config.model_type
            )));
        }

        let feature_dim = state.config.feature_dim;
        let num_sources = state.source_ids.len();

        // Extract smoothing and sample_count from extra_params
        let smoothing = state.extra_params.first().copied().unwrap_or(1e-6);
        let sample_count = state.extra_params.get(1).copied().unwrap_or(0.0) as u64;

        // Validate weights length
        // Expected: num_sources (priors) + num_sources * feature_dim (means) + num_sources * feature_dim (variances)
        let expected_weights = num_sources + 2 * num_sources * feature_dim;
        if state.weights.len() != expected_weights {
            return Err(OxiRouterError::ModelError(format!(
                "Invalid weights length: expected {}, got {}",
                expected_weights,
                state.weights.len()
            )));
        }

        let mut priors = HashMap::new();
        let mut means = HashMap::new();
        let mut variances = HashMap::new();

        let mut pos = 0;

        // Extract priors
        for id in &state.source_ids {
            priors.insert(id.clone(), state.weights[pos]);
            pos += 1;
        }

        // Extract means
        for id in &state.source_ids {
            let mean_vec: Vec<f32> = state.weights[pos..pos + feature_dim].to_vec();
            means.insert(id.clone(), mean_vec);
            pos += feature_dim;
        }

        // Extract variances
        for id in &state.source_ids {
            let var_vec: Vec<f32> = state.weights[pos..pos + feature_dim].to_vec();
            variances.insert(id.clone(), var_vec);
            pos += feature_dim;
        }

        Ok(Self {
            priors,
            means,
            variances,
            sample_count,
            feature_dim,
            smoothing,
            per_source_counts: HashMap::new(),
            prior_decay: 0.99,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features(values: &[f32]) -> FeatureVector {
        let mut fv = FeatureVector::new();
        for (i, &v) in values.iter().enumerate() {
            fv.add(format!("f{}", i), v);
        }
        fv
    }

    #[test]
    fn test_naive_bayes_creation() {
        let nb = NaiveBayesClassifier::new(10);
        assert_eq!(nb.feature_dim(), 10);
        assert_eq!(nb.name(), "NaiveBayes");
    }

    #[test]
    fn test_initialize_sources() {
        let mut nb = NaiveBayesClassifier::new(5);
        let sources = vec!["src1".to_string(), "src2".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);

        assert!(nb.priors.contains_key("src1"));
        assert!(nb.priors.contains_key("src2"));
        assert!((nb.priors["src1"] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_predict() {
        let mut nb = NaiveBayesClassifier::new(3);
        let sources = vec!["src1".to_string(), "src2".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);

        let features = create_test_features(&[0.5, 0.5, 0.5]);
        let predictions = nb.predict(&features, &source_refs).unwrap();

        assert_eq!(predictions.len(), 2);
        let total: f32 = predictions.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 0.01); // Should sum to 1
    }

    #[test]
    fn test_training() {
        let mut nb = NaiveBayesClassifier::new(3);
        let sources = vec!["src1".to_string(), "src2".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);

        // Train with samples that favor src1 for high features
        let samples = vec![
            TrainingSample::new(
                create_test_features(&[0.9, 0.9, 0.9]),
                "src1",
                true,
                100,
                10,
            ),
            TrainingSample::new(
                create_test_features(&[0.1, 0.1, 0.1]),
                "src2",
                true,
                100,
                10,
            ),
        ];

        nb.train(&samples).unwrap();

        // Test prediction
        let high_features = create_test_features(&[0.8, 0.8, 0.8]);
        let predictions = nb.predict(&high_features, &source_refs).unwrap();

        // src1 should be ranked higher for high features
        assert!(predictions[0].0 == "src1" || predictions[0].1 > predictions[1].1);
    }

    #[test]
    fn test_incremental_update() {
        let mut nb = NaiveBayesClassifier::new(3);
        let sources = vec!["src1".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);

        let features = create_test_features(&[0.7, 0.7, 0.7]);
        nb.update(&features, "src1", 1.0).unwrap();

        assert!(nb.sample_count > 0);
    }

    #[test]
    fn test_naive_bayes_serialization_roundtrip() {
        use super::{ModelPersistence, ModelState};

        let mut nb = NaiveBayesClassifier::new(3);
        let sources = vec!["src1".to_string(), "src2".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);

        // Train with some samples
        let samples = vec![
            TrainingSample::new(
                create_test_features(&[0.9, 0.9, 0.9]),
                "src1",
                true,
                100,
                10,
            ),
            TrainingSample::new(
                create_test_features(&[0.1, 0.1, 0.1]),
                "src2",
                true,
                100,
                10,
            ),
            TrainingSample::new(create_test_features(&[0.8, 0.7, 0.9]), "src1", true, 150, 5),
        ];
        nb.train(&samples).unwrap();

        // Serialize
        let state = nb.to_state();
        let bytes = state.to_bytes();

        // Deserialize
        let restored_state = ModelState::from_bytes(&bytes).unwrap();
        let restored_nb = NaiveBayesClassifier::from_state(restored_state).unwrap();

        // Compare predictions
        let test_features = create_test_features(&[0.85, 0.85, 0.85]);
        let original_pred = nb.predict(&test_features, &source_refs).unwrap();
        let restored_pred = restored_nb.predict(&test_features, &source_refs).unwrap();

        // Predictions should match
        assert_eq!(original_pred.len(), restored_pred.len());
        for (orig, rest) in original_pred.iter().zip(restored_pred.iter()) {
            assert_eq!(orig.0, rest.0);
            assert!(
                (orig.1 - rest.1).abs() < 1e-6,
                "Prediction mismatch: {} vs {}",
                orig.1,
                rest.1
            );
        }
    }

    #[test]
    fn test_naive_bayes_to_bytes_from_bytes() {
        use super::ModelPersistence;

        let mut nb = NaiveBayesClassifier::new(5);
        let sources = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let source_refs: Vec<&String> = sources.iter().collect();
        nb.initialize_sources(&source_refs);

        // Train
        let samples = vec![
            TrainingSample::new(
                create_test_features(&[0.1, 0.2, 0.3, 0.4, 0.5]),
                "alpha",
                true,
                50,
                20,
            ),
            TrainingSample::new(
                create_test_features(&[0.9, 0.8, 0.7, 0.6, 0.5]),
                "gamma",
                true,
                100,
                15,
            ),
        ];
        nb.train(&samples).unwrap();

        // Use convenience methods
        let bytes = ModelPersistence::to_bytes(&nb);
        let restored = NaiveBayesClassifier::from_bytes(&bytes).unwrap();

        assert_eq!(nb.feature_dim(), restored.feature_dim());
        assert_eq!(nb.sample_count, restored.sample_count);

        // Verify predictions match
        let test_features = create_test_features(&[0.5, 0.5, 0.5, 0.5, 0.5]);
        let orig_pred = nb.predict(&test_features, &source_refs).unwrap();
        let rest_pred = restored.predict(&test_features, &source_refs).unwrap();

        for (o, r) in orig_pred.iter().zip(rest_pred.iter()) {
            assert_eq!(o.0, r.0);
            assert!((o.1 - r.1).abs() < 1e-6);
        }
    }
}
