//! Ensemble classifier combining multiple ML models with configurable weights.
//!
//! Predictions are aggregated per source as weighted sums of each component's
//! confidence scores, then renormalized. Online training propagates updates to
//! all component models.
//!
//! ## Serialization format (version 3)
//!
//! The ensemble uses a dedicated binary format identified by magic byte `0x03`:
//!
//! ```text
//! [u32 le] version = 3
//! [u8]     model_type = 2 (Ensemble)
//! [u32 le] feature_dim
//! [u16 le] n_components
//! [f32 le] * n_components  component weights
//! For each component:
//!   [u16 le] name_len
//!   [u8]  *  name bytes (UTF-8)
//!   [u32 le] body_len
//!   [u8]  *  body bytes (recursive v2 format — depth limited to 1)
//! ```

#[cfg(feature = "alloc")]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::core::error::{OxiRouterError, Result};
use crate::ml::{
    FeatureVector, Model, ModelConfig, ModelPersistence, ModelState, ModelType, TrainingSample,
};

/// A single component in the ensemble.
struct EnsembleComponent {
    /// The component model (must also support ModelPersistence).
    model: Box<dyn ModelPersistence>,
    /// Non-negative weight for this component.
    weight: f32,
    /// Human-readable name (for debugging and serialization).
    name: String,
}

/// Ensemble classifier combining multiple component models via weighted averaging.
///
/// Components are `Box<dyn ModelPersistence>` so they can be serialized for
/// federated sharing and roundtrip through `to_bytes` / `from_bytes`.
///
/// Predictions are aggregated per source as weighted sums, then divided by the
/// total weight. Components with weight ≤ 0 are skipped.
pub struct EnsembleClassifier {
    components: Vec<EnsembleComponent>,
    feature_dim: usize,
    /// Whether to renormalize combined confidences so they sum to 1.0.
    pub normalize: bool,
}

impl EnsembleClassifier {
    /// Create a new empty ensemble with the given feature dimension.
    #[must_use]
    pub fn new(feature_dim: usize) -> Self {
        Self {
            components: Vec::new(),
            feature_dim,
            normalize: true,
        }
    }

    /// Add a component model with the given weight.
    ///
    /// # Errors
    ///
    /// Returns an error if the component's `feature_dim` does not match the ensemble's.
    pub fn add_component(
        mut self,
        model: Box<dyn ModelPersistence>,
        weight: f32,
        name: impl Into<String>,
    ) -> Result<Self> {
        if model.feature_dim() != self.feature_dim {
            return Err(OxiRouterError::ModelError(format!(
                "Component feature dim {} != ensemble feature dim {}",
                model.feature_dim(),
                self.feature_dim
            )));
        }
        self.components.push(EnsembleComponent {
            model,
            weight: weight.max(0.0),
            name: name.into(),
        });
        Ok(self)
    }

    /// Returns the number of component models.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components.len()
    }
}

impl Model for EnsembleClassifier {
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, features, source_ids),
            fields(classifiers_count = self.components.len())
        )
    )]
    fn predict(
        &self,
        features: &FeatureVector,
        source_ids: &[&String],
    ) -> Result<Vec<(String, f32)>> {
        if self.components.is_empty() {
            return Err(OxiRouterError::ModelError(
                "Ensemble has no components".to_string(),
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

        let total_weight: f32 = self
            .components
            .iter()
            .filter(|c| c.weight > 0.0)
            .map(|c| c.weight)
            .sum();

        if total_weight <= 0.0 {
            return Err(OxiRouterError::ModelError(
                "All component weights are zero or negative".to_string(),
            ));
        }

        // Accumulate weighted confidences per source using an inline vec to avoid hashbrown dep
        let mut aggregated: Vec<(String, f32)> = Vec::with_capacity(source_ids.len());

        for component in &self.components {
            if component.weight <= 0.0 {
                continue;
            }
            let predictions = component.model.predict(features, source_ids)?;
            for (source_id, confidence) in predictions {
                if let Some(existing) = aggregated.iter_mut().find(|(id, _)| *id == source_id) {
                    existing.1 += component.weight * confidence;
                } else {
                    aggregated.push((source_id, component.weight * confidence));
                }
            }
        }

        // Normalize by total weight
        for (_, conf) in &mut aggregated {
            *conf /= total_weight;
        }

        // Sort by confidence descending
        aggregated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        #[cfg(all(feature = "observability", feature = "std"))]
        {
            let elapsed_us = predict_start.elapsed().as_micros() as f64;
            metrics::histogram!("oxirouter.ml.predict.duration_us", "model" => "ensemble")
                .record(elapsed_us);
        }

        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "EnsembleClassifier"
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn train(&mut self, samples: &[TrainingSample]) -> Result<()> {
        for component in &mut self.components {
            component.model.train(samples)?;
        }
        Ok(())
    }

    fn update(&mut self, features: &FeatureVector, source_id: &str, reward: f32) -> Result<()> {
        for component in &mut self.components {
            component.model.update(features, source_id, reward)?;
        }
        Ok(())
    }

    fn to_bytes(&self) -> Vec<u8> {
        <Self as ModelPersistence>::to_bytes(self)
    }

    fn model_type(&self) -> &'static str {
        "ensemble"
    }
}

impl ModelPersistence for EnsembleClassifier {
    fn to_state(&self) -> ModelState {
        // Minimal state for type identification; full serialization is in to_bytes().
        // from_state is a stub — always use from_bytes() for Ensemble.
        ModelState {
            config: ModelConfig {
                model_type: ModelType::Ensemble,
                feature_dim: self.feature_dim,
                num_classes: 0,
                learning_rate: 0.0,
                regularization: 0.0,
            },
            weights: vec![],
            source_ids: vec![],
            iterations: 0,
            extra_params: vec![self.components.len() as f32],
            layer_dims: vec![],
            activation_types: vec![],
            optimizer_type: None,
            optimizer_state: None,
            lr_schedule: None,
            epoch: 0,
            early_stopping_config: None,
            early_stopping_state: None,
        }
    }

    fn from_state(_state: ModelState) -> Result<Self>
    where
        Self: Sized,
    {
        // EnsembleClassifier cannot be reconstructed from a flat ModelState alone
        // because component models are stored as nested byte blobs in to_bytes().
        // Always deserialize via from_bytes() instead.
        Err(OxiRouterError::ModelError(
            "EnsembleClassifier must be deserialized via from_bytes(), not from_state()"
                .to_string(),
        ))
    }

    /// Serialize using the ensemble v3 binary format (see module doc).
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version 3 — ensemble-specific format
        bytes.extend_from_slice(&3u32.to_le_bytes());

        // Model type byte = 2 (Ensemble)
        bytes.push(2u8);

        // feature_dim
        bytes.extend_from_slice(&(self.feature_dim as u32).to_le_bytes());

        // n_components
        let n = self.components.len() as u16;
        bytes.extend_from_slice(&n.to_le_bytes());

        // Component weights
        for component in &self.components {
            bytes.extend_from_slice(&component.weight.to_le_bytes());
        }

        // Component bodies
        for component in &self.components {
            // Name
            let name_bytes = component.name.as_bytes();
            bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            bytes.extend_from_slice(name_bytes);

            // Body: delegate to the component's own to_bytes() via ModelPersistence
            let body = ModelPersistence::to_bytes(component.model.as_ref());
            bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&body);
        }

        bytes
    }

    /// Deserialize from the ensemble v3 binary format.
    ///
    /// Depth-limited: component bodies must not themselves be Ensemble (model_type byte ≠ 2
    /// at version-3 magic byte ≠ 3), preventing infinite recursion.
    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized,
    {
        use crate::ml::{NaiveBayesClassifier, NeuralNetwork};

        let err = |msg: &str| OxiRouterError::ModelError(msg.to_string());

        if bytes.len() < 4 {
            return Err(err("EnsembleClassifier bytes too short"));
        }

        let version = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|_| err("Invalid version bytes"))?,
        );

        if version != 3 {
            return Err(OxiRouterError::ModelError(format!(
                "EnsembleClassifier expects version 3, got {version}"
            )));
        }

        let mut pos = 4;

        // Model type byte (should be 2)
        if pos >= bytes.len() {
            return Err(err("Missing model_type byte"));
        }
        if bytes[pos] != 2 {
            return Err(OxiRouterError::ModelError(format!(
                "Expected Ensemble model_type byte 2, got {}",
                bytes[pos]
            )));
        }
        pos += 1;

        // feature_dim
        if pos + 4 > bytes.len() {
            return Err(err("Missing feature_dim"));
        }
        let feature_dim = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| err("Invalid feature_dim"))?,
        ) as usize;
        pos += 4;

        // n_components
        if pos + 2 > bytes.len() {
            return Err(err("Missing n_components"));
        }
        let n_components = u16::from_le_bytes(
            bytes[pos..pos + 2]
                .try_into()
                .map_err(|_| err("Invalid n_components"))?,
        ) as usize;
        pos += 2;

        // Weights
        if pos + n_components * 4 > bytes.len() {
            return Err(err("Weights truncated"));
        }
        let mut weights = Vec::with_capacity(n_components);
        for _ in 0..n_components {
            let w = f32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| err("Invalid weight f32"))?,
            );
            weights.push(w);
            pos += 4;
        }

        // Component bodies
        let mut ensemble = EnsembleClassifier::new(feature_dim);

        for i in 0..n_components {
            // Name
            if pos + 2 > bytes.len() {
                return Err(err("Missing name_len"));
            }
            let name_len = u16::from_le_bytes(
                bytes[pos..pos + 2]
                    .try_into()
                    .map_err(|_| err("Invalid name_len"))?,
            ) as usize;
            pos += 2;

            if pos + name_len > bytes.len() {
                return Err(err("Name bytes truncated"));
            }
            let name = String::from_utf8(bytes[pos..pos + name_len].to_vec())
                .map_err(|_| err("Invalid UTF-8 in component name"))?;
            pos += name_len;

            // Body length
            if pos + 4 > bytes.len() {
                return Err(err("Missing body_len"));
            }
            let body_len = u32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| err("Invalid body_len"))?,
            ) as usize;
            pos += 4;

            if pos + body_len > bytes.len() {
                return Err(err("Body bytes truncated"));
            }
            let body = &bytes[pos..pos + body_len];
            pos += body_len;

            // Depth limit: reject nested Ensemble components.
            // An ensemble body has version byte ∈ {2,3} AND model_type byte (index 4) == 2.
            // Note: version 3 NaiveBayes/NeuralNetwork bodies also have version=3 but
            // model_type byte 0 or 1 — so we must check both version AND model_type byte.
            let is_ensemble = if body.len() >= 5 {
                let v = u32::from_le_bytes(body[0..4].try_into().unwrap_or([0; 4]));
                (v == 2 || v == 3) && body[4] == 2
            } else {
                false
            };
            if is_ensemble {
                return Err(OxiRouterError::ModelError(
                    "Nested Ensemble components are not allowed (depth limit 1)".to_string(),
                ));
            }

            // Deserialize component: try NaiveBayes first, then NeuralNetwork
            let component_model: Box<dyn ModelPersistence> = NaiveBayesClassifier::from_bytes(body)
                .map(|m| Box::new(m) as Box<dyn ModelPersistence>)
                .or_else(|_| {
                    NeuralNetwork::from_bytes(body)
                        .map(|m| Box::new(m) as Box<dyn ModelPersistence>)
                })
                .map_err(|e| OxiRouterError::ModelError(format!("Component {i} ({name}): {e}")))?;

            ensemble = ensemble.add_component(component_model, weights[i], name)?;
        }

        Ok(ensemble)
    }
}
