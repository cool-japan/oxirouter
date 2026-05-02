//! Model abstraction and serialization

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use super::FeatureVector;
use super::optimizer::{OptimizerState, OptimizerType};
use super::schedule::{EarlyStoppingConfig, EarlyStoppingState, LearningRateSchedule};
use crate::core::error::Result;

/// Trait for ML models used in source selection
pub trait Model: Send + Sync {
    /// Predict scores for each source given features
    ///
    /// # Errors
    ///
    /// Returns error if prediction fails
    fn predict(
        &self,
        features: &FeatureVector,
        source_ids: &[&String],
    ) -> Result<Vec<(String, f32)>>;

    /// Get the model name
    fn name(&self) -> &str;

    /// Get the expected feature dimension
    fn feature_dim(&self) -> usize;

    /// Train the model with labeled data
    ///
    /// # Errors
    ///
    /// Returns error if training fails
    fn train(&mut self, samples: &[TrainingSample]) -> Result<()>;

    /// Update model incrementally with new feedback
    ///
    /// # Errors
    ///
    /// Returns error if update fails
    fn update(&mut self, features: &FeatureVector, source_id: &str, reward: f32) -> Result<()>;

    /// Serialize the model's current (live) state to bytes.
    fn to_bytes(&self) -> Vec<u8>;

    /// A stable discriminant identifying the concrete model type.
    fn model_type(&self) -> &'static str;
}

/// A training sample for supervised learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Feature vector
    pub features: FeatureVector,
    /// Source that was selected
    pub selected_source: String,
    /// Whether the selection was successful
    pub success: bool,
    /// Latency in milliseconds
    pub latency_ms: u32,
    /// Number of results returned
    pub result_count: u32,
}

impl TrainingSample {
    /// Create a new training sample
    #[must_use]
    pub fn new(
        features: FeatureVector,
        selected_source: impl Into<String>,
        success: bool,
        latency_ms: u32,
        result_count: u32,
    ) -> Self {
        Self {
            features,
            selected_source: selected_source.into(),
            success,
            latency_ms,
            result_count,
        }
    }

    /// Calculate reward signal for this sample
    #[must_use]
    pub fn reward(&self) -> f32 {
        if !self.success {
            return 0.0;
        }

        // Base reward for success
        let mut reward = 1.0;

        // Latency penalty (normalized to 0-10 second range)
        let latency_penalty = (self.latency_ms as f32 / 10000.0).min(1.0);
        reward *= 1.0 - (latency_penalty * 0.5);

        // Bonus for returning results
        if self.result_count > 0 {
            reward *= 1.0 + (self.result_count.min(1000) as f32 / 10000.0);
        }

        reward.clamp(0.0, 1.0)
    }
}

/// Serializable model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type identifier
    pub model_type: ModelType,
    /// Feature dimension
    pub feature_dim: usize,
    /// Number of output classes (sources)
    pub num_classes: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Regularization strength
    pub regularization: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::NaiveBayes,
            feature_dim: 48,
            num_classes: 10,
            learning_rate: 0.01,
            regularization: 0.001,
        }
    }
}

/// Type of ML model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Naive Bayes classifier
    NaiveBayes,
    /// Neural network (MLP)
    NeuralNetwork,
    /// Ensemble of multiple models
    Ensemble,
}

/// Model state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Model configuration
    pub config: ModelConfig,
    /// Model weights (flattened)
    pub weights: Vec<f32>,
    /// Source ID mapping
    pub source_ids: Vec<String>,
    /// Training iteration count
    pub iterations: u64,
    /// Additional parameters (e.g., layer dimensions, biases)
    pub extra_params: Vec<f32>,
    /// Layer architecture for neural networks (input_dim, output_dim pairs)
    pub layer_dims: Vec<(usize, usize)>,
    /// Activation types per layer (encoded as u8: 0=ReLU, 1=Sigmoid, 2=Tanh, 3=Linear)
    pub activation_types: Vec<u8>,
    // v3: optimizer and schedule state fields (None for legacy v1/v2 blobs)
    /// Optimizer type (v3+)
    #[serde(default)]
    pub optimizer_type: Option<OptimizerType>,
    /// Adam/Momentum optimizer moment state (v3+)
    #[serde(default)]
    pub optimizer_state: Option<OptimizerState>,
    /// Learning rate schedule (v3+)
    #[serde(default)]
    pub lr_schedule: Option<LearningRateSchedule>,
    /// Current epoch counter (v3+)
    #[serde(default)]
    pub epoch: u64,
    /// Early stopping configuration (v3+)
    #[serde(default)]
    pub early_stopping_config: Option<EarlyStoppingConfig>,
    /// Early stopping runtime state (v3+)
    #[serde(default)]
    pub early_stopping_state: Option<EarlyStoppingState>,
}

/// Trait for model persistence (serialization/deserialization)
pub trait ModelPersistence: Model {
    /// Convert model to serializable state
    fn to_state(&self) -> ModelState;

    /// Restore model from serialized state
    ///
    /// # Errors
    ///
    /// Returns error if the state is invalid or incompatible
    fn from_state(state: ModelState) -> Result<Self>
    where
        Self: Sized;

    /// Serialize model to bytes
    fn to_bytes(&self) -> Vec<u8> {
        self.to_state().to_bytes()
    }

    /// Deserialize model from bytes
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails
    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized,
    {
        let state = ModelState::from_bytes(bytes)?;
        Self::from_state(state)
    }
}

/// v3 extension blob — JSON-encoded and appended after the v2 binary data.
/// Fields default to None/0 so old readers gracefully ignore missing keys.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct V3Extension {
    #[serde(default)]
    optimizer_type: Option<OptimizerType>,
    #[serde(default)]
    optimizer_state: Option<OptimizerState>,
    #[serde(default)]
    lr_schedule: Option<LearningRateSchedule>,
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    early_stopping_config: Option<EarlyStoppingConfig>,
    #[serde(default)]
    early_stopping_state: Option<EarlyStoppingState>,
}

impl ModelState {
    /// Create a new model state
    #[must_use]
    pub fn new(config: ModelConfig, source_ids: Vec<String>) -> Self {
        Self {
            config,
            weights: Vec::new(),
            source_ids,
            iterations: 0,
            extra_params: Vec::new(),
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

    /// Serialize to bytes (version 3 format)
    ///
    /// Binary format:
    /// - 4 bytes: version (3)
    /// - 1 byte: model_type
    /// - 4 bytes: feature_dim
    /// - 4 bytes: num_classes
    /// - 4 bytes: learning_rate (f32)
    /// - 4 bytes: regularization (f32)
    /// - 4 bytes: weight count
    /// - weights as f32
    /// - 4 bytes: extra_params count
    /// - extra_params as f32
    /// - 4 bytes: layer_dims count
    /// - layer_dims as (u32, u32) pairs
    /// - 4 bytes: activation_types count
    /// - activation_types as u8
    /// - 4 bytes: source_ids count
    /// - source IDs as length-prefixed strings
    /// - 8 bytes: iterations
    /// - 4 bytes: v3 extension blob length (0 = no extension)
    /// - N bytes: serde_json-encoded V3Extension struct
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version 3: optimizer_state and early_stopping_state are now persisted
        bytes.extend_from_slice(&3u32.to_le_bytes());

        // Model type
        let model_type_byte: u8 = match self.config.model_type {
            ModelType::NaiveBayes => 0,
            ModelType::NeuralNetwork => 1,
            ModelType::Ensemble => 2,
        };
        bytes.push(model_type_byte);

        // Config
        bytes.extend_from_slice(&(self.config.feature_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_classes as u32).to_le_bytes());
        bytes.extend_from_slice(&self.config.learning_rate.to_le_bytes());
        bytes.extend_from_slice(&self.config.regularization.to_le_bytes());

        // Weights
        bytes.extend_from_slice(&(self.weights.len() as u32).to_le_bytes());
        for &w in &self.weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        // Extra params (biases, etc.)
        bytes.extend_from_slice(&(self.extra_params.len() as u32).to_le_bytes());
        for &p in &self.extra_params {
            bytes.extend_from_slice(&p.to_le_bytes());
        }

        // Layer dimensions
        bytes.extend_from_slice(&(self.layer_dims.len() as u32).to_le_bytes());
        for &(input_dim, output_dim) in &self.layer_dims {
            bytes.extend_from_slice(&(input_dim as u32).to_le_bytes());
            bytes.extend_from_slice(&(output_dim as u32).to_le_bytes());
        }

        // Activation types
        bytes.extend_from_slice(&(self.activation_types.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.activation_types);

        // Source IDs
        bytes.extend_from_slice(&(self.source_ids.len() as u32).to_le_bytes());
        for id in &self.source_ids {
            let id_bytes = id.as_bytes();
            bytes.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(id_bytes);
        }

        // Iterations
        bytes.extend_from_slice(&self.iterations.to_le_bytes());

        // v3 extension: JSON blob for optimizer/schedule/early_stopping state
        let has_extension = self.optimizer_type.is_some()
            || self.optimizer_state.is_some()
            || self.lr_schedule.is_some()
            || self.early_stopping_config.is_some()
            || self.early_stopping_state.is_some()
            || self.epoch > 0;

        if has_extension {
            let ext = V3Extension {
                optimizer_type: self.optimizer_type.clone(),
                optimizer_state: self.optimizer_state.clone(),
                lr_schedule: self.lr_schedule.clone(),
                epoch: self.epoch,
                early_stopping_config: self.early_stopping_config.clone(),
                early_stopping_state: self.early_stopping_state.clone(),
            };
            // Best-effort JSON encode; if it fails, write a zero-length blob
            match serde_json::to_vec(&ext) {
                Ok(json_bytes) => {
                    bytes.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(&json_bytes);
                }
                Err(_) => {
                    bytes.extend_from_slice(&0u32.to_le_bytes());
                }
            }
        } else {
            // No extension data
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use crate::core::error::OxiRouterError;

        if bytes.len() < 4 {
            return Err(OxiRouterError::ModelError(
                "Invalid model state: too short".to_string(),
            ));
        }

        let version = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid version bytes".to_string()))?,
        );

        match version {
            1 => Self::from_bytes_v1(bytes),
            2 => Self::from_bytes_v2(bytes),
            3 => Self::from_bytes_v3(bytes),
            _ => Err(OxiRouterError::ModelError(format!(
                "Unsupported model state version: {}",
                version
            ))),
        }
    }

    /// Deserialize from version 3 format (current)
    /// v3 = v2 layout + trailing JSON blob with optimizer/schedule/early_stopping state
    fn from_bytes_v3(bytes: &[u8]) -> Result<Self> {
        use crate::core::error::OxiRouterError;

        // Parse v2 core layout (same structure, just different version byte)
        let mut state = Self::from_bytes_v2_core(bytes)?;

        // Now parse the trailing v3 extension blob
        // After the v2 fields, there should be 4 bytes for the blob length
        let v2_end = Self::v2_end_position(bytes)?;

        if v2_end + 4 <= bytes.len() {
            let blob_len =
                u32::from_le_bytes(bytes[v2_end..v2_end + 4].try_into().map_err(|_| {
                    OxiRouterError::ModelError("Invalid v3 blob length".to_string())
                })?) as usize;

            if blob_len > 0 && v2_end + 4 + blob_len <= bytes.len() {
                let json_bytes = &bytes[v2_end + 4..v2_end + 4 + blob_len];
                match serde_json::from_slice::<V3Extension>(json_bytes) {
                    Ok(ext) => {
                        state.optimizer_type = ext.optimizer_type;
                        state.optimizer_state = ext.optimizer_state;
                        state.lr_schedule = ext.lr_schedule;
                        state.epoch = ext.epoch;
                        state.early_stopping_config = ext.early_stopping_config;
                        state.early_stopping_state = ext.early_stopping_state;
                    }
                    Err(_) => {
                        // Extension parse failure — treat as no extension (safe default)
                    }
                }
            }
        }

        Ok(state)
    }

    /// Compute position just after the v2 core data ends (= position of the v3 extension length)
    fn v2_end_position(bytes: &[u8]) -> Result<usize> {
        use crate::core::error::OxiRouterError;

        let mut pos = 4; // Skip version

        // Model type byte
        if pos >= bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at model_type".to_string(),
            ));
        }
        pos += 1;

        // Config: 4 fields x 4 bytes
        pos += 16;
        if pos > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at config".to_string(),
            ));
        }

        // Weights
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at weight count".to_string(),
            ));
        }
        let weight_count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("v3: invalid weight count".to_string()))?,
        ) as usize;
        pos += 4 + weight_count * 4;
        if pos > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated in weights".to_string(),
            ));
        }

        // Extra params
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at extra_params count".to_string(),
            ));
        }
        let extra_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
            OxiRouterError::ModelError("v3: invalid extra_params count".to_string())
        })?) as usize;
        pos += 4 + extra_count * 4;
        if pos > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated in extra_params".to_string(),
            ));
        }

        // Layer dims
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at layer_dims count".to_string(),
            ));
        }
        let layer_count =
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
                OxiRouterError::ModelError("v3: invalid layer_dims count".to_string())
            })?) as usize;
        pos += 4 + layer_count * 8; // 2 x u32 per entry
        if pos > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated in layer_dims".to_string(),
            ));
        }

        // Activation types
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at activation count".to_string(),
            ));
        }
        let activation_count =
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
                OxiRouterError::ModelError("v3: invalid activation count".to_string())
            })?) as usize;
        pos += 4 + activation_count;
        if pos > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated in activations".to_string(),
            ));
        }

        // Source IDs
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at source_ids count".to_string(),
            ));
        }
        let id_count =
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
                OxiRouterError::ModelError("v3: invalid source_ids count".to_string())
            })?) as usize;
        pos += 4;
        for _ in 0..id_count {
            if pos + 4 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "v3: truncated at source ID length".to_string(),
                ));
            }
            let id_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
                OxiRouterError::ModelError("v3: invalid source ID length".to_string())
            })?) as usize;
            pos += 4 + id_len;
            if pos > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "v3: truncated in source ID".to_string(),
                ));
            }
        }

        // Iterations (8 bytes)
        pos += 8;
        if pos > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "v3: truncated at iterations".to_string(),
            ));
        }

        Ok(pos)
    }

    /// Parse the v2 core layout, ignoring extension data
    /// (Works for both v2 and v3 since they share the same core layout)
    fn from_bytes_v2_core(bytes: &[u8]) -> Result<Self> {
        use crate::core::error::OxiRouterError;

        let mut pos = 4; // Skip version

        // Model type
        if pos >= bytes.len() {
            return Err(OxiRouterError::ModelError("Missing model type".to_string()));
        }
        let model_type = match bytes[pos] {
            0 => ModelType::NaiveBayes,
            1 => ModelType::NeuralNetwork,
            2 => ModelType::Ensemble,
            _ => return Err(OxiRouterError::ModelError("Unknown model type".to_string())),
        };
        pos += 1;

        // Config
        if pos + 16 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Missing config fields".to_string(),
            ));
        }
        let feature_dim = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid feature_dim".to_string()))?,
        ) as usize;
        pos += 4;
        let num_classes = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid num_classes".to_string()))?,
        ) as usize;
        pos += 4;
        let learning_rate = f32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid learning_rate".to_string()))?,
        );
        pos += 4;
        let regularization = f32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid regularization".to_string()))?,
        );
        pos += 4;

        // Weights
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Missing weight count".to_string(),
            ));
        }
        let weight_count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid weight count".to_string()))?,
        ) as usize;
        pos += 4;
        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            if pos + 4 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of weights".to_string(),
                ));
            }
            let w = f32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid weight".to_string()))?,
            );
            weights.push(w);
            pos += 4;
        }

        // Extra params
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Missing extra_params count".to_string(),
            ));
        }
        let extra_count =
            u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
                OxiRouterError::ModelError("Invalid extra_params count".to_string())
            })?) as usize;
        pos += 4;
        let mut extra_params = Vec::with_capacity(extra_count);
        for _ in 0..extra_count {
            if pos + 4 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of extra_params".to_string(),
                ));
            }
            let p = f32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid extra param".to_string()))?,
            );
            extra_params.push(p);
            pos += 4;
        }

        // Layer dimensions
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Missing layer_dims count".to_string(),
            ));
        }
        let layer_count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid layer_dims count".to_string()))?,
        ) as usize;
        pos += 4;
        let mut layer_dims = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            if pos + 8 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of layer_dims".to_string(),
                ));
            }
            let input_dim = u32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid input_dim".to_string()))?,
            ) as usize;
            pos += 4;
            let output_dim = u32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid output_dim".to_string()))?,
            ) as usize;
            pos += 4;
            layer_dims.push((input_dim, output_dim));
        }

        // Activation types
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Missing activation_types count".to_string(),
            ));
        }
        let activation_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().map_err(|_| {
            OxiRouterError::ModelError("Invalid activation_types count".to_string())
        })?) as usize;
        pos += 4;
        if pos + activation_count > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Unexpected end of activation_types".to_string(),
            ));
        }
        let activation_types = bytes[pos..pos + activation_count].to_vec();
        pos += activation_count;

        // Source IDs
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Missing source_ids count".to_string(),
            ));
        }
        let id_count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid source_ids count".to_string()))?,
        ) as usize;
        pos += 4;
        let mut source_ids = Vec::with_capacity(id_count);
        for _ in 0..id_count {
            if pos + 4 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of source ID length".to_string(),
                ));
            }
            let id_len = u32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid ID length".to_string()))?,
            ) as usize;
            pos += 4;
            if pos + id_len > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of source ID".to_string(),
                ));
            }
            let id = String::from_utf8(bytes[pos..pos + id_len].to_vec()).map_err(|_| {
                OxiRouterError::ModelError("Invalid UTF-8 in source ID".to_string())
            })?;
            source_ids.push(id);
            pos += id_len;
        }

        // Iterations
        let iterations = if pos + 8 <= bytes.len() {
            u64::from_le_bytes(
                bytes[pos..pos + 8]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid iterations".to_string()))?,
            )
        } else {
            0
        };

        Ok(Self {
            config: ModelConfig {
                model_type,
                feature_dim,
                num_classes,
                learning_rate,
                regularization,
            },
            weights,
            source_ids,
            iterations,
            extra_params,
            layer_dims,
            activation_types,
            optimizer_type: None,
            optimizer_state: None,
            lr_schedule: None,
            epoch: 0,
            early_stopping_config: None,
            early_stopping_state: None,
        })
    }

    /// Deserialize from version 1 format (legacy)
    fn from_bytes_v1(bytes: &[u8]) -> Result<Self> {
        use crate::core::error::OxiRouterError;

        if bytes.len() < 20 {
            return Err(OxiRouterError::ModelError(
                "Invalid v1 model state: too short".to_string(),
            ));
        }

        let mut pos = 4; // Skip version

        // Config
        let feature_dim = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid feature_dim".to_string()))?,
        ) as usize;
        pos += 4;
        let num_classes = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid num_classes".to_string()))?,
        ) as usize;
        pos += 4;

        // Weights
        let weight_count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid weight count".to_string()))?,
        ) as usize;
        pos += 4;

        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            if pos + 4 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of weights".to_string(),
                ));
            }
            let w = f32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid weight".to_string()))?,
            );
            weights.push(w);
            pos += 4;
        }

        // Source IDs
        if pos + 4 > bytes.len() {
            return Err(OxiRouterError::ModelError(
                "Unexpected end of source IDs".to_string(),
            ));
        }
        let id_count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| OxiRouterError::ModelError("Invalid source ID count".to_string()))?,
        ) as usize;
        pos += 4;

        let mut source_ids = Vec::with_capacity(id_count);
        for _ in 0..id_count {
            if pos + 4 > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of source ID length".to_string(),
                ));
            }
            let id_len = u32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid ID length".to_string()))?,
            ) as usize;
            pos += 4;

            if pos + id_len > bytes.len() {
                return Err(OxiRouterError::ModelError(
                    "Unexpected end of source ID".to_string(),
                ));
            }
            let id = String::from_utf8(bytes[pos..pos + id_len].to_vec()).map_err(|_| {
                OxiRouterError::ModelError("Invalid UTF-8 in source ID".to_string())
            })?;
            source_ids.push(id);
            pos += id_len;
        }

        // Iterations
        let iterations = if pos + 8 <= bytes.len() {
            u64::from_le_bytes(
                bytes[pos..pos + 8]
                    .try_into()
                    .map_err(|_| OxiRouterError::ModelError("Invalid iterations".to_string()))?,
            )
        } else {
            0
        };

        Ok(Self {
            config: ModelConfig {
                model_type: ModelType::NaiveBayes,
                feature_dim,
                num_classes,
                learning_rate: 0.01,
                regularization: 0.001,
            },
            weights,
            source_ids,
            iterations,
            extra_params: Vec::new(),
            layer_dims: Vec::new(),
            activation_types: Vec::new(),
            optimizer_type: None,
            optimizer_state: None,
            lr_schedule: None,
            epoch: 0,
            early_stopping_config: None,
            early_stopping_state: None,
        })
    }

    /// Deserialize from version 2 format (delegates to shared core parser)
    fn from_bytes_v2(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_v2_core(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    use alloc::vec;

    #[test]
    fn test_training_sample_reward() {
        let features = FeatureVector::new();

        // Successful with good latency
        let sample = TrainingSample::new(features.clone(), "src1", true, 100, 50);
        assert!(sample.reward() > 0.8);

        // Failed query
        let failed = TrainingSample::new(features.clone(), "src1", false, 100, 0);
        assert_eq!(failed.reward(), 0.0);

        // Slow query
        let slow = TrainingSample::new(features, "src1", true, 9000, 50);
        assert!(slow.reward() < 0.7);
    }

    #[test]
    fn test_model_state_serialization() {
        let config = ModelConfig::default();
        let mut state = ModelState::new(config, vec!["src1".to_string(), "src2".to_string()]);
        state.weights = vec![0.1, 0.2, 0.3, 0.4];
        state.extra_params = vec![0.5, 0.6];
        state.layer_dims = vec![(10, 5), (5, 2)];
        state.activation_types = vec![0, 3]; // ReLU, Linear
        state.iterations = 100;

        let bytes = state.to_bytes();
        let restored = ModelState::from_bytes(&bytes).unwrap();

        assert_eq!(restored.weights, state.weights);
        assert_eq!(restored.source_ids, state.source_ids);
        assert_eq!(restored.iterations, state.iterations);
        assert_eq!(restored.extra_params, state.extra_params);
        assert_eq!(restored.layer_dims, state.layer_dims);
        assert_eq!(restored.activation_types, state.activation_types);
        assert_eq!(restored.config.model_type, state.config.model_type);
        assert_eq!(restored.config.learning_rate, state.config.learning_rate);
    }
}
