//! ML inference engine for learned source selection
//!
//! This module provides lightweight machine learning models for
//! intelligent SPARQL source selection.

mod activation;
mod ensemble;
pub mod feature;
pub mod federated;
mod layer;
pub mod model;
mod naive_bayes;
pub mod neural;
pub mod optimizer;
pub mod schedule;

pub use ensemble::EnsembleClassifier;
pub use feature::FeatureVector;
pub use federated::{MergeStrategy, merge_states};
pub use model::{Model, ModelConfig, ModelPersistence, ModelState, ModelType, TrainingSample};
pub use naive_bayes::NaiveBayesClassifier;
pub use neural::NeuralNetwork;
