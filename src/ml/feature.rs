//! Feature vector construction for ML inference

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use serde::{Deserialize, Serialize};

use crate::context::CombinedContext;
use crate::core::error::Result;
use crate::core::query::Query;

/// Maximum feature vector dimension
pub const MAX_FEATURES: usize = 64;

/// Feature vector for ML model input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Raw feature values (normalized to 0.0-1.0)
    pub values: Vec<f32>,
    /// Feature names for interpretability
    pub names: Vec<String>,
}

impl FeatureVector {
    /// Create an empty feature vector
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: Vec::with_capacity(MAX_FEATURES),
            names: Vec::with_capacity(MAX_FEATURES),
        }
    }

    /// Create a feature vector with given capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            names: Vec::with_capacity(capacity),
        }
    }

    /// Add a feature to the vector
    pub fn add(&mut self, name: impl Into<String>, value: f32) {
        self.names.push(name.into());
        self.values.push(value);
    }

    /// Add a feature with automatic normalization
    pub fn add_normalized(&mut self, name: impl Into<String>, value: f32, min: f32, max: f32) {
        let normalized = if max > min {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.add(name, normalized);
    }

    /// Get the number of features
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the feature vector is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get feature value by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }

    /// Get feature value by index
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Create feature vector from query
    ///
    /// # Errors
    ///
    /// Returns error if feature extraction fails
    pub fn from_query(query: &Query) -> Result<Self> {
        let mut features = Self::with_capacity(24);

        // Query type features (one-hot)
        features.add(
            "query_type_select",
            if query.query_type == crate::core::query::QueryType::Select {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "query_type_construct",
            if query.query_type == crate::core::query::QueryType::Construct {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "query_type_ask",
            if query.query_type == crate::core::query::QueryType::Ask {
                1.0
            } else {
                0.0
            },
        );
        features.add(
            "query_type_describe",
            if query.query_type == crate::core::query::QueryType::Describe {
                1.0
            } else {
                0.0
            },
        );

        // Triple pattern count (normalized to 0-50 range)
        features.add_normalized(
            "triple_count",
            query.triple_patterns.len() as f32,
            0.0,
            50.0,
        );

        // Predicate count
        features.add_normalized("predicate_count", query.predicates.len() as f32, 0.0, 20.0);

        // Type count
        features.add_normalized("type_count", query.types.len() as f32, 0.0, 10.0);

        // Query features (binary)
        features.add("has_optional", if query.has_optional { 1.0 } else { 0.0 });
        features.add("has_union", if query.has_union { 1.0 } else { 0.0 });
        features.add("has_filter", if query.has_filter { 1.0 } else { 0.0 });
        features.add(
            "has_aggregation",
            if query.has_aggregation { 1.0 } else { 0.0 },
        );
        features.add(
            "has_property_paths",
            if query.has_property_paths { 1.0 } else { 0.0 },
        );
        features.add("has_subquery", if query.has_subquery { 1.0 } else { 0.0 });
        features.add("has_service", if query.has_service { 1.0 } else { 0.0 });

        // Complexity score
        features.add("complexity", query.complexity);

        // Limit/offset features
        features.add("has_limit", if query.limit.is_some() { 1.0 } else { 0.0 });
        features.add_normalized("limit_value", query.limit.unwrap_or(0) as f32, 0.0, 10000.0);

        // Predicate hash (normalized to 0-1)
        let hash = query.predicate_hash();
        features.add("predicate_hash_low", (hash & 0xFFFF) as f32 / 65535.0);
        features.add(
            "predicate_hash_high",
            ((hash >> 16) & 0xFFFF) as f32 / 65535.0,
        );

        // Triple pattern statistics
        let mut total_selectivity = 0.0;
        let mut variable_sum = 0;
        for pattern in &query.triple_patterns {
            total_selectivity += pattern.selectivity();
            variable_sum += pattern.variable_count() as usize;
        }
        let pattern_count = query.triple_patterns.len().max(1) as f32;
        features.add("avg_selectivity", total_selectivity / pattern_count);
        features.add_normalized(
            "avg_variables",
            variable_sum as f32 / pattern_count,
            0.0,
            3.0,
        );

        // SPARQL 1.1 requirement
        features.add(
            "requires_sparql_1_1",
            if query.requires_sparql_1_1() {
                1.0
            } else {
                0.0
            },
        );

        // SPARQL AST structural features (10 dims, only when `sparql` feature enabled)
        #[cfg(feature = "sparql")]
        {
            let af = query.ast_features.unwrap_or_default();
            features.add("sparql_join_depth", af.join_depth);
            features.add("sparql_optional_count", af.optional_count);
            features.add("sparql_filter_count", af.filter_count);
            features.add("sparql_union_branches", af.union_branch_count);
            features.add("sparql_has_distinct", af.has_distinct);
            features.add("sparql_has_having", af.has_having);
            features.add("sparql_subquery_count", af.subquery_count);
            features.add("sparql_path_exprs", af.path_expr_count);
            features.add("sparql_literal_count", af.literal_count);
            features.add("sparql_blank_nodes", af.blank_node_count);
        }

        Ok(features)
    }

    /// Create feature vector from query and context
    ///
    /// # Errors
    ///
    /// Returns error if feature extraction fails
    pub fn from_query_and_context(
        query: &Query,
        context: Option<&CombinedContext>,
    ) -> Result<Self> {
        let mut features = Self::from_query(query)?;

        // Add context features if available
        if let Some(ctx) = context {
            features.add_context_features(ctx);
        } else {
            // Add placeholder context features
            features.add_placeholder_context_features();
        }

        Ok(features)
    }

    /// Add context features to the vector
    fn add_context_features(&mut self, ctx: &CombinedContext) {
        // ctx is used by feature-gated blocks below; suppress unused warning
        // when none of the context features are active.
        let _ = ctx;
        // Geo features
        #[cfg(feature = "geo")]
        if let Some(ref geo) = ctx.geo {
            self.add("has_geo_context", 1.0);
            self.add("is_eu_region", if geo.is_eu_region() { 1.0 } else { 0.0 });

            if let Some((lon, lat)) = geo.position {
                // Normalize coordinates
                self.add_normalized("geo_lon", lon as f32, -180.0, 180.0);
                self.add_normalized("geo_lat", lat as f32, -90.0, 90.0);
            } else {
                self.add("geo_lon", 0.5);
                self.add("geo_lat", 0.5);
            }
        } else {
            self.add("has_geo_context", 0.0);
            self.add("is_eu_region", 0.0);
            self.add("geo_lon", 0.5);
            self.add("geo_lat", 0.5);
        }

        // Device features
        #[cfg(any(feature = "device", feature = "std"))]
        if let Some(ref device) = ctx.device {
            self.add("has_device_context", 1.0);
            self.add(
                "device_constrained",
                if device.is_constrained() { 1.0 } else { 0.0 },
            );
            self.add("network_quality", device.network_quality());
            self.add("resource_availability", device.resource_availability());
        } else {
            self.add("has_device_context", 0.0);
            self.add("device_constrained", 0.0);
            self.add("network_quality", 0.5);
            self.add("resource_availability", 1.0);
        }

        // Load features
        #[cfg(any(feature = "load", feature = "std"))]
        if let Some(ref load) = ctx.load {
            self.add("has_load_context", 1.0);
            self.add("global_load", load.global_load);
            self.add(
                "is_overloaded",
                if load.is_overloaded() { 1.0 } else { 0.0 },
            );
        } else {
            self.add("has_load_context", 0.0);
            self.add("global_load", 0.0);
            self.add("is_overloaded", 0.0);
        }

        // Legal features
        #[cfg(any(feature = "legal", feature = "std"))]
        if let Some(ref legal) = ctx.legal {
            self.add("has_legal_context", 1.0);
            self.add("gdpr_region", if legal.gdpr_region { 1.0 } else { 0.0 });
            self.add(
                "data_transfer_allowed",
                if legal.data_transfer_allowed {
                    1.0
                } else {
                    0.0
                },
            );
            self.add("compliance_score", legal.compliance_score());
        } else {
            self.add("has_legal_context", 0.0);
            self.add("gdpr_region", 0.0);
            self.add("data_transfer_allowed", 1.0);
            self.add("compliance_score", 1.0);
        }
    }

    /// Add placeholder features when context is not available
    fn add_placeholder_context_features(&mut self) {
        // Geo placeholders
        self.add("has_geo_context", 0.0);
        self.add("is_eu_region", 0.0);
        self.add("geo_lon", 0.5);
        self.add("geo_lat", 0.5);

        // Device placeholders
        self.add("has_device_context", 0.0);
        self.add("device_constrained", 0.0);
        self.add("network_quality", 0.5);
        self.add("resource_availability", 1.0);

        // Load placeholders
        self.add("has_load_context", 0.0);
        self.add("global_load", 0.0);
        self.add("is_overloaded", 0.0);

        // Legal placeholders
        self.add("has_legal_context", 0.0);
        self.add("gdpr_region", 0.0);
        self.add("data_transfer_allowed", 1.0);
        self.add("compliance_score", 1.0);
    }

    /// Compute dot product with weights
    #[must_use]
    pub fn dot(&self, weights: &[f32]) -> f32 {
        self.values
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    /// Compute L2 norm
    #[must_use]
    pub fn norm(&self) -> f32 {
        #[cfg(feature = "ml")]
        {
            libm::sqrtf(self.values.iter().map(|v| v * v).sum())
        }
        #[cfg(not(feature = "ml"))]
        {
            self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
        }
    }

    /// Normalize the feature vector to unit length
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_creation() {
        let mut fv = FeatureVector::new();
        fv.add("feature1", 0.5);
        fv.add("feature2", 0.8);

        assert_eq!(fv.len(), 2);
        assert_eq!(fv.get("feature1"), Some(0.5));
        assert_eq!(fv.get("feature2"), Some(0.8));
    }

    #[test]
    fn test_normalization() {
        let mut fv = FeatureVector::new();
        fv.add_normalized("test", 50.0, 0.0, 100.0);
        assert!((fv.get("test").unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_from_query() {
        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let features = FeatureVector::from_query(&query).unwrap();

        assert!(!features.is_empty());
        assert_eq!(features.get("query_type_select"), Some(1.0));
    }

    #[test]
    fn test_dot_product() {
        let mut fv = FeatureVector::new();
        fv.add("a", 1.0);
        fv.add("b", 2.0);
        fv.add("c", 3.0);

        let weights = [0.5, 0.5, 0.5];
        assert!((fv.dot(&weights) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let mut fv = FeatureVector::new();
        fv.add("x", 3.0);
        fv.add("y", 4.0);
        fv.normalize();

        assert!((fv.norm() - 1.0).abs() < 0.001);
    }
}
