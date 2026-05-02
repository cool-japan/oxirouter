//! Result aggregation for federated queries

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::executor::{QueryResult, ResultFormat};
use crate::core::error::{OxiRouterError, Result};

/// Strategy for aggregating results from multiple sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Use first successful result
    First,
    /// Merge all results (union)
    Union,
    /// Intersect results (only common bindings)
    Intersect,
    /// Concatenate results preserving source info
    Concat,
    /// Use result with most bindings
    Largest,
    /// Use result with fastest response
    Fastest,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::First
    }
}

/// Configuration for result aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
    /// Whether to deduplicate results
    pub deduplicate: bool,
    /// Maximum total results
    pub max_results: u32,
    /// Whether to include source provenance
    pub include_provenance: bool,
    /// Output format
    pub output_format: ResultFormat,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            strategy: AggregationStrategy::First,
            deduplicate: true,
            max_results: 10000,
            include_provenance: false,
            output_format: ResultFormat::Json,
        }
    }
}

/// A single SPARQL binding value (e.g., URI, literal, blank node)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct SparqlValue {
    /// Value type: "uri", "literal", "bnode", "typed-literal"
    #[serde(rename = "type")]
    pub value_type: String,
    /// The actual value
    pub value: String,
    /// Optional language tag for literals
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Optional datatype IRI for typed literals
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
}

impl SparqlValue {
    /// Create a new URI value
    #[must_use]
    pub fn uri(value: impl Into<String>) -> Self {
        Self {
            value_type: "uri".to_string(),
            value: value.into(),
            lang: None,
            datatype: None,
        }
    }

    /// Create a new literal value
    #[must_use]
    pub fn literal(value: impl Into<String>) -> Self {
        Self {
            value_type: "literal".to_string(),
            value: value.into(),
            lang: None,
            datatype: None,
        }
    }

    /// Create a new typed literal value
    #[must_use]
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self {
            value_type: "typed-literal".to_string(),
            value: value.into(),
            lang: None,
            datatype: Some(datatype.into()),
        }
    }

    /// Create a new language-tagged literal
    #[must_use]
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        Self {
            value_type: "literal".to_string(),
            value: value.into(),
            lang: Some(lang.into()),
            datatype: None,
        }
    }

    /// Create a blank node
    #[must_use]
    pub fn bnode(value: impl Into<String>) -> Self {
        Self {
            value_type: "bnode".to_string(),
            value: value.into(),
            lang: None,
            datatype: None,
        }
    }

    /// Convert to canonical string for comparison/hashing
    #[must_use]
    pub fn to_canonical_string(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.value_type);
        s.push(':');
        s.push_str(&self.value);
        if let Some(ref lang) = self.lang {
            s.push('@');
            s.push_str(lang);
        }
        if let Some(ref dt) = self.datatype {
            s.push_str("^^");
            s.push_str(dt);
        }
        s
    }
}

/// A single SPARQL result binding (one row in the result set)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlBinding {
    /// Map from variable name to value
    #[serde(flatten)]
    pub values: HashMap<String, SparqlValue>,
}

impl SparqlBinding {
    /// Create a new empty binding
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Add a value to the binding
    pub fn insert(&mut self, var: impl Into<String>, value: SparqlValue) {
        self.values.insert(var.into(), value);
    }

    /// Get a value from the binding
    #[must_use]
    pub fn get(&self, var: &str) -> Option<&SparqlValue> {
        self.values.get(var)
    }

    /// Check if binding has a variable
    #[must_use]
    pub fn contains(&self, var: &str) -> bool {
        self.values.contains_key(var)
    }

    /// Get all variable names in this binding
    #[must_use]
    pub fn variables(&self) -> Vec<&String> {
        self.values.keys().collect()
    }

    /// Convert to canonical string for deduplication
    #[must_use]
    pub fn to_canonical_string(&self) -> String {
        let mut parts: Vec<_> = self
            .values
            .iter()
            .map(|(k, v)| format!("{}={}", k, v.to_canonical_string()))
            .collect();
        parts.sort();
        parts.join("|")
    }
}

impl Default for SparqlBinding {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed SPARQL JSON result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlJsonResult {
    /// Variable names from head.vars
    pub variables: Vec<String>,
    /// Result bindings
    pub bindings: Vec<SparqlBinding>,
}

impl SparqlJsonResult {
    /// Create a new empty result
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            bindings: Vec::new(),
        }
    }

    /// Create with variables
    #[must_use]
    pub fn with_variables(variables: Vec<String>) -> Self {
        Self {
            variables,
            bindings: Vec::new(),
        }
    }

    /// Parse from JSON bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Ok(Self::new());
        }

        let json: Value = serde_json::from_slice(data)
            .map_err(|e| OxiRouterError::ExecutionError(format!("JSON parse error: {}", e)))?;

        Self::parse_value(&json)
    }

    /// Parse from serde_json Value
    pub fn parse_value(json: &Value) -> Result<Self> {
        let obj = json
            .as_object()
            .ok_or_else(|| OxiRouterError::ExecutionError("Expected JSON object".to_string()))?;

        // Parse head.vars
        let variables = Self::parse_head_vars(obj)?;

        // Parse results.bindings
        let bindings = Self::parse_bindings(obj)?;

        Ok(Self {
            variables,
            bindings,
        })
    }

    /// Parse head.vars array
    fn parse_head_vars(obj: &Map<String, Value>) -> Result<Vec<String>> {
        let Some(head_val) = obj.get("head") else {
            return Ok(Vec::new());
        };

        let head = head_val
            .as_object()
            .ok_or_else(|| OxiRouterError::ExecutionError("head must be an object".to_string()))?;

        let Some(vars_val) = head.get("vars") else {
            return Ok(Vec::new());
        };

        let vars = vars_val
            .as_array()
            .ok_or_else(|| OxiRouterError::ExecutionError("head.vars must be array".to_string()))?;

        let mut result = Vec::new();
        for var in vars {
            if let Some(s) = var.as_str() {
                result.push(s.to_string());
            }
        }

        Ok(result)
    }

    /// Parse results.bindings array
    fn parse_bindings(obj: &Map<String, Value>) -> Result<Vec<SparqlBinding>> {
        let Some(results_val) = obj.get("results") else {
            return Ok(Vec::new());
        };

        let results = results_val.as_object().ok_or_else(|| {
            OxiRouterError::ExecutionError("results must be an object".to_string())
        })?;

        let Some(bindings_val) = results.get("bindings") else {
            return Ok(Vec::new());
        };

        let bindings = bindings_val.as_array().ok_or_else(|| {
            OxiRouterError::ExecutionError("results.bindings must be array".to_string())
        })?;

        let mut result = Vec::new();
        for binding_val in bindings {
            let binding_obj = binding_val.as_object().ok_or_else(|| {
                OxiRouterError::ExecutionError("binding must be an object".to_string())
            })?;

            let binding = Self::parse_single_binding(binding_obj)?;
            result.push(binding);
        }

        Ok(result)
    }

    /// Parse a single binding object
    fn parse_single_binding(obj: &Map<String, Value>) -> Result<SparqlBinding> {
        let mut binding = SparqlBinding::new();

        for (var, val_obj) in obj {
            let val = val_obj.as_object().ok_or_else(|| {
                OxiRouterError::ExecutionError(format!("Value for {} must be object", var))
            })?;

            let value_type = val
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("literal")
                .to_string();

            let value = val
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let lang = val
                .get("xml:lang")
                .and_then(|v| v.as_str())
                .map(String::from);

            let datatype = val
                .get("datatype")
                .and_then(|v| v.as_str())
                .map(String::from);

            binding.insert(
                var.clone(),
                SparqlValue {
                    value_type,
                    value,
                    lang,
                    datatype,
                },
            );
        }

        Ok(binding)
    }

    /// Convert back to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        let json = self.to_json_value();
        serde_json::to_vec(&json)
            .map_err(|e| OxiRouterError::ExecutionError(format!("JSON serialize error: {}", e)))
    }

    /// Convert to serde_json Value
    #[must_use]
    pub fn to_json_value(&self) -> Value {
        let mut head = Map::new();
        head.insert(
            "vars".to_string(),
            Value::Array(
                self.variables
                    .iter()
                    .map(|v| Value::String(v.clone()))
                    .collect(),
            ),
        );

        let bindings_arr: Vec<Value> = self
            .bindings
            .iter()
            .map(|b| {
                let mut obj = Map::new();
                for (var, val) in &b.values {
                    let mut val_obj = Map::new();
                    val_obj.insert("type".to_string(), Value::String(val.value_type.clone()));
                    val_obj.insert("value".to_string(), Value::String(val.value.clone()));
                    if let Some(ref lang) = val.lang {
                        val_obj.insert("xml:lang".to_string(), Value::String(lang.clone()));
                    }
                    if let Some(ref dt) = val.datatype {
                        val_obj.insert("datatype".to_string(), Value::String(dt.clone()));
                    }
                    obj.insert(var.clone(), Value::Object(val_obj));
                }
                Value::Object(obj)
            })
            .collect();

        let mut results = Map::new();
        results.insert("bindings".to_string(), Value::Array(bindings_arr));

        let mut root = Map::new();
        root.insert("head".to_string(), Value::Object(head));
        root.insert("results".to_string(), Value::Object(results));

        Value::Object(root)
    }

    /// Merge bindings from another result (union)
    pub fn merge(&mut self, other: &SparqlJsonResult) {
        // Merge variables
        for var in &other.variables {
            if !self.variables.contains(var) {
                self.variables.push(var.clone());
            }
        }

        // Append bindings
        self.bindings.extend(other.bindings.iter().cloned());
    }

    /// Deduplicate bindings using canonical string comparison
    pub fn deduplicate(&mut self) {
        let mut seen: HashSet<String> = HashSet::new();
        let mut unique = Vec::new();

        for binding in &self.bindings {
            let canonical = binding.to_canonical_string();
            if !seen.contains(&canonical) {
                seen.insert(canonical);
                unique.push(binding.clone());
            }
        }

        self.bindings = unique;
    }

    /// Intersect with another result (keep only common bindings)
    pub fn intersect(&mut self, other: &SparqlJsonResult) {
        // Build set of canonical strings from other
        let other_set: HashSet<String> = other
            .bindings
            .iter()
            .map(SparqlBinding::to_canonical_string)
            .collect();

        // Keep only bindings present in both
        self.bindings
            .retain(|b| other_set.contains(&b.to_canonical_string()));
    }

    /// Add source provenance to all bindings
    pub fn add_provenance(&mut self, source_id: &str) {
        // Add _source to variables if not present
        let source_var = "_source".to_string();
        if !self.variables.contains(&source_var) {
            self.variables.push(source_var.clone());
        }

        // Add source to each binding
        for binding in &mut self.bindings {
            binding.insert(source_var.clone(), SparqlValue::literal(source_id));
        }
    }

    /// Truncate bindings to max count
    pub fn truncate(&mut self, max: usize) {
        if self.bindings.len() > max {
            self.bindings.truncate(max);
        }
    }

    /// Get binding count
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl Default for SparqlJsonResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated result from multiple sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedResult {
    /// Combined result data
    pub data: Vec<u8>,
    /// Result format
    pub format: ResultFormat,
    /// Total row count
    pub row_count: u32,
    /// Sources that contributed
    pub sources: Vec<String>,
    /// Total execution time (max of all sources)
    pub total_latency_ms: u32,
    /// Number of sources that succeeded
    pub successful_sources: u32,
    /// Number of sources that failed
    pub failed_sources: u32,
    /// Whether results were deduplicated
    pub deduplicated: bool,
    /// Whether results were truncated
    pub truncated: bool,
}

impl AggregatedResult {
    /// Create an empty result
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            data: Vec::new(),
            format: ResultFormat::Json,
            row_count: 0,
            sources: Vec::new(),
            total_latency_ms: 0,
            successful_sources: 0,
            failed_sources: 0,
            deduplicated: false,
            truncated: false,
        }
    }

    /// Check if result is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Get result size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Result aggregator
pub struct Aggregator {
    /// Configuration
    config: AggregationConfig,
}

impl Aggregator {
    /// Create a new aggregator
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: AggregationConfig {
                strategy: AggregationStrategy::First,
                deduplicate: true,
                max_results: 10000,
                include_provenance: false,
                output_format: ResultFormat::Json,
            },
        }
    }

    /// Create aggregator with config
    #[must_use]
    pub const fn with_config(config: AggregationConfig) -> Self {
        Self { config }
    }

    /// Set aggregation strategy, consuming and returning self (builder pattern)
    #[must_use]
    pub fn with_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Get the configuration
    #[must_use]
    pub const fn config(&self) -> &AggregationConfig {
        &self.config
    }

    /// Aggregate multiple query results
    ///
    /// # Errors
    ///
    /// Returns error if aggregation fails
    pub fn aggregate(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        if results.is_empty() {
            return Ok(AggregatedResult::empty());
        }

        match self.config.strategy {
            AggregationStrategy::First => self.aggregate_first(results),
            AggregationStrategy::Union => self.aggregate_union(results),
            AggregationStrategy::Intersect => self.aggregate_intersect(results),
            AggregationStrategy::Concat => self.aggregate_concat(results),
            AggregationStrategy::Largest => self.aggregate_largest(results),
            AggregationStrategy::Fastest => self.aggregate_fastest(results),
        }
    }

    /// Use first successful result
    fn aggregate_first(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        let successful: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
        let failed_count = results.len() - successful.len();

        if successful.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "All sources failed".to_string(),
            ));
        }

        let first = &successful[0];

        Ok(AggregatedResult {
            data: first.data.clone(),
            format: first.format,
            row_count: first.row_count,
            sources: vec![first.source_id.clone()],
            total_latency_ms: first.latency_ms,
            successful_sources: 1,
            failed_sources: failed_count as u32,
            deduplicated: false,
            truncated: first.truncated,
        })
    }

    /// Merge all results (union) with proper SPARQL JSON parsing
    fn aggregate_union(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        let successful: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
        let failed_count = results.len() - successful.len();

        if successful.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "All sources failed".to_string(),
            ));
        }

        // Parse all results
        let mut combined = SparqlJsonResult::new();
        let mut max_latency = 0u32;
        let mut sources = Vec::new();

        for result in &successful {
            max_latency = max_latency.max(result.latency_ms);
            sources.push(result.source_id.clone());

            // Parse JSON result
            if let Ok(parsed) = SparqlJsonResult::parse(&result.data) {
                combined.merge(&parsed);
            }
        }

        // Apply deduplication if configured
        if self.config.deduplicate {
            combined.deduplicate();
        }

        // Apply max results limit
        let truncated = combined.len() > self.config.max_results as usize;
        if truncated {
            combined.truncate(self.config.max_results as usize);
        }

        // Convert back to JSON
        let data = combined.to_json_bytes()?;
        let row_count = combined.len() as u32;

        Ok(AggregatedResult {
            data,
            format: ResultFormat::Json,
            row_count,
            sources,
            total_latency_ms: max_latency,
            successful_sources: successful.len() as u32,
            failed_sources: failed_count as u32,
            deduplicated: self.config.deduplicate,
            truncated,
        })
    }

    /// Intersect results (common bindings only) with proper SPARQL JSON parsing
    fn aggregate_intersect(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        let successful: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
        let failed_count = results.len() - successful.len();

        if successful.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "All sources failed".to_string(),
            ));
        }

        // Parse first result as base
        let mut combined = SparqlJsonResult::parse(&successful[0].data)?;
        let mut max_latency = successful[0].latency_ms;
        let mut sources = vec![successful[0].source_id.clone()];

        // Intersect with remaining results
        for result in successful.iter().skip(1) {
            max_latency = max_latency.max(result.latency_ms);
            sources.push(result.source_id.clone());

            if let Ok(parsed) = SparqlJsonResult::parse(&result.data) {
                combined.intersect(&parsed);
            }
        }

        // Apply deduplication if configured (intersection already removes duplicates across sources)
        if self.config.deduplicate {
            combined.deduplicate();
        }

        // Apply max results limit
        let truncated = combined.len() > self.config.max_results as usize;
        if truncated {
            combined.truncate(self.config.max_results as usize);
        }

        // Convert back to JSON
        let data = combined.to_json_bytes()?;
        let row_count = combined.len() as u32;

        Ok(AggregatedResult {
            data,
            format: ResultFormat::Json,
            row_count,
            sources,
            total_latency_ms: max_latency,
            successful_sources: successful.len() as u32,
            failed_sources: failed_count as u32,
            deduplicated: self.config.deduplicate,
            truncated,
        })
    }

    /// Concatenate results with source provenance
    fn aggregate_concat(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        let successful: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
        let failed_count = results.len() - successful.len();

        if successful.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "All sources failed".to_string(),
            ));
        }

        // Parse all results and add provenance
        let mut combined = SparqlJsonResult::new();
        let mut max_latency = 0u32;
        let mut sources = Vec::new();

        for result in &successful {
            max_latency = max_latency.max(result.latency_ms);
            sources.push(result.source_id.clone());

            // Parse JSON result
            if let Ok(mut parsed) = SparqlJsonResult::parse(&result.data) {
                // Add source provenance to each binding
                parsed.add_provenance(&result.source_id);
                combined.merge(&parsed);
            }
        }

        // Apply deduplication if configured
        if self.config.deduplicate {
            combined.deduplicate();
        }

        // Apply max results limit
        let truncated = combined.len() > self.config.max_results as usize;
        if truncated {
            combined.truncate(self.config.max_results as usize);
        }

        // Convert back to JSON
        let data = combined.to_json_bytes()?;
        let row_count = combined.len() as u32;

        Ok(AggregatedResult {
            data,
            format: ResultFormat::Json,
            row_count,
            sources,
            total_latency_ms: max_latency,
            successful_sources: successful.len() as u32,
            failed_sources: failed_count as u32,
            deduplicated: self.config.deduplicate,
            truncated,
        })
    }

    /// Use result with most bindings
    fn aggregate_largest(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        let successful: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
        let failed_count = results.len() - successful.len();

        if successful.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "All sources failed".to_string(),
            ));
        }

        let largest = successful
            .iter()
            .max_by_key(|r| r.row_count)
            .expect("successful is non-empty, checked above");

        Ok(AggregatedResult {
            data: largest.data.clone(),
            format: largest.format,
            row_count: largest.row_count,
            sources: vec![largest.source_id.clone()],
            total_latency_ms: largest.latency_ms,
            successful_sources: 1,
            failed_sources: failed_count as u32,
            deduplicated: false,
            truncated: largest.truncated,
        })
    }

    /// Use result with fastest response
    fn aggregate_fastest(&self, results: &[QueryResult]) -> Result<AggregatedResult> {
        let successful: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
        let failed_count = results.len() - successful.len();

        if successful.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "All sources failed".to_string(),
            ));
        }

        let fastest = successful
            .iter()
            .min_by_key(|r| r.latency_ms)
            .expect("successful is non-empty, checked above");

        Ok(AggregatedResult {
            data: fastest.data.clone(),
            format: fastest.format,
            row_count: fastest.row_count,
            sources: vec![fastest.source_id.clone()],
            total_latency_ms: fastest.latency_ms,
            successful_sources: 1,
            failed_sources: failed_count as u32,
            deduplicated: false,
            truncated: fastest.truncated,
        })
    }
}

impl Default for Aggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result(source_id: &str, row_count: u32, latency: u32) -> QueryResult {
        QueryResult::success(source_id, vec![], row_count, latency)
    }

    fn create_sparql_json_result(bindings: &[(&str, &str, &str)]) -> Vec<u8> {
        let mut result = SparqlJsonResult::with_variables(vec![
            "s".to_string(),
            "p".to_string(),
            "o".to_string(),
        ]);

        for (s, p, o) in bindings {
            let mut binding = SparqlBinding::new();
            binding.insert("s", SparqlValue::uri(*s));
            binding.insert("p", SparqlValue::uri(*p));
            binding.insert("o", SparqlValue::literal(*o));
            result.bindings.push(binding);
        }

        result.to_json_bytes().unwrap()
    }

    fn create_json_query_result(
        source_id: &str,
        bindings: &[(&str, &str, &str)],
        latency: u32,
    ) -> QueryResult {
        let data = create_sparql_json_result(bindings);
        let row_count = bindings.len() as u32;
        QueryResult::success(source_id, data, row_count, latency)
    }

    #[test]
    fn test_sparql_value_creation() {
        let uri = SparqlValue::uri("http://example.org/s");
        assert_eq!(uri.value_type, "uri");
        assert_eq!(uri.value, "http://example.org/s");

        let lit = SparqlValue::literal("hello");
        assert_eq!(lit.value_type, "literal");
        assert_eq!(lit.value, "hello");

        let typed = SparqlValue::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert_eq!(typed.value_type, "typed-literal");
        assert_eq!(
            typed.datatype,
            Some("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );

        let lang_lit = SparqlValue::lang_literal("hello", "en");
        assert_eq!(lang_lit.lang, Some("en".to_string()));
    }

    #[test]
    fn test_sparql_binding_operations() {
        let mut binding = SparqlBinding::new();
        binding.insert("s", SparqlValue::uri("http://example.org/s1"));
        binding.insert("p", SparqlValue::uri("http://example.org/p1"));

        assert!(binding.contains("s"));
        assert!(!binding.contains("o"));

        let val = binding.get("s").unwrap();
        assert_eq!(val.value, "http://example.org/s1");
    }

    #[test]
    fn test_sparql_json_parse() {
        let json = r#"{
            "head": {"vars": ["s", "p", "o"]},
            "results": {
                "bindings": [
                    {
                        "s": {"type": "uri", "value": "http://example.org/s1"},
                        "p": {"type": "uri", "value": "http://example.org/p1"},
                        "o": {"type": "literal", "value": "value1"}
                    }
                ]
            }
        }"#;

        let result = SparqlJsonResult::parse(json.as_bytes()).unwrap();
        assert_eq!(result.variables, vec!["s", "p", "o"]);
        assert_eq!(result.bindings.len(), 1);

        let binding = &result.bindings[0];
        assert_eq!(binding.get("s").unwrap().value, "http://example.org/s1");
    }

    #[test]
    fn test_sparql_json_parse_empty() {
        let result = SparqlJsonResult::parse(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparql_json_roundtrip() {
        let mut original = SparqlJsonResult::with_variables(vec!["s".to_string(), "p".to_string()]);
        let mut binding = SparqlBinding::new();
        binding.insert("s", SparqlValue::uri("http://example.org/s1"));
        binding.insert("p", SparqlValue::uri("http://example.org/p1"));
        original.bindings.push(binding);

        let bytes = original.to_json_bytes().unwrap();
        let parsed = SparqlJsonResult::parse(&bytes).unwrap();

        assert_eq!(parsed.variables.len(), original.variables.len());
        assert_eq!(parsed.bindings.len(), original.bindings.len());
    }

    #[test]
    fn test_sparql_json_merge() {
        let mut result1 = SparqlJsonResult::with_variables(vec!["s".to_string()]);
        let mut b1 = SparqlBinding::new();
        b1.insert("s", SparqlValue::uri("http://example.org/s1"));
        result1.bindings.push(b1);

        let mut result2 = SparqlJsonResult::with_variables(vec!["s".to_string()]);
        let mut b2 = SparqlBinding::new();
        b2.insert("s", SparqlValue::uri("http://example.org/s2"));
        result2.bindings.push(b2);

        result1.merge(&result2);
        assert_eq!(result1.bindings.len(), 2);
    }

    #[test]
    fn test_sparql_json_deduplicate() {
        let mut result = SparqlJsonResult::with_variables(vec!["s".to_string()]);

        let mut b1 = SparqlBinding::new();
        b1.insert("s", SparqlValue::uri("http://example.org/s1"));
        result.bindings.push(b1);

        let mut b2 = SparqlBinding::new();
        b2.insert("s", SparqlValue::uri("http://example.org/s1")); // duplicate
        result.bindings.push(b2);

        let mut b3 = SparqlBinding::new();
        b3.insert("s", SparqlValue::uri("http://example.org/s2")); // different
        result.bindings.push(b3);

        result.deduplicate();
        assert_eq!(result.bindings.len(), 2);
    }

    #[test]
    fn test_sparql_json_intersect() {
        let mut result1 = SparqlJsonResult::with_variables(vec!["s".to_string()]);
        let mut b1a = SparqlBinding::new();
        b1a.insert("s", SparqlValue::uri("http://example.org/common"));
        result1.bindings.push(b1a);
        let mut b1b = SparqlBinding::new();
        b1b.insert("s", SparqlValue::uri("http://example.org/only1"));
        result1.bindings.push(b1b);

        let mut result2 = SparqlJsonResult::with_variables(vec!["s".to_string()]);
        let mut b2a = SparqlBinding::new();
        b2a.insert("s", SparqlValue::uri("http://example.org/common"));
        result2.bindings.push(b2a);
        let mut b2b = SparqlBinding::new();
        b2b.insert("s", SparqlValue::uri("http://example.org/only2"));
        result2.bindings.push(b2b);

        result1.intersect(&result2);
        assert_eq!(result1.bindings.len(), 1);
        assert_eq!(
            result1.bindings[0].get("s").unwrap().value,
            "http://example.org/common"
        );
    }

    #[test]
    fn test_sparql_json_add_provenance() {
        let mut result = SparqlJsonResult::with_variables(vec!["s".to_string()]);
        let mut binding = SparqlBinding::new();
        binding.insert("s", SparqlValue::uri("http://example.org/s1"));
        result.bindings.push(binding);

        result.add_provenance("source1");

        assert!(result.variables.contains(&"_source".to_string()));
        assert_eq!(result.bindings[0].get("_source").unwrap().value, "source1");
    }

    #[test]
    fn test_aggregate_first() {
        let aggregator = Aggregator::new();
        let results = vec![
            create_test_result("src1", 10, 100),
            create_test_result("src2", 20, 50),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 10);
        assert_eq!(agg.sources, vec!["src1"]);
    }

    #[test]
    fn test_aggregate_largest() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Largest,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![
            create_test_result("src1", 10, 100),
            create_test_result("src2", 50, 200),
            create_test_result("src3", 30, 150),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 50);
        assert_eq!(agg.sources, vec!["src2"]);
    }

    #[test]
    fn test_aggregate_fastest() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Fastest,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![
            create_test_result("src1", 10, 100),
            create_test_result("src2", 50, 50),
            create_test_result("src3", 30, 150),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.total_latency_ms, 50);
        assert_eq!(agg.sources, vec!["src2"]);
    }

    #[test]
    fn test_aggregate_union_with_json() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Union,
            deduplicate: true,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![
            create_json_query_result(
                "src1",
                &[
                    ("http://ex.org/s1", "http://ex.org/p", "val1"),
                    ("http://ex.org/s2", "http://ex.org/p", "val2"),
                ],
                100,
            ),
            create_json_query_result(
                "src2",
                &[
                    ("http://ex.org/s2", "http://ex.org/p", "val2"), // duplicate
                    ("http://ex.org/s3", "http://ex.org/p", "val3"),
                ],
                200,
            ),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 3); // 4 total, 1 duplicate removed
        assert_eq!(agg.sources.len(), 2);
        assert!(agg.deduplicated);
        assert_eq!(agg.total_latency_ms, 200);
    }

    #[test]
    fn test_aggregate_union_no_dedup() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Union,
            deduplicate: false,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![
            create_json_query_result(
                "src1",
                &[("http://ex.org/s1", "http://ex.org/p", "val1")],
                100,
            ),
            create_json_query_result(
                "src2",
                &[("http://ex.org/s1", "http://ex.org/p", "val1")], // duplicate kept
                200,
            ),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 2); // duplicates kept
        assert!(!agg.deduplicated);
    }

    #[test]
    fn test_aggregate_intersect_with_json() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Intersect,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![
            create_json_query_result(
                "src1",
                &[
                    ("http://ex.org/common", "http://ex.org/p", "val"),
                    ("http://ex.org/only1", "http://ex.org/p", "val1"),
                ],
                100,
            ),
            create_json_query_result(
                "src2",
                &[
                    ("http://ex.org/common", "http://ex.org/p", "val"),
                    ("http://ex.org/only2", "http://ex.org/p", "val2"),
                ],
                200,
            ),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 1); // only common binding
        assert_eq!(agg.sources.len(), 2);

        // Parse result to verify content
        let parsed = SparqlJsonResult::parse(&agg.data).unwrap();
        assert_eq!(
            parsed.bindings[0].get("s").unwrap().value,
            "http://ex.org/common"
        );
    }

    #[test]
    fn test_aggregate_concat_with_provenance() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Concat,
            deduplicate: false,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![
            create_json_query_result(
                "src1",
                &[("http://ex.org/s1", "http://ex.org/p", "val1")],
                100,
            ),
            create_json_query_result(
                "src2",
                &[("http://ex.org/s2", "http://ex.org/p", "val2")],
                200,
            ),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 2);

        // Parse result to verify provenance
        let parsed = SparqlJsonResult::parse(&agg.data).unwrap();
        assert!(parsed.variables.contains(&"_source".to_string()));
        assert_eq!(parsed.bindings[0].get("_source").unwrap().value, "src1");
        assert_eq!(parsed.bindings[1].get("_source").unwrap().value, "src2");
    }

    #[test]
    fn test_aggregate_with_failures() {
        let aggregator = Aggregator::new();
        let results = vec![
            QueryResult::error("src1", "Failed", 0),
            create_test_result("src2", 20, 100),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.successful_sources, 1);
        assert_eq!(agg.failed_sources, 1);
    }

    #[test]
    fn test_aggregate_all_failures() {
        let aggregator = Aggregator::new();
        let results = vec![
            QueryResult::error("src1", "Failed", 0),
            QueryResult::error("src2", "Failed", 0),
        ];

        let result = aggregator.aggregate(&results);
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_union_truncation() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Union,
            deduplicate: false,
            max_results: 2,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        let results = vec![create_json_query_result(
            "src1",
            &[
                ("http://ex.org/s1", "http://ex.org/p", "val1"),
                ("http://ex.org/s2", "http://ex.org/p", "val2"),
                ("http://ex.org/s3", "http://ex.org/p", "val3"),
            ],
            100,
        )];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 2);
        assert!(agg.truncated);
    }

    #[test]
    fn test_aggregate_empty_results() {
        let aggregator = Aggregator::new();
        let results: Vec<QueryResult> = vec![];

        let agg = aggregator.aggregate(&results).unwrap();
        assert!(agg.is_empty());
    }

    #[test]
    fn test_canonical_string_comparison() {
        let val1 = SparqlValue::uri("http://example.org/s1");
        let val2 = SparqlValue::uri("http://example.org/s1");
        assert_eq!(val1.to_canonical_string(), val2.to_canonical_string());

        let val3 = SparqlValue::lang_literal("hello", "en");
        let val4 = SparqlValue::lang_literal("hello", "de");
        assert_ne!(val3.to_canonical_string(), val4.to_canonical_string());
    }

    #[test]
    fn test_aggregate_intersect_empty() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Intersect,
            ..Default::default()
        };
        let aggregator = Aggregator::with_config(config);

        // Two results with no common bindings
        let results = vec![
            create_json_query_result(
                "src1",
                &[("http://ex.org/s1", "http://ex.org/p", "val1")],
                100,
            ),
            create_json_query_result(
                "src2",
                &[("http://ex.org/s2", "http://ex.org/p", "val2")],
                200,
            ),
        ];

        let agg = aggregator.aggregate(&results).unwrap();
        assert_eq!(agg.row_count, 0);
    }

    #[test]
    fn test_sparql_value_bnode() {
        let bnode = SparqlValue::bnode("b0");
        assert_eq!(bnode.value_type, "bnode");
        assert_eq!(bnode.value, "b0");
        assert_eq!(bnode.to_canonical_string(), "bnode:b0");
    }

    #[test]
    fn test_aggregate_config_default() {
        let config = AggregationConfig::default();
        assert_eq!(config.strategy, AggregationStrategy::First);
        assert!(config.deduplicate);
        assert_eq!(config.max_results, 10000);
    }

    #[test]
    fn test_aggregated_result_size() {
        let mut result = AggregatedResult::empty();
        assert_eq!(result.size_bytes(), 0);

        result.data = vec![1, 2, 3, 4, 5];
        assert_eq!(result.size_bytes(), 5);
    }
}
