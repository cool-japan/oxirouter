//! wasm-bindgen interfaces for JavaScript
//!
//! Exposes `OxiRouter` as a fully-featured WASM class with:
//! - Source management (add / remove / mark up-down)
//! - SPARQL query routing (`route_query`, `route_and_log_js`)
//! - RL feedback loop (`learn_from_outcome_js`)
//! - ML model loading / export (`load_model_bytes_js`, `save_model_bytes_js`)
//! - Context injection from JavaScript (`set_*_context_js`)
//! - Query log analytics (`query_log_summary_js`)

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use wasm_bindgen::prelude::*;

use crate::core::query::Query;
use crate::core::router::Router;
use crate::core::source::{DataSource, SelectionReason, SourceCapabilities};
use crate::wasm::context_provider::{WasmContextProvider, WasmRouterContextAdapter};

/// OxiRouter WASM interface.
#[wasm_bindgen]
pub struct OxiRouter {
    inner: Router<WasmRouterContextAdapter>,
    provider: Arc<WasmContextProvider>,
    #[cfg(feature = "ml")]
    loaded_model_bytes: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl OxiRouter {
    /// Create a new `OxiRouter` WASM instance with default configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        let provider = Arc::new(WasmContextProvider::new());
        let adapter = WasmRouterContextAdapter(Arc::clone(&provider));
        Self {
            inner: Router::with_context_provider(adapter),
            provider,
            #[cfg(feature = "ml")]
            loaded_model_bytes: None,
        }
    }

    /// Add a SPARQL endpoint as a data source with full capabilities.
    #[wasm_bindgen]
    pub fn add_source(&mut self, id: &str, endpoint: &str) {
        let source = DataSource::new(id, endpoint).with_capabilities(SourceCapabilities::full());
        self.inner.add_source(source);
    }

    /// Add a SPARQL endpoint with an associated geographic region tag.
    #[wasm_bindgen]
    pub fn add_source_with_region(&mut self, id: &str, endpoint: &str, region: &str) {
        let source = DataSource::new(id, endpoint)
            .with_capabilities(SourceCapabilities::full())
            .with_region(region);
        self.inner.add_source(source);
    }

    /// Add a SPARQL endpoint with an associated vocabulary namespace.
    #[wasm_bindgen]
    pub fn add_source_with_vocabulary(&mut self, id: &str, endpoint: &str, vocabulary: &str) {
        let source = DataSource::new(id, endpoint)
            .with_capabilities(SourceCapabilities::full())
            .with_vocabulary(vocabulary);
        self.inner.add_source(source);
    }

    /// Remove a data source by ID. Returns `true` if it existed and was removed.
    #[wasm_bindgen]
    pub fn remove_source(&mut self, id: &str) -> bool {
        self.inner.remove_source(id).is_some()
    }

    /// Return the number of registered data sources.
    #[wasm_bindgen]
    pub fn source_count(&self) -> usize {
        self.inner.source_count()
    }

    /// List all registered sources as a JS array of `{id, endpoint, available}` objects.
    #[wasm_bindgen]
    pub fn list_sources(&self) -> Result<js_sys::Array, JsValue> {
        let arr = js_sys::Array::new();
        for source in self.inner.sources() {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &"id".into(), &source.id.clone().into())
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
            js_sys::Reflect::set(&obj, &"endpoint".into(), &source.endpoint.clone().into())
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
            js_sys::Reflect::set(&obj, &"available".into(), &source.available.into())
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
            arr.push(&obj);
        }
        Ok(arr)
    }

    /// Route a SPARQL query and return ranking information as a JS object.
    #[wasm_bindgen]
    pub fn route_query(&self, query_str: &str) -> Result<JsValue, JsValue> {
        let query = Query::parse(query_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ranking = self
            .inner
            .route(&query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = js_sys::Object::new();
        let sources = build_sources_array(&ranking.sources)?;
        js_sys::Reflect::set(&result, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &result,
            &"processingTimeUs".into(),
            &(ranking.processing_time_us as f64).into(),
        )?;
        js_sys::Reflect::set(&result, &"mlUsed".into(), &ranking.ml_used.into())?;
        js_sys::Reflect::set(&result, &"contextUsed".into(), &ranking.context_used.into())?;
        Ok(result.into())
    }

    /// Route a query and persist the decision to the internal query log.
    #[wasm_bindgen]
    pub fn route_and_log_js(&mut self, query_str: &str) -> Result<JsValue, JsValue> {
        let query = Query::parse(query_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let ranking = self
            .inner
            .route_and_log(&query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = js_sys::Object::new();
        let sources = build_sources_array(&ranking.sources)?;
        js_sys::Reflect::set(&result, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &result,
            &"queryId".into(),
            &(query.predicate_hash() as f64).into(),
        )?;
        js_sys::Reflect::set(
            &result,
            &"processingTimeUs".into(),
            &(ranking.processing_time_us as f64).into(),
        )?;
        js_sys::Reflect::set(&result, &"mlUsed".into(), &ranking.ml_used.into())?;
        js_sys::Reflect::set(&result, &"contextUsed".into(), &ranking.context_used.into())?;
        Ok(result.into())
    }

    /// Record the outcome of a previously routed query for RL-driven adaptation.
    #[wasm_bindgen]
    pub fn learn_from_outcome_js(
        &mut self,
        query_id: f64,
        source_id: &str,
        success: bool,
        latency_ms: u32,
        result_count: u32,
    ) -> Result<(), JsValue> {
        self.inner
            .learn_from_outcome(
                query_id as u64,
                source_id,
                success,
                latency_ms,
                result_count,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Enable the RL policy. policy_type: 0=UCB, 1=EpsilonGreedy, 2=Thompson.
    #[cfg(feature = "rl")]
    #[wasm_bindgen]
    pub fn enable_rl_js(&mut self, policy_type: u8, _num_sources: u32) -> Result<(), JsValue> {
        use crate::rl::Policy;
        let policy = match policy_type {
            1 => Policy::epsilon_greedy(0.1),
            2 => Policy::thompson_sampling(),
            _ => Policy::ucb(),
        };
        self.inner.set_policy(policy);
        Ok(())
    }

    /// Inject geographic context from JavaScript (lon=longitude, lat=latitude, WGS-84).
    #[wasm_bindgen]
    pub fn set_geo_context_js(&self, country_code: &str, lon: f64, lat: f64) {
        #[cfg(feature = "geo")]
        self.provider.set_geo(country_code, lon, lat);
        #[cfg(not(feature = "geo"))]
        {
            let _ = (country_code, lon, lat);
        }
    }

    /// Inject device context. network_type: 0=Offline,1=Wifi,2=4G,3=5G,4=3G,5=2G,6=Ethernet,7=Satellite.
    #[wasm_bindgen]
    pub fn set_device_context_js(
        &self,
        battery_pct: u8,
        network_type: u8,
        bandwidth_kbps: u32,
        rtt_ms: u32,
    ) {
        self.provider
            .set_device(battery_pct, network_type, bandwidth_kbps, rtt_ms);
    }

    /// Inject load/workload context. global_load is clamped to [0.0, 1.0].
    #[wasm_bindgen]
    pub fn set_load_context_js(&self, global_load: f32, pending_tasks: u32) {
        self.provider.set_load(global_load, pending_tasks);
    }

    /// Inject legal/compliance context. blocked_regions_csv is comma-separated ISO 3166-1 codes.
    #[wasm_bindgen]
    pub fn set_legal_context_js(
        &self,
        gdpr_region: bool,
        ccpa_applies: bool,
        blocked_regions_csv: &str,
    ) {
        self.provider
            .set_legal(gdpr_region, ccpa_applies, blocked_regions_csv);
    }

    /// Load an ML model from serialized bytes (NaiveBayes or NeuralNetwork).
    #[cfg(feature = "ml")]
    #[wasm_bindgen]
    pub fn load_model_bytes_js(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.inner
            .load_model_from_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.loaded_model_bytes = Some(bytes.to_vec());
        Ok(())
    }

    /// Serialize the currently loaded ML model to bytes.
    #[cfg(feature = "ml")]
    #[wasm_bindgen]
    pub fn save_model_bytes_js(&self) -> Result<Box<[u8]>, JsValue> {
        self.loaded_model_bytes
            .as_ref()
            .map(|b| b.clone().into_boxed_slice())
            .ok_or_else(|| JsValue::from_str("No model loaded — call load_model_bytes_js first"))
    }

    /// Update historical performance statistics for a source after query execution.
    #[wasm_bindgen]
    pub fn update_feedback(
        &mut self,
        source_id: &str,
        latency_ms: u32,
        success: bool,
        result_count: u32,
    ) -> Result<(), JsValue> {
        self.inner
            .update_source_stats(source_id, latency_ms, success, result_count)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Mark a source as unavailable so the router will skip it until re-enabled.
    #[wasm_bindgen]
    pub fn mark_unavailable(&mut self, source_id: &str) -> Result<(), JsValue> {
        self.inner
            .mark_unavailable(source_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Re-enable a source that was previously marked unavailable.
    #[wasm_bindgen]
    pub fn mark_available(&mut self, source_id: &str) -> Result<(), JsValue> {
        self.inner
            .mark_available(source_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Set the maximum number of candidate sources returned per routing decision.
    #[wasm_bindgen]
    pub fn set_max_sources(&mut self, max: usize) {
        self.inner.config_mut().max_sources = max;
    }

    /// Set the minimum confidence threshold a source must exceed to be included in results.
    #[wasm_bindgen]
    pub fn set_min_confidence(&mut self, min: f32) {
        self.inner.config_mut().min_confidence = min;
    }

    /// Enable or disable ML-based source scoring (falls back to heuristics when disabled).
    #[wasm_bindgen]
    pub fn set_use_ml(&mut self, use_ml: bool) {
        self.inner.config_mut().use_ml = use_ml;
    }

    /// Enable or disable context-aware routing (geo, device, load, legal signals).
    #[wasm_bindgen]
    pub fn set_use_context(&mut self, use_context: bool) {
        self.inner.config_mut().use_context = use_context;
    }

    /// Retrieve live performance statistics for a single source as a JS object.
    #[wasm_bindgen]
    pub fn get_source_stats(&self, source_id: &str) -> Result<JsValue, JsValue> {
        let source = self
            .inner
            .get_source(source_id)
            .ok_or_else(|| JsValue::from_str("Source not found"))?;
        let stats = js_sys::Object::new();
        js_sys::Reflect::set(
            &stats,
            &"totalQueries".into(),
            &source.stats.total_queries.into(),
        )?;
        js_sys::Reflect::set(
            &stats,
            &"successfulQueries".into(),
            &source.stats.successful_queries.into(),
        )?;
        js_sys::Reflect::set(
            &stats,
            &"totalResults".into(),
            &(source.stats.total_results as f64).into(),
        )?;
        js_sys::Reflect::set(
            &stats,
            &"avgLatencyMs".into(),
            &source.stats.avg_latency_ms.into(),
        )?;
        js_sys::Reflect::set(
            &stats,
            &"successRate".into(),
            &source.stats.success_rate.into(),
        )?;
        Ok(stats.into())
    }

    /// Return a summary of the internal query log as a JS object.
    #[wasm_bindgen]
    pub fn query_log_summary_js(&self) -> Result<JsValue, JsValue> {
        let log = self.inner.query_log();
        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &"totalRecorded".into(),
            &(log.total_recorded as f64).into(),
        )
        .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
        let by_sources = js_sys::Array::new();
        for (source_id, score) in log.ranked_sources() {
            let entry = js_sys::Object::new();
            js_sys::Reflect::set(&entry, &"sourceId".into(), &source_id.clone().into())
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
            if let Some(stats) = log.source_stats(&source_id) {
                js_sys::Reflect::set(
                    &entry,
                    &"totalRouted".into(),
                    &(stats.total_routed as f64).into(),
                )
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
                js_sys::Reflect::set(&entry, &"successRate".into(), &stats.success_rate().into())
                    .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
                js_sys::Reflect::set(
                    &entry,
                    &"avgLatencyMs".into(),
                    &stats.avg_latency_ms().into(),
                )
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
                js_sys::Reflect::set(&entry, &"avgReward".into(), &stats.avg_reward().into())
                    .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
            }
            js_sys::Reflect::set(&entry, &"score".into(), &f64::from(score).into())
                .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
            by_sources.push(&entry);
        }
        js_sys::Reflect::set(&result, &"bySources".into(), &by_sources)
            .map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?;
        Ok(result.into())
    }

    /// Parse and analyze a SPARQL query, returning structural features as a JS object.
    #[wasm_bindgen]
    pub fn analyze_query(&self, query_str: &str) -> Result<JsValue, JsValue> {
        let query = Query::parse(query_str).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let analysis = js_sys::Object::new();
        let query_type = match query.query_type {
            crate::core::query::QueryType::Select => "SELECT",
            crate::core::query::QueryType::Construct => "CONSTRUCT",
            crate::core::query::QueryType::Ask => "ASK",
            crate::core::query::QueryType::Describe => "DESCRIBE",
        };
        js_sys::Reflect::set(&analysis, &"queryType".into(), &query_type.into())?;
        js_sys::Reflect::set(
            &analysis,
            &"triplePatternCount".into(),
            &query.triple_patterns.len().into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"predicateCount".into(),
            &query.predicates.len().into(),
        )?;
        js_sys::Reflect::set(&analysis, &"complexity".into(), &query.complexity.into())?;
        js_sys::Reflect::set(&analysis, &"hasOptional".into(), &query.has_optional.into())?;
        js_sys::Reflect::set(&analysis, &"hasUnion".into(), &query.has_union.into())?;
        js_sys::Reflect::set(&analysis, &"hasFilter".into(), &query.has_filter.into())?;
        js_sys::Reflect::set(
            &analysis,
            &"hasAggregation".into(),
            &query.has_aggregation.into(),
        )?;
        js_sys::Reflect::set(
            &analysis,
            &"requiresSparql11".into(),
            &query.requires_sparql_1_1().into(),
        )?;
        let predicates = js_sys::Array::new();
        for pred in &query.predicates {
            predicates.push(&pred.clone().into());
        }
        js_sys::Reflect::set(&analysis, &"predicates".into(), &predicates)?;
        let types = js_sys::Array::new();
        for t in &query.types {
            types.push(&t.clone().into());
        }
        js_sys::Reflect::set(&analysis, &"types".into(), &types)?;
        Ok(analysis.into())
    }

    /// Route a raw SPARQL string with real prefix expansion (requires `sparql` feature).
    ///
    /// Returns a JS object matching the shape of the standard `route_query` response.
    #[cfg(feature = "sparql")]
    #[wasm_bindgen]
    pub fn route_sparql_js(&self, sparql: &str) -> Result<JsValue, JsValue> {
        let ranking = self
            .inner
            .route_sparql(sparql)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = js_sys::Object::new();
        let sources = build_sources_array(&ranking.sources)?;
        js_sys::Reflect::set(&result, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &result,
            &"processingTimeUs".into(),
            &(ranking.processing_time_us as f64).into(),
        )?;
        js_sys::Reflect::set(&result, &"mlUsed".into(), &ranking.ml_used.into())?;
        js_sys::Reflect::set(&result, &"contextUsed".into(), &ranking.context_used.into())?;
        Ok(result.into())
    }

    /// Return a JSON string containing per-feature scoring breakdowns for every
    /// registered source given a SPARQL query string.
    ///
    /// Parses the query, calls `Router::explain`, and serializes the resulting
    /// `Vec<RoutingExplanation>` as a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` string error if SPARQL parsing, routing, or
    /// JSON serialization fails.
    #[wasm_bindgen]
    pub fn explain_query_js(&self, sparql: &str) -> Result<String, JsValue> {
        let query = Query::parse(sparql).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let explanations = self
            .inner
            .explain(&query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_json::to_string(&explanations).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Serialize the router's full learnable state (sources, ML model, RL policy,
    /// query log) to bytes.
    ///
    /// The returned `Vec<u8>` is marshalled as a `Uint8Array` by wasm-bindgen.
    /// Pass the bytes to [`OxiRouter::load_state_js`] to restore the router.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` string error if serialization fails.
    #[wasm_bindgen]
    pub fn save_state_js(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .save_state()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Restore the router state from bytes produced by [`OxiRouter::save_state_js`].
    ///
    /// Replaces sources, ML model, RL policy, and query log.  The router
    /// configuration (max sources, confidence threshold, etc.) is preserved.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` string error on magic/version mismatch or parse failure.
    #[wasm_bindgen]
    pub fn load_state_js(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.inner
            .load_state(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Update the circuit breaker configuration.
    ///
    /// - `failure_threshold`: consecutive failures before a source is tripped
    ///   (0 disables the circuit breaker entirely).
    /// - `cooldown_ms`: milliseconds a tripped source stays skipped.
    ///
    /// On WASM the clock function (`now_ms`) is set to `None` (disabled) because
    /// `SystemTime` is not available; the threshold still gates the trip but
    /// auto-recovery is disabled until `load_state_js` restores a fresh router.
    #[wasm_bindgen]
    pub fn set_circuit_breaker_config_js(
        &mut self,
        failure_threshold: u32,
        cooldown_ms: u32,
    ) -> Result<(), JsValue> {
        use crate::core::router::CircuitBreakerConfig;
        let cfg = CircuitBreakerConfig {
            failure_threshold,
            cooldown_ms: u64::from(cooldown_ms),
            now_ms: None, // no SystemTime in WASM no_std
        };
        self.inner.set_circuit_breaker_config(cfg);
        Ok(())
    }

    /// Run a federated query across all ranked sources and aggregate results.
    ///
    /// On WASM targets without a full HTTP runtime, federation returns mock empty
    /// bindings. For real federation, a host-side fetch shim is required.
    ///
    /// `top_n` is accepted for API symmetry but ignored on WASM — the router uses
    /// `max_sources` from its configuration.
    ///
    /// `strategy` must be one of: `"first"`, `"union"`, `"intersect"`, `"concat"`,
    /// `"largest"`, or `"fastest"`.
    #[cfg(feature = "http")]
    #[wasm_bindgen]
    pub fn federated_query_js(
        &self,
        query_str: &str,
        _top_n: usize,
        strategy: &str,
    ) -> Result<JsValue, JsValue> {
        use crate::federation::AggregationStrategy;
        let agg_strategy = match strategy {
            "first" => AggregationStrategy::First,
            "union" => AggregationStrategy::Union,
            "intersect" => AggregationStrategy::Intersect,
            "concat" => AggregationStrategy::Concat,
            "largest" => AggregationStrategy::Largest,
            "fastest" => AggregationStrategy::Fastest,
            other => return Err(JsValue::from_str(&format!("Unknown strategy: {other}"))),
        };
        let query = crate::core::query::Query::parse(query_str)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let aggregated = self
            .inner
            .federated_query(&query, agg_strategy)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let json =
            serde_json::to_string(&aggregated).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }

    /// Route and execute a query, returning raw per-source results without aggregation.
    ///
    /// `top_n` is accepted for API symmetry but ignored on WASM — the router uses
    /// `max_sources` from its configuration.
    ///
    /// On WASM targets without a full HTTP runtime, execution returns mock empty
    /// results per source. For real execution, a host-side fetch shim is required.
    #[cfg(feature = "http")]
    #[wasm_bindgen]
    pub fn route_and_execute_js(&self, query_str: &str, _top_n: usize) -> Result<JsValue, JsValue> {
        let query = crate::core::query::Query::parse(query_str)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let results = self
            .inner
            .route_and_execute(&query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let json =
            serde_json::to_string(&results).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }

    /// Parse a VoID/Turtle document and register all `void:Dataset` entries as sources.
    ///
    /// Requires the `void` feature. Returns `Ok("ok")` on success.
    #[cfg(feature = "void")]
    #[wasm_bindgen]
    pub fn register_from_void_ttl_js(&mut self, ttl: &str) -> Result<JsValue, JsValue> {
        self.inner
            .register_from_void_ttl(ttl)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str("ok"))
    }

    /// Return the crate version string.
    #[wasm_bindgen]
    pub fn version() -> String {
        crate::VERSION.to_string()
    }
}

impl Default for OxiRouter {
    fn default() -> Self {
        Self::new()
    }
}

fn selection_reason_str(reason: SelectionReason) -> &'static str {
    match reason {
        SelectionReason::ModelPrediction => "model_prediction",
        SelectionReason::VocabularyMatch => "vocabulary_match",
        SelectionReason::GeographicProximity => "geographic_proximity",
        SelectionReason::HistoricalPerformance => "historical_performance",
        SelectionReason::Fallback => "fallback",
        SelectionReason::UserPreference => "user_preference",
        SelectionReason::ComplianceRequired => "compliance_required",
    }
}

fn build_sources_array(
    selections: &[crate::core::source::SourceSelection],
) -> Result<js_sys::Array, JsValue> {
    let sources = js_sys::Array::new();
    for sel in selections {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"sourceId".into(), &sel.source_id.clone().into())?;
        js_sys::Reflect::set(&obj, &"confidence".into(), &sel.confidence.into())?;
        js_sys::Reflect::set(
            &obj,
            &"estimatedLatencyMs".into(),
            &sel.estimated_latency_ms.into(),
        )?;
        js_sys::Reflect::set(
            &obj,
            &"reason".into(),
            &selection_reason_str(sel.reason).into(),
        )?;
        sources.push(&obj);
    }
    Ok(sources)
}

/// WASM module entry point — installs the panic hook on load.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[cfg(test)]
mod tests {
    use crate::core::router::Router;
    use crate::core::source::DataSource;

    /// Validate that `Router::save_state` → `Router::load_state` round-trips
    /// the source count and a registered source correctly.
    ///
    /// This test uses the inner `Router` API directly (no JS interface) so it
    /// runs on all platforms when the `wasm` feature is compiled in.
    #[test]
    fn test_save_load_state_roundtrip() {
        let mut router = Router::new();
        router.add_source(
            DataSource::new("sparql-ep", "https://endpoint.example.org/sparql")
                .with_vocabulary("https://schema.org/"),
        );

        let bytes = router.save_state().expect("save_state failed");
        assert!(!bytes.is_empty(), "save_state returned empty bytes");

        let mut restored = Router::new();
        restored.load_state(&bytes).expect("load_state failed");

        assert_eq!(
            restored.source_count(),
            1,
            "restored router should have 1 source"
        );
        assert!(
            restored.get_source("sparql-ep").is_some(),
            "restored router should contain 'sparql-ep'"
        );
    }
}
