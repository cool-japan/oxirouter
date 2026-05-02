//! Query execution and dispatch

#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use crate::core::error::{OxiRouterError, Result};
use crate::core::query::Query;
use crate::core::source::{DataSource, SourceRanking};

/// Query result from execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Source that returned the result
    pub source_id: String,
    /// Raw result data (serialized)
    pub data: Vec<u8>,
    /// Result format
    pub format: ResultFormat,
    /// Number of rows/bindings
    pub row_count: u32,
    /// Execution time in milliseconds
    pub latency_ms: u32,
    /// Whether result was truncated
    pub truncated: bool,
    /// Error message (if partial failure)
    pub error: Option<String>,
}

impl QueryResult {
    /// Create a successful result
    #[must_use]
    pub fn success(
        source_id: impl Into<String>,
        data: Vec<u8>,
        row_count: u32,
        latency_ms: u32,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            data,
            format: ResultFormat::Json,
            row_count,
            latency_ms,
            truncated: false,
            error: None,
        }
    }

    /// Create an error result
    #[must_use]
    pub fn error(source_id: impl Into<String>, error: impl Into<String>, latency_ms: u32) -> Self {
        Self {
            source_id: source_id.into(),
            data: Vec::new(),
            format: ResultFormat::Unknown,
            row_count: 0,
            latency_ms,
            truncated: false,
            error: Some(error.into()),
        }
    }

    /// Check if result is successful
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.error.is_none()
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

/// Result format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultFormat {
    /// JSON (SPARQL JSON Results)
    Json,
    /// XML (SPARQL XML Results)
    Xml,
    /// CSV
    Csv,
    /// TSV
    Tsv,
    /// RDF (for CONSTRUCT queries)
    Rdf,
    /// Unknown format
    Unknown,
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Timeout per source in milliseconds
    pub timeout_ms: u32,
    /// Maximum retries per source
    pub max_retries: u8,
    /// Whether to execute sources in parallel
    pub parallel: bool,
    /// Maximum concurrent requests
    pub max_concurrency: u8,
    /// Preferred result format
    pub preferred_format: ResultFormat,
    /// Maximum response body size in bytes. Streaming read aborts early if exceeded.
    /// Default: 64 MiB.
    pub max_response_bytes: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            max_retries: 2,
            parallel: true,
            max_concurrency: 4,
            preferred_format: ResultFormat::Json,
            max_response_bytes: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

/// Query executor for federation
pub struct Executor {
    /// Execution configuration
    config: ExecutionConfig,
}

impl Executor {
    /// Create a new executor
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: ExecutionConfig {
                timeout_ms: 30_000,
                max_retries: 2,
                parallel: true,
                max_concurrency: 4,
                preferred_format: ResultFormat::Json,
                max_response_bytes: 64 * 1024 * 1024,
            },
        }
    }

    /// Create executor with config
    #[must_use]
    pub const fn with_config(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    #[must_use]
    pub const fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Execute query against selected sources
    ///
    /// # Errors
    ///
    /// Returns error if execution fails for all sources
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, query, sources, ranking),
            fields(sources_count = sources.len())
        )
    )]
    pub fn execute(
        &self,
        query: &Query,
        sources: &[&DataSource],
        ranking: &SourceRanking,
    ) -> Result<Vec<QueryResult>> {
        if sources.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "No sources to execute".to_string(),
            ));
        }

        // Collect the ordered sources we'll query
        let selected: Vec<&DataSource> = ranking
            .top(self.config.max_concurrency as usize)
            .iter()
            .filter_map(|sel| sources.iter().find(|s| s.id == sel.source_id).copied())
            .collect();

        if selected.is_empty() {
            return Err(OxiRouterError::ExecutionError(
                "No results obtained".to_string(),
            ));
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if self.config.parallel && selected.len() > 1 {
            #[cfg(feature = "observability")]
            let exec_start = std::time::Instant::now();
            let results = self.execute_parallel(query, &selected);
            #[cfg(feature = "observability")]
            {
                let elapsed = exec_start.elapsed();
                metrics::histogram!("oxirouter.federation.execute.duration_ms")
                    .record(elapsed.as_secs_f64() * 1000.0);
            }
            return Ok(results);
        }

        #[cfg(feature = "observability")]
        let exec_start = std::time::Instant::now();

        // Serial fallback (also used on wasm32 and when parallel=false)
        let results: Vec<QueryResult> = selected
            .into_iter()
            .map(|source| self.execute_single(query, source))
            .collect();

        #[cfg(feature = "observability")]
        {
            let elapsed = exec_start.elapsed();
            metrics::histogram!("oxirouter.federation.execute.duration_ms")
                .record(elapsed.as_secs_f64() * 1000.0);
        }

        Ok(results)
    }

    /// Execute query against multiple sources concurrently using scoped threads.
    ///
    /// Each source gets a per-source deadline of `timeout_ms`.  An overall
    /// budget of `2 * timeout_ms` (minimum `timeout_ms + 1000 ms`) guards
    /// against runaway stragglers.  Result order matches `sources` order.
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    fn execute_parallel(&self, query: &Query, sources: &[&DataSource]) -> Vec<QueryResult> {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let n = sources.len();
        if n == 0 {
            return vec![];
        }

        // End-to-end budget: at least timeout_ms + 1 s, otherwise 2 × timeout_ms.
        // The per-source HTTP timeout is enforced at the ureq agent layer.
        let total_budget = Duration::from_millis(
            u64::from(self.config.timeout_ms)
                .saturating_mul(2)
                .max(u64::from(self.config.timeout_ms) + 1000),
        );
        let total_deadline = Instant::now() + total_budget;

        // Indexed channel: each thread sends (source_index, result).
        let (tx, rx) = mpsc::channel::<(usize, QueryResult)>();

        let mut results: Vec<Option<QueryResult>> = (0..n).map(|_| None).collect();

        // `thread::scope` borrows `self` and `sources` from the enclosing
        // stack frame — no Arc needed because Executor is just plain config data.
        std::thread::scope(|scope| {
            for (idx, source) in sources.iter().enumerate() {
                let tx = tx.clone();
                let source_ref: &DataSource = source;

                scope.spawn(move || {
                    let result = self.execute_single(query, source_ref);
                    let _ = tx.send((idx, result));
                });
            }

            // Drop the last sender so the channel closes once all threads finish.
            drop(tx);

            // Collect results until all arrive or the total budget expires.
            loop {
                if results.iter().all(Option::is_some) {
                    break;
                }

                let remaining = total_deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }

                match rx.recv_timeout(remaining) {
                    Ok((idx, result)) => {
                        if idx < results.len() {
                            results[idx] = Some(result);
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        // Fill missing slots with a timeout error result.
        results
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                r.unwrap_or_else(|| {
                    QueryResult::error(
                        sources[idx].id.clone(),
                        "timeout: end-to-end budget exceeded",
                        u64::from(self.config.timeout_ms).min(u64::from(u32::MAX)) as u32,
                    )
                })
            })
            .collect()
    }

    /// Execute query against a single source
    #[cfg(feature = "http")]
    pub(crate) fn execute_single(&self, query: &Query, source: &DataSource) -> QueryResult {
        let start = Self::get_time_ms();

        // Check capability constraints first
        if let Some(err) = self.check_capabilities(query, source) {
            let latency = (Self::get_time_ms() - start) as u32;
            return QueryResult::error(&source.id, err, latency);
        }

        // Execute with retries
        let mut last_error = String::new();
        let mut attempt = 0;

        while attempt <= self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff: 100ms, 200ms, 400ms, ...
                let delay_ms = 100 * (1 << (attempt - 1));
                Self::sleep_ms(delay_ms);
            }

            match self.execute_http_request(query, source) {
                Ok(result) => {
                    return result;
                }
                Err(err) => {
                    last_error = err.to_string();
                    attempt += 1;
                }
            }
        }

        let latency = (Self::get_time_ms() - start) as u32;
        QueryResult::error(&source.id, last_error, latency)
    }

    /// Execute query against a single source (mock implementation for no-http)
    #[cfg(not(feature = "http"))]
    pub(crate) fn execute_single(&self, query: &Query, source: &DataSource) -> QueryResult {
        // Mock implementation for testing/WASM when HTTP feature is disabled
        let start = Self::get_time_ms();

        // Check capability constraints
        if let Some(err) = self.check_capabilities(query, source) {
            let latency = (Self::get_time_ms() - start) as u32;
            return QueryResult::error(&source.id, err, latency);
        }

        // Simulate execution based on source stats
        let latency = if source.stats.has_history() {
            source.stats.avg_latency_ms as u32
        } else {
            500 // Default 500ms
        };

        // Create mock result
        let mock_data = r#"{"head":{"vars":["s","p","o"]},"results":{"bindings":[]}}"#;

        let actual_latency = Self::get_time_ms() - start;

        QueryResult::success(
            &source.id,
            mock_data.as_bytes().to_vec(),
            0,
            actual_latency.max(u64::from(latency)) as u32,
        )
    }

    /// Check if the source supports the query type and features
    fn check_capabilities(&self, query: &Query, source: &DataSource) -> Option<String> {
        use crate::core::query::QueryType;

        let can_execute = match query.query_type {
            QueryType::Select => true,
            QueryType::Construct => source.capabilities.construct,
            QueryType::Ask => source.capabilities.ask,
            QueryType::Describe => source.capabilities.describe,
        };

        if !can_execute {
            return Some("Source does not support this query type".to_string());
        }

        if query.requires_sparql_1_1() && !source.capabilities.sparql_1_1 {
            return Some("Source does not support SPARQL 1.1 features".to_string());
        }

        None
    }

    /// Execute HTTP request to SPARQL endpoint
    #[cfg(feature = "http")]
    fn execute_http_request(
        &self,
        query: &Query,
        source: &DataSource,
    ) -> std::result::Result<QueryResult, OxiRouterError> {
        use std::io::Read;
        use std::time::Duration;

        let start = Self::get_time_ms();

        // Build the agent with timeout
        let timeout = Duration::from_millis(u64::from(self.config.timeout_ms));
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build()
            .new_agent();

        // Build URL with query parameter
        let url = Self::build_url(&source.endpoint, query);

        // Execute the request
        let response = agent
            .get(&url)
            .header(
                "Accept",
                Self::accept_header_for_format(self.config.preferred_format),
            )
            .header("User-Agent", "OxiRouter/0.1")
            .call()
            .map_err(|e| OxiRouterError::ExecutionError(Self::map_http_error(e)))?;

        // Stream response body with a bounded chunk-by-chunk reader.
        // We abort early if the cumulative size exceeds `max_response_bytes`
        // rather than buffering the whole body first.
        let limit = self.config.max_response_bytes;
        let mut reader = response.into_body().into_reader();
        let mut buf: Vec<u8> = Vec::with_capacity(8192);
        let mut tmp = [0u8; 8192];

        loop {
            let n = reader
                .read(&mut tmp)
                .map_err(|e| OxiRouterError::ExecutionError(format!("HTTP read error: {e}")))?;
            if n == 0 {
                break;
            }
            let new_len = (buf.len() as u64).saturating_add(n as u64);
            if new_len > limit {
                return Err(OxiRouterError::ResponseTooLarge {
                    source_id: source.id.clone(),
                    observed_bytes: new_len,
                    limit_bytes: limit,
                });
            }
            buf.extend_from_slice(&tmp[..n]);
        }

        let latency = (Self::get_time_ms() - start) as u32;

        // Parse JSON to get row count
        let row_count = Self::count_bindings(&buf);

        Ok(QueryResult {
            source_id: source.id.clone(),
            data: buf,
            format: self.config.preferred_format,
            row_count,
            latency_ms: latency,
            truncated: false,
            error: None,
        })
    }

    /// Map ureq error to string description
    #[cfg(feature = "http")]
    fn map_http_error(err: ureq::Error) -> String {
        match err {
            ureq::Error::Timeout(kind) => {
                format!("Request timeout: {kind:?}")
            }
            ureq::Error::HostNotFound => "Host not found".to_string(),
            ureq::Error::Tls(e) => format!("TLS error: {e}"),
            ureq::Error::Io(e) => format!("I/O error: {e}"),
            ureq::Error::ConnectionFailed => "Connection failed".to_string(),
            ureq::Error::TooManyRedirects => "Too many redirects".to_string(),
            ureq::Error::StatusCode(code) => {
                format!("HTTP error: status {code}")
            }
            ureq::Error::Http(e) => format!("HTTP protocol error: {e}"),
            ureq::Error::BadUri(uri) => {
                format!("Invalid URI: {uri}")
            }
            ureq::Error::BodyExceedsLimit(limit) => {
                format!("Response body exceeds limit: {limit}")
            }
            ureq::Error::Protocol(e) => format!("Protocol error: {e}"),
            ureq::Error::RedirectFailed => "Redirect failed".to_string(),
            ureq::Error::Rustls(e) => format!("TLS (rustls) error: {e}"),
            _ => format!("HTTP error: {err}"),
        }
    }

    /// Get Accept header value for the preferred format
    #[cfg(feature = "http")]
    const fn accept_header_for_format(format: ResultFormat) -> &'static str {
        match format {
            ResultFormat::Json => "application/sparql-results+json",
            ResultFormat::Xml => "application/sparql-results+xml",
            ResultFormat::Csv => "text/csv",
            ResultFormat::Tsv => "text/tab-separated-values",
            ResultFormat::Rdf => "application/rdf+xml",
            ResultFormat::Unknown => "*/*",
        }
    }

    /// Count bindings in SPARQL JSON results
    #[cfg(feature = "http")]
    fn count_bindings(data: &[u8]) -> u32 {
        // Simple JSON parsing to count bindings
        // Look for "bindings":[ and count objects at depth 1 of the array
        let text = match std::str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => return 0,
        };

        // Find bindings array
        if let Some(pos) = text.find("\"bindings\"") {
            let remaining = &text[pos..];
            if let Some(bracket_pos) = remaining.find('[') {
                let array_start = &remaining[bracket_pos..];
                // Count opening braces at depth 1 of the bindings array
                // depth 0 = before entering the array
                // depth 1 = inside the bindings array (where binding objects are)
                // depth 2+ = inside a binding object or nested structure
                let mut count = 0u32;
                let mut array_depth = 0i32;
                let mut object_depth = 0i32;
                let mut in_string = false;
                let mut escape_next = false;

                for c in array_start.chars() {
                    if escape_next {
                        escape_next = false;
                        continue;
                    }
                    match c {
                        '\\' if in_string => escape_next = true,
                        '"' => in_string = !in_string,
                        '[' if !in_string => array_depth += 1,
                        ']' if !in_string => {
                            array_depth -= 1;
                            if array_depth == 0 {
                                break;
                            }
                        }
                        '{' if !in_string => {
                            // Only count objects directly inside the bindings array (array_depth == 1, object_depth == 0)
                            if array_depth == 1 && object_depth == 0 {
                                count += 1;
                            }
                            object_depth += 1;
                        }
                        '}' if !in_string => {
                            object_depth -= 1;
                        }
                        _ => {}
                    }
                }
                return count;
            }
        }
        0
    }

    /// Sleep for given milliseconds
    #[cfg(all(feature = "http", feature = "std"))]
    fn sleep_ms(ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }

    #[cfg(all(feature = "http", not(feature = "std")))]
    fn sleep_ms(_ms: u64) {
        // No-op in no_std environment
    }

    /// Get current time in milliseconds
    fn get_time_ms() -> u64 {
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        }

        #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
        {
            0
        }
    }

    /// Build SPARQL endpoint URL with query
    #[must_use]
    pub fn build_url(endpoint: &str, query: &Query) -> String {
        // URL encode the query
        let encoded = Self::url_encode(&query.raw);
        format!("{}?query={}", endpoint, encoded)
    }

    /// Simple URL encoding
    fn url_encode(s: &str) -> String {
        let mut result = String::with_capacity(s.len() * 3);

        for c in s.chars() {
            match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                    result.push(c);
                }
                ' ' => result.push_str("%20"),
                '<' => result.push_str("%3C"),
                '>' => result.push_str("%3E"),
                '#' => result.push_str("%23"),
                '%' => result.push_str("%25"),
                '{' => result.push_str("%7B"),
                '}' => result.push_str("%7D"),
                '|' => result.push_str("%7C"),
                '^' => result.push_str("%5E"),
                '[' => result.push_str("%5B"),
                ']' => result.push_str("%5D"),
                '`' => result.push_str("%60"),
                '"' => result.push_str("%22"),
                '\'' => result.push_str("%27"),
                '?' => result.push_str("%3F"),
                '&' => result.push_str("%26"),
                '=' => result.push_str("%3D"),
                _ => {
                    for byte in c.to_string().bytes() {
                        result.push_str(&format!("%{:02X}", byte));
                    }
                }
            }
        }

        result
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::source::SourceCapabilities;
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    use alloc::vec;

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new();
        assert_eq!(executor.config().timeout_ms, 30_000);
    }

    #[test]
    fn test_query_result_success() {
        let result = QueryResult::success("src1", vec![1, 2, 3], 10, 100);
        assert!(result.is_success());
        assert!(!result.is_empty());
        assert_eq!(result.row_count, 10);
    }

    #[test]
    fn test_query_result_error() {
        let result = QueryResult::error("src1", "Connection failed", 0);
        assert!(!result.is_success());
        assert!(result.is_empty());
    }

    #[test]
    fn test_execute_single() {
        let executor = Executor::new();
        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let source = DataSource::new("test", "http://example.com/sparql")
            .with_capabilities(SourceCapabilities::full());

        let result = executor.execute_single(&query, &source);
        // Without http feature, this uses mock; with http feature it would fail (no real endpoint)
        #[cfg(not(feature = "http"))]
        assert!(result.is_success());
        #[cfg(feature = "http")]
        assert!(!result.is_success()); // Expected to fail with no real endpoint
    }

    #[test]
    fn test_url_encoding() {
        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let url = Executor::build_url("http://example.com/sparql", &query);

        assert!(url.starts_with("http://example.com/sparql?query="));
        assert!(url.contains("%20")); // Space encoding
    }

    #[test]
    fn test_check_capabilities_select() {
        let executor = Executor::new();
        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let source = DataSource::new("test", "http://example.com/sparql")
            .with_capabilities(SourceCapabilities::basic());

        let result = executor.check_capabilities(&query, &source);
        assert!(result.is_none()); // SELECT is always supported
    }

    #[test]
    fn test_check_capabilities_construct_unsupported() {
        let executor = Executor::new();
        let query = Query::parse("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }").unwrap();
        let mut caps = SourceCapabilities::basic();
        caps.construct = false;
        let source = DataSource::new("test", "http://example.com/sparql").with_capabilities(caps);

        let result = executor.check_capabilities(&query, &source);
        assert!(result.is_some());
        assert!(result.unwrap().contains("does not support"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_count_bindings_empty() {
        let json = r#"{"head":{"vars":["s","p","o"]},"results":{"bindings":[]}}"#;
        let count = Executor::count_bindings(json.as_bytes());
        assert_eq!(count, 0);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_count_bindings_with_results() {
        let json = r#"{"head":{"vars":["s"]},"results":{"bindings":[{"s":{"type":"uri","value":"http://example.org/1"}},{"s":{"type":"uri","value":"http://example.org/2"}}]}}"#;
        let count = Executor::count_bindings(json.as_bytes());
        assert_eq!(count, 2);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_count_bindings_nested_objects() {
        let json = r#"{"head":{"vars":["s","o"]},"results":{"bindings":[{"s":{"type":"uri","value":"http://example.org/1"},"o":{"type":"literal","value":"test"}}]}}"#;
        let count = Executor::count_bindings(json.as_bytes());
        assert_eq!(count, 1);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_accept_header_json() {
        let header = Executor::accept_header_for_format(ResultFormat::Json);
        assert_eq!(header, "application/sparql-results+json");
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_accept_header_xml() {
        let header = Executor::accept_header_for_format(ResultFormat::Xml);
        assert_eq!(header, "application/sparql-results+xml");
    }

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert_eq!(config.timeout_ms, 30_000);
        assert_eq!(config.max_retries, 2);
        assert!(config.parallel);
        assert_eq!(config.max_concurrency, 4);
    }

    #[test]
    fn test_result_format_equality() {
        assert_eq!(ResultFormat::Json, ResultFormat::Json);
        assert_ne!(ResultFormat::Json, ResultFormat::Xml);
    }
}

/// Integration tests that require network access
#[cfg(all(test, feature = "http"))]
mod integration_tests {
    use super::*;
    use crate::core::source::SourceCapabilities;

    /// Test against DBpedia SPARQL endpoint (requires network)
    /// This test is ignored by default - run with `cargo test -- --ignored`
    #[test]
    #[ignore = "requires network access to DBpedia"]
    fn test_real_sparql_endpoint() {
        let executor = Executor::with_config(ExecutionConfig {
            timeout_ms: 10_000,
            max_retries: 1,
            ..ExecutionConfig::default()
        });

        let query =
            Query::parse("SELECT ?s WHERE { ?s a <http://dbpedia.org/ontology/Country> } LIMIT 5")
                .unwrap();
        let source = DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_capabilities(SourceCapabilities::full());

        let result = executor.execute_single(&query, &source);

        if result.is_success() {
            assert!(result.row_count <= 5);
            assert!(!result.data.is_empty());
            println!(
                "DBpedia returned {} results in {}ms",
                result.row_count, result.latency_ms
            );
        } else {
            // Network issues are acceptable in CI
            println!("DBpedia test skipped: {:?}", result.error);
        }
    }

    /// Test timeout handling
    #[test]
    #[ignore = "requires network access to an external endpoint"]
    fn test_timeout_handling() {
        let executor = Executor::with_config(ExecutionConfig {
            timeout_ms: 1, // 1ms timeout - should fail
            max_retries: 0,
            ..ExecutionConfig::default()
        });

        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1").unwrap();
        let source = DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_capabilities(SourceCapabilities::full());

        let result = executor.execute_single(&query, &source);
        // Should fail due to timeout
        assert!(!result.is_success());
    }

    /// Test error handling for invalid endpoint
    #[test]
    fn test_invalid_endpoint() {
        let executor = Executor::with_config(ExecutionConfig {
            timeout_ms: 5_000,
            max_retries: 0,
            ..ExecutionConfig::default()
        });

        let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
        let source = DataSource::new("invalid", "http://invalid.endpoint.local/sparql")
            .with_capabilities(SourceCapabilities::full());

        let result = executor.execute_single(&query, &source);
        assert!(!result.is_success());
        assert!(result.error.is_some());
    }
}
