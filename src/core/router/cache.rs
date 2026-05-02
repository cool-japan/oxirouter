//! Router cache methods — only available with the `cache` feature.

#[cfg(feature = "cache")]
use crate::cache::{CacheManager, CacheStats, CombinedCacheStats};

#[cfg(feature = "cache")]
use alloc::string::String;

#[cfg(feature = "cache")]
use crate::context::{CombinedContext, ContextProvider};

#[cfg(feature = "cache")]
use super::Router;

#[cfg(feature = "cache")]
impl<C: ContextProvider> Router<C> {
    /// Get a cached query result by query hash
    ///
    /// Returns `None` if the entry doesn't exist, has expired, or caching is disabled.
    pub fn get_cached_result(&mut self, query_hash: u64) -> Option<&crate::cache::CacheEntry> {
        if !self.config.cache_enabled || !self.cache.is_enabled() {
            return None;
        }
        self.cache.query_cache.get(query_hash)
    }

    /// Store a query result in the cache
    ///
    /// Does nothing if caching is disabled.
    pub fn cache_result(&mut self, query_hash: u64, result: String, source_id: String) {
        if !self.config.cache_enabled || !self.cache.is_enabled() {
            return;
        }
        self.cache.query_cache.put(query_hash, result, source_id);
    }

    /// Store a query result with custom TTL
    pub fn cache_result_with_ttl(
        &mut self,
        query_hash: u64,
        result: String,
        source_id: String,
        ttl_ms: u64,
    ) {
        if !self.config.cache_enabled || !self.cache.is_enabled() {
            return;
        }
        self.cache
            .query_cache
            .put_with_ttl(query_hash, result, source_id, ttl_ms);
    }

    /// Invalidate a cached query result
    pub fn invalidate_cached_result(&mut self, query_hash: u64) {
        self.cache.query_cache.invalidate(query_hash);
    }

    /// Check if a query result is cached (without updating LRU)
    #[must_use]
    pub fn is_result_cached(&self, query_hash: u64) -> bool {
        if !self.config.cache_enabled || !self.cache.is_enabled() {
            return false;
        }
        self.cache.query_cache.contains(query_hash)
    }

    /// Get cached context for a provider
    pub fn get_cached_context(&mut self, provider_id: &str) -> Option<&CombinedContext> {
        if !self.config.cache_enabled || !self.cache.is_enabled() {
            return None;
        }
        self.cache.context_cache.get(provider_id)
    }

    /// Cache a context
    pub fn cache_context(&mut self, provider_id: impl Into<String>, context: CombinedContext) {
        if !self.config.cache_enabled || !self.cache.is_enabled() {
            return;
        }
        self.cache.context_cache.put(provider_id, context);
    }

    /// Get cache statistics
    #[must_use]
    pub fn cache_stats(&self) -> CombinedCacheStats {
        self.cache.combined_stats()
    }

    /// Get query cache statistics
    #[must_use]
    pub fn query_cache_stats(&self) -> &CacheStats {
        self.cache.query_cache.stats()
    }

    /// Get the query cache hit rate
    #[must_use]
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache.query_cache.stats().hit_rate()
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.cache.clear_all();
    }

    /// Clear only the query cache
    pub fn clear_query_cache(&mut self) {
        self.cache.query_cache.clear();
    }

    /// Cleanup expired cache entries
    ///
    /// Returns the total number of entries removed.
    pub fn cleanup_cache(&mut self) -> usize {
        self.cache.cleanup_all()
    }

    /// Enable caching
    pub fn enable_cache(&mut self) {
        self.config.cache_enabled = true;
        self.cache.enable();
    }

    /// Disable caching
    pub fn disable_cache(&mut self) {
        self.config.cache_enabled = false;
        self.cache.disable();
    }

    /// Check if caching is enabled
    #[must_use]
    pub const fn is_cache_enabled(&self) -> bool {
        self.config.cache_enabled
    }

    /// Get the number of cached query results
    #[must_use]
    pub fn cached_query_count(&self) -> usize {
        self.cache.query_cache.len()
    }

    /// Get a mutable reference to the cache manager
    pub fn cache_mut(&mut self) -> &mut CacheManager {
        &mut self.cache
    }

    /// Get a reference to the cache manager
    #[must_use]
    pub const fn cache(&self) -> &CacheManager {
        &self.cache
    }
}
