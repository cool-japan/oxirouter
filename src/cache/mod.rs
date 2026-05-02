//! Caching layer for OxiRouter
//!
//! This module provides caching functionality for query results, context data,
//! and source capabilities. All caches support TTL-based expiration and are
//! designed to be no_std compatible.
//!
//! ## Features
//!
//! - **QueryCache**: Cache query results by query hash with LRU eviction
//! - **ContextCache**: Cache context data with TTL support
//! - **SourceCache**: Cache source capabilities and performance stats
//!
//! ## Thread Safety
//!
//! When the `std` feature is enabled, caches can be wrapped in `Arc<RwLock<>>`
//! for thread-safe access. The module also provides convenience types for this.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::context::CombinedContext;
use crate::core::source::{SourceCapabilities, SourceStats};

/// Default maximum entries for QueryCache
pub const DEFAULT_MAX_ENTRIES: usize = 1000;

/// Default TTL in milliseconds (5 minutes)
pub const DEFAULT_TTL_MS: u64 = 300_000;

/// Default context TTL in milliseconds (1 minute)
pub const DEFAULT_CONTEXT_TTL_MS: u64 = 60_000;

// ============================================================================
// QueryCache
// ============================================================================

/// A cached query result entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// JSON result string
    pub result: String,
    /// Source ID that produced this result
    pub source_id: String,
    /// Timestamp when entry was cached (Unix epoch milliseconds)
    pub cached_at: u64,
    /// Time-to-live in milliseconds
    pub ttl_ms: u64,
    /// Number of cache hits for this entry
    pub hits: u32,
    /// Last access timestamp (Unix epoch milliseconds)
    pub last_accessed: u64,
}

impl CacheEntry {
    /// Create a new cache entry
    #[must_use]
    pub fn new(result: String, source_id: String, ttl_ms: u64) -> Self {
        let now = get_time_ms();
        Self {
            result,
            source_id,
            cached_at: now,
            ttl_ms,
            hits: 0,
            last_accessed: now,
        }
    }

    /// Check if the entry has expired
    #[must_use]
    pub fn is_expired(&self, current_time_ms: u64) -> bool {
        current_time_ms.saturating_sub(self.cached_at) > self.ttl_ms
    }

    /// Get the age of the entry in milliseconds
    #[must_use]
    pub fn age_ms(&self, current_time_ms: u64) -> u64 {
        current_time_ms.saturating_sub(self.cached_at)
    }

    /// Get the remaining TTL in milliseconds (0 if expired)
    #[must_use]
    pub fn remaining_ttl_ms(&self, current_time_ms: u64) -> u64 {
        let age = self.age_ms(current_time_ms);
        self.ttl_ms.saturating_sub(age)
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits = self.hits.saturating_add(1);
        self.last_accessed = get_time_ms();
    }
}

/// LRU node for tracking access order
#[derive(Debug, Clone)]
struct LruNode {
    /// Query hash
    key: u64,
    /// Previous node in LRU list (None if head)
    prev: Option<usize>,
    /// Next node in LRU list (None if tail)
    next: Option<usize>,
}

/// Cache for query results with LRU eviction
///
/// Uses FNV-1a hash of query predicates as the cache key.
/// Implements LRU eviction when max_entries is reached.
#[derive(Debug)]
pub struct QueryCache {
    /// Cache entries indexed by query hash
    entries: HashMap<u64, (CacheEntry, usize)>,
    /// LRU linked list nodes
    lru_nodes: Vec<Option<LruNode>>,
    /// Index of the most recently used node
    lru_head: Option<usize>,
    /// Index of the least recently used node
    lru_tail: Option<usize>,
    /// Free list for recycling node indices
    free_list: Vec<usize>,
    /// Maximum number of entries
    max_entries: usize,
    /// Default TTL in milliseconds
    default_ttl_ms: u64,
    /// Cache statistics
    stats: CacheStats,
}

impl QueryCache {
    /// Create a new query cache with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DEFAULT_MAX_ENTRIES, DEFAULT_TTL_MS)
    }

    /// Create a new query cache with custom configuration
    #[must_use]
    pub fn with_config(max_entries: usize, default_ttl_ms: u64) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            lru_nodes: Vec::with_capacity(max_entries),
            lru_head: None,
            lru_tail: None,
            free_list: Vec::new(),
            max_entries,
            default_ttl_ms,
            stats: CacheStats::default(),
        }
    }

    /// Get a cached entry by query hash
    ///
    /// Returns `None` if the entry doesn't exist or has expired.
    /// Updates LRU order and hit count on access.
    pub fn get(&mut self, query_hash: u64) -> Option<&CacheEntry> {
        let now = get_time_ms();

        // Check if entry exists
        if let Some((entry, node_idx)) = self.entries.get_mut(&query_hash) {
            if entry.is_expired(now) {
                // Entry expired, remove it
                self.stats.expirations += 1;
                let node_idx = *node_idx;
                self.remove_from_lru(node_idx);
                self.entries.remove(&query_hash);
                self.stats.misses += 1;
                return None;
            }

            // Record hit and move to front of LRU
            entry.record_hit();
            self.stats.hits += 1;
            let node_idx = *node_idx;
            self.move_to_front(node_idx);

            // Re-borrow immutably
            return self.entries.get(&query_hash).map(|(e, _)| e);
        }

        self.stats.misses += 1;
        None
    }

    /// Put a result in the cache
    ///
    /// Uses the default TTL. Evicts LRU entry if cache is full.
    pub fn put(&mut self, query_hash: u64, result: String, source_id: String) {
        self.put_with_ttl(query_hash, result, source_id, self.default_ttl_ms);
    }

    /// Put a result in the cache with a custom TTL
    pub fn put_with_ttl(
        &mut self,
        query_hash: u64,
        result: String,
        source_id: String,
        ttl_ms: u64,
    ) {
        // Check if we need to evict
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&query_hash) {
            self.evict_lru();
        }

        // If entry already exists, update it and move to front
        if let Some((entry, node_idx)) = self.entries.get_mut(&query_hash) {
            *entry = CacheEntry::new(result, source_id, ttl_ms);
            let idx = *node_idx;
            self.move_to_front(idx);
            return;
        }

        // Create new entry
        let entry = CacheEntry::new(result, source_id, ttl_ms);
        let node_idx = self.allocate_node(query_hash);
        self.entries.insert(query_hash, (entry, node_idx));
        self.add_to_front(node_idx);
        self.stats.inserts += 1;
    }

    /// Invalidate a specific cache entry
    pub fn invalidate(&mut self, query_hash: u64) {
        if let Some((_, node_idx)) = self.entries.remove(&query_hash) {
            self.remove_from_lru(node_idx);
            self.stats.invalidations += 1;
        }
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_nodes.clear();
        self.lru_head = None;
        self.lru_tail = None;
        self.free_list.clear();
        self.stats.clears += 1;
    }

    /// Get the number of cached entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub const fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get the maximum number of entries
    #[must_use]
    pub const fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Get the default TTL in milliseconds
    #[must_use]
    pub const fn default_ttl_ms(&self) -> u64 {
        self.default_ttl_ms
    }

    /// Remove expired entries
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let now = get_time_ms();
        let expired: Vec<u64> = self
            .entries
            .iter()
            .filter(|(_, (entry, _))| entry.is_expired(now))
            .map(|(hash, _)| *hash)
            .collect();

        let count = expired.len();
        for hash in expired {
            self.invalidate(hash);
            self.stats.expirations += 1;
        }
        count
    }

    /// Check if a query hash exists in the cache (without updating LRU)
    #[must_use]
    pub fn contains(&self, query_hash: u64) -> bool {
        if let Some((entry, _)) = self.entries.get(&query_hash) {
            !entry.is_expired(get_time_ms())
        } else {
            false
        }
    }

    // -------------------------------------------------------------------------
    // LRU Implementation
    // -------------------------------------------------------------------------

    /// Allocate a new LRU node index
    fn allocate_node(&mut self, key: u64) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.lru_nodes[idx] = Some(LruNode {
                key,
                prev: None,
                next: None,
            });
            idx
        } else {
            let idx = self.lru_nodes.len();
            self.lru_nodes.push(Some(LruNode {
                key,
                prev: None,
                next: None,
            }));
            idx
        }
    }

    /// Add a node to the front of the LRU list
    fn add_to_front(&mut self, node_idx: usize) {
        if let Some(old_head) = self.lru_head {
            if let Some(ref mut node) = self.lru_nodes[node_idx] {
                node.next = Some(old_head);
                node.prev = None;
            }
            if let Some(ref mut old_head_node) = self.lru_nodes[old_head] {
                old_head_node.prev = Some(node_idx);
            }
        }
        self.lru_head = Some(node_idx);

        if self.lru_tail.is_none() {
            self.lru_tail = Some(node_idx);
        }
    }

    /// Move a node to the front of the LRU list
    fn move_to_front(&mut self, node_idx: usize) {
        if self.lru_head == Some(node_idx) {
            return; // Already at front
        }

        self.remove_from_lru_list_only(node_idx);
        self.add_to_front(node_idx);
    }

    /// Remove a node from the LRU list (but don't free it)
    fn remove_from_lru_list_only(&mut self, node_idx: usize) {
        let (prev, next) = if let Some(ref node) = self.lru_nodes[node_idx] {
            (node.prev, node.next)
        } else {
            return;
        };

        // Update prev node's next pointer
        if let Some(prev_idx) = prev {
            if let Some(ref mut prev_node) = self.lru_nodes[prev_idx] {
                prev_node.next = next;
            }
        } else {
            // This was the head
            self.lru_head = next;
        }

        // Update next node's prev pointer
        if let Some(next_idx) = next {
            if let Some(ref mut next_node) = self.lru_nodes[next_idx] {
                next_node.prev = prev;
            }
        } else {
            // This was the tail
            self.lru_tail = prev;
        }
    }

    /// Remove a node from the LRU list and free it
    fn remove_from_lru(&mut self, node_idx: usize) {
        self.remove_from_lru_list_only(node_idx);
        self.lru_nodes[node_idx] = None;
        self.free_list.push(node_idx);
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(tail_idx) = self.lru_tail {
            if let Some(ref node) = self.lru_nodes[tail_idx] {
                let key = node.key;
                self.entries.remove(&key);
                self.remove_from_lru(tail_idx);
                self.stats.evictions += 1;
            }
        }
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ContextCache
// ============================================================================

/// Cached context entry with TTL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCacheEntry {
    /// The cached combined context
    pub context: CombinedContext,
    /// Timestamp when entry was cached (Unix epoch milliseconds)
    pub cached_at: u64,
    /// Time-to-live in milliseconds
    pub ttl_ms: u64,
    /// Number of cache hits
    pub hits: u32,
    /// Last access timestamp
    pub last_accessed: u64,
}

impl ContextCacheEntry {
    /// Create a new context cache entry
    #[must_use]
    pub fn new(context: CombinedContext, ttl_ms: u64) -> Self {
        let now = get_time_ms();
        Self {
            context,
            cached_at: now,
            ttl_ms,
            hits: 0,
            last_accessed: now,
        }
    }

    /// Check if the entry has expired
    #[must_use]
    pub fn is_expired(&self, current_time_ms: u64) -> bool {
        current_time_ms.saturating_sub(self.cached_at) > self.ttl_ms
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits = self.hits.saturating_add(1);
        self.last_accessed = get_time_ms();
    }
}

/// Cache for context data with TTL support
///
/// Supports automatic expiration and optional refresh on access.
#[derive(Debug)]
pub struct ContextCache {
    /// Cached contexts keyed by provider ID
    entries: HashMap<String, ContextCacheEntry>,
    /// Default TTL in milliseconds
    default_ttl_ms: u64,
    /// Whether to refresh TTL on access
    refresh_on_access: bool,
    /// Cache statistics
    stats: CacheStats,
}

impl ContextCache {
    /// Create a new context cache with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DEFAULT_CONTEXT_TTL_MS, false)
    }

    /// Create a new context cache with custom configuration
    #[must_use]
    pub fn with_config(default_ttl_ms: u64, refresh_on_access: bool) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_ms,
            refresh_on_access,
            stats: CacheStats::default(),
        }
    }

    /// Get a cached context by provider ID
    ///
    /// Returns `None` if the entry doesn't exist or has expired.
    pub fn get(&mut self, provider_id: &str) -> Option<&CombinedContext> {
        let now = get_time_ms();

        // Check if entry exists and is valid
        if let Some(entry) = self.entries.get_mut(provider_id) {
            if entry.is_expired(now) {
                self.stats.expirations += 1;
                self.entries.remove(provider_id);
                self.stats.misses += 1;
                return None;
            }

            entry.record_hit();
            self.stats.hits += 1;

            // Optionally refresh TTL
            if self.refresh_on_access {
                entry.cached_at = now;
            }

            // Re-borrow
            return self.entries.get(provider_id).map(|e| &e.context);
        }

        self.stats.misses += 1;
        None
    }

    /// Get a cloned context (useful when you need ownership)
    pub fn get_cloned(&mut self, provider_id: &str) -> Option<CombinedContext> {
        self.get(provider_id).cloned()
    }

    /// Put a context in the cache with the default TTL
    pub fn put(&mut self, provider_id: impl Into<String>, context: CombinedContext) {
        self.put_with_ttl(provider_id, context, self.default_ttl_ms);
    }

    /// Put a context in the cache with a custom TTL
    pub fn put_with_ttl(
        &mut self,
        provider_id: impl Into<String>,
        context: CombinedContext,
        ttl_ms: u64,
    ) {
        let entry = ContextCacheEntry::new(context, ttl_ms);
        self.entries.insert(provider_id.into(), entry);
        self.stats.inserts += 1;
    }

    /// Invalidate a specific context entry
    pub fn invalidate(&mut self, provider_id: &str) {
        if self.entries.remove(provider_id).is_some() {
            self.stats.invalidations += 1;
        }
    }

    /// Clear all context entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.clears += 1;
    }

    /// Remove expired entries
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let now = get_time_ms();
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(now))
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            self.entries.remove(&id);
            self.stats.expirations += 1;
        }
        count
    }

    /// Get the number of cached contexts
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub const fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Check if refresh on access is enabled
    #[must_use]
    pub const fn refresh_on_access(&self) -> bool {
        self.refresh_on_access
    }

    /// Enable or disable refresh on access
    pub fn set_refresh_on_access(&mut self, enabled: bool) {
        self.refresh_on_access = enabled;
    }
}

impl Default for ContextCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SourceCache
// ============================================================================

/// Cached source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCacheEntry {
    /// Source capabilities
    pub capabilities: SourceCapabilities,
    /// Source statistics
    pub stats: SourceStats,
    /// Timestamp when entry was cached (Unix epoch milliseconds)
    pub cached_at: u64,
    /// Time-to-live in milliseconds
    pub ttl_ms: u64,
    /// Number of cache hits
    pub hits: u32,
}

impl SourceCacheEntry {
    /// Create a new source cache entry
    #[must_use]
    pub fn new(capabilities: SourceCapabilities, stats: SourceStats, ttl_ms: u64) -> Self {
        Self {
            capabilities,
            stats,
            cached_at: get_time_ms(),
            ttl_ms,
            hits: 0,
        }
    }

    /// Check if the entry has expired
    #[must_use]
    pub fn is_expired(&self, current_time_ms: u64) -> bool {
        current_time_ms.saturating_sub(self.cached_at) > self.ttl_ms
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits = self.hits.saturating_add(1);
    }
}

/// Cache for source capabilities and statistics
///
/// Useful for caching expensive capability discovery operations.
#[derive(Debug)]
pub struct SourceCache {
    /// Cached source info keyed by source ID
    entries: HashMap<String, SourceCacheEntry>,
    /// Default TTL in milliseconds (longer than context, as capabilities change rarely)
    default_ttl_ms: u64,
    /// Cache statistics
    stats: CacheStats,
}

/// Default source cache TTL (10 minutes)
pub const DEFAULT_SOURCE_TTL_MS: u64 = 600_000;

impl SourceCache {
    /// Create a new source cache with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_SOURCE_TTL_MS)
    }

    /// Create a new source cache with custom TTL
    #[must_use]
    pub fn with_ttl(default_ttl_ms: u64) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_ms,
            stats: CacheStats::default(),
        }
    }

    /// Get cached source capabilities
    pub fn get_capabilities(&mut self, source_id: &str) -> Option<&SourceCapabilities> {
        let now = get_time_ms();

        if let Some(entry) = self.entries.get_mut(source_id) {
            if entry.is_expired(now) {
                self.stats.expirations += 1;
                self.entries.remove(source_id);
                self.stats.misses += 1;
                return None;
            }

            entry.record_hit();
            self.stats.hits += 1;
            return self.entries.get(source_id).map(|e| &e.capabilities);
        }

        self.stats.misses += 1;
        None
    }

    /// Get cached source statistics
    pub fn get_stats(&mut self, source_id: &str) -> Option<&SourceStats> {
        let now = get_time_ms();

        if let Some(entry) = self.entries.get_mut(source_id) {
            if entry.is_expired(now) {
                self.stats.expirations += 1;
                self.entries.remove(source_id);
                self.stats.misses += 1;
                return None;
            }

            entry.record_hit();
            self.stats.hits += 1;
            return self.entries.get(source_id).map(|e| &e.stats);
        }

        self.stats.misses += 1;
        None
    }

    /// Get the full cached entry
    pub fn get(&mut self, source_id: &str) -> Option<&SourceCacheEntry> {
        let now = get_time_ms();

        if let Some(entry) = self.entries.get_mut(source_id) {
            if entry.is_expired(now) {
                self.stats.expirations += 1;
                self.entries.remove(source_id);
                self.stats.misses += 1;
                return None;
            }

            entry.record_hit();
            self.stats.hits += 1;
            return self.entries.get(source_id);
        }

        self.stats.misses += 1;
        None
    }

    /// Put source info in the cache
    pub fn put(
        &mut self,
        source_id: impl Into<String>,
        capabilities: SourceCapabilities,
        stats: SourceStats,
    ) {
        self.put_with_ttl(source_id, capabilities, stats, self.default_ttl_ms);
    }

    /// Put source info in the cache with custom TTL
    pub fn put_with_ttl(
        &mut self,
        source_id: impl Into<String>,
        capabilities: SourceCapabilities,
        stats: SourceStats,
        ttl_ms: u64,
    ) {
        let entry = SourceCacheEntry::new(capabilities, stats, ttl_ms);
        self.entries.insert(source_id.into(), entry);
        self.stats.inserts += 1;
    }

    /// Update just the statistics for a source
    pub fn update_stats(&mut self, source_id: &str, stats: SourceStats) {
        if let Some(entry) = self.entries.get_mut(source_id) {
            entry.stats = stats;
        }
    }

    /// Invalidate a specific source entry
    pub fn invalidate(&mut self, source_id: &str) {
        if self.entries.remove(source_id).is_some() {
            self.stats.invalidations += 1;
        }
    }

    /// Clear all source entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.clears += 1;
    }

    /// Remove expired entries
    pub fn cleanup_expired(&mut self) -> usize {
        let now = get_time_ms();
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(now))
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            self.entries.remove(&id);
            self.stats.expirations += 1;
        }
        count
    }

    /// Get the number of cached sources
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub const fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

impl Default for SourceCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CacheStats
// ============================================================================

/// Statistics for cache operations
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of inserts
    pub inserts: u64,
    /// Number of evictions (LRU)
    pub evictions: u64,
    /// Number of explicit invalidations
    pub invalidations: u64,
    /// Number of TTL expirations
    pub expirations: u64,
    /// Number of clear operations
    pub clears: u64,
}

impl CacheStats {
    /// Get the cache hit rate (0.0 - 1.0)
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get the total number of requests
    #[must_use]
    pub const fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ============================================================================
// CacheManager - Combined Cache Management
// ============================================================================

/// Combined cache manager for all cache types
///
/// Provides a unified interface for managing query, context, and source caches.
#[derive(Debug)]
pub struct CacheManager {
    /// Query result cache
    pub query_cache: QueryCache,
    /// Context cache
    pub context_cache: ContextCache,
    /// Source cache
    pub source_cache: SourceCache,
    /// Whether caching is enabled
    enabled: bool,
}

impl CacheManager {
    /// Create a new cache manager with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            query_cache: QueryCache::new(),
            context_cache: ContextCache::new(),
            source_cache: SourceCache::new(),
            enabled: true,
        }
    }

    /// Create a new cache manager with custom configuration
    #[must_use]
    pub fn with_config(
        query_max_entries: usize,
        query_ttl_ms: u64,
        context_ttl_ms: u64,
        source_ttl_ms: u64,
    ) -> Self {
        Self {
            query_cache: QueryCache::with_config(query_max_entries, query_ttl_ms),
            context_cache: ContextCache::with_config(context_ttl_ms, false),
            source_cache: SourceCache::with_ttl(source_ttl_ms),
            enabled: true,
        }
    }

    /// Check if caching is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable caching
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable caching
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        self.query_cache.clear();
        self.context_cache.clear();
        self.source_cache.clear();
    }

    /// Cleanup expired entries in all caches
    ///
    /// Returns the total number of entries removed.
    pub fn cleanup_all(&mut self) -> usize {
        let mut count = 0;
        count += self.query_cache.cleanup_expired();
        count += self.context_cache.cleanup_expired();
        count += self.source_cache.cleanup_expired();
        count
    }

    /// Get combined statistics from all caches
    #[must_use]
    pub fn combined_stats(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            query: *self.query_cache.stats(),
            context: *self.context_cache.stats(),
            source: *self.source_cache.stats(),
        }
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined statistics from all caches
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CombinedCacheStats {
    /// Query cache statistics
    pub query: CacheStats,
    /// Context cache statistics
    pub context: CacheStats,
    /// Source cache statistics
    pub source: CacheStats,
}

impl CombinedCacheStats {
    /// Get the total hits across all caches
    #[must_use]
    pub const fn total_hits(&self) -> u64 {
        self.query.hits + self.context.hits + self.source.hits
    }

    /// Get the total misses across all caches
    #[must_use]
    pub const fn total_misses(&self) -> u64 {
        self.query.misses + self.context.misses + self.source.misses
    }

    /// Get the overall hit rate
    #[must_use]
    pub fn overall_hit_rate(&self) -> f64 {
        let total = self.total_hits() + self.total_misses();
        if total == 0 {
            0.0
        } else {
            self.total_hits() as f64 / total as f64
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current time in milliseconds (platform-independent)
#[inline]
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
        0 // No time tracking in no_std/WASM - caller must provide timestamps
    }
}

/// Compute FNV-1a hash for a string (used for query hashing)
#[must_use]
pub fn fnv1a_hash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

// ============================================================================
// Thread-Safe Wrappers (std only)
// ============================================================================

#[cfg(feature = "std")]
pub mod sync {
    //! Thread-safe cache wrappers using parking_lot or std sync primitives

    use super::*;
    use std::sync::{Arc, RwLock};

    /// Thread-safe query cache
    pub type SharedQueryCache = Arc<RwLock<QueryCache>>;

    /// Thread-safe context cache
    pub type SharedContextCache = Arc<RwLock<ContextCache>>;

    /// Thread-safe source cache
    pub type SharedSourceCache = Arc<RwLock<SourceCache>>;

    /// Thread-safe cache manager
    pub type SharedCacheManager = Arc<RwLock<CacheManager>>;

    /// Create a new shared query cache
    #[must_use]
    pub fn shared_query_cache() -> SharedQueryCache {
        Arc::new(RwLock::new(QueryCache::new()))
    }

    /// Create a new shared context cache
    #[must_use]
    pub fn shared_context_cache() -> SharedContextCache {
        Arc::new(RwLock::new(ContextCache::new()))
    }

    /// Create a new shared source cache
    #[must_use]
    pub fn shared_source_cache() -> SharedSourceCache {
        Arc::new(RwLock::new(SourceCache::new()))
    }

    /// Create a new shared cache manager
    #[must_use]
    pub fn shared_cache_manager() -> SharedCacheManager {
        Arc::new(RwLock::new(CacheManager::new()))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // QueryCache Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_query_cache_basic() {
        let mut cache = QueryCache::new();

        // Put and get
        cache.put(12345, "result".to_string(), "source1".to_string());
        let entry = cache.get(12345).unwrap();
        assert_eq!(entry.result, "result");
        assert_eq!(entry.source_id, "source1");
        assert_eq!(entry.hits, 1);
    }

    #[test]
    fn test_query_cache_miss() {
        let mut cache = QueryCache::new();
        assert!(cache.get(99999).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_query_cache_lru_eviction() {
        let mut cache = QueryCache::with_config(3, DEFAULT_TTL_MS);

        // Fill cache
        cache.put(1, "r1".to_string(), "s1".to_string());
        cache.put(2, "r2".to_string(), "s2".to_string());
        cache.put(3, "r3".to_string(), "s3".to_string());
        assert_eq!(cache.len(), 3);

        // Access item 1 to make it most recently used
        let _ = cache.get(1);

        // Add item 4 - should evict item 2 (LRU)
        cache.put(4, "r4".to_string(), "s4".to_string());
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(1)); // Most recently accessed
        assert!(!cache.contains(2)); // Should be evicted
        assert!(cache.contains(3));
        assert!(cache.contains(4));
    }

    #[test]
    fn test_query_cache_update() {
        let mut cache = QueryCache::new();

        cache.put(1, "old".to_string(), "s1".to_string());
        cache.put(1, "new".to_string(), "s1".to_string());

        let entry = cache.get(1).unwrap();
        assert_eq!(entry.result, "new");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_query_cache_invalidate() {
        let mut cache = QueryCache::new();

        cache.put(1, "r1".to_string(), "s1".to_string());
        cache.invalidate(1);
        assert!(cache.get(1).is_none());
        assert_eq!(cache.stats().invalidations, 1);
    }

    #[test]
    fn test_query_cache_clear() {
        let mut cache = QueryCache::new();

        cache.put(1, "r1".to_string(), "s1".to_string());
        cache.put(2, "r2".to_string(), "s2".to_string());
        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.stats().clears, 1);
    }

    // -------------------------------------------------------------------------
    // ContextCache Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_context_cache_basic() {
        let mut cache = ContextCache::new();
        let ctx = CombinedContext::new().with_timestamp(12345);

        cache.put("provider1", ctx.clone());
        let cached = cache.get("provider1").unwrap();
        assert_eq!(cached.timestamp, 12345);
    }

    #[test]
    fn test_context_cache_miss() {
        let mut cache = ContextCache::new();
        assert!(cache.get("unknown").is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_context_cache_refresh_on_access() {
        let mut cache = ContextCache::with_config(DEFAULT_CONTEXT_TTL_MS, true);
        let ctx = CombinedContext::new();

        cache.put("p1", ctx);

        // Access should refresh the TTL
        let _ = cache.get("p1");

        // The entry should still be valid
        assert!(cache.get("p1").is_some());
    }

    // -------------------------------------------------------------------------
    // SourceCache Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_source_cache_basic() {
        let mut cache = SourceCache::new();
        let caps = SourceCapabilities::full();
        let stats = SourceStats::default();

        cache.put("source1", caps, stats);

        let cached_caps = cache.get_capabilities("source1").unwrap();
        assert!(cached_caps.sparql_1_1);
    }

    #[test]
    fn test_source_cache_update_stats() {
        let mut cache = SourceCache::new();
        let caps = SourceCapabilities::default();
        let mut stats = SourceStats::default();
        stats.total_queries = 10;

        cache.put("s1", caps, stats);

        let mut new_stats = SourceStats::default();
        new_stats.total_queries = 20;
        cache.update_stats("s1", new_stats);

        let cached_stats = cache.get_stats("s1").unwrap();
        assert_eq!(cached_stats.total_queries, 20);
    }

    // -------------------------------------------------------------------------
    // CacheStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut stats = CacheStats::default();
        stats.hits = 80;
        stats.misses = 20;

        assert!((stats.hit_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_empty() {
        let stats = CacheStats::default();
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // CacheManager Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_manager_basic() {
        let mut manager = CacheManager::new();

        // Use all caches
        manager
            .query_cache
            .put(1, "r1".to_string(), "s1".to_string());
        manager.context_cache.put("p1", CombinedContext::new());
        manager
            .source_cache
            .put("s1", SourceCapabilities::default(), SourceStats::default());

        assert!(!manager.query_cache.is_empty());
        assert!(!manager.context_cache.is_empty());
        assert!(!manager.source_cache.is_empty());
    }

    #[test]
    fn test_cache_manager_clear_all() {
        let mut manager = CacheManager::new();

        manager
            .query_cache
            .put(1, "r1".to_string(), "s1".to_string());
        manager.context_cache.put("p1", CombinedContext::new());

        manager.clear_all();

        assert!(manager.query_cache.is_empty());
        assert!(manager.context_cache.is_empty());
    }

    #[test]
    fn test_cache_manager_enable_disable() {
        let mut manager = CacheManager::new();

        assert!(manager.is_enabled());
        manager.disable();
        assert!(!manager.is_enabled());
        manager.enable();
        assert!(manager.is_enabled());
    }

    // -------------------------------------------------------------------------
    // Hash Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fnv1a_hash() {
        let hash1 = fnv1a_hash("test");
        let hash2 = fnv1a_hash("test");
        let hash3 = fnv1a_hash("different");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    // -------------------------------------------------------------------------
    // CacheEntry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new("result".to_string(), "source".to_string(), 1000);

        // Entry should not be expired immediately
        assert!(!entry.is_expired(entry.cached_at));

        // Entry should be expired after TTL
        assert!(entry.is_expired(entry.cached_at + 1001));
    }

    #[test]
    fn test_cache_entry_remaining_ttl() {
        let entry = CacheEntry::new("result".to_string(), "source".to_string(), 1000);

        assert_eq!(entry.remaining_ttl_ms(entry.cached_at), 1000);
        assert_eq!(entry.remaining_ttl_ms(entry.cached_at + 500), 500);
        assert_eq!(entry.remaining_ttl_ms(entry.cached_at + 1500), 0);
    }

    // -------------------------------------------------------------------------
    // Thread-Safe Wrapper Tests (std only)
    // -------------------------------------------------------------------------

    #[cfg(feature = "std")]
    #[test]
    fn test_shared_cache_creation() {
        let _cache = sync::shared_query_cache();
        let _ctx_cache = sync::shared_context_cache();
        let _src_cache = sync::shared_source_cache();
        let _manager = sync::shared_cache_manager();
    }
}
