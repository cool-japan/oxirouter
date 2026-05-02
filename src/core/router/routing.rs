//! Routing logic: heuristic and ML-based source selection with scoring.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use super::{Router, RoutingExplanation, ScoreComponent};
use crate::context::{CombinedContext, ContextProvider};
use crate::core::error::{OxiRouterError, Result};
use crate::core::query::Query;
use crate::core::source::{DataSource, SelectionReason, SourceRanking, SourceSelection};

#[cfg(feature = "ml")]
use crate::ml::{FeatureVector, Model};

impl<C: ContextProvider> Router<C> {
    /// Route a query to the best sources
    ///
    /// # Errors
    ///
    /// Returns an error if no sources are available or routing fails
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, query),
            fields(
                predicates = query.predicates.len(),
                sources = self.sources.len()
            )
        )
    )]
    pub fn route(&self, query: &Query) -> Result<SourceRanking> {
        if self.sources.is_empty() {
            let query_vocabs = query.vocabularies();
            let missing_vocabularies: Vec<String> = query_vocabs.into_iter().collect();
            #[cfg(feature = "observability")]
            {
                metrics::counter!("oxirouter.route.errors", "kind" => "no_sources").increment(1);
            }
            return Err(OxiRouterError::NoSources {
                reason: "no sources registered".into(),
                missing_vocabularies,
            });
        }

        let start = Self::get_time_us();
        let mut ranking = SourceRanking::new();

        // Get context if enabled
        let context = if self.config.use_context {
            Some(self.context_provider.get_combined_context())
        } else {
            None
        };
        ranking.context_used = context.is_some();

        // Try ML model first if enabled
        #[cfg(feature = "ml")]
        if self.config.use_ml {
            if let Some(ref model) = self.model {
                if let Ok(ml_ranking) = self.route_with_ml(query, model.as_ref(), context.as_ref())
                {
                    ranking = ml_ranking;
                    ranking.ml_used = true;
                }
            }
        }

        // Fall back to heuristic routing if ML didn't produce results
        if ranking.is_empty() {
            ranking = self.route_heuristic(query, context.as_ref());
        }

        // Apply compliance filtering if we have legal context
        if let Some(ref ctx) = context {
            self.apply_compliance_filter(&mut ranking, ctx, query);
        }

        // Sort by confidence and limit results
        ranking.sort_by_confidence();
        ranking.sources.truncate(self.config.max_sources);

        // Filter by minimum confidence
        ranking
            .sources
            .retain(|s| s.confidence >= self.config.min_confidence);

        ranking.processing_time_us = Self::get_time_us() - start;

        #[cfg(feature = "observability")]
        {
            metrics::counter!("oxirouter.route.total").increment(1);
            if let Some(top) = ranking.sources.first() {
                metrics::histogram!("oxirouter.route.confidence").record(f64::from(top.confidence));
            }
        }

        Ok(ranking)
    }

    /// Route using ML model
    #[cfg(feature = "ml")]
    pub(super) fn route_with_ml(
        &self,
        query: &Query,
        model: &dyn Model,
        context: Option<&CombinedContext>,
    ) -> Result<SourceRanking> {
        let mut ranking = SourceRanking::new();

        // Build feature vector
        let features = FeatureVector::from_query_and_context(query, context)?;

        // Pre-filter source IDs to exclude circuit-tripped sources.
        let source_ids: Vec<&String> = self
            .sources
            .iter()
            .filter(|(_, source)| {
                // Apply circuit breaker: exclude source if it has a trip
                // timestamp, the CB clock is enabled, and cooldown has NOT elapsed.
                if let (Some(tripped_until), Some(now_fn)) = (
                    source.stats.tripped_until_ms,
                    self.config.circuit_breaker.now_ms,
                ) {
                    // true = include (cooldown elapsed), false = exclude (still tripped)
                    return tripped_until <= now_fn();
                }
                true // no trip or CB disabled — include
            })
            .map(|(id, _)| id)
            .collect();

        let predictions = model.predict(&features, &source_ids)?;

        for (source_id, confidence) in predictions {
            #[cfg(feature = "observability")]
            if tracing::enabled!(tracing::Level::DEBUG) {
                tracing::debug!(source_id = %source_id, confidence = confidence, "ml score");
            }
            ranking.add(SourceSelection {
                source_id: source_id.clone(),
                confidence,
                estimated_latency_ms: self
                    .sources
                    .get(&source_id)
                    .map(|s| s.stats.avg_latency_ms as u32)
                    .unwrap_or(1000),
                reason: SelectionReason::ModelPrediction,
            });
        }

        Ok(ranking)
    }

    /// Route using heuristic methods.
    ///
    /// This is a thin wrapper around [`Router::compute_source_components`]: it
    /// applies hard-filter gates (availability, circuit breaker, capability,
    /// load) and then delegates all scoring to the canonical function so that
    /// [`Router::route`] and [`Router::explain`] always agree on scores.
    pub(super) fn route_heuristic(
        &self,
        query: &Query,
        context: Option<&CombinedContext>,
    ) -> SourceRanking {
        let mut ranking = SourceRanking::new();

        for source in self.sources.values() {
            // ── Hard filter: source marked unavailable ───────────────────────
            if !source.available {
                continue;
            }

            // ── Hard filter: circuit breaker ─────────────────────────────────
            // Skip tripped sources only when both a trip timestamp exists AND
            // the clock function is configured (CB is enabled).
            if let (Some(tripped_until), Some(now_fn)) = (
                source.stats.tripped_until_ms,
                self.config.circuit_breaker.now_ms,
            ) {
                if tripped_until > now_fn() {
                    continue; // source is still in cooldown — skip it
                }
                // Cooldown elapsed — auto-recover: include source and let
                // learn_from_outcome clear the trip on the next success.
            }

            // ── Hard filter: capability gate ─────────────────────────────────
            if query.requires_sparql_1_1() && !source.capabilities.sparql_1_1 {
                continue;
            }

            // ── Hard filter: load circuit breaker ────────────────────────────
            // This is a distinct hard gate: if the load manager's circuit is
            // open for this endpoint we cannot send queries to it at all.
            // The availability *score* component in compute_source_components
            // provides a soft penalty for degraded (but reachable) endpoints.
            #[cfg(any(feature = "load", feature = "std"))]
            if let Some(ctx) = context {
                if let Some(ref load) = ctx.load {
                    if !load.is_available(&source.endpoint) {
                        continue; // circuit open — hard skip
                    }
                }
            }

            // ── Score via canonical function ─────────────────────────────────
            let (confidence, components) = self.compute_source_components(source, query, context);
            let confidence = confidence.clamp(0.0, 1.0);

            // ── Derive SelectionReason from highest positive component ────────
            let reason = components
                .iter()
                .filter(|c| c.contribution > 0.0 && c.name != "circuit_breaker")
                .max_by(|a, b| {
                    a.contribution
                        .partial_cmp(&b.contribution)
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .map(|c| match c.name.as_str() {
                    "vocabulary" => SelectionReason::VocabularyMatch,
                    "region" => SelectionReason::GeographicProximity,
                    "performance" => SelectionReason::HistoricalPerformance,
                    _ => SelectionReason::Fallback,
                })
                .unwrap_or(SelectionReason::Fallback);

            #[cfg(feature = "observability")]
            if tracing::enabled!(tracing::Level::DEBUG) {
                tracing::debug!(
                    source_id = %source.id,
                    confidence = confidence,
                    "heuristic score"
                );
            }
            ranking.add(SourceSelection {
                source_id: source.id.clone(),
                confidence,
                estimated_latency_ms: source.stats.avg_latency_ms as u32,
                reason,
            });
        }

        ranking
    }

    /// Calculate vocabulary match score
    pub(super) fn calculate_vocab_score(
        &self,
        query_vocabs: &hashbrown::HashSet<String>,
        source: &DataSource,
    ) -> f32 {
        if query_vocabs.is_empty() || source.vocabularies.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        for vocab in query_vocabs {
            if source.supports_vocabulary(vocab) {
                matches += 1;
            }
        }

        matches as f32 / query_vocabs.len() as f32
    }

    /// Calculate geographic proximity score
    pub(super) fn calculate_geo_score(
        &self,
        source: &DataSource,
        context: &CombinedContext,
    ) -> f32 {
        // Check if we have geo context
        #[cfg(feature = "geo")]
        if let Some(ref geo) = context.geo {
            if let Some(ref country) = geo.country_code {
                // Same country = high score
                if source.in_region(country) {
                    return 1.0;
                }
            }

            if let Some(ref region) = geo.region {
                // Same region = medium score
                if source.in_region(region) {
                    return 0.7;
                }
            }
        }

        // No geo context or no match
        #[cfg(not(feature = "geo"))]
        {
            let _ = source;
            let _ = context;
        }

        0.0
    }

    /// Compute per-feature score components for a single source against a query.
    ///
    /// Returns `(total_score, components)` where `total_score == sum(component.contribution)`.
    ///
    /// The base priority contribution is included as the first component so that
    /// `sum(contribution) == total_score` always holds.
    pub(super) fn compute_source_components(
        &self,
        source: &DataSource,
        query: &Query,
        context: Option<&CombinedContext>,
    ) -> (f32, Vec<ScoreComponent>) {
        let mut components: Vec<ScoreComponent> = Vec::new();

        // ── Base priority ────────────────────────────────────────────────────
        let base_raw = source.priority;
        let base_contribution = base_raw * 0.1_f32;
        components.push(ScoreComponent {
            name: "priority".into(),
            weight: 0.1,
            raw_value: base_raw,
            contribution: base_contribution,
        });

        // ── Vocabulary match ─────────────────────────────────────────────────
        let query_vocabs = query.vocabularies();
        let vocab_raw = self.calculate_vocab_score(&query_vocabs, source);
        let vocab_contribution = vocab_raw * self.config.vocab_weight;
        components.push(ScoreComponent {
            name: "vocabulary".into(),
            weight: self.config.vocab_weight,
            raw_value: vocab_raw,
            contribution: vocab_contribution,
        });

        // ── Historical performance ───────────────────────────────────────────
        let reliability = self
            .query_log
            .combined_reliability(&source.id, source.stats.success_rate);
        let perf_raw =
            if source.stats.has_history() || self.query_log.source_stats(&source.id).is_some() {
                let latency_score = (1.0 - (source.stats.avg_latency_ms / 10_000.0).min(1.0)) * 0.3;
                (reliability * 0.7 + latency_score).clamp(0.0, 1.0)
            } else {
                0.0
            };
        let perf_contribution = perf_raw * self.config.history_weight;
        components.push(ScoreComponent {
            name: "performance".into(),
            weight: self.config.history_weight,
            raw_value: perf_raw,
            contribution: perf_contribution,
        });

        // ── Geographic proximity ─────────────────────────────────────────────
        let geo_raw = context
            .map(|ctx| self.calculate_geo_score(source, ctx))
            .unwrap_or(0.0);
        let geo_contribution = geo_raw * self.config.geo_weight;
        components.push(ScoreComponent {
            name: "region".into(),
            weight: self.config.geo_weight,
            raw_value: geo_raw,
            contribution: geo_contribution,
        });

        // ── Availability (load-aware) ────────────────────────────────────────
        #[cfg(any(feature = "load", feature = "std"))]
        let availability_raw: f32 = context
            .and_then(|ctx| ctx.load.as_ref())
            .map(|load| load.availability_score(&source.endpoint))
            .unwrap_or(1.0);
        #[cfg(not(any(feature = "load", feature = "std")))]
        let availability_raw: f32 = 1.0;

        // Availability is a multiplicative modifier; represent its contribution
        // as a negative delta (penalty) when availability < 1.0, else 0.
        let pre_availability_total = components.iter().map(|c| c.contribution).sum::<f32>();
        let availability_contribution = if availability_raw >= 1.0 {
            0.0
        } else {
            // The score is multiplied by availability_raw; the loss delta is:
            pre_availability_total * (availability_raw - 1.0)
        };
        components.push(ScoreComponent {
            name: "availability".into(),
            weight: 1.0,
            raw_value: availability_raw,
            contribution: availability_contribution,
        });

        // ── Circuit breaker pass-through ─────────────────────────────────────
        // If the source reaches this point it hasn't been tripped (or CB is
        // disabled / cooldown elapsed). Represent as a zero-cost component.
        components.push(ScoreComponent {
            name: "circuit_breaker".into(),
            weight: 0.0,
            raw_value: 1.0,
            contribution: 0.0,
        });

        // ── RL Q-blend ────────────────────────────────────────────────────────
        // Blend 20% RL Q-value signal into the running score total. The delta
        // is (pre_rl * 0.8 + q * 0.2) - pre_rl = q * 0.2 - pre_rl * 0.2.
        #[cfg(feature = "rl")]
        {
            if let Some(ref policy) = self.policy {
                let q = policy.get_q_value(&source.id);
                if q > 0.0 {
                    let pre_rl_total: f32 = components.iter().map(|c| c.contribution).sum();
                    let post_rl_total = pre_rl_total * 0.8 + q * 0.2;
                    let rl_contribution = post_rl_total - pre_rl_total;
                    components.push(ScoreComponent {
                        name: "rl_q_blend".into(),
                        weight: 0.2,
                        raw_value: q,
                        contribution: rl_contribution,
                    });
                }
            }
        }

        // ── Capability-aware components (soft preferences) ───────────────────

        // Capability: federation (SERVICE clause support)
        if query.has_service {
            let raw_value = if source.capabilities.federation {
                1.0_f32
            } else {
                -1.0_f32
            };
            let contribution = 0.2_f32 * raw_value;
            components.push(ScoreComponent {
                name: "capability_federation_match".into(),
                weight: 0.2,
                raw_value,
                contribution,
            });
        }

        // Capability: aggregation (GROUP BY / COUNT / SUM etc.)
        if query.has_aggregation {
            let raw_value = if source.capabilities.aggregation {
                1.0_f32
            } else {
                -1.0_f32
            };
            let contribution = 0.2_f32 * raw_value;
            components.push(ScoreComponent {
                name: "capability_aggregation_match".into(),
                weight: 0.2,
                raw_value,
                contribution,
            });
        }

        // Capability: property paths
        if query.has_property_paths {
            let raw_value = if source.capabilities.property_paths {
                1.0_f32
            } else {
                -1.0_f32
            };
            let contribution = 0.2_f32 * raw_value;
            components.push(ScoreComponent {
                name: "capability_property_paths_match".into(),
                weight: 0.2,
                raw_value,
                contribution,
            });
        }

        // Capability: subqueries (nested SELECT)
        if query.has_subquery {
            let raw_value = if source.capabilities.subqueries {
                1.0_f32
            } else {
                -1.0_f32
            };
            let contribution = 0.2_f32 * raw_value;
            components.push(ScoreComponent {
                name: "capability_subqueries_match".into(),
                weight: 0.2,
                raw_value,
                contribution,
            });
        }

        // Result density: bias toward sources that return results
        {
            let avg = source.stats.avg_results_per_query() as f32;
            let raw_value = (avg / 100.0_f32).min(1.0_f32);
            let contribution = 0.05_f32 * raw_value;
            components.push(ScoreComponent {
                name: "result_density".into(),
                weight: 0.05,
                raw_value,
                contribution,
            });
        }

        // ── P2P kind boost ────────────────────────────────────────────────────
        // Boost or slightly penalise P2P sources based on connectivity quality.
        #[cfg(feature = "p2p")]
        #[cfg(any(feature = "device", feature = "std"))]
        if source.kind.is_p2p() {
            let multiplier: f32 = if let Some(ctx) = context {
                if let Some(dev) = ctx.device.as_ref() {
                    if dev.is_offline() {
                        // P2P shines offline: local-first delivery beats unreachable endpoints
                        1.25_f32
                    } else if dev.network_quality() < 0.3 {
                        // Mild boost on poor but non-zero connectivity
                        1.10_f32
                    } else {
                        // Slight penalty: centralised SPARQL endpoints usually win on good links
                        0.95_f32
                    }
                } else {
                    1.0_f32
                }
            } else {
                1.0_f32
            };
            if (multiplier - 1.0).abs() > f32::EPSILON {
                let pre_p2p_total: f32 = components.iter().map(|c| c.contribution).sum();
                let p2p_contribution = (multiplier - 1.0) * pre_p2p_total;
                components.push(ScoreComponent {
                    name: "p2p_kind_boost".into(),
                    weight: 1.0,
                    raw_value: multiplier,
                    contribution: p2p_contribution,
                });
            }
        }

        // ── Total score ──────────────────────────────────────────────────────
        // Sum all component contributions; clamp to [0.0, 1.0].
        // If the unclamped sum exceeds 1.0, push a "clamp" component with a
        // negative contribution so that sum(contributions) == total_score
        // exactly, preserving the invariant tested by explain tests.
        let unclamped: f32 = components.iter().map(|c| c.contribution).sum();
        let total_score = unclamped.clamp(0.0, 1.0);
        let clamp_delta = total_score - unclamped;
        if clamp_delta.abs() > f32::EPSILON {
            components.push(ScoreComponent {
                name: "clamp".into(),
                weight: 1.0,
                raw_value: total_score,
                contribution: clamp_delta,
            });
        }

        // Derive final total from components to guarantee sum == total_score.
        let derived_total: f32 = components.iter().map(|c| c.contribution).sum();
        (derived_total, components)
    }

    /// Return per-feature scoring breakdowns for every non-tripped source.
    ///
    /// Unlike [`Router::route`] this method does **not** truncate to
    /// `max_sources` — it returns one [`RoutingExplanation`] per eligible
    /// source so callers can inspect the full picture.
    ///
    /// # Errors
    ///
    /// Returns [`OxiRouterError::NoSources`] when no sources are registered.
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, query),
            fields(predicates = query.predicates.len())
        )
    )]
    pub fn explain(&self, query: &Query) -> Result<Vec<RoutingExplanation>> {
        if self.sources.is_empty() {
            let query_vocabs = query.vocabularies();
            let missing_vocabularies: Vec<String> = query_vocabs.into_iter().collect();
            return Err(OxiRouterError::NoSources {
                reason: "no sources registered".into(),
                missing_vocabularies,
            });
        }

        let context = if self.config.use_context {
            Some(self.context_provider.get_combined_context())
        } else {
            None
        };

        let mut explanations: Vec<RoutingExplanation> = Vec::new();
        for source in self.sources.values() {
            // Skip circuit-tripped sources (operational, not capability-related).
            if let (Some(tripped_until), Some(now_fn)) = (
                source.stats.tripped_until_ms,
                self.config.circuit_breaker.now_ms,
            ) {
                if tripped_until > now_fn() {
                    continue;
                }
            }

            // Capability-gate check: produce a diagnostic entry rather than
            // silently omitting the source, so operators can see why it was
            // excluded.
            if query.requires_sparql_1_1() && !source.capabilities.sparql_1_1 {
                explanations.push(RoutingExplanation {
                    source_id: source.id.clone(),
                    total_score: -1.0,
                    components: alloc::vec![ScoreComponent {
                        name: "capability_gate_failed".into(),
                        weight: 1.0,
                        raw_value: -1.0,
                        contribution: -1.0,
                    }],
                });
                continue;
            }

            let (total_score, components) =
                self.compute_source_components(source, query, context.as_ref());
            explanations.push(RoutingExplanation {
                source_id: source.id.clone(),
                total_score,
                components,
            });
        }

        Ok(explanations)
    }

    /// Apply compliance filtering based on legal context
    pub(super) fn apply_compliance_filter(
        &self,
        ranking: &mut SourceRanking,
        context: &CombinedContext,
        _query: &Query,
    ) {
        #[cfg(feature = "legal")]
        if let Some(ref legal) = context.legal {
            // Filter out sources that don't allow data transfer
            if !legal.data_transfer_allowed {
                ranking.sources.retain(|s| {
                    // Only keep sources in compliant regions
                    if let Some(source) = self.sources.get(&s.source_id) {
                        if let Some(ref allowed) = legal.allowed_regions {
                            return source.regions.iter().any(|r| allowed.contains(r));
                        }
                    }
                    true
                });
            }

            // Boost sources in required regions
            if let Some(ref required) = legal.required_regions {
                for selection in &mut ranking.sources {
                    if let Some(source) = self.sources.get(&selection.source_id) {
                        if source.regions.iter().any(|r| required.contains(r)) {
                            selection.confidence = (selection.confidence * 1.5).min(1.0);
                            selection.reason = SelectionReason::ComplianceRequired;
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "legal"))]
        {
            let _ = ranking;
            let _ = context;
        }
    }

    /// Get current time in microseconds (platform-independent)
    pub(super) fn get_time_us() -> u64 {
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0)
        }

        #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
        {
            0 // No time tracking in no_std or WASM
        }
    }
}
