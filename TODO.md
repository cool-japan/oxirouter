# OxiRouter TODO

_Last updated: 2026-05-03 — version 0.1.0 (initial release)_

All Phase 1–4 and Round 8 items are complete; the v0.1.0 milestone is shipped.
Forward-looking work lives in **Proposed follow-ups (Round 9+)** at the bottom of this file.

## Completed

### Infrastructure
- [x] All ecosystem crates resolved via crates.io (oxigdal-core 0.1.4, legalis-core 0.1.5, celers-core 0.2.0, mielin-hal 0.1.0-rc.1, oxirs-ttl deferred)
- [x] Zero warnings across all feature combinations (default, `full`, `full,http`, `cli`, `observability`); `clippy -D warnings` and `rustdoc -D warnings` both clean
- [x] 501 unit/integration tests + 6 doc tests passing (cargo nextest run --all-features)

### Phase 1 – Foundation (COMPLETE)
- [x] Core routing engine (`Router`, `DataSource`, `Query`, `SourceRanking`)
- [x] Heuristic routing: vocabulary match, historical performance, geo proximity
- [x] **QueryLog** – statistics-based routing history tracking (`src/core/query_log.rs`)
  - Per-source aggregated stats (success rate, avg latency, avg reward)
  - `combined_reliability()` blending live stats with log history
  - Ring-buffer eviction with configurable capacity
- [x] **`route_and_log()`** – routing with automatic log recording
- [x] **`learn_from_outcome()`** – post-execution feedback updating stats + RL + log
- [x] Load-aware heuristic routing: `LoadContext.availability_score()` used to downgrade/skip circuit-broken endpoints
- [x] Context modules: GeoContext, DeviceContext, LoadContext, LegalContext, EcosystemContextProvider

### Phase 2 – Intelligence (COMPLETE)
- [x] Feature vector extraction (24 query features + 14 context features = 38 dims)
- [x] Naive Bayes classifier for source selection
- [x] Neural network (MLP) with momentum, dropout, step-decay LR
- [x] RL feedback loop: Policy (UCB, ε-greedy, Thompson), Reward, Feedback
- [x] **RL integration in Router**: `enable_rl()`, `set_policy()`, Q-value blending (20%) in heuristic
- [x] Model serialization (binary v2 format with full roundtrip)
- [x] WASM bindings (wasm-bindgen)
- [x] LRU cache with TTL for queries, context, and source capabilities
- [x] `splitrs` refactor of `src/ml/neural.rs` (completed 2026-04-28)
  - **Goal:** `src/ml/neural.rs` drops from 1993 → ≤900 lines by extracting four natural sub-modules: `activation.rs`, `layer.rs`, `optimizer.rs`, `schedule.rs`. All 402 existing tests pass after the split.
  - **Design:** Extract: `Activation` enum (lines 14–103) → `src/ml/activation.rs`; `Layer`/`LayerCache`/`LayerGradients` (lines 104–273) → `src/ml/layer.rs`; `OptimizerType`/`AdamConfig`/`OptimizerState` (lines 274–358) → `src/ml/optimizer.rs`; `LearningRateSchedule`/`EarlyStoppingConfig`/`EarlyStoppingState`/`DropoutConfig`/`DropoutState` (lines 359–487) → `src/ml/schedule.rs`. Move `#[cfg(test)] mod tests { ... }` (lines 1392–1993) to `src/ml/neural_tests.rs` using `#[cfg(test)] #[path = "neural_tests.rs"] mod tests;` if needed to satisfy ≤1500 acceptance gate. Update `src/ml/mod.rs` to re-export new modules. Adjust visibility for cross-module types.
  - **Files:** NEW `src/ml/activation.rs`, `src/ml/layer.rs`, `src/ml/optimizer.rs`, `src/ml/schedule.rs`; MODIFIED `src/ml/neural.rs` (large reduction), `src/ml/mod.rs`; MAYBE NEW `src/ml/neural_tests.rs`
  - **Tests:** No new tests required. All 402 existing tests must still pass. Acceptance gate: `wc -l src/ml/neural.rs` ≤ 1500.
  - **Risk:** Medium. Visibility adjustments can cascade. Mitigation: run `cargo test --features full --lib` after each extraction.

## Implemented (Round 1–7)

_All items below shipped in v0.1.0. Originally tracked under "Remaining" while the
roadmap was open; retained here as a release-history record of what landed when._

### Phase 1 – Foundation (COMPLETE)
- [x] Real oxigdal integration in `EcosystemContextProvider::get_geo_context()`
  - `StaticOxigdalGeoSensor` and `DynamicOxigdalGeoSensor` via injectable `GeoSensor` trait
  - `EcosystemContextProvider` rewritten with `Option<Box<dyn GeoSensor>>` + Mutex cache
- [x] Real mielin integration in `EcosystemContextProvider::get_device_context()`
  - `MielinDeviceSensor` uses `mielin_hal::capabilities::HardwareProfile` + `platform::detect_platform()`
  - Battery/network left `None`/`Unknown` (mielin_hal does not expose them — documented constraint)
- [x] Real celers integration in `EcosystemContextProvider::get_load_context()`
  - `CelersLoadSensor<F>` closure-based sensor; `from_celers_stats` maps loadavg, pool, broker, error_rate
- [x] Real legalis integration in `EcosystemContextProvider::get_legal_context()`
  - `LegalisPolicyEngine` evaluates `Vec<legalis_core::Statute>`; pre-built GDPR/CCPA/LGPD statute sets
- [x] Router-side circuit breaker with auto-recovery (planned 2026-04-27)
  - **Goal:** Sources that fail N consecutive times are auto-skipped for `cooldown_ms`, then probed again on the next route call. Circuit breaker complements existing live `availability_score` — long memory for repeatedly-broken endpoints, separate from per-call load.
  - **Design:** NEW `CircuitBreakerConfig { failure_threshold: u32 (default 5), cooldown_ms: u64 (default 30_000), now_ms: Option<fn() -> u64> }` field on `Router::config`. Extend `SourceStats` with `consecutive_failures: u32` and `tripped_until_ms: Option<u64>`. `learn_from_outcome` increments `consecutive_failures` on `success=false` (trips when threshold reached), resets to 0 on `success=true`. `Router::route` filters out sources where `tripped_until_ms.map_or(false, |t| t > now_ms())`. On std, default `now_ms` uses `SystemTime`; on no_std defaults to `None` (breaker disabled — backwards-compatible).
  - **Files:** MODIFIED `src/core/source.rs`, `src/core/router.rs`, `src/lib.rs`; NEW `tests/circuit_breaker_test.rs`
  - **Tests:** 6 tests: failure accumulation, trip on threshold, source skipped during cooldown, auto-recovery after cooldown (mock clock), success resets counter, multi-source isolation.
  - **Risk:** Low–Medium. no_std path: inject `now_ms` as `Option<fn() -> u64>` — `None` disables the breaker entirely.
- [x] Router state persistence: `save_state()` / `load_state()` (planned 2026-04-27)
  - **Goal:** A trained Router survives process restart. After `learn_from_outcome` updates the model and query log, `router.save_state()` produces bytes; a fresh `Router::load_state(bytes)` resumes with sources, learned weights, policy state, and query log intact.
  - **Design:** NEW `src/core/state.rs` (~350 lines). `pub struct RouterState { magic: [u8; 4], version: u32, sources: Vec<DataSource>, model_state: Option<Vec<u8>>, rl_state: Option<Vec<u8>>, query_log: QueryLog }`. Magic `OXIR` (4 bytes) + version `0x00000001` (4 bytes LE) prefix. `save_state` calls each component's *live* `to_bytes()` (not cached `original_bytes`). `load_state` validates magic/version → `IncompatibleModel{reason}` on mismatch.
  - **Files:** NEW `src/core/state.rs`; MODIFIED `src/core/router.rs`, `src/core/mod.rs`, `src/lib.rs`; NEW `tests/state_persistence_test.rs`
  - **Tests:** 6 tests: roundtrip default config, roundtrip with model+RL, online-trained weights survive roundtrip, query log preserved, magic mismatch error, version mismatch error.
  - **Risk:** Medium. Live model weights vs cached `original_bytes` — rely on existing `to_bytes`/`from_bytes` patterns; extend only if missing.
- [x] Single source of truth for source scoring (planned 2026-04-28)
  - **Goal:** `route_heuristic` becomes a thin caller that builds `SourceRanking` from `compute_source_components` output. Single source of truth for scoring; `Router::explain()` cannot drift from `Router::route()`. Round-6 follow-up "route_heuristic duplicates scoring" is closed.
  - **Design:** Model every modifier as a `ScoreComponent`: extend `compute_source_components` (router.rs:921) to include RL Q-blend (`"rl_q_blend"`, raw_value=q, weight=0.2, contribution=0.2*(q-prior)), P2P boost (`"p2p_kind_boost"`, contribution=(multiplier-1.0)*prior_total), circuit-breaker (`"circuit_breaker"`), and capability gate (`"capability_gate_failed"`, raw_value=-1.0 — hard filter but emits diagnostic component). Rewrite `route_heuristic` (router.rs:745) as thin caller: for each source call `compute_source_components`, sum contributions, sort, build `SourceSelection`. ~60 lines vs 119 lines current. Snapshot pre-refactor scoring on fixed inputs; bake as frozen-output regression in test 6.
  - **Files:** MODIFIED `src/core/router.rs` (refactor 200+ lines); NEW `tests/scoring_consistency_test.rs`
  - **Tests:** 6 tests: (1) route/explain agree (10 queries × 5 sources, within f32::EPSILON×4); (2) components sum to total_score; (3) rl_blend appears as component; (4) p2p boost appears; (5) capability-filtered source visible in explain but not route; (6) regression on round-5 explain fixtures (vocab/region/sum-ε).
  - **Risk:** High — hot routing path. Mitigation: frozen-output regression + run full test suite before marking [x].

### Phase 2 – Intelligence (COMPLETE)
- [x] Re-enable `sparql` feature once `oxirs-core` upstream bug is fixed (planned 2026-04-27)
  - Bug: `oxirs-core 0.2.2` missing `use scirs2_core::RngExt` — **FIXED in oxirs-core 0.2.4 (verified 2026-04-27)**
  - **Goal:** `cargo build --features sparql` with real SPARQL parsing via `oxirs-core 0.2.4`. New `Query::from_sparql` uses `oxirs_core::sparql::extract_and_expand_prefixes` + `extract_select_variables`. `Router::route_sparql`, `Router::route_sparql_and_log`, agent dispatch path, and WASM `route_sparql_js` all use the SPARQL-aware path when feature enabled; heuristic `Query::parse` stays as the default path.
  - **Design:** Single `oxirs-core = { version = "0.2.4", optional = true, default-features = false, features = ["sparql-12"] }` dep; feature `sparql = ["dep:oxirs-core"]`. New `Query::projection_vars: SmallVec<[String; 4]>` field (`#[serde(default)]`). `#[cfg(feature = "sparql")] impl Query { pub fn from_sparql }`. Convenience methods on `Router<C>`. Feature-gated branches in `agent.rs` execute_route/execute_explain. WASM entry point under `cfg(all(wasm, sparql))`.
  - **Files:** MODIFIED Cargo.toml, src/core/query.rs, src/core/router.rs, src/agent.rs, src/wasm/bindings.rs; NEW tests/sparql_test.rs
  - **Tests:** 6 tests in tests/sparql_test.rs: prefix expansion, projection vars, non-select query, invalid input, route_sparql method, agent route via sparql path.
- [x] WASM: expose ML/RL APIs through `OxiRouter` WASM bindings
  - `route_and_log_js()`, `learn_from_outcome_js()`, `enable_rl_js()`
- [x] Online training: feed `TrainingSample` back into the ML model after each query
- [x] Ensemble model: combine NaiveBayes + NeuralNetwork predictions with configurable weights
- [x] Property-path parsing + path-aware vocabulary scoring (planned 2026-04-27)
  - **Goal:** SPARQL property-path expressions decompose into base IRIs so vocab scoring sees them. Queries like `?s foaf:knows+ ?o`, `?s foaf:knows/foaf:name ?n`, `?s ^dbo:author ?b` correctly contribute base predicates to `Query.predicates`, enabling `Router::calculate_vocab_score` to match them against `DataSource.vocabularies`.
  - **Design:** NEW `PropertyPath` enum in `sparql_ast.rs`: `Iri(String)`, `Inverse(Box<PropertyPath>)`, `Sequence(Vec<PropertyPath>)`, `Alternative(Vec<PropertyPath>)`, `ZeroOrMore(Box<PropertyPath>)`, `OneOrMore(Box<PropertyPath>)`, `ZeroOrOne(Box<PropertyPath>)`, `NegatedPropertySet(Vec<PropertyPath>)`. NEW `pub(crate) fn parse_property_path(s: &str) -> PropertyPath` — recursive descent honouring SPARQL 1.1 §9 precedence (alt < seq < unary). NEW `PropertyPath::base_iris(&self) -> Vec<&str>`. `Query::from_sparql` populates `predicates` via `base_iris()`. Best-effort + infallible: malformed input returns `PropertyPath::Iri(raw)`.
  - **Files:** MODIFIED `src/core/sparql_ast.rs`, `src/core/query.rs`; MODIFIED `tests/sparql_ast_test.rs`
  - **Tests:** 6 unit tests in `sparql_ast.rs` (precedence, inverse, sequence, alternative, quantifier, negated set); 3 integration tests (vocab decomposition, `path_expr_count` correctness, vocab scoring picks up base IRIs).
  - **Risk:** Medium. SPARQL 1.1 §9 precedence corners. Mitigation: scope to eight operators; best-effort return on malformed input.
- [x] `Router::source_stats()` public API + `Model::model_type()` discriminant (planned 2026-04-27)
  - **Goal:** (R-1) Public delegators on `Router` for `QueryLog` stats: `source_stats(id)`, `ranked_sources_from_log()`, `best_source_from_log()`, `query_log_len()`. (R-2) `Model` trait gains `fn model_type(&self) -> &'static str` for polymorphic deserialization dispatch instead of try-then-fallback.
  - **Design:** R-1: Add four thin delegators to `Router<C>` in `src/core/router.rs` that call `self.query_log.*`. Re-export `SourceLogStats` from `src/lib.rs`. R-2: Add `model_type(&self) -> &'static str` to `Model` trait; implement `"naive_bayes"`, `"neural"`, `"ensemble"` on respective structs. Refactor `Router::load_model_from_bytes` to inspect the `ModelState.config.model_type` discriminant directly, keeping the try-fallback as a safety net for old blobs.
  - **Files:** MODIFIED `src/core/router.rs`, `src/ml/model.rs`, `src/ml/naive_bayes.rs`, `src/ml/neural.rs` (or `activation.rs`/split file after Block N), `src/ml/ensemble.rs`, `src/lib.rs`; NEW `tests/router_stats_test.rs`
  - **Tests:** 4 tests: `test_router_source_stats_after_routing`, `test_router_ranked_sources_from_log`, `test_model_type_naive_bayes`, `test_load_model_dispatches_by_type`.
  - **Risk:** Low. Both halves are pure additive; fallback path preserved.
- [x] Real RNG for RL (planned 2026-04-28)
  - **Goal:** Epsilon-greedy actually explores randomly; Thompson sampling actually samples; UCB no_std uses real log and sqrt. The bandit becomes a bandit.
  - **Design:** (T-1) Add `PolicyRng` struct to `RoutingPolicy` (field `#[serde(skip, default = "default_rng_state")]`). Under `std`: `SmallRng` seeded from `SystemTime`; under `no_std`: inline wyrand seeded by `0xdead_beef ^ total_visits`. Add `rand = { version = "0.9", default-features = false, features = ["small_rng"] }` to Cargo.toml, gated on `rl` feature. Add `RoutingPolicy::with_seed(u64)` for deterministic tests. (T-2) Replace `select_epsilon_greedy` (policy.rs:128–139): `r: f32 = rng.next_f32()`, compare to epsilon; if explore, pick random index via `rng.next_range(len)`. Changes `&self` → `&mut self` — cascade through all callers; if interior mutability needed, wrap rng_state in `Cell<PolicyRng>`. (T-3) Real Thompson via Beta(α,β)=X/(X+Y), X~Gamma(α), Y~Gamma(β); track per-source `visits_success` and `visits_failure`; use `rand_distr = "0.5"` on std, Marsaglia–Tsang Gamma on no_std. (T-4) Real UCB no_std (policy.rs:167–171): replace `exploration_constant * 0.5` with `libm::sqrtf(libm::logf(total) / n)` — libm already a dep.
  - **Files:** MODIFIED `Cargo.toml`, `src/rl/policy.rs`, `src/rl/mod.rs`; NEW `tests/rl_randomness_test.rs`
  - **Tests:** 7 tests: uniform exploration at ε=1.0 (χ² with ≥2200/4 each), pure exploitation at ε=0.0, Thompson distribution ≥90% favoring high-success source, UCB no_std real-log correctness, seeded reproducibility, serialize-skips-rng-state (selections differ post-load), compile-time mut-self cascade.
  - **Risk:** Medium-High. `&self` → `&mut self` cascade. Mitigation: wrap rng_state in Cell if needed; all tests gated to prevent flakiness.
- [x] Adam optimizer state survives save/load (planned 2026-04-28)
  - **Goal:** `train(batch_1) → save → load → train(batch_2)` produces same weights as `train(batch_1 ∪ batch_2)` within floating-point ε. Closes round-6 follow-up "save → load → learn → save → load → route".
  - **Design:** Remove `#[serde(skip)]` from these NeuralNetwork fields (neural.rs:37+): `optimizer_state`, `lr_schedule`, `early_stopping`, `early_stopping_state`, `dropout_config`, `dropout_state`; replace each with `#[serde(default)]` for v2 backward compat. Verify `OptimizerType`, `AdamConfig`, `OptimizerState` (optimizer.rs), `LearningRateSchedule`, `EarlyStoppingConfig`, `EarlyStoppingState`, `DropoutConfig`, `DropoutState` (schedule.rs) all derive `Serialize, Deserialize` — add if missing, with `#[serde(default)]` on Optional fields. Fix `from_state` (neural.rs:~856–873) to restore all fields (currently discards them). Bump `ModelState.config.version` by 1; v2 blobs load with defaults (serde(default) fills missing fields).
  - **Files:** MODIFIED `src/ml/neural.rs`, `src/ml/optimizer.rs`, `src/ml/schedule.rs`, `src/ml/model.rs`; NEW `tests/continuous_improvement_test.rs`
  - **Tests:** 5 tests: Adam moments survive save/load (exact equality); LR schedule step count preserved; early-stopping patience counter preserved; continuous-improvement equivalence (A/B split within 1e-4 or configured tolerance); legacy v2 blob loads with optimizer_state=None.
  - **Risk:** Medium. Backward compat via serde(default). Test 4 (equivalence) may need a tolerance gate.
- [x] Per-source mean/variance update for Naive Bayes (planned 2026-04-28)
  - **Goal:** `update_statistics` (naive_bayes.rs:175) uses per-source sample counts, not global `sample_count`. Online learning curves are mathematically correct.
  - **Design:** Bug at line 185: `let n = self.sample_count as f32 + 1.0;` — global, wrong. Fix: (1) Add `per_source_counts: HashMap<String, u32>` to `NaiveBayesClassifier` with `#[serde(default)]`. (2) Increment per-source count in `update_statistics`. (3) Use `n = (*self.per_source_counts.get(source_id).unwrap_or(&0) as f32) + 1.0` for both means (line 192) and variances (line 205) updates. Global `sample_count` stays for prior decay. (4) Bonus: make decay configurable — add `prior_decay: f32` (default 0.99) to `NaiveBayesConfig`.
  - **Files:** MODIFIED `src/ml/naive_bayes.rs` only
  - **Tests:** 4 tests: per_source_count after updates (A=10,B=5); means converge independently (source-A→0.8, source-B→0.2, alternating 50 each, within 0.05); legacy v1 blob loads with per_source_counts={}; configurable decay halves convergence speed.
  - **Risk:** Low. Single file, local change. Test 2 proves the fix empirically.

### Phase 3 – Autonomy (COMPLETE)
- [x] Drone/IoT deployment support: pre-compiled WASM module distribution (planned 2026-04-27)
  - **Goal:** Production-ready edge WASM build pipeline. `cargo build --profile release-edge --target wasm32-unknown-unknown` produces a stripped, size-optimized `.wasm`. `scripts/build-wasm.sh` automates the build with env-var overrides and surfaces the artifact size.
  - **Design:** Add `[profile.release-edge]` (inherits release; lto=fat, panic=abort, opt-level=z, codegen-units=1, strip=true, incremental=false) to Cargo.toml. NEW `scripts/build-wasm.sh` (bash, set -euo pipefail): env-var defaults (OXIROUTER_FEATURES, OXIROUTER_TARGET, OXIROUTER_PROFILE), target-install check, cargo build invocation, size reporting, optional wasm-opt pass.
  - **Files:** MODIFIED Cargo.toml; NEW scripts/build-wasm.sh
  - **Tests:** bash -n scripts/build-wasm.sh; cargo check --profile release-edge --no-default-features --features alloc
- [x] P2P node integration: IPFS/libp2p source type
- [x] Oxi-Agent integration: expose routing as an agent action primitive (planned 2026-04-27)
  - **Goal:** Routing as self-describing agent action primitives (JSON schema in/out). Three actions: `oxirouter.route`, `oxirouter.learn`, `oxirouter.explain`. Stable `AgentAction` trait consumable by `oxi-agent` (when it ships) and any LLM agent runtime.
  - **Design:** NEW `src/agent.rs` (≤900 lines, behind `agent` feature): `AgentAction` trait + `AgentActionMeta` struct + `RouterAgent<C: ContextProvider>` host with `dispatch()` + `list_actions()`. Action input/output structs (Serialize/Deserialize). JSON-Schema constants for each action. Reason string mapping from routing internals.
  - **Files:** NEW src/agent.rs, tests/agent_test.rs; MODIFIED Cargo.toml (add agent=[]), src/lib.rs
  - **Tests:** 9 tests in tests/agent_test.rs covering route/learn/explain dispatch, error cases, schema validity, and output roundtrip.
- [x] Federated model sharing: exchange learned weights between edge nodes
- [x] Concurrent fan-out + per-source deadline in federation executor (planned 2026-04-27)
  - **Goal:** `ExecutionConfig.parallel = true` actually parallelizes source dispatch. Per-source and end-to-end deadlines enforced. Currently `execute_parallel` at `src/federation/executor.rs:185` is a serial for-loop.
  - **Design:** Rewrite `Executor::execute_parallel` using `std::thread::scope` (stable since Rust 1.63 — no extra deps). Each source dispatched to a scoped thread; per-source deadline via `mpsc::channel().recv_timeout(per_source_timeout)`. Overall budget via `Instant::now() + total_timeout`; non-finishing threads contribute `FederationResult::Timeout`. Result vector preserves input source order via index tags. Native-only (`http` feature implies std).
  - **Files:** MODIFIED `src/federation/executor.rs`; NEW `tests/parallel_exec_test.rs` (gated `feature = "http"`)
  - **Tests:** 4 tests using `127.0.0.1:0` listeners: (1) two sources run concurrently (wall time ≤80% of serial); (2) per-source deadline trips slow source; (3) overall timeout abandons stragglers; (4) source order preserved.
  - **Risk:** Medium. Timing tests use ≤80% bound (not exact 50%) to avoid flakiness on loaded CI.
- [x] Rich routing explanation + actionable `NoSources` error (planned 2026-04-27)
  - **Goal:** `oxirouter.explain` agent action returns per-feature scoring components (vocabulary / region / performance / circuit-breaker contributions). `OxiRouterError::NoSources` names the unmatched vocabulary/region.
  - **Design:** NEW `RoutingExplanation { source_id, total_score, components: Vec<ScoreComponent { name, weight, raw_value, contribution }> }` in `src/core/router.rs`. NEW `Router::explain(&self, &Query) -> Vec<RoutingExplanation>`. Existing `route()` refactored to share component computation internally — no API break, still returns `SourceRanking`. `OxiRouterError::NoSources` becomes struct variant: `NoSources { reason: String, missing_vocabularies: Vec<String> }`. `agent::execute_explain` calls `Router::explain`.
  - **Files:** MODIFIED `src/core/router.rs`, `src/core/error.rs`, `src/agent.rs`; NEW `tests/explain_test.rs`
  - **Tests:** 5 tests: components sum to `total_score` (within ε), vocab component matches manual calc, region component matches manual calc, `NoSources` names missing vocab, agent JSON contains components array.
  - **Risk:** Low–Medium. `NoSources` unit→struct is a SemVer break — acceptable at 0.1.x with no external users.
- [x] `Router::federated_query()` — wire Executor + Aggregator (planned 2026-04-27)
  - **Goal:** A single `Router::federated_query(query, strategy) -> Result<AggregatedResult>` call: routes → executes via `Executor` → aggregates via `Aggregator`. Fixes the gap where Aggregator (1371 lines) and Executor (832 lines) are never invoked from `Router`.
  - **Design:** Add two methods gated `#[cfg(feature = "http")]`: `federated_query(&self, query: &Query, strategy: AggregationStrategy) -> Result<AggregatedResult>` chains `route()` → `Executor::execute()` → `Aggregator::aggregate()`; `route_and_execute(&self, query: &Query) -> Result<Vec<QueryResult>>` returns raw per-source results. Re-export `AggregationStrategy`, `AggregatedResult`, `QueryResult` from `src/lib.rs`.
  - **Files:** MODIFIED `src/core/router.rs`, `src/lib.rs`; NEW `tests/federated_query_test.rs`
  - **Tests:** 5 tests gated `#[cfg(feature = "http")]` using `127.0.0.1:0` listeners: first-strategy, union-strategy, all-sources-fail, route-and-execute per-source, no-sources error.
  - **Risk:** Medium. Touches Executor (parallel) and Aggregator (multi-format merging). Mitigation: local listeners with fixed JSON.

### Phase 4 – Descriptors & Deployment (COMPLETE)

- [x] Edge-deployment README.md (planned 2026-04-27)
  - **Goal:** Create a top-level `README.md` covering: project overview, feature matrix, quick-start routing example, edge profile build via `scripts/build-wasm.sh`, WASM distribution, agent action (route/learn/explain) quickstart, and SPARQL integration snippet. ≤250 lines.
  - **Design:** Single markdown file. Feature table with rows: `default`, `http`, `sparql`, `void`, `wasm`, `agent`, `p2p`, `ml`, `rl`, `cache`, `geo`, `device`, `load`, `legal`, `full`. Code blocks use existing `DataSource::new().with_vocabulary().with_region()` API and `Router::route()` patterns visible in integration tests.
  - **Files:** NEW `README.md`
  - **Tests:** None (documentation only).
  - **Risk:** Low — documentation only, no code changes.

- [x] Pure-Rust VoID/Turtle source-capability descriptors (planned 2026-04-27)
  - **Goal:** Declarative source registration via an `oxirouter.ttl` file using VoID vocabulary. `Router::register_from_void_ttl(&mut self, ttl: &str) -> Result<()>` parses a Turtle document, maps VoID terms to `DataSource` fields, and calls `self.add_source()` for each `void:Dataset` found.
  - **Design:** NEW `src/core/turtle.rs` (~500 lines): pure-Rust Turtle subset scanner (6-state machine mirroring `sparql.rs` style); handles `@prefix`/`PREFIX`, IRI refs `<...>`, prefixed names `pfx:local`, `;`/`,`/`.` separators, blank-node property lists `[ … ]`, `#` comments, string literals `"..."`. Returns `(HashMap<String,String>, Vec<Triple>)` where `Triple` = (subject, predicate, object) with all names expanded to full IRIs. NEW `src/core/void.rs` (~200 lines): `pub fn parse_oxirouter_ttl(ttl: &str) -> Result<Vec<DataSource>>` consuming turtle output; maps `void:sparqlEndpoint` → `DataSource.endpoint`, `void:vocabulary` → `.vocabularies`, `dcterms:spatial` → `.regions`, custom `oxirouter:kind` → `.kind`. New feature flag `void = []` in `Cargo.toml`. `Router::register_from_void_ttl` behind `#[cfg(feature="void")]`.
  - **Files:** NEW `src/core/turtle.rs`; NEW `src/core/void.rs`; MODIFIED `src/core/mod.rs`; MODIFIED `src/core/router.rs`; MODIFIED `src/lib.rs`; MODIFIED `Cargo.toml`; NEW `tests/void_test.rs`
  - **Tests:** 8–10 in `tests/void_test.rs`: single dataset, multiple datasets, vocabulary expansion, region mapping, kind mapping, round-trip via `register_from_void_ttl`, invalid TTL error, blank-node property-list syntax.
  - **Risk:** Medium. Turtle blank-node property-list syntax is non-trivial; scope-limited to the VoID subset needed (no full-Turtle conformance required).

- [x] Full SPARQL AST integration + ML feature expansion (planned 2026-04-27)
  - **Goal:** `src/core/sparql_ast.rs` (pure-Rust SPARQL structural parser gated on `sparql` feature) extracts 10 new AST-derived features. `FeatureVector::from_query_and_context` with `sparql` feature enabled appends these 10 dims after context dims (positions 38–47), expanding FeatureVector from 38→48 dims and exactly matching `ModelConfig.feature_dim` default of 48. Old models fail with a clear feature-dim mismatch error instead of silently truncating.
  - **Design:** NEW `src/core/sparql_ast.rs` (~700 lines, `#![cfg(feature="sparql")]`): `SparqlAst`, `GraphPattern` enum (Bgp | Optional | Union | Filter | GroupBy | Subquery | Service), `TriplePattern`, `SparqlAstFeatures` (10 normalized f32 fields: `join_depth`, `optional_count`, `filter_count`, `union_branch_count`, `has_distinct`, `has_having`, `subquery_count`, `path_expr_count`, `literal_count`, `blank_node_count`); `pub(crate) fn parse_sparql_ast(&str) -> SparqlAst` (best-effort, never panics); `pub(crate) fn extract_ast_features(&SparqlAst) -> SparqlAstFeatures`. `Query.from_sparql` populates new `#[serde(default)] pub ast_features: Option<SparqlAstFeatures>` field. `FeatureVector::from_query_and_context` appends all 10 dims (real or 0.0) under `#[cfg(feature="sparql")]`. `src/ml/model.rs` graceful `Err` on feature_dim mismatch.
  - **Files:** NEW `src/core/sparql_ast.rs`; MODIFIED `src/core/query.rs`; MODIFIED `src/ml/feature.rs`; MODIFIED `src/ml/model.rs`; MODIFIED `src/core/mod.rs`; NEW `tests/sparql_ast_test.rs`
  - **Tests:** 8–10 in `tests/sparql_ast_test.rs`: OPTIONAL detected, FILTER counted, UNION counted, GROUP BY detected, subquery detected, feature vector length with sparql feature = 48, routing with AST features succeeds, model mismatch error on wrong dim.
  - **Risk:** Medium. FeatureVector length change must be consistent across all code paths; mitigation: always pad with 0.0 when AST unavailable so vector length is deterministic per feature combo.
- [x] Examples directory + minimal CLI binary (planned 2026-04-27)
  - **Goal:** A new user can `cargo run --example quickstart` and see routing work; can `cargo run --bin oxirouter-cli -- route --query "..."` for ad-hoc routing without writing Rust.
  - **Design:** NEW examples gated via `[[example]] required-features`: `examples/quickstart.rs` (no features), `examples/sparql_routing.rs` (`sparql`), `examples/void_descriptor.rs` (`void`), `examples/agent_actions.rs` (`agent,sparql`). NEW `src/bin/oxirouter-cli.rs` with hand-rolled arg parsing (no clap — Pure Rust + zero new deps). Subcommands: `route --query "..." [--max N]`, `explain --query "..."`, `void-import --file path.ttl --query "..."`. NEW feature flag `cli = ["std", "agent", "sparql", "void"]`.
  - **Files:** NEW `examples/quickstart.rs`, `examples/sparql_routing.rs`, `examples/void_descriptor.rs`, `examples/agent_actions.rs`, `src/bin/oxirouter-cli.rs`; MODIFIED `Cargo.toml`
  - **Tests:** Build tests — `cargo build --examples --features full,sparql,void,agent` and `cargo build --bin oxirouter-cli --features cli` must compile clean.
  - **Risk:** Low. Pure additive — new files only, no behavior changes to existing modules.
- [x] WASM bindings for round-5 APIs: `explain_query_js`, `save_state_js`, `load_state_js`, `set_circuit_breaker_config_js` (planned 2026-04-27)
  - **Goal:** JS users can call all four methods from the WASM build. These APIs exist in Rust since round 5 but were not exposed via wasm-bindgen.
  - **Design:** Add four `#[wasm_bindgen]` methods to `src/wasm/bindings.rs` on the existing `OxiRouter` struct: `explain_query_js(&self, sparql: &str) -> Result<String, JsValue>` (calls `self.inner.explain(&q)`, serializes as JSON string); `save_state_js(&self) -> Result<Vec<u8>, JsValue>`; `load_state_js(&mut self, bytes: &[u8]) -> Result<(), JsValue>`; `set_circuit_breaker_config_js(&mut self, failure_threshold: u32, cooldown_ms: u32) -> Result<(), JsValue>`. Add `Router::set_circuit_breaker_config` setter if missing. Return `Vec<u8>` is marshalled as `Uint8Array` by wasm-bindgen automatically.
  - **Files:** MODIFIED `src/wasm/bindings.rs`; possibly MODIFIED `src/core/router.rs`
  - **Tests:** WASM compile check: `cargo check --target wasm32-unknown-unknown --no-default-features --features alloc,wasm,ml,rl,cache,sparql`. Plus a native unit test in bindings.rs that round-trips save_state → load_state.
  - **Risk:** Low. Pure additive; mirrors existing patterns in bindings.rs.
- [x] CLI completeness: `explain` uses `router.explain()`, `--json` flag, `state save/load` subcommands (planned 2026-04-27)
  - **Goal:** `oxirouter-cli explain` calls `router.explain()` (not `route()`); `--json` flag emits parseable output; `state save --file <path>` and `state load --file <path>` subcommands provide model persistence from the shell. Bonus: `--query -` reads SPARQL from stdin.
  - **Design:** Modify `src/bin/oxirouter-cli.rs`: (1) `explain` calls `router.explain(&query)?` and prints component breakdown; (2) top-level `--json` flag serializes all output as JSON via `serde_json::to_string_pretty`; (3) `state save/load` subcommands call `router.save_state()`/`router.load_state(&bytes)` with `std::fs::write`/`std::fs::read`. Output enum: `CliOutput { Route, Explain, StateSaved, StateLoaded }` with `#[serde(tag = "kind")]`.
  - **Files:** MODIFIED `src/bin/oxirouter-cli.rs`; NEW `tests/cli_smoke_test.rs`
  - **Tests:** Build: `cargo build --bin oxirouter-cli --features cli`. Smoke test invokes binary with `--help` and validates subcommand names in output.
  - **Risk:** Low. Pure additive plus one method-call change on the explain path.
- [x] Tracing + metrics observability (planned 2026-04-28)
  - **Goal:** Operators run `RUST_LOG=oxirouter=debug` and see structured spans for routing/learning/federation. Prometheus-compatible counters/histograms on the hot path. New `observability` feature gate keeps cost off default profile.
  - **Design:** Add `tracing = { version = "0.1", default-features = false, optional = true }` and `metrics = { version = "0.24", optional = true }` to Cargo.toml. New feature `observability = ["dep:tracing", "dep:metrics", "std"]`. Add `tracing-subscriber` as dev-dependency. Decorate under `#[cfg(feature = "observability")]` with `#[tracing::instrument]`: `Router::route` (fields: predicates_count, sources_len, result_confidence), `Router::route_and_log`, `Router::learn_from_outcome` (source_id, success, latency_ms), `Router::federated_query` (strategy, sources_len), `Router::explain`, `Executor::execute_parallel`, `NeuralNetwork::predict`, `NaiveBayesClassifier::predict`, `EnsembleClassifier::predict`. Inside scoring loops: `tracing::debug!` events gated on `tracing::enabled!(Level::DEBUG)`. Metrics: `oxirouter.route.total` (counter), `oxirouter.route.confidence` (histogram), `oxirouter.federation.execute.duration_ms` (histogram), `oxirouter.federation.execute.errors` (counter), `oxirouter.ml.predict.duration_us{model=...}` (histogram), `oxirouter.rl.explore.total` / `exploit.total` (counters), `oxirouter.circuit_breaker.tripped{source=...}` (counter). metrics recorder pluggable; we install none — no-op without recorder.
  - **Files:** MODIFIED `Cargo.toml`, `src/core/router.rs`, `src/federation/executor.rs`, `src/ml/neural.rs`, `src/ml/naive_bayes.rs`, `src/ml/ensemble.rs`, `src/rl/policy.rs`, `src/lib.rs`; NEW `tests/observability_test.rs`
  - **Tests:** 4 tests gated `observability`: route emits tracing span (TestWriter capture); route increments counter (custom 30-line test recorder); federated_query path histogram has ≥1 observation; default features compile without tracing/metrics symbols.
  - **Risk:** Low–Medium. Pure additive. cfg(feature) compile-time gate ensures zero cost on default builds.
- [x] RouterConfig serialization + Timeout context + RouterState v2 (planned 2026-04-28)
  - **Goal:** `RouterConfig` deserializable from JSON file; `OxiRouterError::Timeout` carries source_id/operation/elapsed_ms/deadline_ms; `RouterState` v1→v2 includes active config with v1 migration path.
  - **Design:** (X-1) Derive `Serialize, Deserialize` on `RouterConfig` (router.rs:86) and all transitive types (`CircuitBreakerConfig`, `RoutingStrategy`, etc.). Mark `now_ms: Option<fn()->u64>` with `#[serde(skip, default="default_now_ms")]`. Add `Router::with_config_file<P: AsRef<Path>>(path: P) -> Result<Self>` gated `#[cfg(feature="std")]`: reads file, `serde_json::from_slice` → `RouterConfig` → `Router::with_config`. Re-export `RouterConfig` from `src/lib.rs`. (X-2) Convert `OxiRouterError::Timeout` (error.rs:36) from unit variant to: `Timeout { source_id: String, operation: String, elapsed_ms: u64, deadline_ms: u64 }`. Update Display to print all four fields. Update all call sites (primarily executor.rs parallel-fan-out) to populate the struct. (X-3) Bump `STATE_VERSION` from 1 to 2 (state.rs:20). Split `RouterState` into a versioned wrapper + a `RouterStateBody` that carries `config: Option<RouterConfig>` (new). `from_bytes`: if ver==1, decode v1 JSON (no config field) and set config=None; if ver==2, decode full layout. `to_bytes` always writes v2. `Router::load_state` applies loaded config if Some, else keeps existing config.
  - **Files:** MODIFIED `src/core/router.rs`, `src/core/error.rs`, `src/core/state.rs`, `src/federation/executor.rs`, `src/lib.rs`; NEW `tests/config_persistence_test.rs`
  - **Tests:** 6 tests: RouterConfig JSON roundtrip; with_config_file from tempfile; Timeout error carries all four fields; Timeout Display contains all fields; v1 state loads with config=None; v2 roundtrip with non-default config.
  - **Risk:** Medium. Timeout SemVer break + state wire format version bump. Mitigation: at 0.1.x acceptable; v1 migration path explicit.


## Round 8 – From Ranker to Router (COMPLETE, planned 2026-04-28)

### Phase 0 – File hygiene (prerequisite) (COMPLETE)
- [x] splitrs prerequisite: split `router.rs` and `sparql_ast.rs` (planned 2026-04-28)
  - **Goal:** `src/core/router.rs` (1905 lines) and `src/core/sparql_ast.rs` (1528 lines) are split via `splitrs` into directory modules. No file in the post-split tree exceeds 800 lines. All public re-exports preserved. No-op for external semantics.
  - **Design:** Run `splitrs src/core/router.rs --max-lines 800 --output src/core/router/` and `splitrs src/core/sparql_ast.rs --max-lines 800 --output src/core/sparql_ast/`. Both become directory modules; `mod.rs` re-exports the public surface. Verify Round-7 `#[cfg_attr(feature="observability", tracing::instrument(...))]` macros survive split. Wave 1 and Wave 2 blocks reference symbols, not paths.
  - **Files:** `src/core/router.rs` → `src/core/router/`*.rs; `src/core/sparql_ast.rs` → `src/core/sparql_ast/`*.rs
  - **Tests:** No new tests — full 431-test suite is the regression gate.
  - **Risk:** Medium — splitrs may mis-handle `#[cfg_attr]` stacking. Run `cargo nextest run --features full,http,p2p,agent,sparql,void` + clippy before declaring done.

### Phase 1 – Per-triple terms (Wave 1) (COMPLETE)
- [x] Per-triple structured SPARQL term parser (planned 2026-04-28)
  - **Goal:** Every `Query` parsed from SPARQL carries a `Vec<StructuredTriple>` alongside existing `Vec<TriplePattern>`. Each `StructuredTriple` holds actual `Term` values (resolved IRIs / variable names / literals), not just type tags. API foundation for BGP decomposition (Block Z).
  - **Design:** Locked Y→Z contract — `pub struct StructuredTriple { pub subject: Term, pub predicate: Term, pub object: Term }` and `pub enum Term { Variable(String), Iri(String), PrefixedName(String,String), Literal(String), BlankNode(String) }` plus `impl Term { pub fn resolve(&self, prefix_map: &HashMap<String,String>) -> Self }`. Add `pub structured_triples: Vec<StructuredTriple>` field to `Query` (`#[serde(default)]` for backward compat). Add `pub prefixes: HashMap<String,String>` field (from PREFIX declarations). Extend `from_sparql` to populate both vectors in same parse pass. For `from_predicates_and_keywords`, `structured_triples` stays empty (Z fallback handles this). Property-path predicates each produce a `StructuredTriple` with the leaf IRI as predicate.
  - **Files:** MODIFIED `src/core/query.rs` (or post-AF split); NEW `src/core/term.rs` (if extracted); MODIFIED `src/core/mod.rs`, `src/lib.rs`; NEW `tests/structured_triple_test.rs`
  - **Tests:** 7 tests: simple BGP parse; prefix resolution; multiple triples; Variable term; Literal term; property-path leaf extraction; serialize roundtrip.
  - **Risk:** Medium — touches parser exercised by every SPARQL test. `structured_triples` is purely additive with `#[serde(default)]`.

### Phase 2 – Production (Wave 1) (COMPLETE)
- [x] WASM bindings expansion + Reflect::set silent-drop fixes (planned 2026-04-28)
  - **Goal:** Round-6/7 APIs reachable from WASM. `Reflect::set` errors propagate as `JsValue` errors instead of being silently swallowed.
  - **Design:** Add 3 new WASM functions on `OxiRouter`: `federated_query_js(&self, query_str, top_n, strategy) -> Result<JsValue,JsValue>` (strategy: "first"|"union"|"intersect"|"concat"|"largest"|"fastest"), `route_and_execute_js(&self, query_str, top_n)`, `register_from_void_ttl_js(&mut self, ttl)`. On WASM, executor mocks federation (no real HTTP — documented in doc-comment). Fix all ~12 `let _ = Reflect::set(...)` in `query_log_summary_js` (lines 263–280) and `source_to_js` (lines 92–94): replace with `Reflect::set(...).map_err(|e| JsValue::from_str(&format!("Reflect::set failed: {:?}", e)))?`. Update return types and callers accordingly.
  - **Files:** MODIFIED `src/wasm/bindings.rs`; NEW `tests/wasm_bindings_test.rs` (gated `#[cfg(target_arch = "wasm32")]`)
  - **Tests:** 4 wasm_bindgen_test gated tests: federated_query_js round-trip; route_and_execute_js round-trip; register_from_void_ttl_js; error propagation from Reflect::set.
  - **Risk:** Low–Medium. Return type changes cascade to callers. WASM tests compile-only fallback if wasm-pack not available.
- [x] Proptest infrastructure for parsers and wire formats (planned 2026-04-28)
  - **Goal:** Property-based tests sweep adversarial inputs across SPARQL parser, Turtle parser, property-path parser, and RouterState v2 wire format. Crashes on malformed input caught; roundtrip invariants verified.
  - **Design:** Add `proptest = "1"` to `[dev-dependencies]` in Cargo.toml. New `tests/proptest_parsers.rs` with 4 modules: `sparql_parser_props` (5 properties: no-panic on any UTF-8, predicate-set superset, whitespace invariance, prefix-order invariance, query-type classification); `turtle_parser_props` (3: empty/comment ok, random single-triple no panic, roundtrip); `property_path_parser_props` (3: no-panic, `<http://e.org/p>` → Iri, deep nesting safe); `router_state_wire_props` (3: v2 roundtrip, v1 load succeeds, truncation → Err not panic). All configs use `proptest::test_runner::Config { cases: 256, .. Default::default() }`.
  - **Files:** MODIFIED `Cargo.toml`; NEW `tests/proptest_parsers.rs`
  - **Tests:** 14 properties (5+3+3+3).
  - **Risk:** Low. Proptest is Pure Rust and deterministic.
- [x] Streaming HTTP response with bounded buffer (planned 2026-04-28)
  - **Goal:** `Executor::execute_http_request` reads chunk-by-chunk; aborts if response exceeds `config.max_response_bytes` (default 64 MiB) with a clear `OxiRouterError::ResponseTooLarge`.
  - **Design:** Add `max_response_bytes: u64` to `RouterConfig` (default `64 * 1024 * 1024`, `#[serde(default = "default_max_response_bytes")]`). New `OxiRouterError::ResponseTooLarge { source_id: String, observed_bytes: u64, limit_bytes: u64 }` with Display. Replace `response.into_body().read_to_vec()` in executor.rs:411–470 with chunked reader: 8192-byte tmp buf, check `buf.len() + n > limit` before extending — abort early if exceeded. Connection dropped via RAII.
  - **Files:** MODIFIED `src/federation/executor.rs`, `src/core/router.rs` (post-AF split, config field), `src/core/error.rs`, `src/lib.rs`; NEW `tests/streaming_response_test.rs`
  - **Tests:** 4 tests (gated `#[cfg(all(feature="http",feature="std",not(target_arch="wasm32")))]`): small response succeeds; oversized aborts early; exact-limit edge case; Display contains all three fields.
  - **Risk:** Medium — HTTP test infrastructure. Use hand-rolled TcpListener mock server (~30 lines).

### Phase 3 – Federation (Wave 2) (COMPLETE)
- [x] True federated query planner: BGP decomposition + UnionAll (planned 2026-04-28)
  - **Goal:** `Router::federated_query` decomposes BGP into per-source sub-queries based on vocabulary coverage, dispatches them, and unions results at row level. No cross-source joins (deferred to Round 9). Depends on Block Y (StructuredTriple API).
  - **Design:** NEW `src/federation/planner.rs` with `pub struct FederatedPlan { pub sub_plans: Vec<SubPlan>, pub fallback_used: bool }`, `pub struct SubPlan { pub source_id: String, pub triples: Vec<StructuredTriple>, pub sub_query: Query, pub confidence: f32 }`, `pub trait FederatedPlanner { fn plan(&self, q: &Query, sources: &[DataSource]) -> Result<FederatedPlan> }`, `pub struct DefaultPlanner { pub min_triple_confidence: f32 }`. Algorithm: (1) empty structured_triples → fallback (same query to all sources, `fallback_used=true`); (2) per-triple: resolve predicate IRI via prefixes, score per source (1.0 if namespace in vocabularies, 0.0 else; Variable predicate → score by success_rate), pick best, error if < min_triple_confidence; (3) group by source; (4) synthesize sub_query.raw as `SELECT * WHERE { <s> <p> <o> . ... }` using Term Display (variables as `?v`, IRIs as `<...>`, prefixed names as `prefix:local`); (5) sort sub_plans by confidence descending. Rewrite `Router::federated_query` to use planner. Add `planner: Box<dyn FederatedPlanner>` to Router. NOT in scope: cross-source joins, SERVICE clause, PREFIX headers in sub_query, cost-based planning.
  - **Files:** NEW `src/federation/planner.rs`; MODIFIED `src/federation/mod.rs`, `src/core/router.rs` (post-AF split), `src/lib.rs`; NEW `tests/federated_planner_test.rs`
  - **Tests:** 8 tests: single-triple routes to best source; multi-triple decomposition; same-source grouping; unmatched triple → NoSources; empty structured_triples fallback; sub_query.raw valid SPARQL; variable-predicate routes by reliability; confidence sort order.
  - **Risk:** High — headline block, hot federation path, depends on Y API. Mitigation: fallback path lands first so existing tests pass; Z codes against locked Y contract.
- [x] Capability-aware scoring + total_results wired into scoring (planned 2026-04-28)
  - **Goal:** `compute_source_components` consults 4 unused SourceCapabilities fields (federation, aggregation, property_paths, subqueries) and `SourceStats.avg_results_per_query()`. Routing decisions reflect these signals.
  - **Design:** In `compute_source_components` (post-AF split), add components: `capability_federation_match` (weight 0.2, +if source.capabilities.federation matches query.has_service_clause(), -0.2 if mismatch); `capability_aggregation_match` (weight 0.2, query.has_aggregation() checks GROUP BY / COUNT( / SUM(, case-insensitive); `capability_property_paths_match` (weight 0.2, query uses property-path predicates); `capability_subqueries_match` (weight 0.2, query.has_subquery() checks nested SELECT). Add `result_density` component (weight 0.05, raw_value = (avg_results_per_query()/100.0).min(1.0)). Capability signals are soft preferences (negative contribution on mismatch), not hard gates. Add helper methods `has_service_clause(&self)`, `has_aggregation(&self)`, `has_subquery(&self)` on `Query`.
  - **Files:** MODIFIED `src/core/router.rs` (post-AF split, compute_source_components), `src/core/query.rs` (post-AF split, new helpers); NEW `tests/capability_scoring_test.rs`
  - **Tests:** 6 tests: SERVICE query increases federation-capable source score; GROUP BY routes to aggregation-capable; property-path routes to path-capable; nested SELECT routes to subquery-capable; avg_results_per_query produces result_density component; mismatch produces negative contribution.
  - **Risk:** Low–Medium. Pure additive; round-7 Block S SoT invariant catches any drift in test_route_and_explain_agree.

### Phase 4 – Test coverage (Wave 2) (COMPLETE)
- [x] Property-path E2E routing test (planned 2026-04-28)
  - **Goal:** Empirical proof that `?s foaf:knows+ ?o` routes foaf-vocab source above non-foaf source. Closes round-7 follow-up.
  - **Design:** NEW `tests/property_path_routing_test.rs`. Register foaf-source (vocabularies: foaf) and schema-source (vocabularies: schema.org). Parse query with `foaf:knows+`, call `router.route()`, assert foaf_idx < schema_idx in result. Four cases: simple `+`, chained `/`, inverse `^`, alternation `|`.
  - **Files:** NEW `tests/property_path_routing_test.rs`
  - **Tests:** 4 tests. If routing bug found, fix is in scope (likely in `extract_path_predicates` or `vocabularies()`).
  - **Risk:** Low — pure coverage; if test fails it surfaces a real bug.

## Proposed follow-ups (Round 9+)

- Cross-source joins (symmetric hash join, bind join) — full federation join planner (~1500+ LOC, `src/federation/join.rs`).
- `RankingCache` — cache routing decisions alongside existing QueryCache/ContextCache/SourceCache (~200 LOC).
- Remaining 5 capability fields: `bind`, `values`, `max_results`, `full_text_search`, `geospatial`.
- cargo-fuzz targets — long-running mutation fuzzing (CI side-path).
- Async/tokio HTTP path — `Executor` uses ureq/thread::scope; tokio path is Round-9.
- SERVICE clause synthesis — Block Z uses plain SELECT WHERE; Round 9 adds `SERVICE <ep>` wrapping.
- Cost-based planning — Block Z is greedy; Round 9 may add cardinality estimates.
- Streaming JSON parse — Block AE bounds bytes; Round 9 parses SPARQL JSON stream incrementally.
- WASM real federation — Block AB exposes API; executor on WASM still mocks without web_sys::fetch.
