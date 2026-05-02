# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-03

### Added

- Initial release of OxiRouter — Autonomous Semantic Federation Engine for the Edge
- Learned source selection for SPARQL federated queries with context-awareness
- Pure-Rust SPARQL parser (`sparql` feature): prefix expansion, SELECT-variable projection
- VoID/Turtle source-capability descriptors (`void` feature): pure-Rust Turtle subset parser
- ML inference engine (`ml` feature): neural network, naive Bayes, ensemble, federated learning
- Reinforcement learning exploration (`rl` feature): ε-greedy policy with SmallRng
- Query log and source statistics tracking with routing score feedback
- Circuit-breaker per source (configurable failure threshold + cooldown)
- Cache layer for context and routing results (`cache` feature)
- HTTP federation executor (`http` feature) using `ureq` (sync, no async)
- WASM bindings (`wasm` feature) with `wasm-bindgen` and `js-sys`
- Agent action interface (`agent` feature) — LLM tool-call protocol compatible
- Ecosystem integrations: geo/oxigdal (`geo`), device/mielin (`device`), load/celers (`load`), legal/legalis (`legal`)
- P2P/IPFS source kind support (`p2p` feature)
- Observability spans and metrics (`observability` feature) via `tracing` + `metrics`
- CLI binary `oxirouter-cli` (`cli` feature): route, explain, void-import, state save/load
- `no_std` compatible core (with `alloc` feature)
- Wire-format state persistence: magic-prefixed `OXIR` binary format with serde_json body
- 501 unit and integration tests, all passing

[0.1.0]: https://github.com/cool-japan/oxirouter/releases/tag/v0.1.0
