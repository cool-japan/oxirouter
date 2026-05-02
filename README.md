# OxiRouter

**Autonomous Semantic Federation Engine for the Edge** — Learned source selection
for SPARQL federated queries with context-awareness.

<!-- badges -->

[![docs.rs](https://docs.rs/oxirouter/badge.svg)](https://docs.rs/oxirouter)
[![crates.io](https://img.shields.io/crates/v/oxirouter.svg)](https://crates.io/crates/oxirouter)
[![License: Apache-2.0](https://img.shields.io/crates/l/oxirouter.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Status

| | |
|---|---|
| Version | 0.1.0 (initial release, 2026-05-03) |
| Status | Alpha — feature-complete for 0.1.x, API may evolve |
| Tests | 501 unit/integration + 6 doc tests (all passing) |
| Lines of code | 23,361 Rust (96 files) |
| Edition / MSRV | 2024 / 1.85 |
| License | Apache-2.0 |
| CLI | `oxirouter-cli` — `route`, `explain`, `void-import`, `state save/load` |

## Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `std` | yes | Standard library support (`thiserror`, `serde/std`) |
| `alloc` | yes | Heap allocation support for `no_std` targets |
| `http` | no | HTTP client for live federation execution via `ureq` |
| `wasm` | no | WebAssembly bindings (`wasm-bindgen`, `js-sys`) |
| `edge` | no | Edge device compile-time gates |
| `ml` | no | Neural-network inference engine (no-std compatible via `libm`) |
| `dropout` | no | Dropout regularization layer (requires `ml`) |
| `rl` | no | Reinforcement learning policy (UCB, Q-learning) |
| `cache` | no | In-memory query result cache with TTL |
| `p2p` | no | P2P / IPFS source kind — type-level scoring, no network deps |
| `agent` | no | Self-describing action primitives for LLM agent runtimes |
| `sparql` | no | Pure-Rust SPARQL prefix expansion and projection extraction |
| `void` | no | Declarative source registration via VoID/Turtle (`Router::register_from_void_ttl`) |
| `geo` | no | Physical-brain context via `oxigdal-core` / `oxigdal-proj` |
| `device` | no | Body-brain context via `mielin-hal` |
| `load` | no | Situation-brain context via `celers-core` |
| `legal` | no | Social-brain context via `legalis-core` |
| `ecosystem` | no | All four brains: `geo` + `device` + `load` + `legal` |
| `full` | no | `ml` + `rl` + `cache` + `ecosystem` |
| `observability` | no | `tracing` spans + `metrics` counters/histograms (zero-cost when disabled) |
| `cli` | no | Activates everything required by the `oxirouter-cli` binary (`std` + `agent` + `sparql` + `void`) |

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
oxirouter = "0.1"
```

```rust
use oxirouter::core::source::SourceCapabilities;
use oxirouter::prelude::*;

fn main() -> Result<(), OxiRouterError> {
    let mut router = Router::new();

    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_capabilities(SourceCapabilities::full())
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );

    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_capabilities(SourceCapabilities::full())
            .with_vocabulary("http://www.wikidata.org/")
            .with_region("EU"),
    );

    let query = Query::parse(
        r"
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?name WHERE {
            ?person a dbo:Person .
            ?person dbo:name ?name .
        }
        LIMIT 100
        ",
    )?;

    let ranking = router.route(&query)?;

    if let Some(best) = ranking.best() {
        println!("Best source: {} (confidence {:.2})", best.source_id, best.confidence);
    }
    Ok(())
}
```

### Learning from outcomes

```rust
let ranking = router.route_and_log(&query)?;
let query_id = query.predicate_hash();

// After executing the query, feed back the observed outcome:
router.learn_from_outcome(query_id, "dbpedia", true, 150, 100)?;
```

## SPARQL Integration

Enable with `features = ["sparql"]`.

`Query::from_sparql` performs accurate PREFIX-declaration parsing and projection
extraction. `Router::route_sparql` is the single-call convenience wrapper:

```rust
#[cfg(feature = "sparql")]
fn route_sparql_example(router: &oxirouter::core::router::Router)
    -> oxirouter::core::error::Result<()>
{
    let ranking = router.route_sparql(
        r"
        PREFIX schema: <http://schema.org/>
        SELECT ?name WHERE { ?s schema:name ?name }
        ",
    )?;

    for sel in ranking.sources.iter() {
        println!("{}: {:.2}", sel.source_id, sel.confidence);
    }
    Ok(())
}
```

## Agent Actions

Enable with `features = ["agent"]`. OxiRouter exposes three self-describing
actions compatible with the OpenAI tool-call and Anthropic tool-use protocols.

| Action name | Description |
|-------------|-------------|
| `oxirouter.route` | Rank sources for a SPARQL query |
| `oxirouter.learn` | Record an observed query outcome |
| `oxirouter.explain` | Return a human-readable routing explanation |

### `oxirouter.route` input schema (excerpt)

```json
{
  "type": "object",
  "properties": {
    "query":            { "type": "string",  "description": "SPARQL query string to route" },
    "max_sources":      { "type": "integer", "minimum": 1 },
    "domain":           { "type": "string",  "description": "Domain hint, e.g. 'biology'" },
    "expected_results": { "type": "integer", "minimum": 0 }
  },
  "required": ["query"]
}
```

### `oxirouter.learn` input fields

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | `u64` | Hash from a prior `route` call (`Query::predicate_hash()`) |
| `source_id` | `string` | Source that was used |
| `success` | `bool` | Whether the query executed successfully |
| `latency_ms` | `u32` | Observed latency in milliseconds |
| `result_count` | `u32` | Number of results returned |

### `oxirouter.explain` output fields

| Field | Type | Description |
|-------|------|-------------|
| `explanation` | `string` | Human-readable routing rationale |
| `ranked_sources` | `array` | Ranked source objects (`id`, `endpoint`, `confidence`, `reason`) |

## Edge / WASM Deployment

OxiRouter targets `wasm32-unknown-unknown` with the `release-edge` profile
(fat LTO, `opt-level = "z"`, `panic = "abort"`, stripped symbols).

```bash
# Install the WASM target once
rustup target add wasm32-unknown-unknown

# Standard edge build (features: alloc,wasm,ml,rl,cache)
./scripts/build-wasm.sh

# Minimal build without ML (smaller binary)
OXIROUTER_FEATURES=wasm ./scripts/build-wasm.sh

# Use the lighter release-wasm profile
OXIROUTER_PROFILE=release-wasm ./scripts/build-wasm.sh
```

The script validates the target, runs `cargo build --no-default-features
--features "alloc,${OXIROUTER_FEATURES}"`, reports the artifact size, and
runs `wasm-opt -Oz --strip-debug` automatically when `binaryen` is available.

Output artifact: `target/wasm32-unknown-unknown/release-edge/oxirouter.wasm`

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
