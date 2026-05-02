# OxiRouter WASM Demo

Interactive browser demo for OxiRouter - Autonomous Semantic Federation Engine.

## Quick Start

### 1. Build WASM Module

```bash
# From project root
chmod +x examples/wasm-demo/build.sh
./examples/wasm-demo/build.sh
```

Or manually:
```bash
wasm-pack build --target web --out-dir examples/wasm-demo/pkg \
    --features "wasm,ml,rl" -- --profile release-wasm
```

### 2. Serve the Demo

```bash
cd examples/wasm-demo
python3 -m http.server 8080
```

Or with Node.js:
```bash
npx http-server examples/wasm-demo -p 8080 --cors
```

### 3. Open Browser

Navigate to `http://localhost:8080`

## Features Demo

### Source Management
- Add SPARQL endpoints manually
- Load pre-configured demo sources (DBpedia, Wikidata, etc.)
- Remove sources
- Toggle source availability

### Query Routing
- Write SPARQL queries in the editor
- Route queries to get ranked source recommendations
- View confidence scores and estimated latency

### Query Analysis
- Parse and analyze SPARQL queries
- View query type (SELECT, CONSTRUCT, ASK, DESCRIBE)
- See complexity metrics
- Detect SPARQL 1.1 features (OPTIONAL, UNION, FILTER, etc.)

### Configuration
- Adjust max sources to return
- Set minimum confidence threshold
- Toggle ML-based routing
- Toggle context-aware routing

## API Reference

```typescript
// Create router
const router = new OxiRouter();

// Add sources
router.add_source("dbpedia", "https://dbpedia.org/sparql");
router.add_source_with_region("wikidata", "https://query.wikidata.org/sparql", "EU");
router.add_source_with_vocabulary("schema", "https://schema.example.com/sparql", "http://schema.org/");

// Configure
router.set_max_sources(5);
router.set_min_confidence(0.1);
router.set_use_ml(true);
router.set_use_context(true);

// Route query
const result = router.route_query(`
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?name WHERE { ?s dbo:name ?name }
`);
// Returns: { sources: [...], processingTimeUs, mlUsed, contextUsed }

// Analyze query
const analysis = router.analyze_query("SELECT ?s WHERE { ?s ?p ?o }");
// Returns: { queryType, complexity, hasOptional, hasUnion, ... }

// Update feedback after query execution
router.update_feedback("dbpedia", 150, true, 100);

// Get stats
const stats = router.get_source_stats("dbpedia");
// Returns: { totalQueries, successfulQueries, avgLatencyMs, successRate, ... }

// Mark unavailable/available
router.mark_unavailable("dbpedia");
router.mark_available("dbpedia");

// Version
OxiRouter.version();  // "0.1.0"
```

## Browser Compatibility

- Chrome/Edge 89+
- Firefox 89+
- Safari 15+

Requires WebAssembly and ES modules support.
