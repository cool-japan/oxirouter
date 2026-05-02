#!/bin/bash
# Build script for OxiRouter WASM demo

set -e

echo "Building OxiRouter WASM module..."

# Navigate to project root
cd "$(dirname "$0")/../.."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build WASM with wasm-pack (web target for ES modules)
wasm-pack build --target web --out-dir examples/wasm-demo/pkg \
    --features "wasm,ml,rl"

echo ""
echo "Build complete! WASM files are in examples/wasm-demo/pkg/"
echo ""
WASM_SIZE=$(du -h examples/wasm-demo/pkg/oxirouter_bg.wasm | cut -f1)
echo "WASM module size: $WASM_SIZE"
echo ""
echo "To run the demo:"
echo "  cd examples/wasm-demo"
echo "  python3 -m http.server 8080"
echo "  # Then open http://localhost:8080"
echo ""
echo "Or with Node.js http-server:"
echo "  npx http-server examples/wasm-demo -p 8080"
