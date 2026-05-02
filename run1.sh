#!/bin/bash
cd "$(dirname "$0")/examples/wasm-demo"
echo "Starting OxiRouter WASM demo at http://localhost:8080"
python3 -m http.server 8081
