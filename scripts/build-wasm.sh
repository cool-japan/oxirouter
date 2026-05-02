#!/usr/bin/env bash
# build-wasm.sh — OxiRouter edge WASM build helper
#
# Env vars (all optional, with defaults):
#   OXIROUTER_FEATURES   — cargo features to enable (default: wasm,ml,rl,cache)
#   OXIROUTER_TARGET     — rust target triple (default: wasm32-unknown-unknown)
#   OXIROUTER_PROFILE    — cargo build profile (default: release-edge)
#
# Usage:
#   ./scripts/build-wasm.sh               # default: edge wasm build
#   OXIROUTER_FEATURES=wasm ./scripts/build-wasm.sh
#   OXIROUTER_TARGET=wasm32-wasi ./scripts/build-wasm.sh
#   ./scripts/build-wasm.sh --help

set -euo pipefail

OXIROUTER_FEATURES="${OXIROUTER_FEATURES:-wasm,ml,rl,cache}"
OXIROUTER_TARGET="${OXIROUTER_TARGET:-wasm32-unknown-unknown}"
OXIROUTER_PROFILE="${OXIROUTER_PROFILE:-release-edge}"

show_help() {
    cat <<'HELP'
OxiRouter Edge WASM Builder

USAGE:
    ./scripts/build-wasm.sh [OPTIONS]

OPTIONS:
    --help    Show this help message

ENV VARS:
    OXIROUTER_FEATURES   Comma-separated cargo features (default: wasm,ml,rl,cache)
    OXIROUTER_TARGET     Rust target triple (default: wasm32-unknown-unknown)
    OXIROUTER_PROFILE    Cargo build profile (default: release-edge)

EXAMPLES:
    # Standard edge build (stripped, fat LTO)
    ./scripts/build-wasm.sh

    # Minimal WASM without ML (smaller binary)
    OXIROUTER_FEATURES=wasm ./scripts/build-wasm.sh

    # Use a different profile
    OXIROUTER_PROFILE=release-wasm ./scripts/build-wasm.sh
HELP
}

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "error: unknown argument: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# Check cargo is available
if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found on PATH. Install Rust from https://rustup.rs" >&2
    exit 1
fi

# Check that the target is installed
if ! rustup target list --installed 2>/dev/null | grep -q "^${OXIROUTER_TARGET}$"; then
    echo "error: target '${OXIROUTER_TARGET}' is not installed." >&2
    echo "Install it with: rustup target add ${OXIROUTER_TARGET}" >&2
    exit 1
fi

# Locate the workspace root (the directory containing Cargo.toml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> OxiRouter edge WASM build"
echo "    target:   ${OXIROUTER_TARGET}"
echo "    profile:  ${OXIROUTER_PROFILE}"
echo "    features: alloc,${OXIROUTER_FEATURES}"
echo ""

cd "$WORKSPACE_ROOT"

cargo build \
    --target "${OXIROUTER_TARGET}" \
    --profile "${OXIROUTER_PROFILE}" \
    --no-default-features \
    --features "alloc,${OXIROUTER_FEATURES}"

# Profile directory: cargo uses the profile name as-is for custom profiles.
# Built artifact is at target/<target>/<profile>/<crate_name>.wasm
WASM_PATH="target/${OXIROUTER_TARGET}/${OXIROUTER_PROFILE}/oxirouter.wasm"

if [ ! -f "$WASM_PATH" ]; then
    echo "error: expected WASM artifact not found at ${WASM_PATH}" >&2
    exit 1
fi

RAW_BYTES="$(wc -c < "$WASM_PATH" | tr -d ' ')"
RAW_KIB=$(( RAW_BYTES / 1024 ))
echo ""
echo "==> Build succeeded"
echo "    artifact: ${WASM_PATH}"
echo "    size:     ${RAW_BYTES} bytes (${RAW_KIB} KiB)"

# Optional wasm-opt pass (informational if missing)
if command -v wasm-opt >/dev/null 2>&1; then
    OPT_PATH="${WASM_PATH}.opt"
    echo ""
    echo "==> Running wasm-opt -Oz --strip-debug ..."
    wasm-opt -Oz --strip-debug -o "$OPT_PATH" "$WASM_PATH"
    OPT_BYTES="$(wc -c < "$OPT_PATH" | tr -d ' ')"
    OPT_KIB=$(( OPT_BYTES / 1024 ))
    SAVINGS=$(( RAW_BYTES - OPT_BYTES ))
    mv "$OPT_PATH" "$WASM_PATH"
    echo "    optimized size: ${OPT_BYTES} bytes (${OPT_KIB} KiB, saved ${SAVINGS} bytes)"
else
    echo ""
    echo "    note: wasm-opt not found on PATH; skipping optimization."
    echo "          Install binaryen (https://github.com/WebAssembly/binaryen) for smaller output."
fi

echo ""
echo "==> Done. Deploy ${WASM_PATH} to your edge runtime."
