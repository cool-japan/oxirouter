//! WebAssembly bindings for OxiRouter
//!
//! This module provides JavaScript-accessible interfaces for
//! running OxiRouter in browsers and edge runtimes.
//!
//! The `context_provider` sub-module is always available (it is pure Rust and
//! can be tested on any host target).  The `bindings` sub-module requires
//! the `wasm32` target architecture because it depends on `wasm-bindgen` and
//! `js-sys`, which are only linked when compiling for WebAssembly.

pub mod context_provider;

#[cfg(target_arch = "wasm32")]
mod bindings;

#[cfg(target_arch = "wasm32")]
pub use bindings::*;

pub use context_provider::{WasmContextProvider, WasmRouterContextAdapter};
