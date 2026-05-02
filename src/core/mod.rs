//! Core module - Platform-agnostic core logic for OxiRouter
//!
//! This module contains the fundamental types and logic for:
//! - Data source definitions and management
//! - SPARQL query parsing and feature extraction
//! - Main routing engine

pub mod error;
pub mod query;
pub mod query_log;
pub mod router;
pub mod source;
pub mod term;

#[cfg(feature = "alloc")]
pub(crate) mod state;

#[cfg(feature = "sparql")]
pub(crate) mod sparql;

#[cfg(feature = "sparql")]
pub(crate) mod sparql_ast;

#[cfg(feature = "void")]
pub(crate) mod turtle;

#[cfg(feature = "void")]
pub(crate) mod void;
