//! Federation execution module
//!
//! This module handles query dispatch to selected sources and
//! result aggregation.

pub mod aggregator;
mod executor;
pub mod planner;

pub use aggregator::{
    AggregatedResult, AggregationConfig, AggregationStrategy, Aggregator, SparqlBinding,
    SparqlJsonResult, SparqlValue,
};
pub use executor::{ExecutionConfig, Executor, QueryResult, ResultFormat};
pub use planner::{DefaultPlanner, FederatedPlan, FederatedPlanner, SubPlan};
