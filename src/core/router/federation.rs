//! Federation, SPARQL, and VoID extension methods for Router.

#[cfg(any(feature = "http", feature = "sparql", feature = "void"))]
use crate::context::ContextProvider;

#[cfg(any(feature = "http", feature = "sparql"))]
use crate::core::error::Result;

#[cfg(any(feature = "http", feature = "sparql"))]
use crate::core::query::Query;

#[cfg(any(feature = "http", feature = "sparql"))]
use crate::core::source::SourceRanking;

#[cfg(any(feature = "http", feature = "sparql", feature = "void"))]
use super::Router;

#[cfg(feature = "http")]
impl<C: ContextProvider> Router<C> {
    /// One-shot federated query: decompose BGP into per-source sub-queries via the
    /// [`FederatedPlanner`][crate::federation::planner::FederatedPlanner], dispatch
    /// each sub-query, and aggregate results.
    ///
    /// When `query.structured_triples` is empty the planner falls back to sending
    /// the full query to the top-N sources ordered by success rate.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails, no sources are available, execution fails
    /// for all sub-plans, or aggregation fails.
    #[cfg_attr(
        feature = "observability",
        tracing::instrument(
            skip(self, query),
            fields(
                strategy = tracing::field::debug(strategy),
                sources_count = tracing::field::Empty
            )
        )
    )]
    pub fn federated_query(
        &self,
        query: &Query,
        strategy: crate::federation::AggregationStrategy,
    ) -> crate::core::error::Result<crate::federation::AggregatedResult> {
        use crate::core::error::OxiRouterError;
        use crate::federation::{Aggregator, Executor};

        // Collect sources into a slice for the planner.
        let sources_vec: Vec<crate::core::source::DataSource> =
            self.sources.values().cloned().collect();

        let plan = self.planner.plan(query, &sources_vec)?;

        #[cfg(feature = "observability")]
        {
            tracing::Span::current().record("sources_count", plan.sub_plans.len());
        }

        let executor = Executor::new();
        let mut results = Vec::with_capacity(plan.sub_plans.len());

        for sub_plan in &plan.sub_plans {
            let source = self
                .sources
                .get(&sub_plan.source_id)
                .ok_or_else(|| OxiRouterError::SourceNotFound(sub_plan.source_id.clone()))?;

            let result = executor.execute_single(&sub_plan.sub_query, source);
            results.push(result);
        }

        let aggregation_result = Aggregator::new()
            .with_strategy(strategy)
            .aggregate(&results)
            .inspect_err(|_e| {
                #[cfg(feature = "observability")]
                {
                    metrics::counter!("oxirouter.federation.execute.errors").increment(1);
                }
            })?;

        Ok(aggregation_result)
    }

    /// Routes and executes, returning raw per-source results without aggregation.
    ///
    /// # Errors
    ///
    /// Returns an error if no sources are available, routing fails, or execution fails.
    pub fn route_and_execute(
        &self,
        query: &Query,
    ) -> crate::core::error::Result<Vec<crate::federation::QueryResult>> {
        use crate::federation::Executor;
        let ranking = self.route(query)?;
        let sources: Vec<&crate::core::source::DataSource> = ranking
            .sources
            .iter()
            .filter_map(|sel| self.sources.get(&sel.source_id))
            .collect();
        Executor::new().execute(query, &sources, &ranking)
    }
}

#[cfg(feature = "sparql")]
impl<C: ContextProvider> Router<C> {
    /// Route a raw SPARQL string with real prefix expansion.
    ///
    /// Parses via [`Query::from_sparql`] (accurate PREFIX declarations, projection
    /// variables) then routes using the standard heuristic + ML pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if SPARQL parsing fails or no sources are configured.
    pub fn route_sparql(&self, sparql: &str) -> Result<SourceRanking> {
        let query = Query::from_sparql(sparql)?;
        self.route(&query)
    }

    /// Parse, route, and log a raw SPARQL string.
    ///
    /// Combines [`Query::from_sparql`] with [`Router::route_and_log`].
    ///
    /// # Errors
    ///
    /// Returns an error if SPARQL parsing fails or no sources are configured.
    pub fn route_sparql_and_log(&mut self, sparql: &str) -> Result<SourceRanking> {
        let query = Query::from_sparql(sparql)?;
        self.route_and_log(&query)
    }
}

#[cfg(feature = "void")]
impl<C: ContextProvider> Router<C> {
    /// Parse a VoID/Turtle source descriptor document and register all
    /// `void:Dataset` entries as sources in this router.
    ///
    /// # Errors
    ///
    /// Returns an error if the Turtle document cannot be parsed.
    pub fn register_from_void_ttl(&mut self, ttl: &str) -> crate::core::error::Result<()> {
        let sources = crate::core::void::parse_oxirouter_ttl(ttl)?;
        for source in sources {
            self.add_source(source);
        }
        Ok(())
    }
}
