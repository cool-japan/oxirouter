//! `oxirouter-cli` — command-line interface for OxiRouter.
//!
//! Requires the `cli` feature (which activates `sparql`, `void`, `agent`, and `std`).
//!
//! # Global flags
//!
//! - `--json`   Emit parseable JSON instead of human-readable text
//!
//! # Subcommands
//!
//! - `route   --query <SPARQL>`              — route and print ranked sources
//! - `explain --query <SPARQL>`              — explain routing decision (component scores)
//! - `void-import --file <path.ttl> --query <SPARQL>` — load TTL, then route query
//! - `state save --file <path>`              — serialize router state to file
//! - `state load --file <path>`              — restore router state from file

use std::fs;
use std::io::Read as _;

use oxirouter::core::source::SourceSelection;
use oxirouter::{DataSource, OxiRouterError, Router, RoutingExplanation};
use serde::Serialize;

// ─────────────────────────────────────────────────────────────────────────────
// Output enum — serialized when --json is active
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CliOutput<'a> {
    Route {
        sources: &'a [SourceSelection],
        processing_time_us: u64,
    },
    Explain {
        explanations: &'a [RoutingExplanation],
    },
    VoidImport {
        sources_loaded: usize,
        route_sources: &'a [SourceSelection],
    },
    StateSaved {
        bytes_written: usize,
        file: &'a str,
    },
    StateLoaded {
        sources_count: usize,
        file: &'a str,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), OxiRouterError> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();

    if raw_args.is_empty() {
        print_usage();
        std::process::exit(1);
    }

    // Strip global flags before subcommand dispatch
    let mut json_mode = false;
    let mut args: Vec<String> = Vec::with_capacity(raw_args.len());
    for arg in raw_args {
        if arg == "--json" {
            json_mode = true;
        } else {
            args.push(arg);
        }
    }

    if args.is_empty() {
        print_usage();
        std::process::exit(1);
    }

    let subcommand = args.remove(0);

    match subcommand.as_str() {
        "route" => cmd_route(&args, json_mode),
        "explain" => cmd_explain(&args, json_mode),
        "void-import" => cmd_void_import(&args, json_mode),
        "state" => cmd_state(&args, json_mode),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        "--version" | "-V" => {
            println!("oxirouter-cli {}", oxirouter::VERSION);
            Ok(())
        }
        other => {
            eprintln!("Unknown subcommand: {other}");
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Argument parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the value that immediately follows a named flag (e.g. `--query VALUE`).
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter.next().map(String::as_str);
        }
    }
    None
}

/// Resolve a query string: if `raw == "-"`, read from stdin; otherwise return it verbatim.
fn resolve_query(raw: &str) -> Result<String, OxiRouterError> {
    if raw == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| OxiRouterError::QueryParse(format!("stdin read error: {e}")))?;
        Ok(buf)
    } else {
        Ok(raw.to_owned())
    }
}

/// Emit `output` as pretty JSON, mapping the serde error into `OxiRouterError`.
fn emit_json<T: Serialize>(output: &T) -> Result<(), OxiRouterError> {
    let text = serde_json::to_string_pretty(output)
        .map_err(|e| OxiRouterError::ModelError(format!("JSON serialisation: {e}")))?;
    println!("{text}");
    Ok(())
}

/// Print usage to stderr.
fn print_usage() {
    eprintln!("oxirouter-cli {}", oxirouter::VERSION);
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  oxirouter-cli [--json] route        --query <SPARQL|->");
    eprintln!("  oxirouter-cli [--json] explain      --query <SPARQL|->");
    eprintln!("  oxirouter-cli [--json] void-import  --file <path.ttl> --query <SPARQL|->");
    eprintln!("  oxirouter-cli [--json] state save   --file <path>");
    eprintln!("  oxirouter-cli [--json] state load   --file <path>");
    eprintln!("  oxirouter-cli --version");
    eprintln!("  oxirouter-cli --help");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --json             Emit JSON instead of human-readable text");
    eprintln!("  --query <SPARQL>   SPARQL query string; use '-' to read from stdin");
    eprintln!("  --file  <PATH>     Path to a VoID/Turtle descriptor or state snapshot");
}

// ─────────────────────────────────────────────────────────────────────────────
// Build a default router with a few well-known endpoints
// ─────────────────────────────────────────────────────────────────────────────

fn default_router() -> Router {
    let mut router = Router::new();
    router.add_source(
        DataSource::new("dbpedia", "https://dbpedia.org/sparql")
            .with_vocabulary("http://dbpedia.org/ontology/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("wikidata", "https://query.wikidata.org/sparql")
            .with_vocabulary("http://www.wikidata.org/prop/direct/")
            .with_vocabulary("http://www.wikidata.org/entity/")
            .with_region("EU"),
    );
    router.add_source(
        DataSource::new("schema-org", "https://schema.org/sparql")
            .with_vocabulary("http://schema.org/")
            .with_region("US"),
    );
    router
}

// ─────────────────────────────────────────────────────────────────────────────
// Subcommand implementations
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_route(args: &[String], json_mode: bool) -> Result<(), OxiRouterError> {
    let raw_query = flag_value(args, "--query").ok_or_else(|| {
        eprintln!("Missing required flag: --query");
        eprintln!("Usage: oxirouter-cli route --query <SPARQL|->");
        OxiRouterError::QueryParse("--query flag is required".to_string())
    })?;

    let sparql = resolve_query(raw_query)?;
    let router = default_router();
    let query = oxirouter::Query::from_sparql(&sparql)?;
    let ranking = router.route(&query)?;

    if json_mode {
        let output = CliOutput::Route {
            sources: &ranking.sources,
            processing_time_us: ranking.processing_time_us,
        };
        return emit_json(&output);
    }

    if ranking.is_empty() {
        println!("No sources matched the query.");
        return Ok(());
    }

    println!("Ranked sources for query:");
    println!("  {sparql}");
    println!();
    for (i, sel) in ranking.sources.iter().enumerate() {
        println!(
            "  {}. {}  confidence={:.3}  latency_est={}ms  reason={:?}",
            i + 1,
            sel.source_id,
            sel.confidence,
            sel.estimated_latency_ms,
            sel.reason,
        );
    }
    println!("\nProcessing time: {}μs", ranking.processing_time_us);

    Ok(())
}

fn cmd_explain(args: &[String], json_mode: bool) -> Result<(), OxiRouterError> {
    let raw_query = flag_value(args, "--query").ok_or_else(|| {
        eprintln!("Missing required flag: --query");
        eprintln!("Usage: oxirouter-cli explain --query <SPARQL|->");
        OxiRouterError::QueryParse("--query flag is required".to_string())
    })?;

    let sparql = resolve_query(raw_query)?;
    let router = default_router();
    let query = oxirouter::Query::from_sparql(&sparql)?;
    let explanations = router.explain(&query)?;

    if json_mode {
        let output = CliOutput::Explain {
            explanations: &explanations,
        };
        return emit_json(&output);
    }

    println!("OxiRouter routing explanation");
    println!("Query: {}", sparql.lines().next().unwrap_or(&sparql));
    println!();

    if explanations.is_empty() {
        println!("  No sources to explain.");
        return Ok(());
    }

    for expl in &explanations {
        println!(
            "Source: {} (total score: {:.3})",
            expl.source_id, expl.total_score
        );
        for comp in &expl.components {
            println!(
                "  {}: raw={:.3} weight={:.3} contribution={:.3}",
                comp.name, comp.raw_value, comp.weight, comp.contribution,
            );
        }
        println!();
    }

    Ok(())
}

fn cmd_void_import(args: &[String], json_mode: bool) -> Result<(), OxiRouterError> {
    let file_path = flag_value(args, "--file").ok_or_else(|| {
        eprintln!("Missing required flag: --file");
        eprintln!("Usage: oxirouter-cli void-import --file <path.ttl> --query <SPARQL|->");
        OxiRouterError::QueryParse("--file flag is required".to_string())
    })?;

    let raw_query = flag_value(args, "--query").ok_or_else(|| {
        eprintln!("Missing required flag: --query");
        eprintln!("Usage: oxirouter-cli void-import --file <path.ttl> --query <SPARQL|->");
        OxiRouterError::QueryParse("--query flag is required".to_string())
    })?;

    // Read TTL file
    let ttl = fs::read_to_string(file_path).map_err(|e| {
        eprintln!("Failed to read file '{file_path}': {e}");
        OxiRouterError::QueryParse(format!("cannot read TTL file: {e}"))
    })?;

    // Import sources from VoID descriptor
    let mut router = Router::new();
    router.register_from_void_ttl(&ttl)?;

    let sources_loaded = router.source_count();

    if sources_loaded == 0 {
        println!("No void:Dataset with a void:sparqlEndpoint found in descriptor.");
        return Ok(());
    }

    let sparql = resolve_query(raw_query)?;
    let query = oxirouter::Query::from_sparql(&sparql)?;
    let ranking = router.route(&query)?;

    if json_mode {
        let output = CliOutput::VoidImport {
            sources_loaded,
            route_sources: &ranking.sources,
        };
        return emit_json(&output);
    }

    println!("Loaded {} source(s) from '{file_path}'.", sources_loaded);
    println!("\nRanked sources for query:");
    println!("  {sparql}");
    println!();

    if ranking.is_empty() {
        println!("No sources matched the query.");
        return Ok(());
    }

    for (i, sel) in ranking.sources.iter().enumerate() {
        println!(
            "  {}. {}  confidence={:.3}  reason={:?}",
            i + 1,
            sel.source_id,
            sel.confidence,
            sel.reason,
        );
    }

    Ok(())
}

fn cmd_state(args: &[String], json_mode: bool) -> Result<(), OxiRouterError> {
    let sub = args.first().map(String::as_str).ok_or_else(|| {
        eprintln!("Missing state sub-subcommand. Use: state save | state load");
        OxiRouterError::QueryParse("state sub-subcommand required".to_string())
    })?;

    let rest = &args[1..];

    match sub {
        "save" => cmd_state_save(rest, json_mode),
        "load" => cmd_state_load(rest, json_mode),
        other => {
            eprintln!("Unknown state sub-subcommand: {other}");
            eprintln!("Use: oxirouter-cli state save --file <path>");
            eprintln!("     oxirouter-cli state load --file <path>");
            std::process::exit(1);
        }
    }
}

fn cmd_state_save(args: &[String], json_mode: bool) -> Result<(), OxiRouterError> {
    let file_path = flag_value(args, "--file").ok_or_else(|| {
        eprintln!("Missing required flag: --file");
        eprintln!("Usage: oxirouter-cli state save --file <path>");
        OxiRouterError::QueryParse("--file flag is required".to_string())
    })?;

    let router = default_router();
    let bytes = router.save_state()?;
    let bytes_written = bytes.len();

    fs::write(file_path, &bytes).map_err(|e| {
        OxiRouterError::ModelError(format!("failed to write state to '{file_path}': {e}"))
    })?;

    if json_mode {
        let output = CliOutput::StateSaved {
            bytes_written,
            file: file_path,
        };
        return emit_json(&output);
    }

    println!("State saved: {bytes_written} bytes to '{file_path}'");
    Ok(())
}

fn cmd_state_load(args: &[String], json_mode: bool) -> Result<(), OxiRouterError> {
    let file_path = flag_value(args, "--file").ok_or_else(|| {
        eprintln!("Missing required flag: --file");
        eprintln!("Usage: oxirouter-cli state load --file <path>");
        OxiRouterError::QueryParse("--file flag is required".to_string())
    })?;

    let bytes = fs::read(file_path).map_err(|e| {
        OxiRouterError::ModelError(format!("failed to read state from '{file_path}': {e}"))
    })?;

    let mut router = Router::new();
    router.load_state(&bytes)?;
    let sources_count = router.source_count();

    if json_mode {
        let output = CliOutput::StateLoaded {
            sources_count,
            file: file_path,
        };
        return emit_json(&output);
    }

    println!("State loaded from '{file_path}' ({sources_count} sources)");
    Ok(())
}
