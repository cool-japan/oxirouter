//! VoID/Turtle source-capability descriptor parser for the `void` feature.
//!
//! Converts a Turtle document that uses the VoID vocabulary and the
//! OxiRouter-specific namespace into a `Vec<DataSource>`.

#![cfg(feature = "void")]

#[cfg(feature = "alloc")]
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use hashbrown::HashSet;

use crate::core::error::Result;
use crate::core::source::{DataSource, SourceKind};
use crate::core::turtle::parse_turtle;

// Infallible: parse_turtle never returns Err; Result is here for API forward-compat

// ─────────────────────────────────────────────────────────────────────────────
// Vocabulary constants
// ─────────────────────────────────────────────────────────────────────────────

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const VOID_DATASET: &str = "http://rdfs.org/ns/void#Dataset";
const VOID_SPARQL_ENDPOINT: &str = "http://rdfs.org/ns/void#sparqlEndpoint";
const VOID_VOCABULARY: &str = "http://rdfs.org/ns/void#vocabulary";
const DCTERMS_SPATIAL: &str = "http://purl.org/dc/terms/spatial";
const DCTERMS_TITLE: &str = "http://purl.org/dc/terms/title";
const OXIROUTER_KIND: &str = "http://oxirouter.rs/ns/kind";
const OXIROUTER_MULTIADDR: &str = "http://oxirouter.rs/ns/multiaddr";
const OXIROUTER_PEER_ID: &str = "http://oxirouter.rs/ns/peerId";
const OXIROUTER_PRIORITY: &str = "http://oxirouter.rs/ns/priority";
const OXIROUTER_KIND_P2P_IPFS: &str = "http://oxirouter.rs/ns/P2pIpfs";
const OXIROUTER_KIND_P2P_LIBP2P: &str = "http://oxirouter.rs/ns/P2pLibp2p";

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a VoID/Turtle source descriptor document and return a
/// `Vec<DataSource>` for every `void:Dataset` that has a `void:sparqlEndpoint`.
///
/// Datasets without an endpoint triple are silently skipped (they are not
/// actionable routing targets).
///
/// # Errors
///
/// Returns an error if the Turtle input cannot be parsed at all (e.g., a
/// structural tokenization failure).
pub fn parse_oxirouter_ttl(ttl: &str) -> Result<Vec<DataSource>> {
    let doc = parse_turtle(ttl);

    // Collect all subjects typed as void:Dataset
    let dataset_subjects: HashSet<String> = doc
        .triples
        .iter()
        .filter(|t| t.predicate == RDF_TYPE && t.object == VOID_DATASET)
        .map(|t| t.subject.clone())
        .collect();

    let mut sources: Vec<DataSource> = Vec::new();

    for subject in &dataset_subjects {
        // Collect all triples for this subject
        let subject_triples: Vec<_> = doc
            .triples
            .iter()
            .filter(|t| &t.subject == subject)
            .collect();

        // Mandatory: sparqlEndpoint
        let endpoint = match subject_triples
            .iter()
            .find(|t| t.predicate == VOID_SPARQL_ENDPOINT)
            .map(|t| t.object.clone())
        {
            Some(ep) => ep,
            None => continue, // no endpoint → not a usable routing target
        };

        // ID: dcterms:title if present, else the subject IRI
        let id = subject_triples
            .iter()
            .find(|t| t.predicate == DCTERMS_TITLE)
            .map(|t| strip_string_quotes(&t.object))
            .unwrap_or_else(|| subject.clone());

        let mut source = DataSource::new(id, endpoint);

        // Vocabularies
        for t in subject_triples
            .iter()
            .filter(|t| t.predicate == VOID_VOCABULARY)
        {
            source = source.with_vocabulary(t.object.clone());
        }

        // Regions
        for t in subject_triples
            .iter()
            .filter(|t| t.predicate == DCTERMS_SPATIAL)
        {
            source = source.with_region(strip_string_quotes(&t.object));
        }

        // Priority
        if let Some(prio_triple) = subject_triples
            .iter()
            .find(|t| t.predicate == OXIROUTER_PRIORITY)
        {
            let raw = strip_string_quotes(&prio_triple.object);
            if let Ok(p) = raw.parse::<f32>() {
                source = source.with_priority(p);
            }
        }

        // Kind
        if let Some(kind_triple) = subject_triples
            .iter()
            .find(|t| t.predicate == OXIROUTER_KIND)
        {
            let kind_iri = &kind_triple.object;
            if kind_iri == OXIROUTER_KIND_P2P_IPFS {
                let multiaddr = subject_triples
                    .iter()
                    .find(|t| t.predicate == OXIROUTER_MULTIADDR)
                    .map(|t| strip_string_quotes(&t.object))
                    .unwrap_or_default();
                source = source.with_kind(SourceKind::P2pIpfs { multiaddr });
            } else if kind_iri == OXIROUTER_KIND_P2P_LIBP2P {
                let peer_id = subject_triples
                    .iter()
                    .find(|t| t.predicate == OXIROUTER_PEER_ID)
                    .map(|t| strip_string_quotes(&t.object))
                    .unwrap_or_default();
                source = source.with_kind(SourceKind::P2pLibp2p { peer_id });
            }
            // Otherwise keep default SourceKind::Sparql
        }

        sources.push(source);
    }

    Ok(sources)
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Strip surrounding quotes from a string if present.
///
/// The Turtle tokenizer already unescapes literal content and emits it
/// without quotes, so this is defensive — it handles any case where the
/// outer quotes were forwarded verbatim.
fn strip_string_quotes(s: &str) -> String {
    let s = s.trim();
    let has_outer_quotes =
        (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\''));
    if has_outer_quotes && s.len() >= 2 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}
