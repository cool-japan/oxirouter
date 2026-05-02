//! Tests for [`oxirouter::core::term::StructuredTriple`] and [`oxirouter::core::term::Term`]
//! and for [`oxirouter::core::query::Query::from_sparql`] structured-triple population.
//!
//! All tests in this file require the `sparql` feature.

#![cfg(feature = "sparql")]

use oxirouter::core::query::Query;
use oxirouter::core::term::Term;

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 – Simple BGP parse produces one StructuredTriple
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_simple_bgp_parse() {
    let sparql = "SELECT ?s WHERE { ?s foaf:knows ?o }";
    let q = Query::from_sparql(sparql).expect("parse failed");
    assert_eq!(
        q.structured_triples.len(),
        1,
        "expected exactly one structured triple, got: {:?}",
        q.structured_triples
    );
    let t = &q.structured_triples[0];
    assert_eq!(t.subject, Term::Variable("s".to_string()));
    assert_eq!(
        t.predicate,
        Term::PrefixedName("foaf".to_string(), "knows".to_string())
    );
    assert_eq!(t.object, Term::Variable("o".to_string()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 – Prefix resolution via Term::resolve
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_prefix_resolution() {
    let sparql = r"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?s WHERE { ?s foaf:knows ?o }
    ";
    let q = Query::from_sparql(sparql).expect("parse failed");

    // prefixes map should contain bare label "foaf"
    assert!(
        q.prefixes.contains_key("foaf"),
        "prefixes map missing 'foaf': {:?}",
        q.prefixes
    );
    assert_eq!(
        q.prefixes.get("foaf").map(String::as_str),
        Some("http://xmlns.com/foaf/0.1/")
    );

    // Find the foaf:knows predicate and resolve it
    let foaf_triple = q
        .structured_triples
        .iter()
        .find(|t| matches!(&t.predicate, Term::PrefixedName(p, _) if p == "foaf"))
        .expect("no foaf triple found");

    let resolved = foaf_triple.predicate.resolve(&q.prefixes);
    assert_eq!(
        resolved,
        Term::Iri("http://xmlns.com/foaf/0.1/knows".to_string()),
        "resolved IRI mismatch"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 – Multiple triples preserved in source order
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multiple_triples() {
    let sparql = r"
        SELECT ?s ?name ?age WHERE {
            ?s rdf:type foaf:Person .
            ?s foaf:name ?name .
            ?s foaf:age ?age .
        }
    ";
    let q = Query::from_sparql(sparql).expect("parse failed");
    assert_eq!(
        q.structured_triples.len(),
        3,
        "expected 3 structured triples, got: {:?}",
        q.structured_triples
    );

    // Verify source-order preservation
    let preds: Vec<&Term> = q.structured_triples.iter().map(|t| &t.predicate).collect();
    assert!(
        matches!(preds[0], Term::PrefixedName(p, l) if p == "rdf" && l == "type"),
        "first predicate should be rdf:type, got {:?}",
        preds[0]
    );
    assert!(
        matches!(preds[1], Term::PrefixedName(p, l) if p == "foaf" && l == "name"),
        "second predicate should be foaf:name, got {:?}",
        preds[1]
    );
    assert!(
        matches!(preds[2], Term::PrefixedName(p, l) if p == "foaf" && l == "age"),
        "third predicate should be foaf:age, got {:?}",
        preds[2]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 – Variable term strips leading '?'
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_variable_term() {
    let t = Term::from_token("?subject");
    assert_eq!(t, Term::Variable("subject".to_string()));

    let t2 = Term::from_token("$other");
    assert_eq!(t2, Term::Variable("other".to_string()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 – Literal in object position
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_literal_term() {
    let sparql = r#"SELECT ?s WHERE { ?s foaf:name "Alice" }"#;
    let q = Query::from_sparql(sparql).expect("parse failed");
    let obj = q
        .structured_triples
        .iter()
        .find(|t| matches!(&t.predicate, Term::PrefixedName(p, l) if p == "foaf" && l == "name"))
        .map(|t| &t.object)
        .expect("triple with foaf:name not found");
    assert!(
        matches!(obj, Term::Literal(_)),
        "expected Literal, got {:?}",
        obj
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 – Property-path leaf extraction
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_property_path_leaf_extraction() {
    let sparql = "SELECT ?s ?o WHERE { ?s foaf:knows+ ?o }";
    let q = Query::from_sparql(sparql).expect("parse failed");
    // There should be at least one StructuredTriple with predicate = PrefixedName("foaf","knows")
    let has_leaf = q
        .structured_triples
        .iter()
        .any(|t| matches!(&t.predicate, Term::PrefixedName(p, l) if p == "foaf" && l == "knows"));
    assert!(
        has_leaf,
        "expected PrefixedName(foaf, knows) leaf in structured_triples: {:?}",
        q.structured_triples
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7 – Serialize / deserialize round-trip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_serialize_roundtrip() {
    let sparql = r"
        PREFIX schema: <http://schema.org/>
        SELECT ?s ?name WHERE {
            ?s schema:name ?name .
        }
    ";
    let q = Query::from_sparql(sparql).expect("parse failed");
    assert!(!q.structured_triples.is_empty(), "no structured triples");

    // Serialize to JSON and deserialize back
    let json = serde_json::to_string(&q).expect("serialize failed");
    let q2: Query = serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(
        q.structured_triples, q2.structured_triples,
        "structured_triples round-trip mismatch"
    );
    assert_eq!(q.prefixes, q2.prefixes, "prefixes round-trip mismatch");
}
