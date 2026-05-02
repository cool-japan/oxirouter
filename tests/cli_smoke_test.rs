//! Smoke tests for the `oxirouter-cli` binary.
//!
//! These tests locate the compiled binary relative to the test binary and
//! invoke it as a subprocess. They are skipped (not failed) when the binary
//! has not been compiled, which happens in test-only runs without `--bin`.

#![cfg(feature = "cli")]

use std::path::PathBuf;
use std::process::Command;

/// Locate the `oxirouter-cli` binary next to the test runner.
fn oxirouter_cli_bin() -> PathBuf {
    let mut path = std::env::current_exe().expect("could not determine test binary path");
    path.pop(); // remove test binary name
    if path.ends_with("deps") {
        path.pop(); // remove deps/
    }
    path.join("oxirouter-cli")
}

#[test]
fn test_cli_help_mentions_subcommands() {
    let bin = oxirouter_cli_bin();
    if !bin.exists() {
        // Binary not built — skip (happens in test-only runs without `--bin`)
        return;
    }
    let output = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("failed to run oxirouter-cli --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    assert!(combined.contains("route"), "help should mention 'route'");
    assert!(
        combined.contains("explain"),
        "help should mention 'explain'"
    );
    assert!(combined.contains("state"), "help should mention 'state'");
}

#[test]
fn test_cli_version_flag() {
    let bin = oxirouter_cli_bin();
    if !bin.exists() {
        return;
    }
    let output = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("failed to run oxirouter-cli --version");
    assert!(
        output.status.success(),
        "oxirouter-cli --version should exit 0"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("oxirouter-cli"),
        "version output should mention binary name"
    );
}

#[test]
fn test_cli_state_save_load_roundtrip() {
    let bin = oxirouter_cli_bin();
    if !bin.exists() {
        return;
    }

    let state_file = std::env::temp_dir().join("oxirouter_cli_smoke_state.bin");

    // state save
    let save_out = Command::new(&bin)
        .args([
            "state",
            "save",
            "--file",
            state_file.to_str().expect("valid path"),
        ])
        .output()
        .expect("failed to run state save");
    assert!(
        save_out.status.success(),
        "state save failed: {}",
        String::from_utf8_lossy(&save_out.stderr),
    );
    assert!(state_file.exists(), "state file should exist after save");

    // state load
    let load_out = Command::new(&bin)
        .args([
            "state",
            "load",
            "--file",
            state_file.to_str().expect("valid path"),
        ])
        .output()
        .expect("failed to run state load");
    assert!(
        load_out.status.success(),
        "state load failed: {}",
        String::from_utf8_lossy(&load_out.stderr),
    );
    let stdout = String::from_utf8_lossy(&load_out.stdout);
    assert!(
        stdout.contains("sources"),
        "load output should mention sources count"
    );

    // Cleanup
    let _ = std::fs::remove_file(&state_file);
}

#[test]
fn test_cli_state_save_json_output() {
    let bin = oxirouter_cli_bin();
    if !bin.exists() {
        return;
    }

    let state_file = std::env::temp_dir().join("oxirouter_cli_smoke_json_state.bin");

    let save_out = Command::new(&bin)
        .args([
            "--json",
            "state",
            "save",
            "--file",
            state_file.to_str().expect("valid path"),
        ])
        .output()
        .expect("failed to run --json state save");
    assert!(
        save_out.status.success(),
        "state save --json failed: {}",
        String::from_utf8_lossy(&save_out.stderr),
    );

    let stdout = String::from_utf8_lossy(&save_out.stdout);
    // Verify the output is valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json output should be valid JSON");
    assert_eq!(
        parsed["kind"], "state_saved",
        "JSON kind should be 'state_saved'"
    );
    assert!(
        parsed["bytes_written"].is_number(),
        "JSON should include bytes_written"
    );

    let _ = std::fs::remove_file(&state_file);
}
