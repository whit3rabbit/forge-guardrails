use super::*;
use indexmap::IndexMap;
use serde_json::{json, Value};

fn safe_config() -> ToolOutputCompressionConfig {
    ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Safe)
}

#[test]
fn mode_parse_accepts_expected_values() {
    assert_eq!(
        "disabled".parse::<ToolOutputCompressionMode>().unwrap(),
        ToolOutputCompressionMode::Disabled
    );
    assert_eq!(
        "safe".parse::<ToolOutputCompressionMode>().unwrap(),
        ToolOutputCompressionMode::Safe
    );
    assert_eq!(
        "standard".parse::<ToolOutputCompressionMode>().unwrap(),
        ToolOutputCompressionMode::Standard
    );
    assert_eq!(
        "aggressive".parse::<ToolOutputCompressionMode>().unwrap(),
        ToolOutputCompressionMode::Aggressive
    );
}

#[test]
fn method_parse_accepts_expected_values() {
    assert_eq!(
        "lzw".parse::<ToolOutputCompressionMethod>().unwrap(),
        ToolOutputCompressionMethod::Lzw
    );
    assert_eq!(
        "repair".parse::<ToolOutputCompressionMethod>().unwrap(),
        ToolOutputCompressionMethod::Repair
    );
    assert_eq!(
        "auto".parse::<ToolOutputCompressionMethod>().unwrap(),
        ToolOutputCompressionMethod::Auto
    );
    assert!("gzip".parse::<ToolOutputCompressionMethod>().is_err());
}

#[test]
fn disabled_returns_original_output() {
    let result = compress_tool_output(
        "bash",
        None,
        None,
        "unchanged",
        &ToolOutputCompressionConfig::disabled(),
        None,
    );
    assert_eq!(result.output, "unchanged");
    assert_eq!(result.saved_tokens, 0);
}

#[test]
fn safe_redacts_secret_like_values() {
    let result = compress_tool_output(
        "bash",
        None,
        None,
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        &safe_config(),
        None,
    );
    assert!(result.redacted);
    assert!(result.output.contains("[REDACTED_SECRET]"));
    assert!(!result.output.contains("sk-abcdefghijklmnopqrstuvwxyz"));
}

#[test]
fn safe_redaction_can_be_disabled() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Safe,
        redact_secrets: false,
        ..ToolOutputCompressionConfig::default()
    };
    let result = compress_tool_output(
        "bash",
        None,
        None,
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        &config,
        None,
    );
    assert!(!result.redacted);
    assert!(result.output.contains("sk-abcdefghijklmnopqrstuvwxyz"));
}

#[test]
fn safe_strips_ansi_sequences() {
    let result = compress_tool_output(
        "bash",
        None,
        None,
        "\u{1b}[31merror\u{1b}[0m",
        &safe_config(),
        None,
    );
    assert_eq!(result.output, "error");
    assert!(result.strategies.contains(&"strip_ansi".to_string()));
}

#[test]
fn safe_redacts_secret_split_by_ansi_sequences() {
    let result = compress_tool_output(
        "bash",
        None,
        None,
        "token sk-\u{1b}[31mabcdefghijklmnopqrstuvwxyz\u{1b}[0m",
        &safe_config(),
        None,
    );
    assert!(result.redacted);
    assert!(result.output.contains("[REDACTED_SECRET]"));
    assert!(!result.output.contains("sk-abcdefghijklmnopqrstuvwxyz"));
}

#[test]
fn safe_redacts_single_line_private_key_without_swallowing_rest() {
    let raw = "-----BEGIN PRIVATE KEY-----MIIEvQIBADANBg-----END PRIVATE KEY-----\nimportant output line\n";
    let result = compress_tool_output("bash", None, None, raw, &safe_config(), None);
    assert!(result.redacted);
    assert!(result.output.contains("[REDACTED_PRIVATE_KEY]"));
    assert!(!result.output.contains("MIIEvQIBADANBg"));
    assert!(result.output.contains("important output line"));
}

#[test]
fn safe_redacts_github_fine_grained_and_slack_tokens() {
    let raw = "pat github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef slack xoxb-1234567890-abcdefghijk";
    let result = compress_tool_output("bash", None, None, raw, &safe_config(), None);
    assert!(result.redacted);
    assert!(!result
        .output
        .contains("github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"));
    assert!(!result.output.contains("xoxb-1234567890-abcdefghijk"));
    assert_eq!(result.output.matches("[REDACTED_SECRET]").count(), 2);
}

#[test]
fn safe_preserves_thinking_blocks_from_tool_output() {
    let raw = "visible before\n<thinking>\nprivate chain\n</thinking>\nvisible after\n";
    let result = compress_tool_output("bash", None, None, raw, &safe_config(), None);
    assert_eq!(result.output, raw);
    assert!(!result.strategies.contains(&"strip_thinking".to_string()));
}

#[test]
fn safe_suppresses_binary_output() {
    let result = compress_tool_output("bash", None, None, "abc\0def", &safe_config(), None);
    assert!(result.capped);
    assert!(result.output.contains("Binary output suppressed"));
}

#[test]
fn safe_caps_oversized_output() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Safe,
        max_output_bytes: 20,
        ..ToolOutputCompressionConfig::default()
    };
    let result = compress_tool_output("bash", None, None, "a".repeat(200).as_str(), &config, None);
    assert!(result.capped);
    assert!(result.output.contains("Tool output capped"));
}

#[test]
fn safe_caps_oversized_unicode_output_on_char_boundaries() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Safe,
        max_output_bytes: 13,
        ..ToolOutputCompressionConfig::default()
    };
    let raw = "alpha😀beta😀gamma😀delta";

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(result.capped);
    assert!(result.output.contains("Tool output capped"));
    assert!(result.output.is_char_boundary(result.output.len()));
    assert!(!result.output.contains('\u{fffd}'));
}

#[test]
fn aggressive_binary_suppression_skips_later_filters() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = format!("{}{}", "abc\0def", "2026-05-28T12:34:56Z ".repeat(20));

    let result = compress_tool_output("bash", None, None, &raw, &config, None);

    assert!(result.capped);
    assert_eq!(
        result.output,
        format!("[Binary output suppressed: {} bytes]", raw.len())
    );
    assert!(result
        .strategies
        .contains(&"binary_suppression".to_string()));
    assert!(!result
        .strategies
        .contains(&"normalize_dynamic_log_noise".to_string()));
    assert!(!is_dictionary_compressed_output(&result.output));
}

#[test]
fn standard_minifies_json_when_smaller() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let result = compress_tool_output(
        "read",
        None,
        None,
        "{\n  \"a\": 1,\n  \"b\": 2\n}",
        &config,
        None,
    );
    assert_eq!(result.output, "{\"a\":1,\"b\":2}");
}

#[test]
fn standard_routes_grep_output_by_file() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let result = compress_tool_output(
        "search",
        None,
        None,
        "src/a.rs:10:fn alpha()\nsrc/a.rs:20:fn beta()\ntarget/x.rs:1:noise\n",
        &config,
        None,
    );
    assert!(result.output.contains("src/a.rs:"));
    assert!(!result.output.contains("target/x.rs"));
}

#[test]
fn standard_grep_handles_windows_paths_and_column_fields() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "\
C:\\repo\\src\\a.rs:10:5:fn alpha()
C:\\repo\\src\\a.rs:11:fn beta()
target\\debug\\noise.rs:1:ignored
";

    let result = compress_tool_output("grep", None, None, raw, &config, None);

    assert!(result.output.contains("C:\\repo\\src\\a.rs:"));
    assert!(result.output.contains("10: fn alpha()"));
    assert!(result.output.contains("11: fn beta()"));
    assert!(!result.output.contains("target\\debug\\noise.rs"));
}

#[test]
fn standard_grep_preserves_diagnostics_with_matches() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "\
rg: warning: skipped hidden file
src/a.rs:10:needle alpha
src/a.rs:11:needle beta
src/a.rs:12:needle gamma
target/noise.rs:1:needle ignored
";

    let result = compress_tool_output("grep", None, None, raw, &config, None);

    assert!(result.output.contains("diagnostics:"));
    assert!(result.output.contains("rg: warning: skipped hidden file"));
    assert!(result.output.contains("src/a.rs:"));
    assert!(!result.output.contains("target/noise.rs"));
}

#[test]
fn standard_grep_unknown_output_is_preserved() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "rg: regex parse error:\n    (?:\n    ^\nerror: unclosed group\n";

    let result = compress_tool_output("grep", None, None, raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.output.contains("(no matches)"));
}

#[test]
fn standard_glob_all_noise_paths_are_preserved() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "target/debug/build.log\nnode_modules/pkg/index.js\n";

    let result = compress_tool_output("glob", None, None, raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.output.contains("(no matches)"));
}

#[test]
fn standard_cargo_unknown_success_output_is_preserved() {
    let mut args = IndexMap::new();
    args.insert("command".to_string(), json!("cargo build"));
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "Compiling demo v0.1.0\nFinished dev [unoptimized] target(s) in 0.1s\n";

    let result = compress_tool_output("bash", None, Some(&args), raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.output.contains("compiled successfully"));
}

#[test]
fn standard_cargo_json_diagnostics_are_summarized() {
    let mut args = IndexMap::new();
    args.insert(
        "command".to_string(),
        json!("cargo check --message-format=json"),
    );
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "{\"reason\":\"compiler-message\",\"message\":{\"level\":\"error\",\"rendered\":\"error[E0425]: missing value\\n --> src/lib.rs:1:1\\n\"}}\n{\"reason\":\"build-finished\",\"success\":false}\n";

    let result = compress_tool_output("bash", None, Some(&args), raw, &config, None);

    assert!(result.output.starts_with("Errors (1):\nerror[E0425]"));
    assert!(!result.output.contains("\"reason\""));
}

#[test]
fn standard_test_unknown_success_output_is_preserved() {
    let mut args = IndexMap::new();
    args.insert("command".to_string(), json!("python custom_harness.py"));
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "custom harness completed without standard summary\n";

    let result = compress_tool_output("bash", None, Some(&args), raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.output.contains("all tests passed"));
}

#[test]
fn standard_read_large_source_returns_outline() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let mut args = IndexMap::new();
    args.insert("path".to_string(), json!("src/example.rs"));
    let mut source = String::new();
    source.push_str("use crate::thing;\n");
    for idx in 0..300 {
        source.push_str(&format!("let value_{idx} = {idx};\n"));
    }
    source.push_str("pub struct Widget {\n    id: String,\n}\n");
    source.push_str("fn run_widget() {\n    println!(\"run\");\n}\n");

    let result = compress_tool_output("read_file", None, Some(&args), &source, &config, None);

    assert!(result.output.starts_with("// src/example.rs ("));
    assert!(result.output.contains("L1: use crate"));
    assert!(result.output.contains("pub struct Widget"));
    assert!(result.output.contains("fn run_widget"));
    assert!(!result.output.contains("let value_299"));
}

#[test]
fn standard_read_keeps_config_files_verbatim() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let mut args = IndexMap::new();
    args.insert("path".to_string(), json!("Cargo.toml"));
    let raw = "[package]\nname = \"forge\"\n\n[dependencies]\nserde = \"1\"\n";

    let result = compress_tool_output("read_file", None, Some(&args), raw, &config, None);

    assert_eq!(result.output, raw);
}

#[test]
fn standard_detects_bash_family_from_command_args() {
    let mut args = IndexMap::new();
    args.insert("command".to_string(), json!("cargo test"));
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let result = compress_tool_output(
        "shell",
        None,
        Some(&args),
        "Compiling x\nerror: failed\nlots of noise\n",
        &config,
        None,
    );
    assert_eq!(result.canonical_tool, "bash");
    assert_eq!(result.family, "cargo");
    assert!(result.output.contains("error: failed"));
}

#[test]
fn standard_bash_git_diff_keeps_hunks_and_changed_lines() {
    let mut args = IndexMap::new();
    args.insert("command".to_string(), json!("git diff"));
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "\
diff --git a/src/lib.rs b/src/lib.rs
index 111..222 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,3 @@
 context line
-old value
+new value
";

    let result = compress_tool_output("bash", None, Some(&args), raw, &config, None);

    assert!(result.output.contains("Files changed: 1"));
    assert!(result.output.contains("src/lib.rs"));
    assert!(result.output.contains("@@ -1,3 +1,3 @@"));
    assert!(result.output.contains("-old value"));
    assert!(result.output.contains("+new value"));
    assert!(!result.output.contains("diff --git"));
    assert!(!result.output.contains("context line"));
}

#[test]
fn standard_glob_drops_noise_paths_when_useful() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let mut lines = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
    for idx in 0..40 {
        lines.push(format!("target/debug/deps/generated_artifact_{idx}.rs"));
        lines.push(format!("node_modules/pkg_{idx}/index.js"));
    }
    let raw = lines.join("\n");

    let result = compress_tool_output("glob", None, None, &raw, &config, None);

    assert!(result.output.contains("src/main.rs"));
    assert!(result.output.contains("src/lib.rs"));
    assert!(!result.output.contains("node_modules"));
    assert!(!result.output.contains("target/debug"));
}

#[test]
fn opentoken_filter_fixture_cases_match_expected_outputs() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/parity/fixtures/opentoken_tool_output_filters.json"
    ))
    .expect("valid OpenToken tool-output fixture");
    let cases = fixture["cases"].as_array().expect("fixture cases");
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);

    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let tool = case["tool"].as_str().expect("case tool");
        let input = fixture_input(case);
        let args = fixture_args(case);
        let result = compress_tool_output(tool, None, args.as_ref(), &input, &config, None);
        let expected = case["expected_output"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: missing expected_output"));
        assert_eq!(
            result.output, expected,
            "{name}: OpenToken fixture mismatch"
        );
    }
}

#[test]
fn compression_golden_fixture_cases_match_expected_outputs() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/compression/golden.json"))
            .expect("valid compression golden fixture");
    let cases = fixture["cases"].as_array().expect("fixture cases");

    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let tool = case["tool"].as_str().expect("case tool");
        let mode = case["mode"]
            .as_str()
            .expect("case mode")
            .parse::<ToolOutputCompressionMode>()
            .unwrap_or_else(|err| panic!("{name}: invalid mode: {err}"));
        let input = case["input"].as_str().expect("case input");
        let expected = case["expected_output"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: missing expected_output"));
        let expected_strategies = case["expected_strategies"]
            .as_array()
            .expect("case expected_strategies")
            .iter()
            .map(|value| value.as_str().expect("strategy string").to_string())
            .collect::<Vec<_>>();

        let config = ToolOutputCompressionConfig::from_mode(mode);
        let result = compress_tool_output(tool, None, None, input, &config, None);

        assert_eq!(result.output, expected, "{name}: output mismatch");
        assert_eq!(
            result.strategies, expected_strategies,
            "{name}: strategy mismatch"
        );
    }
}

#[test]
fn standard_filter_does_not_grow_output() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = "short";
    let result = compress_tool_output("bash", None, None, raw, &config, None);
    assert_eq!(result.output, raw);
}

#[test]
fn aggressive_normalizes_timestamps_and_hashes() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = "2026-05-28T12:34:56Z completed artifact 0123456789abcdef0123456789abcdef\n";

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(result.output.contains("[timestamp]"));
    assert!(result.output.contains("[hash]"));
    assert!(result
        .strategies
        .contains(&"normalize_dynamic_log_noise".to_string()));
}

#[test]
fn aggressive_converts_json_array_to_tabular_form_when_smaller() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_status_name":"passed","long_duration_ms":10,"long_file_path":"src/a.rs"},
  {"long_status_name":"failed","long_duration_ms":20,"long_file_path":"src/b.rs"},
  {"long_status_name":"passed","long_duration_ms":30,"long_file_path":"src/c.rs"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(result
        .output
        .starts_with("[3]{long_status_name:string,long_duration_ms:int,long_file_path:string}\n"));
    assert!(result.output.contains("failed,20,src/b.rs"));
    assert!(result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_converts_sparse_json_array_to_nullable_schema_table() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_status":"passed"},
  {"long_identifier":2,"long_owner":"alice"},
  {"long_identifier":3,"long_status":"failed","long_owner":"bob"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(result
        .output
        .starts_with("[3]{long_identifier:int,long_status:string?,long_owner:string?}\n"));
    assert!(result.output.contains("1,passed,"));
    assert!(result.output.contains("2,,alice"));
    assert!(result.output.contains("3,failed,bob"));
    assert!(result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_quotes_csv_strings() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_message":"alice, lead"},
  {"long_identifier":2,"long_message":"she said \"hi\""},
  {"long_identifier":3,"long_message":"plain"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(result.output.contains(r#""alice, lead""#));
    assert!(result.output.contains(r#""she said ""hi""""#));
    assert!(result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_records_scalar_types() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_score":1.5,"long_ok":true,"long_name":"alpha"},
  {"long_identifier":2,"long_score":2.25,"long_ok":false,"long_name":"beta"},
  {"long_identifier":3,"long_score":3.75,"long_ok":true,"long_name":"gamma"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(result
        .output
        .starts_with("[3]{long_identifier:int,long_score:float,long_ok:bool,long_name:string}\n"));
    assert!(result.output.contains("2,2.25,false,beta"));
    assert!(result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_skips_nested_values() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_payload":{"nested":true}},
  {"long_identifier":2,"long_payload":{"nested":false}},
  {"long_identifier":3,"long_payload":{"nested":true}}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(!result.output.starts_with("[3]{"));
    assert!(!result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_skips_mixed_arrays() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_status":"passed"},
  2,
  {"long_identifier":3,"long_status":"failed"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(!result.output.starts_with("[3]{"));
    assert!(!result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_skips_small_inputs() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[{"a":1},{"a":2}]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_skips_mixed_type_columns() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_status_value":5,"long_file_path":"src/alpha.rs"},
  {"long_identifier":2,"long_status_value":"5","long_file_path":"src/beta.rs"},
  {"long_identifier":3,"long_status_value":7,"long_file_path":"src/gamma.rs"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    assert!(!result.output.starts_with("[3]{"));
    assert!(!result.strategies.contains(&"toon_table".to_string()));
}

#[test]
fn aggressive_json_array_table_distinguishes_null_from_empty_string() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = r#"[
  {"long_identifier":1,"long_message":""},
  {"long_identifier":2,"long_message":null},
  {"long_identifier":3,"long_message":"plain text value"}
]"#;

    let result = compress_tool_output("bash", None, None, raw, &config, None);

    let lines = result.output.lines().collect::<Vec<_>>();
    assert_eq!(lines[0], "[3]{long_identifier:int,long_message:string?}");
    assert_eq!(lines[1], "1,\"\"");
    assert_eq!(lines[2], "2,");
    assert_eq!(lines[3], "3,plain text value");
}

#[test]
fn standard_does_not_apply_lzw() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
    let raw = repeated_lzw_output();

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.strategies.contains(&"lzw_dictionary".to_string()));
}

#[test]
fn standard_ignores_dictionary_method() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        method: ToolOutputCompressionMethod::Repair,
        ..ToolOutputCompressionConfig::default()
    };
    let raw = repeated_lzw_output();

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert_eq!(result.output, raw);
    assert!(!result.output.starts_with("[Forge RePair Dictionary]"));
    assert!(!result.strategies.contains(&"repair_dictionary".to_string()));
}

#[test]
fn aggressive_lzw_records_strategy() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = repeated_lzw_output();

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert!(result.output.starts_with("[Forge LZW Dictionary]"));
    assert!(result.strategies.contains(&"lzw_dictionary".to_string()));
}

#[test]
fn aggressive_repair_records_strategy() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Aggressive,
        method: ToolOutputCompressionMethod::Repair,
        ..ToolOutputCompressionConfig::default()
    };
    let raw = repeated_lzw_output();

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert!(result.output.starts_with("[Forge RePair Dictionary]"));
    assert!(result.strategies.contains(&"repair_dictionary".to_string()));
}

#[test]
fn aggressive_auto_uses_smaller_dictionary_output() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Aggressive,
        method: ToolOutputCompressionMethod::Auto,
        ..ToolOutputCompressionConfig::default()
    };
    let raw = repeated_lzw_output();
    let lzw = compress_lzw_dictionary(&raw).expect("lzw output");
    let repair = compress_repair_dictionary(&raw).expect("repair output");

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert_eq!(result.output.len(), lzw.len().min(repair.len()));
    assert!(result.strategies.contains(&"auto_dictionary".to_string()));
}

#[test]
fn aggressive_lzw_can_fire_after_table_whitespace_minimization() {
    let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let raw = repeated_table_dictionary_output();

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert!(result.output.starts_with("[Forge LZW Dictionary]"));
    assert!(result
        .strategies
        .contains(&"minimize_table_whitespace".to_string()));
    assert!(result.strategies.contains(&"lzw_dictionary".to_string()));
}

#[test]
fn aggressive_repair_can_fire_after_table_whitespace_minimization() {
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Aggressive,
        method: ToolOutputCompressionMethod::Repair,
        ..ToolOutputCompressionConfig::default()
    };
    let raw = repeated_table_dictionary_output();

    let result = compress_tool_output("custom_tool", None, None, &raw, &config, None);

    assert!(result.output.starts_with("[Forge RePair Dictionary]"));
    assert!(result
        .strategies
        .contains(&"minimize_table_whitespace".to_string()));
    assert!(result.strategies.contains(&"repair_dictionary".to_string()));
}

#[test]
fn aggressive_auto_can_choose_dictionary_after_table_whitespace_minimization() {
    let lzw_config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
    let repair_config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Aggressive,
        method: ToolOutputCompressionMethod::Repair,
        ..ToolOutputCompressionConfig::default()
    };
    let auto_config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Aggressive,
        method: ToolOutputCompressionMethod::Auto,
        ..ToolOutputCompressionConfig::default()
    };
    let raw = repeated_table_dictionary_output();
    let lzw = compress_tool_output("custom_tool", None, None, &raw, &lzw_config, None);
    let repair = compress_tool_output("custom_tool", None, None, &raw, &repair_config, None);

    let result = compress_tool_output("custom_tool", None, None, &raw, &auto_config, None);

    assert_eq!(
        result.output.len(),
        lzw.output.len().min(repair.output.len())
    );
    assert!(result
        .strategies
        .contains(&"minimize_table_whitespace".to_string()));
    assert!(result.strategies.contains(&"auto_dictionary".to_string()));
}

#[test]
fn dedup_returns_bounded_marker_for_repeated_output() {
    let state = ToolOutputCompressionState::new();
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some("s1".to_string()),
        ..ToolOutputCompressionConfig::default()
    };
    let raw = (0..200)
        .map(|idx| format!("unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let first = compress_tool_output("bash", Some("call_1"), None, &raw, &config, Some(&state));
    let second = compress_tool_output("bash", Some("call_2"), None, &raw, &config, Some(&state));
    assert!(!first.deduped);
    assert!(second.deduped);
    assert!(second.output.contains("Duplicate of tool call call_1"));
}

#[test]
fn dedup_keeps_content_when_same_tool_call_id_is_resent() {
    let state = ToolOutputCompressionState::new();
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some("s1".to_string()),
        ..ToolOutputCompressionConfig::default()
    };
    let raw = (0..200)
        .map(|idx| format!("unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");

    // The proxy re-walks the full conversation every request, so the same
    // tool result re-arrives under the same call id and must keep its content.
    let first = compress_tool_output("bash", Some("call_1"), None, &raw, &config, Some(&state));
    let resent = compress_tool_output("bash", Some("call_1"), None, &raw, &config, Some(&state));
    let duplicate = compress_tool_output("bash", Some("call_2"), None, &raw, &config, Some(&state));

    assert!(!first.deduped);
    assert!(!resent.deduped);
    assert_eq!(resent.output, first.output);
    assert!(duplicate.deduped);
}

#[test]
fn dedup_skips_results_without_tool_call_id() {
    let state = ToolOutputCompressionState::new();
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some("s1".to_string()),
        ..ToolOutputCompressionConfig::default()
    };
    let raw = (0..200)
        .map(|idx| format!("unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");

    let first = compress_tool_output("bash", None, None, &raw, &config, Some(&state));
    let second = compress_tool_output("bash", None, None, &raw, &config, Some(&state));

    assert!(!first.deduped);
    assert!(!second.deduped);
}

#[test]
fn dedup_is_scoped_by_session() {
    let state = ToolOutputCompressionState::new();
    let raw = (0..200)
        .map(|idx| format!("unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let first_config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some("s1".to_string()),
        ..ToolOutputCompressionConfig::default()
    };
    let second_config = ToolOutputCompressionConfig {
        session_id: Some("s2".to_string()),
        ..first_config.clone()
    };

    let first = compress_tool_output(
        "bash",
        Some("call_1"),
        None,
        &raw,
        &first_config,
        Some(&state),
    );
    let second = compress_tool_output(
        "bash",
        Some("call_2"),
        None,
        &raw,
        &second_config,
        Some(&state),
    );
    let third = compress_tool_output(
        "bash",
        Some("call_3"),
        None,
        &raw,
        &first_config,
        Some(&state),
    );

    assert!(!first.deduped);
    assert!(!second.deduped);
    assert!(third.deduped);
}

#[test]
fn dedup_is_scoped_by_tool_name() {
    let state = ToolOutputCompressionState::new();
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some("s1".to_string()),
        ..ToolOutputCompressionConfig::default()
    };
    let raw = (0..200)
        .map(|idx| format!("unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");

    let first = compress_tool_output("bash", Some("call_1"), None, &raw, &config, Some(&state));
    let second = compress_tool_output("read", Some("call_2"), None, &raw, &config, Some(&state));
    let third = compress_tool_output("bash", Some("call_3"), None, &raw, &config, Some(&state));

    assert!(!first.deduped);
    assert!(!second.deduped);
    assert!(third.deduped);
}

#[test]
fn dedup_evicts_oldest_entries_per_session() {
    let state = ToolOutputCompressionState::new();
    let config = ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some("s1".to_string()),
        max_dedup_entries_per_session: 1,
        ..ToolOutputCompressionConfig::default()
    };
    let first_raw = (0..200)
        .map(|idx| format!("first unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let second_raw = (0..200)
        .map(|idx| format!("second unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !compress_tool_output(
            "bash",
            Some("call_1"),
            None,
            &first_raw,
            &config,
            Some(&state)
        )
        .deduped
    );
    assert!(
        !compress_tool_output(
            "bash",
            Some("call_2"),
            None,
            &second_raw,
            &config,
            Some(&state)
        )
        .deduped
    );
    assert!(
        !compress_tool_output(
            "bash",
            Some("call_3"),
            None,
            &first_raw,
            &config,
            Some(&state)
        )
        .deduped
    );
    assert!(
        compress_tool_output(
            "bash",
            Some("call_4"),
            None,
            &first_raw,
            &config,
            Some(&state)
        )
        .deduped
    );
}

#[test]
fn dedup_evicts_oldest_sessions() {
    let state = ToolOutputCompressionState::new();
    let raw = (0..200)
        .map(|idx| format!("unique long content line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let config_for_session = |session_id: &str| ToolOutputCompressionConfig {
        mode: ToolOutputCompressionMode::Standard,
        session_id: Some(session_id.to_string()),
        max_dedup_sessions: 1,
        ..ToolOutputCompressionConfig::default()
    };

    let s1 = config_for_session("s1");
    let s2 = config_for_session("s2");

    assert!(!compress_tool_output("bash", Some("call_1"), None, &raw, &s1, Some(&state)).deduped);
    assert!(!compress_tool_output("bash", Some("call_2"), None, &raw, &s2, Some(&state)).deduped);
    assert!(!compress_tool_output("bash", Some("call_3"), None, &raw, &s1, Some(&state)).deduped);
    assert!(compress_tool_output("bash", Some("call_4"), None, &raw, &s1, Some(&state)).deduped);
}

fn repeated_lzw_output() -> String {
    (0..24)
        .map(|idx| {
            format!(
                "error: repeated dependency resolution failure in workspace crate alpha at module_{idx}\n"
            )
        })
        .collect::<String>()
}

fn repeated_table_dictionary_output() -> String {
    (0..48)
        .map(|idx| {
            format!(
                "ROW-{idx:04} | service=checkout | status=degraded | \
                 message=database connection pool timeout while processing payment authorization | \
                 recommended_action=retry with exponential backoff and preserve request correlation id\n"
            )
        })
        .collect::<String>()
}

fn fixture_args(case: &Value) -> Option<IndexMap<String, Value>> {
    let args = case.get("args")?.as_object()?;
    Some(
        args.iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn fixture_input(case: &Value) -> String {
    if let Some(input) = case.get("input").and_then(Value::as_str) {
        return input.to_string();
    }
    let mut result = String::new();
    for part in case["input_parts"]
        .as_array()
        .expect("input or input_parts")
    {
        let text = part["text"].as_str().expect("input part text");
        let repeat = part
            .get("repeat")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .try_into()
            .expect("repeat fits usize");
        for _ in 0..repeat {
            result.push_str(text);
        }
    }
    result
}

#[test]
fn test_looks_like_jsonl_strict() {
    use super::postcall::looks_like_jsonl;

    // Valid JSONL
    assert!(looks_like_jsonl("{\"a\": 1}\n{\"b\": 2}"));
    assert!(looks_like_jsonl("  {\"a\": 1}  \n  {\"b\": 2}  "));

    // Single valid JSON object
    assert!(looks_like_jsonl("{\n  \"a\": 1\n}"));

    // Invalid: Starts with '{' but contains plain text / table data later
    assert!(!looks_like_jsonl("{\n  \"a\": 1\n}\nplain text here"));
    assert!(!looks_like_jsonl("{\n  \"a\": 1\n}\n| col1 | col2 |"));

    // Invalid: Plain text
    assert!(!looks_like_jsonl("plain text"));
}

#[test]
fn test_cargo_json_messages_with_mixed_lines() {
    use super::families::cargo::filter_cargo_output;

    // Mixed output: Compile status text + JSON compiler message
    let raw = "\
Compiling demo v0.1.0
{\"reason\":\"compiler-message\",\"message\":{\"level\":\"error\",\"rendered\":\"error[E0425]: missing value\\n\"}}
Finished dev [unoptimized] target(s) in 0.1s
";
    let result = filter_cargo_output("cargo build", raw);
    // If it successfully parses JSON, it should output:
    // "Errors (1):\nerror[E0425]: missing value"
    assert!(result.starts_with("Errors (1):"));
    assert!(result.contains("error[E0425]"));
}

#[test]
fn test_windows_path_compatibility() {
    use super::basename;
    use super::filters::filter_glob_output;
    use super::filters::is_noise_path;

    // 1. is_noise_path matches Windows paths
    assert!(is_noise_path("node_modules\\lodash\\index.js"));
    assert!(is_noise_path("target\\debug\\build.log"));

    // 2. basename extracts command names on Windows paths
    assert_eq!(
        basename("C:\\Users\\admin\\.cargo\\bin\\cargo.exe"),
        "cargo.exe"
    );

    // 3. filter_glob_output groups Windows paths under directory
    let mut lines = Vec::new();
    for idx in 0..105 {
        lines.push(format!("src\\file_{idx}.rs"));
    }
    let raw_glob = lines.join("\n");
    let filtered = filter_glob_output(&raw_glob);
    assert!(filtered.contains("src/: 105 files"));
}
