use super::*;
use crate::cli::{Cli, CliBackend};
use clap::Parser;

fn parse(args: &[&str]) -> Cli {
    Cli::try_parse_from(std::iter::once("forge-guardrails-proxy").chain(args.iter().copied()))
        .expect("parse")
}

fn sample_config() -> ProxyConfig {
    ProxyConfig {
        host: DEFAULT_ENV_HOST.to_string(),
        port: DEFAULT_PROXY_PORT,
        default_model: DEFAULT_INTERNAL_MODEL.to_string(),
        default_model_explicit: false,
        context_tokens: DEFAULT_ENV_CONTEXT_TOKENS,
        max_retries: DEFAULT_MAX_RETRIES,
        rescue_enabled: true,
        serialize_requests: false,
        verbose: false,
        classifier_dir: None,
        classifier_mode: ScorerMode::Shadow,
        classifier_model: ClassifierModelKind::Quantized,
        classifier_auto_download: false,
        classifier_max_latency_ms: None,
        final_response_classifier_dir: None,
        final_response_classifier_mode: ScorerMode::Shadow,
        final_response_classifier_model: ClassifierModelKind::Quantized,
        final_response_classifier_max_latency_ms: None,
        tool_output_compression: ToolOutputCompressionConfig::disabled(),
        tool_call_policy: ToolCallPolicyConfig::disabled(),
    }
}

#[test]
fn env_fallback_accepts_safe_cli_overrides() {
    let cli = parse(&[
        "--host",
        "127.0.0.1",
        "--port",
        "9090",
        "--model",
        "env-override",
        "--budget-tokens",
        "2048",
        "--max-retries",
        "0",
        "--no-rescue",
        "--serialize",
        "-v",
    ]);
    let mut config = sample_config();
    apply_env_cli_overrides(&mut config, &cli).expect("overrides");
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 9090);
    assert_eq!(config.default_model, "env-override");
    assert!(config.default_model_explicit);
    assert_eq!(config.context_tokens, 2048);
    assert_eq!(config.max_retries, 0);
    assert!(!config.rescue_enabled);
    assert!(config.serialize_requests);
    assert!(config.verbose);
    assert_eq!(config.classifier_dir, None);
    assert_eq!(config.classifier_mode, ScorerMode::Shadow);
    assert_eq!(config.classifier_model, ClassifierModelKind::Quantized);
    assert!(!config.classifier_auto_download);
    assert_eq!(config.classifier_max_latency_ms, None);
    assert_eq!(config.final_response_classifier_dir, None);
    assert_eq!(config.final_response_classifier_mode, ScorerMode::Shadow);
    assert_eq!(
        config.final_response_classifier_model,
        ClassifierModelKind::Quantized
    );
    assert_eq!(config.final_response_classifier_max_latency_ms, None);
    assert_eq!(
        config.tool_output_compression.mode,
        ToolOutputCompressionMode::Disabled
    );
    assert_eq!(config.tool_call_policy.mode, ToolCallPolicyMode::Disabled);
}

#[test]
fn classifier_cli_overrides_include_final_response_settings() {
    let cli = parse(&[
        "--classifier-dir",
        "target/classifier-artifacts/onnx",
        "--classifier-mode",
        "advisory",
        "--classifier-model",
        "full",
        "--classifier-max-latency-ms",
        "25",
        "--final-response-classifier-dir",
        "target/final-response-artifacts/onnx",
        "--final-response-classifier-mode",
        "enforce",
        "--final-response-classifier-model",
        "quantized",
        "--final-response-classifier-max-latency-ms",
        "40",
    ]);
    let mut config = sample_config();

    apply_env_cli_overrides(&mut config, &cli).expect("overrides");

    assert_eq!(
        config.classifier_dir.as_deref(),
        Some("target/classifier-artifacts/onnx")
    );
    assert_eq!(config.classifier_mode, ScorerMode::Advisory);
    assert_eq!(config.classifier_model, ClassifierModelKind::Full);
    assert!(!config.classifier_auto_download);
    assert_eq!(config.classifier_max_latency_ms, Some(25));
    assert_eq!(
        config.final_response_classifier_dir.as_deref(),
        Some("target/final-response-artifacts/onnx")
    );
    assert_eq!(config.final_response_classifier_mode, ScorerMode::Enforce);
    assert_eq!(
        config.final_response_classifier_model,
        ClassifierModelKind::Quantized
    );
    assert_eq!(config.final_response_classifier_max_latency_ms, Some(40));
}

#[test]
fn tool_output_compression_cli_override_sets_mode() {
    let cli = parse(&["--tool-output-compression", "standard"]);
    let mut config = sample_config();

    apply_env_cli_overrides(&mut config, &cli).expect("overrides");

    assert_eq!(
        config.tool_output_compression.mode,
        ToolOutputCompressionMode::Standard
    );
}

#[test]
fn tool_output_compression_cli_override_sets_method() {
    let cli = parse(&[
        "--tool-output-compression",
        "aggressive",
        "--tool-output-compression-method",
        "repair",
    ]);
    let mut config = sample_config();

    apply_env_cli_overrides(&mut config, &cli).expect("overrides");

    assert_eq!(
        config.tool_output_compression.mode,
        ToolOutputCompressionMode::Aggressive
    );
    assert_eq!(
        config.tool_output_compression.method,
        ToolOutputCompressionMethod::Repair
    );
}

#[test]
fn tool_output_compression_cli_rejects_invalid_method() {
    let cli = parse(&["--tool-output-compression-method", "gzip"]);
    let mut config = sample_config();

    let err = apply_env_cli_overrides(&mut config, &cli).expect_err("invalid method");

    assert!(err.contains("method must be lzw, repair, or auto"));
}

#[test]
fn tool_output_compression_env_and_cli_method_precedence() {
    let cli = parse(&["--tool-output-compression-method", "auto"]);
    let mut config = tool_output_compression_from_env_values(
        Some("aggressive".to_string()),
        Some("repair".to_string()),
    )
    .expect("env config");

    apply_tool_output_compression_cli_overrides(&mut config, &cli).expect("cli override");

    assert_eq!(config.mode, ToolOutputCompressionMode::Aggressive);
    assert_eq!(config.method, ToolOutputCompressionMethod::Auto);
}

#[test]
fn tool_output_compression_env_rejects_invalid_method() {
    let err = tool_output_compression_from_env_values(
        Some("aggressive".to_string()),
        Some("gzip".to_string()),
    )
    .expect_err("invalid method");

    assert!(err.contains("method must be lzw, repair, or auto"));
}

#[test]
fn tool_call_policy_cli_override_sets_mode() {
    let cli = parse(&["--tool-call-policy", "standard"]);
    let mut config = sample_config();

    apply_env_cli_overrides(&mut config, &cli).expect("overrides");

    assert_eq!(config.tool_call_policy.mode, ToolCallPolicyMode::Standard);
    assert!(config.tool_call_policy.lsp_first);
    assert!(config.tool_call_policy.quiet_commands);
    assert!(config.tool_call_policy.write_payload_caps);
}

#[test]
fn managed_mode_serializes_by_default() {
    let cli = parse(&["--backend", "llamaserver", "--gguf", "model.gguf"]);
    assert!(resolve_serialize(&cli, true));
}

#[test]
fn managed_mode_can_disable_serialization() {
    let cli = parse(&[
        "--backend",
        "llamaserver",
        "--gguf",
        "model.gguf",
        "--no-serialize",
    ]);
    assert!(!resolve_serialize(&cli, true));
}

#[cfg(feature = "classifier")]
#[test]
fn classify_shortcut_uses_advisory_quantized_cache_defaults() {
    let cli = parse(&["--classify"]);
    let (dir, mode, model, auto_download) =
        classifier_settings_from_env_cli(&cli).expect("settings");
    assert!(dir
        .as_deref()
        .expect("dir")
        .contains("forge-guardrails/classifiers/tool-call"));
    assert_eq!(mode, ScorerMode::Advisory);
    assert_eq!(model, ClassifierModelKind::Quantized);
    assert!(auto_download);
}

#[cfg(feature = "classifier")]
#[test]
fn classify_shortcut_allows_explicit_dir_mode_and_model() {
    let cli = parse(&[
        "--classify",
        "--classifier-dir",
        "custom/onnx",
        "--classifier-mode",
        "shadow",
        "--classifier-model",
        "full",
    ]);
    let (dir, mode, model, auto_download) =
        classifier_settings_from_env_cli(&cli).expect("settings");
    assert_eq!(dir.as_deref(), Some("custom/onnx"));
    assert_eq!(mode, ScorerMode::Shadow);
    assert_eq!(model, ClassifierModelKind::Full);
    assert!(auto_download);
}

#[cfg(feature = "classifier")]
#[test]
fn classify_shortcut_rejects_disabled_mode() {
    let cli = parse(&["--classify", "--classifier-mode", "disabled"]);
    let err = classifier_settings_from_env_cli(&cli).expect_err("disabled");
    assert!(err.contains("--classify cannot be combined"));
}

#[test]
fn default_proxy_port_matches_python_reference() {
    assert_eq!(DEFAULT_PROXY_PORT, 8081);
}

#[test]
fn backend_as_str_matches_setup_backend_names() {
    assert_eq!(CliBackend::Llamaserver.as_str(), "llamaserver");
    assert_eq!(CliBackend::Llamafile.as_str(), "llamafile");
    assert_eq!(CliBackend::Ollama.as_str(), "ollama");
}
