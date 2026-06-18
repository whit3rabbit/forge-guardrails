use std::str::FromStr;

use crate::cli::Cli;
#[cfg(feature = "classifier")]
use forge_guardrails::default_tool_call_classifier_artifact_dir;
use forge_guardrails::{
    ClassifierModelKind, SchemaCompressionMode, ScorerMode, ToolCallPolicyConfig,
    ToolCallPolicyMode, ToolOutputCompressionConfig, ToolOutputCompressionMethod,
    ToolOutputCompressionMode,
};

pub(crate) const DEFAULT_PROXY_PORT: u16 = 8081;
pub(crate) const DEFAULT_BACKEND_PORT: u16 = 8080;
pub(crate) const DEFAULT_ENV_CONTEXT_TOKENS: i64 = 128_000;
pub(crate) const DEFAULT_EXTERNAL_CONTEXT_TOKENS: i64 = 8192;
pub(crate) const DEFAULT_ENV_HOST: &str = "0.0.0.0";
pub(crate) const DEFAULT_CLI_HOST: &str = "127.0.0.1";
pub(crate) const DEFAULT_INTERNAL_MODEL: &str = "forge-guardrails-unset";
pub(crate) const DEFAULT_MAX_RETRIES: i32 = 3;

mod env_helpers;
mod validation;

#[cfg(test)]
mod tests;

use env_helpers::{
    env_bool, env_classifier_model, env_final_response_classifier_model, env_first_string, env_i32,
    env_i64, env_optional_string, env_optional_u64, env_redact_secrets, env_schema_compression,
    env_scoring_mode, env_string, env_tool_call_policy, env_tool_output_compression,
};

#[cfg(test)]
pub(super) use env_helpers::{
    redact_secrets_from_env_value, tool_output_compression_from_env_values,
};

pub(crate) use validation::{
    cli_host, cli_max_retries, cli_model, cli_port, normalized_extra_flags, require_cli_gguf,
    require_cli_llamafile_runtime, require_cli_model, resolve_serialize, validate_nonempty,
    validate_nonnegative_i32, validate_nonzero_u16, validate_optional_positive_i64,
    validate_positive_i64,
};

#[derive(Clone)]
pub(crate) struct ProxyConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) default_model: String,
    pub(crate) default_model_explicit: bool,
    pub(crate) context_tokens: i64,
    pub(crate) max_retries: i32,
    pub(crate) rescue_enabled: bool,
    pub(crate) serialize_requests: bool,
    pub(crate) verbose: bool,
    pub(crate) classifier_dir: Option<String>,
    pub(crate) classifier_mode: ScorerMode,
    pub(crate) classifier_model: ClassifierModelKind,
    pub(crate) classifier_auto_download: bool,
    pub(crate) classifier_max_latency_ms: Option<u64>,
    pub(crate) final_response_classifier_dir: Option<String>,
    pub(crate) final_response_classifier_mode: ScorerMode,
    pub(crate) final_response_classifier_model: ClassifierModelKind,
    pub(crate) final_response_classifier_max_latency_ms: Option<u64>,
    pub(crate) tool_output_compression: ToolOutputCompressionConfig,
    pub(crate) tool_call_policy: ToolCallPolicyConfig,
    pub(crate) schema_compression: SchemaCompressionMode,
    pub(crate) redact_secrets: bool,
}

impl ProxyConfig {
    pub(crate) fn from_env() -> Result<Self, String> {
        let env_model = env_first_string(&["FORGE_MODEL", "SMALL_MODEL"]);
        Ok(Self {
            host: env_string(&["FORGE_HOST"], DEFAULT_ENV_HOST),
            port: env_helpers::env_u16(
                &["FORGE_PORT", "PORT", "LISTEN_PORT"],
                DEFAULT_PROXY_PORT,
                "FORGE_PORT",
            )?,
            default_model: env_model
                .clone()
                .unwrap_or_else(|| DEFAULT_INTERNAL_MODEL.to_string()),
            default_model_explicit: env_model.is_some(),
            context_tokens: env_i64(
                &["FORGE_CONTEXT_TOKENS"],
                DEFAULT_ENV_CONTEXT_TOKENS,
                "FORGE_CONTEXT_TOKENS",
            )?,
            max_retries: env_i32(
                &["FORGE_MAX_RETRIES"],
                DEFAULT_MAX_RETRIES,
                "FORGE_MAX_RETRIES",
            )?,
            rescue_enabled: env_bool("FORGE_RESCUE_ENABLED", true)?,
            serialize_requests: env_bool("FORGE_SERIALIZE_REQUESTS", false)?,
            verbose: false,
            classifier_dir: env_optional_string("FORGE_CLASSIFIER_DIR"),
            classifier_mode: env_scoring_mode("FORGE_CLASSIFIER_MODE", ScorerMode::Shadow)?,
            classifier_model: env_classifier_model()?,
            classifier_auto_download: false,
            classifier_max_latency_ms: env_optional_u64("FORGE_CLASSIFIER_MAX_LATENCY_MS")?,
            final_response_classifier_dir: env_optional_string(
                "FORGE_FINAL_RESPONSE_CLASSIFIER_DIR",
            ),
            final_response_classifier_mode: env_scoring_mode(
                "FORGE_FINAL_RESPONSE_CLASSIFIER_MODE",
                ScorerMode::Shadow,
            )?,
            final_response_classifier_model: env_final_response_classifier_model()?,
            final_response_classifier_max_latency_ms: env_optional_u64(
                "FORGE_FINAL_RESPONSE_CLASSIFIER_MAX_LATENCY_MS",
            )?,
            tool_output_compression: env_tool_output_compression()?,
            tool_call_policy: env_tool_call_policy()?,
            schema_compression: env_schema_compression()?,
            redact_secrets: env_redact_secrets()?,
        })
    }
}

pub(crate) fn apply_env_cli_overrides(config: &mut ProxyConfig, cli: &Cli) -> Result<(), String> {
    if let Some(host) = cli.host.as_deref() {
        config.host = validate_nonempty(host, "--host")?.to_string();
    }
    if let Some(port) = cli.port {
        config.port = validate_nonzero_u16(port, "--port")?;
    }
    if let Some(model) = cli.model.as_deref() {
        config.default_model = validate_nonempty(model, "--model")?.to_string();
        config.default_model_explicit = true;
    }
    if let Some(tokens) = cli.budget_tokens {
        config.context_tokens = validate_positive_i64(tokens, "--budget-tokens")?;
    }
    if let Some(max_retries) = cli.max_retries {
        config.max_retries = validate_nonnegative_i32(max_retries, "--max-retries")?;
    }
    if cli.no_rescue {
        config.rescue_enabled = false;
    }
    if cli.serialize {
        config.serialize_requests = true;
    } else if cli.no_serialize {
        config.serialize_requests = false;
    }
    if cli.verbose {
        config.verbose = true;
    }
    let (classifier_dir, classifier_mode, classifier_model, classifier_auto_download) =
        classifier_settings_from_env_cli(cli)?;
    config.classifier_dir = classifier_dir;
    config.classifier_mode = classifier_mode;
    config.classifier_model = classifier_model;
    config.classifier_auto_download = classifier_auto_download;
    if let Some(value) = cli.classifier_max_latency_ms {
        config.classifier_max_latency_ms = Some(value);
    }
    if let Some(dir) = cli.final_response_classifier_dir.as_deref() {
        config.final_response_classifier_dir =
            Some(validate_nonempty(dir, "--final-response-classifier-dir")?.to_string());
    }
    if let Some(mode) = cli.final_response_classifier_mode.as_deref() {
        config.final_response_classifier_mode = ScorerMode::from_str(mode)?;
    }
    if let Some(model) = cli.final_response_classifier_model.as_deref() {
        config.final_response_classifier_model = ClassifierModelKind::from_str(model)?;
    }
    if let Some(value) = cli.final_response_classifier_max_latency_ms {
        config.final_response_classifier_max_latency_ms = Some(value);
    }
    config.tool_output_compression = tool_output_compression_from_env_cli(cli)?;
    config.tool_call_policy = tool_call_policy_from_env_cli(cli)?;
    config.schema_compression = schema_compression_from_env_cli(cli)?;
    config.redact_secrets = redact_secrets_from_env_cli(cli)?;
    Ok(())
}

pub(crate) fn tool_output_compression_from_env_cli(
    cli: &Cli,
) -> Result<ToolOutputCompressionConfig, String> {
    let mut config = env_tool_output_compression()?;
    apply_tool_output_compression_cli_overrides(&mut config, cli)?;
    Ok(config)
}

pub(super) fn apply_tool_output_compression_cli_overrides(
    config: &mut ToolOutputCompressionConfig,
    cli: &Cli,
) -> Result<(), String> {
    if let Some(mode) = cli.tool_output_compression.as_deref() {
        config.mode = ToolOutputCompressionMode::from_str(mode)?;
    }
    if let Some(method) = cli.tool_output_compression_method.as_deref() {
        config.method = ToolOutputCompressionMethod::from_str(method)?;
    }
    Ok(())
}

pub(crate) fn tool_call_policy_from_env_cli(cli: &Cli) -> Result<ToolCallPolicyConfig, String> {
    let mut config = env_tool_call_policy()?;
    if let Some(mode) = cli.tool_call_policy.as_deref() {
        config = ToolCallPolicyConfig::from_mode(ToolCallPolicyMode::from_str(mode)?);
    }
    Ok(config)
}

pub(crate) fn schema_compression_from_env_cli(cli: &Cli) -> Result<SchemaCompressionMode, String> {
    let mut mode = env_schema_compression()?;
    if let Some(s) = cli.schema_compression.as_deref() {
        mode = SchemaCompressionMode::from_str(s)?;
    }
    Ok(mode)
}

pub(crate) fn redact_secrets_from_env_cli(cli: &Cli) -> Result<bool, String> {
    let enabled = env_redact_secrets()? || cli.redact_secrets;
    if enabled && !cfg!(feature = "secrets-scanner") {
        return Err(
            "--redact-secrets/FORGE_REDACT_SECRETS requires building with the `secrets-scanner` feature"
                .to_string(),
        );
    }
    Ok(enabled)
}

pub(crate) fn classifier_settings_from_env_cli(
    cli: &Cli,
) -> Result<(Option<String>, ScorerMode, ClassifierModelKind, bool), String> {
    if cli.classify {
        #[cfg(not(feature = "classifier"))]
        {
            return Err("--classify requires building with --features classifier".to_string());
        }

        #[cfg(feature = "classifier")]
        {
            let dir = match cli.classifier_dir.as_deref() {
                Some(raw) => validate_nonempty(raw, "--classifier-dir")?.to_string(),
                None => default_tool_call_classifier_artifact_dir()
                    .map_err(|err| err.to_string())?
                    .to_string_lossy()
                    .into_owned(),
            };
            let mode = match cli.classifier_mode.as_deref() {
                Some(raw) => ScorerMode::from_str(raw)?,
                None => ScorerMode::Advisory,
            };
            if mode == ScorerMode::Disabled {
                return Err(
                    "--classify cannot be combined with --classifier-mode disabled".to_string(),
                );
            }
            let model = match cli.classifier_model.as_deref() {
                Some(raw) => ClassifierModelKind::from_str(raw)?,
                None => ClassifierModelKind::Quantized,
            };
            return Ok((Some(dir), mode, model, true));
        }
    }

    let mut dir = env_optional_string("FORGE_CLASSIFIER_DIR");
    let mut mode = env_scoring_mode("FORGE_CLASSIFIER_MODE", ScorerMode::Shadow)?;
    let mut model = env_classifier_model()?;

    if let Some(raw) = cli.classifier_dir.as_deref() {
        dir = Some(validate_nonempty(raw, "--classifier-dir")?.to_string());
    }
    if let Some(raw) = cli.classifier_mode.as_deref() {
        mode = ScorerMode::from_str(raw)?;
    }
    if let Some(raw) = cli.classifier_model.as_deref() {
        model = ClassifierModelKind::from_str(raw)?;
    }

    Ok((dir, mode, model, false))
}

pub(crate) fn final_response_classifier_settings_from_env_cli(
    cli: &Cli,
) -> Result<(Option<String>, ScorerMode, ClassifierModelKind), String> {
    let mut dir = env_optional_string("FORGE_FINAL_RESPONSE_CLASSIFIER_DIR");
    let mut mode = env_scoring_mode("FORGE_FINAL_RESPONSE_CLASSIFIER_MODE", ScorerMode::Shadow)?;
    let mut model = env_final_response_classifier_model()?;

    if let Some(raw) = cli.final_response_classifier_dir.as_deref() {
        dir = Some(validate_nonempty(raw, "--final-response-classifier-dir")?.to_string());
    }
    if let Some(raw) = cli.final_response_classifier_mode.as_deref() {
        mode = ScorerMode::from_str(raw)?;
    }
    if let Some(raw) = cli.final_response_classifier_model.as_deref() {
        model = ClassifierModelKind::from_str(raw)?;
    }

    Ok((dir, mode, model))
}
