use std::env;
use std::str::FromStr;

use crate::cli::Cli;
use forge_guardrails::{ClassifierModelKind, ScorerMode};

pub(crate) const DEFAULT_PROXY_PORT: u16 = 8081;
pub(crate) const DEFAULT_BACKEND_PORT: u16 = 8080;
pub(crate) const DEFAULT_ENV_CONTEXT_TOKENS: i64 = 128_000;
pub(crate) const DEFAULT_EXTERNAL_CONTEXT_TOKENS: i64 = 8192;
pub(crate) const DEFAULT_ENV_HOST: &str = "0.0.0.0";
pub(crate) const DEFAULT_CLI_HOST: &str = "127.0.0.1";
pub(crate) const DEFAULT_ENV_MODEL: &str = "gpt-4o-mini";
pub(crate) const DEFAULT_EXTERNAL_MODEL: &str = "default";
pub(crate) const DEFAULT_MAX_RETRIES: i32 = 3;

#[derive(Clone)]
pub(crate) struct ProxyConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) default_model: String,
    pub(crate) context_tokens: i64,
    pub(crate) max_retries: i32,
    pub(crate) rescue_enabled: bool,
    pub(crate) serialize_requests: bool,
    pub(crate) verbose: bool,
    pub(crate) classifier_dir: Option<String>,
    pub(crate) classifier_mode: ScorerMode,
    pub(crate) classifier_model: ClassifierModelKind,
}

impl ProxyConfig {
    pub(crate) fn from_env() -> Result<Self, String> {
        Ok(Self {
            host: env_string(&["FORGE_HOST"], DEFAULT_ENV_HOST),
            port: env_u16(
                &["FORGE_PORT", "PORT", "LISTEN_PORT"],
                DEFAULT_PROXY_PORT,
                "FORGE_PORT",
            )?,
            default_model: env_string(&["FORGE_MODEL", "SMALL_MODEL"], DEFAULT_ENV_MODEL),
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
    if let Some(dir) = cli.classifier_dir.as_deref() {
        config.classifier_dir = Some(validate_nonempty(dir, "--classifier-dir")?.to_string());
    }
    if let Some(mode) = cli.classifier_mode.as_deref() {
        config.classifier_mode = ScorerMode::from_str(mode)?;
    }
    if let Some(model) = cli.classifier_model.as_deref() {
        config.classifier_model = ClassifierModelKind::from_str(model)?;
    }
    Ok(())
}

pub(crate) fn classifier_settings_from_env_cli(
    cli: &Cli,
) -> Result<(Option<String>, ScorerMode, ClassifierModelKind), String> {
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

    Ok((dir, mode, model))
}

pub(crate) fn cli_host(cli: &Cli) -> Result<String, String> {
    Ok(cli
        .host
        .as_deref()
        .map(|host| validate_nonempty(host, "--host").map(ToOwned::to_owned))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_CLI_HOST.to_string()))
}

pub(crate) fn cli_port(cli: &Cli) -> Result<u16, String> {
    validate_nonzero_u16(cli.port.unwrap_or(DEFAULT_PROXY_PORT), "--port")
}

pub(crate) fn cli_model(cli: &Cli, default: &str) -> Result<String, String> {
    Ok(cli
        .model
        .as_deref()
        .map(|model| validate_nonempty(model, "--model").map(ToOwned::to_owned))
        .transpose()?
        .unwrap_or_else(|| default.to_string()))
}

pub(crate) fn cli_max_retries(cli: &Cli) -> Result<i32, String> {
    validate_nonnegative_i32(
        cli.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
        "--max-retries",
    )
}

pub(crate) fn require_cli_model(cli: &Cli) -> Result<String, String> {
    cli.model
        .as_deref()
        .map(|model| validate_nonempty(model, "--model").map(ToOwned::to_owned))
        .transpose()?
        .ok_or_else(|| "--backend ollama requires --model".to_string())
}

pub(crate) fn require_cli_gguf(cli: &Cli, backend: &str) -> Result<String, String> {
    cli.gguf
        .as_deref()
        .map(|gguf| validate_nonempty(gguf, "--gguf").map(ToOwned::to_owned))
        .transpose()?
        .ok_or_else(|| format!("--backend {backend} requires --gguf"))
}

pub(crate) fn require_cli_llamafile_runtime(cli: &Cli) -> Result<String, String> {
    cli.llamafile_runtime
        .as_deref()
        .map(|runtime| validate_nonempty(runtime, "--llamafile-runtime").map(ToOwned::to_owned))
        .transpose()?
        .ok_or_else(|| "--backend llamafile requires --llamafile-runtime".to_string())
}

pub(crate) fn validate_nonempty<'a>(value: &'a str, label: &str) -> Result<&'a str, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{label} cannot be empty"))
    } else {
        Ok(trimmed)
    }
}

pub(crate) fn validate_nonzero_u16(value: u16, label: &str) -> Result<u16, String> {
    if value == 0 {
        Err(format!("{label} cannot be 0"))
    } else {
        Ok(value)
    }
}

pub(crate) fn validate_optional_positive_i64(
    value: Option<i64>,
    label: &str,
) -> Result<Option<i64>, String> {
    value
        .map(|tokens| validate_positive_i64(tokens, label))
        .transpose()
}

pub(crate) fn validate_positive_i64(value: i64, label: &str) -> Result<i64, String> {
    if value <= 0 {
        Err(format!("{label} must be positive"))
    } else {
        Ok(value)
    }
}

pub(crate) fn validate_nonnegative_i32(value: i32, label: &str) -> Result<i32, String> {
    if value < 0 {
        Err(format!("{label} must be non-negative"))
    } else {
        Ok(value)
    }
}

pub(crate) fn resolve_serialize(cli: &Cli, default: bool) -> bool {
    if cli.serialize {
        true
    } else if cli.no_serialize {
        false
    } else {
        default
    }
}

pub(crate) fn normalized_extra_flags(flags: &[String]) -> Vec<String> {
    if flags.first().is_some_and(|flag| flag == "--") {
        flags[1..].to_vec()
    } else {
        flags.to_vec()
    }
}

fn env_string(keys: &[&str], default: &str) -> String {
    keys.iter()
        .find_map(|key| env::var(key).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_optional_string(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_scoring_mode(key: &str, default: ScorerMode) -> Result<ScorerMode, String> {
    match env_optional_string(key) {
        Some(raw) => ScorerMode::from_str(&raw),
        None => Ok(default),
    }
}

fn env_classifier_model() -> Result<ClassifierModelKind, String> {
    if let Some(raw) = env_optional_string("FORGE_CLASSIFIER_MODEL") {
        return ClassifierModelKind::from_str(&raw);
    }
    match env::var("FORGE_CLASSIFIER_USE_QUANTIZED") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(ClassifierModelKind::Quantized),
            "0" | "false" | "no" | "off" => Ok(ClassifierModelKind::Full),
            _ => Err(format!(
                "FORGE_CLASSIFIER_USE_QUANTIZED must be true or false, got '{raw}'"
            )),
        },
        Err(_) => Ok(ClassifierModelKind::Quantized),
    }
}

fn env_u16(keys: &[&str], default: u16, label: &str) -> Result<u16, String> {
    match keys.iter().find_map(|key| env::var(key).ok()) {
        Some(raw) => {
            let value = raw
                .parse::<u16>()
                .map_err(|_| format!("{label} must be a number in 1-65535, got '{raw}'"))?;
            if value == 0 {
                return Err(format!("{label} cannot be 0"));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

fn env_i64(keys: &[&str], default: i64, label: &str) -> Result<i64, String> {
    match keys.iter().find_map(|key| env::var(key).ok()) {
        Some(raw) => {
            let value = raw
                .parse::<i64>()
                .map_err(|_| format!("{label} must be a positive integer, got '{raw}'"))?;
            if value <= 0 {
                return Err(format!("{label} must be positive"));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

fn env_i32(keys: &[&str], default: i32, label: &str) -> Result<i32, String> {
    match keys.iter().find_map(|key| env::var(key).ok()) {
        Some(raw) => {
            let value = raw
                .parse::<i32>()
                .map_err(|_| format!("{label} must be a non-negative integer, got '{raw}'"))?;
            if value < 0 {
                return Err(format!("{label} must be non-negative"));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

fn env_bool(key: &str, default: bool) -> Result<bool, String> {
    match env::var(key) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!("{key} must be true or false, got '{raw}'")),
        },
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
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
            default_model: DEFAULT_ENV_MODEL.to_string(),
            context_tokens: DEFAULT_ENV_CONTEXT_TOKENS,
            max_retries: DEFAULT_MAX_RETRIES,
            rescue_enabled: true,
            serialize_requests: false,
            verbose: false,
            classifier_dir: None,
            classifier_mode: ScorerMode::Shadow,
            classifier_model: ClassifierModelKind::Quantized,
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
        assert_eq!(config.context_tokens, 2048);
        assert_eq!(config.max_retries, 0);
        assert!(!config.rescue_enabled);
        assert!(config.serialize_requests);
        assert!(config.verbose);
        assert_eq!(config.classifier_dir, None);
        assert_eq!(config.classifier_mode, ScorerMode::Shadow);
        assert_eq!(config.classifier_model, ClassifierModelKind::Quantized);
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
}
