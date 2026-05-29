use super::{DEFAULT_CLI_HOST, DEFAULT_MAX_RETRIES, DEFAULT_PROXY_PORT};
use crate::cli::Cli;

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
