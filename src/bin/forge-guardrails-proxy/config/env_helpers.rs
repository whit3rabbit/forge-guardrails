use std::env;
use std::str::FromStr;

use forge_guardrails::{
    ClassifierModelKind, SchemaCompressionMode, ScorerMode, ToolCallPolicyConfig,
    ToolCallPolicyMode, ToolOutputCompressionConfig, ToolOutputCompressionMethod,
    ToolOutputCompressionMode,
};

pub(crate) fn env_string(keys: &[&str], default: &str) -> String {
    env_first_string(keys).unwrap_or_else(|| default.to_string())
}

pub(crate) fn env_first_string(keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| env::var(key).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(crate) fn env_tool_output_compression() -> Result<ToolOutputCompressionConfig, String> {
    tool_output_compression_from_env_values(
        env_optional_string("FORGE_TOOL_OUTPUT_COMPRESSION"),
        env_optional_string("FORGE_TOOL_OUTPUT_COMPRESSION_METHOD"),
    )
}

pub(crate) fn tool_output_compression_from_env_values(
    mode: Option<String>,
    method: Option<String>,
) -> Result<ToolOutputCompressionConfig, String> {
    let mut config = match mode {
        Some(mode) => {
            ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::from_str(&mode)?)
        }
        None => ToolOutputCompressionConfig::default(),
    };
    if let Some(method) = method {
        config.method = ToolOutputCompressionMethod::from_str(&method)?;
    }
    Ok(config)
}

pub(crate) fn env_tool_call_policy() -> Result<ToolCallPolicyConfig, String> {
    match env_optional_string("FORGE_TOOL_CALL_POLICY") {
        Some(mode) => Ok(ToolCallPolicyConfig::from_mode(
            ToolCallPolicyMode::from_str(&mode)?,
        )),
        None => Ok(ToolCallPolicyConfig::disabled()),
    }
}

pub(crate) fn env_schema_compression() -> Result<SchemaCompressionMode, String> {
    match env_optional_string("FORGE_SCHEMA_COMPRESSION") {
        Some(mode) => SchemaCompressionMode::from_str(&mode),
        None => Ok(SchemaCompressionMode::Disabled),
    }
}

pub(crate) fn env_redact_secrets() -> Result<bool, String> {
    redact_secrets_from_env_value(env_optional_string("FORGE_REDACT_SECRETS"))
}

pub(crate) fn redact_secrets_from_env_value(value: Option<String>) -> Result<bool, String> {
    parse_optional_bool("FORGE_REDACT_SECRETS", value, false)
}

pub(crate) fn env_optional_string(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(crate) fn env_scoring_mode(key: &str, default: ScorerMode) -> Result<ScorerMode, String> {
    match env_optional_string(key) {
        Some(raw) => ScorerMode::from_str(&raw),
        None => Ok(default),
    }
}

pub(crate) fn env_classifier_model() -> Result<ClassifierModelKind, String> {
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

pub(crate) fn env_final_response_classifier_model() -> Result<ClassifierModelKind, String> {
    if let Some(raw) = env_optional_string("FORGE_FINAL_RESPONSE_CLASSIFIER_MODEL") {
        return ClassifierModelKind::from_str(&raw);
    }
    Ok(ClassifierModelKind::Quantized)
}

pub(crate) fn env_optional_u64(key: &str) -> Result<Option<u64>, String> {
    let Some(raw) = env_optional_string(key) else {
        return Ok(None);
    };
    raw.parse::<u64>()
        .map(Some)
        .map_err(|_| format!("{key} must be a non-negative integer, got '{raw}'"))
}

pub(crate) fn env_u16(keys: &[&str], default: u16, label: &str) -> Result<u16, String> {
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

pub(crate) fn env_i64(keys: &[&str], default: i64, label: &str) -> Result<i64, String> {
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

pub(crate) fn env_i32(keys: &[&str], default: i32, label: &str) -> Result<i32, String> {
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

pub(crate) fn env_bool(key: &str, default: bool) -> Result<bool, String> {
    parse_optional_bool(key, env::var(key).ok(), default)
}

fn parse_optional_bool(key: &str, raw: Option<String>, default: bool) -> Result<bool, String> {
    match raw {
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!("{key} must be true or false, got '{raw}'")),
        },
        None => Ok(default),
    }
}
