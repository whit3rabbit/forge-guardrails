//! Secret redaction for proxy-bound request inputs.
//!
//! This module redacts text that is about to be sent to an upstream LLM. It
//! intentionally does not redact response bodies returned from the LLM.

use serde_json::Value;

/// Fixed marker used when a detected secret is replaced.
pub const SECRET_REDACTION_MARKER: &str = "[REDACTED_SECRET]";

/// Summary of a proxy request redaction pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SecretRedactionSummary {
    /// Number of string fields inspected.
    pub fields_scanned: usize,
    /// Number of string fields whose content changed.
    pub fields_redacted: usize,
    /// Number of findings returned by the scanner.
    pub findings: usize,
    /// Number of string fields where scanner finding caps truncated findings.
    pub findings_truncated: usize,
}

/// Error returned when proxy-bound input redaction cannot safely complete.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SecretRedactionError {
    /// The binary was built without the default `secrets-scanner` feature.
    #[error("secret redaction requires building with the `secrets-scanner` feature")]
    Unavailable,
    /// The bundled scanner could not be initialized.
    #[error("failed to initialize secret scanner: {0}")]
    ScannerInit(String),
    /// Input exceeded the scanner's fail-closed proxy size limit.
    #[error("secret redaction input too large: {size} bytes exceeds max {max}")]
    InputTooLarge {
        /// Size of the rejected input in bytes.
        size: usize,
        /// Configured maximum input size in bytes.
        max: u64,
    },
    /// The scanner rejected its configuration or output conversion failed.
    #[error("secret redaction failed: {0}")]
    Internal(String),
}

impl SecretRedactionError {
    /// HTTP status code to use when redaction happens at the route boundary.
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InputTooLarge { .. } => 413,
            Self::Unavailable | Self::ScannerInit(_) | Self::Internal(_) => 500,
        }
    }
}

/// Redact secret-looking text in proxy-bound request inputs.
///
/// The function mutates only text that is sent as model input: message content,
/// tool-result content, Anthropic system text, and prior assistant tool-call
/// argument values. It preserves model names, roles, IDs, tool names, and tool
/// schemas.
pub fn redact_proxy_request_inputs(
    body: &mut Value,
) -> Result<SecretRedactionSummary, SecretRedactionError> {
    let mut summary = SecretRedactionSummary::default();
    redact_system(body, &mut summary)?;
    redact_messages(body, &mut summary)?;
    Ok(summary)
}

fn redact_system(
    body: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    let Some(system) = body.get_mut("system") else {
        return Ok(());
    };
    match system {
        Value::String(_) => redact_string_value(system, summary),
        Value::Array(blocks) => {
            for block in blocks {
                redact_content_block(block, summary)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn redact_messages(
    body: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for message in messages {
        let Some(obj) = message.as_object_mut() else {
            continue;
        };
        if let Some(content) = obj.get_mut("content") {
            redact_content_value(content, summary)?;
        }
        if let Some(tool_calls) = obj.get_mut("tool_calls").and_then(Value::as_array_mut) {
            redact_openai_tool_calls(tool_calls, summary)?;
        }
    }
    Ok(())
}

fn redact_content_value(
    value: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    match value {
        Value::String(_) => redact_string_value(value, summary),
        Value::Array(blocks) => {
            for block in blocks {
                match block {
                    Value::String(_) => redact_string_value(block, summary)?,
                    Value::Object(_) => redact_content_block(block, summary)?,
                    _ => {}
                }
            }
            Ok(())
        }
        Value::Object(_) => redact_content_block(value, summary),
        _ => Ok(()),
    }
}

fn redact_content_block(
    block: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    let Some(obj) = block.as_object_mut() else {
        return Ok(());
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("text") | Some("input_text") => {
            if let Some(text) = obj.get_mut("text") {
                redact_string_value(text, summary)?;
            }
        }
        Some("tool_result") => {
            if let Some(content) = obj.get_mut("content") {
                redact_content_value(content, summary)?;
            }
        }
        Some("tool_use") => {
            if let Some(input) = obj.get_mut("input") {
                redact_json_string_values(input, summary)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn redact_openai_tool_calls(
    tool_calls: &mut [Value],
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    for call in tool_calls {
        let Some(function) = call.get_mut("function").and_then(Value::as_object_mut) else {
            continue;
        };
        let Some(arguments) = function.get_mut("arguments") else {
            continue;
        };
        redact_tool_arguments(arguments, summary)?;
    }
    Ok(())
}

fn redact_tool_arguments(
    arguments: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    match arguments {
        Value::String(raw) => {
            if let Ok(mut parsed) = serde_json::from_str::<Value>(raw) {
                let fields_redacted_before = summary.fields_redacted;
                redact_json_string_values(&mut parsed, summary)?;
                if summary.fields_redacted != fields_redacted_before {
                    *raw = serde_json::to_string(&parsed)
                        .map_err(|err| SecretRedactionError::Internal(err.to_string()))?;
                }
                Ok(())
            } else {
                let redacted = redact_text(raw, summary)?;
                if redacted != *raw {
                    *raw = redacted;
                }
                Ok(())
            }
        }
        Value::Object(_) | Value::Array(_) => redact_json_string_values(arguments, summary),
        _ => Ok(()),
    }
}

fn redact_json_string_values(
    value: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    match value {
        Value::String(_) => redact_string_value(value, summary),
        Value::Array(items) => {
            for item in items {
                redact_json_string_values(item, summary)?;
            }
            Ok(())
        }
        Value::Object(obj) => {
            for value in obj.values_mut() {
                redact_json_string_values(value, summary)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn redact_string_value(
    value: &mut Value,
    summary: &mut SecretRedactionSummary,
) -> Result<(), SecretRedactionError> {
    let Value::String(text) = value else {
        return Ok(());
    };
    let redacted = redact_text(text, summary)?;
    if redacted != *text {
        *text = redacted;
    }
    Ok(())
}

#[cfg(feature = "secrets-scanner")]
fn redact_text(
    text: &str,
    summary: &mut SecretRedactionSummary,
) -> Result<String, SecretRedactionError> {
    use secrets_scanner::{ProxyError, ScanConfig, Scanner};
    use std::sync::LazyLock;

    static SCANNER: LazyLock<Result<Scanner, String>> = LazyLock::new(|| {
        Scanner::from_bundled()
            .map(|scanner| scanner.with_config(ScanConfig::proxy()))
            .map_err(|err| err.to_string())
    });

    summary.fields_scanned += 1;
    let scanner = SCANNER
        .as_ref()
        .map_err(|err| SecretRedactionError::ScannerInit(err.clone()))?;
    let output = scanner
        .scan_proxy(text.as_bytes())
        .map_err(|err| match err {
            ProxyError::InputTooLarge { size, max } => {
                SecretRedactionError::InputTooLarge { size, max }
            }
            ProxyError::NotHardened => SecretRedactionError::Internal(err.to_string()),
        })?;
    summary.findings += output.findings.len();
    if output.findings_truncated {
        summary.findings_truncated += 1;
    }
    let redacted = String::from_utf8(output.redacted)
        .map_err(|err| SecretRedactionError::Internal(err.to_string()))?;
    if redacted != text {
        summary.fields_redacted += 1;
    }
    Ok(redacted)
}

#[cfg(not(feature = "secrets-scanner"))]
fn redact_text(
    _text: &str,
    _summary: &mut SecretRedactionSummary,
) -> Result<String, SecretRedactionError> {
    Err(SecretRedactionError::Unavailable)
}

#[cfg(feature = "secrets-scanner")]
pub(crate) fn redact_text_best_effort(text: &str) -> String {
    let mut summary = SecretRedactionSummary::default();
    redact_text(text, &mut summary)
        .map(|redacted| redact_private_key_blocks(&redacted))
        .unwrap_or_else(|_| legacy_redact_secrets(text))
}

#[cfg(not(feature = "secrets-scanner"))]
pub(crate) fn redact_text_best_effort(text: &str) -> String {
    legacy_redact_secrets(text)
}

pub(crate) fn legacy_redact_secrets(output: &str) -> String {
    let mut redacted = output.to_string();
    for pattern in legacy_secret_patterns() {
        redacted = pattern
            .replace_all(&redacted, SECRET_REDACTION_MARKER)
            .to_string();
    }
    redact_private_key_blocks(&redacted)
}

fn legacy_secret_patterns() -> &'static [regex_lite::Regex] {
    use std::sync::LazyLock;

    static SECRET_PATTERNS: LazyLock<Vec<regex_lite::Regex>> = LazyLock::new(|| {
        [
            r#"sk-[A-Za-z0-9_-]{20,}"#,
            r#"gh[pousr]_[A-Za-z0-9_]{20,}"#,
            r#"github_pat_[A-Za-z0-9_]{20,}"#,
            r#"xox[abprs]-[A-Za-z0-9-]{10,}"#,
            r#"AKIA[0-9A-Z]{16}"#,
            r#"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*["']?[^"'\s]{8,}"#,
            r#"(?i)(postgres|mysql|mongodb|redis)://[^ \n\r\t]+"#,
        ]
        .iter()
        .map(|pattern| regex_lite::Regex::new(pattern).expect("valid secret regex"))
        .collect()
    });

    SECRET_PATTERNS.as_slice()
}

fn redact_private_key_blocks(output: &str) -> String {
    let mut result = Vec::new();
    let mut in_private_key = false;
    for line in output.lines() {
        if line.contains("-----BEGIN") && line.contains("PRIVATE KEY-----") {
            if !in_private_key {
                result.push("[REDACTED_PRIVATE_KEY]".to_string());
            }
            in_private_key = !line.contains("-----END");
            continue;
        }
        if in_private_key {
            if line.contains("-----END") && line.contains("PRIVATE KEY-----") {
                in_private_key = false;
            }
            continue;
        }
        result.push(line.to_string());
    }
    preserve_trailing_newline(output, result.join("\n"))
}

fn preserve_trailing_newline(original: &str, mut output: String) -> String {
    if original.ends_with('\n') && !output.ends_with('\n') {
        output.push('\n');
    }
    output
}
