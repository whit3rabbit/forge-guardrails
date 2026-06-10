use super::{preserve_trailing_newline, ToolOutputCompressionConfig};
use regex_lite::Regex;
use std::sync::LazyLock;

#[derive(Debug)]
pub(super) struct SafeFilterResult {
    pub(super) output: String,
    pub(super) redacted: bool,
    pub(super) capped: bool,
    pub(super) binary_suppressed: bool,
    pub(super) strategies: Vec<String>,
}

pub(super) fn apply_safe_filters(
    raw_output: &str,
    config: &ToolOutputCompressionConfig,
) -> SafeFilterResult {
    let mut output = raw_output.to_string();
    let mut redacted = false;
    let mut capped = false;
    let mut binary_suppressed = false;
    let mut strategies = Vec::new();

    let stripped = strip_ansi(&output);
    if stripped != output {
        output = stripped;
        strategies.push("strip_ansi".to_string());
    }

    if config.redact_secrets {
        let redacted_output = redact_secrets(&output);
        if redacted_output != output {
            output = redacted_output;
            redacted = true;
            strategies.push("redact_secrets".to_string());
        }
    }

    if looks_binary(&output) {
        let bytes = output.len();
        output = format!("[Binary output suppressed: {bytes} bytes]");
        capped = true;
        binary_suppressed = true;
        strategies.push("binary_suppression".to_string());
        return SafeFilterResult {
            output,
            redacted,
            capped,
            binary_suppressed,
            strategies,
        };
    }

    let capped_output = cap_output(&output, config.max_output_bytes);
    if capped_output != output {
        output = capped_output;
        capped = true;
        strategies.push("cap_oversized".to_string());
    }

    SafeFilterResult {
        output,
        redacted,
        capped,
        binary_suppressed,
        strategies,
    }
}

static SECRET_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        // Also covers sk-ant- prefixed keys.
        r#"sk-[A-Za-z0-9_-]{20,}"#,
        r#"gh[pousr]_[A-Za-z0-9_]{20,}"#,
        r#"github_pat_[A-Za-z0-9_]{20,}"#,
        r#"xox[abprs]-[A-Za-z0-9-]{10,}"#,
        r#"AKIA[0-9A-Z]{16}"#,
        r#"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*["']?[^"'\s]{8,}"#,
        r#"(?i)(postgres|mysql|mongodb|redis)://[^ \n\r\t]+"#,
    ]
    .iter()
    .map(|pattern| Regex::new(pattern).expect("valid secret regex"))
    .collect()
});

static ANSI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\x1b\[[0-9;?]*[ -/]*[@-~]"#).expect("valid ansi regex"));

pub(crate) fn redact_secrets(output: &str) -> String {
    let mut redacted = output.to_string();
    for pattern in SECRET_PATTERNS.iter() {
        redacted = pattern
            .replace_all(&redacted, "[REDACTED_SECRET]")
            .to_string();
    }
    redact_private_key_blocks(&redacted)
}

fn redact_private_key_blocks(output: &str) -> String {
    let mut result = Vec::new();
    let mut in_private_key = false;
    for line in output.lines() {
        if line.contains("-----BEGIN") && line.contains("PRIVATE KEY-----") {
            if !in_private_key {
                result.push("[REDACTED_PRIVATE_KEY]".to_string());
            }
            // A single-line BEGIN...END key must not open a block that
            // swallows the rest of the output.
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

fn looks_binary(output: &str) -> bool {
    if output.contains('\0') {
        return true;
    }
    let control = output
        .chars()
        .filter(|ch| ch.is_control() && !matches!(ch, '\n' | '\r' | '\t' | '\x1b'))
        .count();
    let total = output.chars().count().max(1);
    total > 32 && control * 100 / total > 5
}

fn strip_ansi(output: &str) -> String {
    ANSI_RE.replace_all(output, "").to_string()
}

fn cap_output(output: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || output.len() <= max_bytes {
        return output.to_string();
    }
    let head_limit = max_bytes.saturating_mul(3) / 5;
    let tail_limit = max_bytes.saturating_sub(head_limit);
    let head = take_bytes_on_char_boundary(output, head_limit);
    let tail = take_last_bytes_on_char_boundary(output, tail_limit);
    let removed = output.len().saturating_sub(head.len() + tail.len());
    format!("{head}\n[Tool output capped: {removed} bytes removed]\n{tail}")
}

fn take_bytes_on_char_boundary(value: &str, limit: usize) -> String {
    let mut end = 0;
    for (idx, ch) in value.char_indices() {
        let next = idx + ch.len_utf8();
        if next > limit {
            break;
        }
        end = next;
    }
    value[..end].to_string()
}

fn take_last_bytes_on_char_boundary(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_string();
    }
    let target = value.len().saturating_sub(limit);
    let mut start = value.len();
    for (idx, _) in value.char_indices() {
        if idx >= target {
            start = idx;
            break;
        }
    }
    value[start..].to_string()
}
