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

    let cap_outcome = cap_output(&output, config.max_output_bytes);
    if cap_outcome.output != output {
        output = cap_outcome.output;
        capped = true;
        strategies.push("cap_oversized".to_string());
        if cap_outcome.error_window {
            strategies.push("cap_error_window".to_string());
        }
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

/// Error-signal lines scanned before giving up; bounds pathological inputs.
const ERROR_SIGNAL_SCAN_LIMIT: usize = 64;
/// Head/error-window split used when a signal would otherwise be dropped;
/// the tail keeps the remainder.
const ERROR_AWARE_HEAD_PCT: usize = 45;
const ERROR_AWARE_WINDOW_PCT: usize = 20;

struct CapOutcome {
    output: String,
    error_window: bool,
}

fn cap_output(output: &str, max_bytes: usize) -> CapOutcome {
    if max_bytes == 0 || output.len() <= max_bytes {
        return CapOutcome {
            output: output.to_string(),
            error_window: false,
        };
    }
    let head_limit = max_bytes.saturating_mul(3) / 5;
    let tail_limit = max_bytes.saturating_sub(head_limit);
    let tail_start = output.len().saturating_sub(tail_limit);

    if let Some(signal_start) = first_uncovered_error_signal(output, head_limit, tail_start) {
        if let Some(windowed) = cap_output_error_aware(output, max_bytes, signal_start) {
            return CapOutcome {
                output: windowed,
                error_window: true,
            };
        }
    }

    let head = take_bytes_on_char_boundary(output, head_limit);
    let tail = take_last_bytes_on_char_boundary(output, tail_limit);
    let removed = output.len().saturating_sub(head.len() + tail.len());
    CapOutcome {
        output: format!("{head}\n[Tool output capped: {removed} bytes removed]\n{tail}"),
        error_window: false,
    }
}

/// Cap with a third retained window anchored at the first error signal that
/// the plain head/tail split would drop. Windows that touch merge so at most
/// two cap markers are emitted.
fn cap_output_error_aware(output: &str, max_bytes: usize, signal_start: usize) -> Option<String> {
    let head_limit = max_bytes.saturating_mul(ERROR_AWARE_HEAD_PCT) / 100;
    let window_limit = max_bytes.saturating_mul(ERROR_AWARE_WINDOW_PCT) / 100;
    let tail_limit = max_bytes.saturating_sub(head_limit.saturating_add(window_limit));
    if window_limit == 0 {
        return None;
    }

    let head = take_bytes_on_char_boundary(output, head_limit);
    let tail = take_last_bytes_on_char_boundary(output, tail_limit);
    let tail_start = output.len() - tail.len();

    // Both bounds are char boundaries: head was boundary-trimmed and the
    // signal offset is a line start.
    let window_start = signal_start.max(head.len()).min(tail_start);
    let window = take_bytes_on_char_boundary(&output[window_start..tail_start], window_limit);
    if window.is_empty() {
        return None;
    }
    let window_end = window_start + window.len();

    let head_gap = window_start - head.len();
    let tail_gap = tail_start - window_end;
    let mut parts = vec![head];
    if head_gap > 0 {
        parts.push(format!("[Tool output capped: {head_gap} bytes removed]"));
    }
    parts.push(window.to_string());
    if tail_gap > 0 {
        parts.push(format!("[Tool output capped: {tail_gap} bytes removed]"));
    }
    parts.push(tail);
    Some(parts.join("\n"))
}

/// Byte offset of the first error-signal line not fully covered by the head
/// or tail retention windows, scanning at most `ERROR_SIGNAL_SCAN_LIMIT`
/// signal lines.
fn first_uncovered_error_signal(
    output: &str,
    head_limit: usize,
    tail_start: usize,
) -> Option<usize> {
    let mut signals_seen = 0usize;
    let mut offset = 0usize;
    for line in output.split_inclusive('\n') {
        let start = offset;
        offset += line.len();
        if signals_seen >= ERROR_SIGNAL_SCAN_LIMIT {
            return None;
        }
        let content = line.strip_suffix('\n').unwrap_or(line);
        if !is_error_signal_line(content) {
            continue;
        }
        signals_seen += 1;
        let covered_by_head = start + content.len() <= head_limit;
        let covered_by_tail = start >= tail_start;
        if !covered_by_head && !covered_by_tail {
            return Some(start);
        }
    }
    None
}

fn is_error_signal_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    has_ascii_ci_prefix(trimmed, "error")
        || has_ascii_ci_prefix(trimmed, "fatal:")
        || has_ascii_ci_prefix(trimmed, "panic")
        || line.contains("error:")
        || line.contains("panicked at")
        || line.contains("Traceback (most recent call last)")
        || line.contains("Exception")
        || line.contains("FAILED")
        || line.contains("assertion")
}

fn has_ascii_ci_prefix(value: &str, prefix: &str) -> bool {
    value.len() >= prefix.len()
        && value.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
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
