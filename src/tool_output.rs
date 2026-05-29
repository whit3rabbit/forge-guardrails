//! Tool-output compression for proxy-forwarded tool result messages.
//!
//! This module intentionally compresses tool results, not tool calls. Tool-call
//! names and arguments remain API contracts owned by the client and guardrails.

use indexmap::IndexMap;
use regex_lite::Regex;
use serde_json::Value;
use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::{LazyLock, Mutex};

/// Default maximum tool-output bytes retained before safe capping.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
/// Default maximum dedup records kept per session.
pub const DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION: usize = 128;
/// Default maximum sessions kept in dedup memory.
pub const DEFAULT_MAX_DEDUP_SESSIONS: usize = 64;

/// Opt-in compression level for tool outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ToolOutputCompressionMode {
    /// No compression or mutation.
    #[default]
    Disabled,
    /// Safety-only transforms: redaction, ANSI stripping, binary suppression, capping.
    Safe,
    /// Safe transforms plus conservative structural and tool-family compaction.
    Standard,
    /// Standard transforms plus explicitly lossy/high-aggression transforms.
    Aggressive,
}

impl ToolOutputCompressionMode {
    /// Return the stable lowercase mode name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Safe => "safe",
            Self::Standard => "standard",
            Self::Aggressive => "aggressive",
        }
    }

    /// Return true if this mode can change tool output.
    pub fn enabled(self) -> bool {
        self != Self::Disabled
    }
}

impl fmt::Display for ToolOutputCompressionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ToolOutputCompressionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "disabled" | "off" | "false" | "none" => Ok(Self::Disabled),
            "safe" => Ok(Self::Safe),
            "standard" | "on" | "true" => Ok(Self::Standard),
            "aggressive" => Ok(Self::Aggressive),
            other => Err(format!(
                "tool output compression must be disabled, safe, standard, or aggressive, got '{other}'"
            )),
        }
    }
}

/// Configuration for one tool-output compression pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutputCompressionConfig {
    /// Compression mode.
    pub mode: ToolOutputCompressionMode,
    /// Whether secret-looking values are redacted before other transforms.
    pub redact_secrets: bool,
    /// Whether repeated compressed outputs are replaced with bounded references.
    pub enable_dedup: bool,
    /// Optional client/session key used for dedup.
    pub session_id: Option<String>,
    /// Maximum bytes retained before safe capping.
    pub max_output_bytes: usize,
    /// Maximum dedup records per session.
    pub max_dedup_entries_per_session: usize,
    /// Maximum dedup sessions retained.
    pub max_dedup_sessions: usize,
}

impl Default for ToolOutputCompressionConfig {
    fn default() -> Self {
        Self {
            mode: ToolOutputCompressionMode::Disabled,
            redact_secrets: true,
            enable_dedup: true,
            session_id: None,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_dedup_entries_per_session: DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION,
            max_dedup_sessions: DEFAULT_MAX_DEDUP_SESSIONS,
        }
    }
}

impl ToolOutputCompressionConfig {
    /// Return a disabled compression configuration.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Build a configuration from a mode with safe defaults.
    pub fn from_mode(mode: ToolOutputCompressionMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// Return true if this configuration can change tool output.
    pub fn enabled(&self) -> bool {
        self.mode.enabled()
    }
}

/// Internal metrics and output for a compression pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutputCompressionResult {
    /// Final output to send to the upstream model.
    pub output: String,
    /// Heuristic token estimate before compression.
    pub before_tokens: i64,
    /// Heuristic token estimate after compression.
    pub after_tokens: i64,
    /// Estimated tokens saved.
    pub saved_tokens: i64,
    /// Estimated percentage saved.
    pub saved_pct: i64,
    /// Canonical tool family: bash, read, grep, glob, or generic.
    pub canonical_tool: String,
    /// Content or command family used by filters.
    pub family: String,
    /// Compression mode used.
    pub mode: ToolOutputCompressionMode,
    /// Whether a secret redaction changed output.
    pub redacted: bool,
    /// Whether binary or oversized output was suppressed/capped.
    pub capped: bool,
    /// Whether output was replaced by a dedup reference.
    pub deduped: bool,
    /// Names of transforms that changed output.
    pub strategies: Vec<String>,
}

/// Bounded in-memory dedup state for compressed tool outputs.
#[derive(Debug, Default)]
pub struct ToolOutputCompressionState {
    inner: Mutex<DedupState>,
}

#[derive(Debug, Default)]
struct DedupState {
    sessions: IndexMap<String, VecDeque<DedupRecord>>,
    session_order: VecDeque<String>,
    next_call_index: u64,
}

#[derive(Debug, Clone)]
struct DedupRecord {
    hash: u64,
    tool_name: String,
    call_index: u64,
}

impl ToolOutputCompressionState {
    /// Create empty bounded compression state.
    pub fn new() -> Self {
        Self::default()
    }

    fn deduplicate(
        &self,
        session_id: &str,
        tool_name: &str,
        output: &str,
        max_sessions: usize,
        max_entries_per_session: usize,
    ) -> Option<String> {
        if session_id.is_empty() || output.is_empty() {
            return None;
        }
        let max_sessions = max_sessions.max(1);
        let max_entries_per_session = max_entries_per_session.max(1);
        let hash = hash_output(output);
        let mut state = self.inner.lock().expect("tool output dedup lock");

        if let Some(records) = state.sessions.get(session_id) {
            if let Some(record) = records.iter().find(|record| record.hash == hash) {
                return Some(format!(
                    "[Duplicate of call #{} ({}) - see earlier result]",
                    record.call_index, record.tool_name
                ));
            }
        }

        if !state.sessions.contains_key(session_id) {
            state.session_order.push_back(session_id.to_string());
            state
                .sessions
                .insert(session_id.to_string(), VecDeque::new());
        }

        state.next_call_index = state.next_call_index.saturating_add(1);
        let call_index = state.next_call_index;
        let records = state
            .sessions
            .get_mut(session_id)
            .expect("session inserted above");
        records.push_back(DedupRecord {
            hash,
            tool_name: tool_name.to_string(),
            call_index,
        });
        while records.len() > max_entries_per_session {
            records.pop_front();
        }

        while state.sessions.len() > max_sessions {
            let Some(oldest) = state.session_order.pop_front() else {
                break;
            };
            if oldest != session_id {
                state.sessions.shift_remove(&oldest);
            } else {
                state.session_order.push_back(oldest);
                break;
            }
        }

        None
    }
}

/// Compress one tool output using the requested mode and optional dedup state.
pub fn compress_tool_output(
    tool_name: &str,
    args: Option<&IndexMap<String, Value>>,
    raw_output: &str,
    config: &ToolOutputCompressionConfig,
    state: Option<&ToolOutputCompressionState>,
) -> ToolOutputCompressionResult {
    let canonical_tool = canonical_tool_name(tool_name);
    let command = command_from_args(args);
    let family = detect_family(&canonical_tool, command.as_deref().unwrap_or_default());
    let before_tokens = estimate_tokens(raw_output);

    if !config.enabled() {
        return compression_result(CompressionResultInput {
            output: raw_output.to_string(),
            before_tokens,
            canonical_tool,
            family,
            mode: config.mode,
            redacted: false,
            capped: false,
            deduped: false,
            strategies: Vec::new(),
        });
    }

    let mut output = raw_output.to_string();
    let mut redacted = false;
    let mut capped = false;
    let mut deduped = false;
    let mut strategies = Vec::new();

    let safe = apply_safe_filters(&output, config);
    if safe.output != output {
        output = safe.output;
        redacted = safe.redacted;
        capped = safe.capped;
        strategies.extend(safe.strategies);
    }

    if config.mode >= ToolOutputCompressionMode::Standard && !safe.binary_suppressed {
        output = apply_standard_filters(
            &canonical_tool,
            &family,
            command.as_deref().unwrap_or_default(),
            &output,
            config.max_output_bytes,
            &mut strategies,
        );
    }

    if config.mode >= ToolOutputCompressionMode::Aggressive && !safe.binary_suppressed {
        output = apply_aggressive_filters(&output, &mut strategies);
    }

    if config.enable_dedup {
        if let (Some(state), Some(session_id)) = (state, config.session_id.as_deref()) {
            if let Some(marker) = state.deduplicate(
                session_id,
                &canonical_tool,
                &output,
                config.max_dedup_sessions,
                config.max_dedup_entries_per_session,
            ) {
                if marker.len() < output.len() {
                    output = marker;
                    deduped = true;
                    strategies.push("dedup".to_string());
                }
            }
        }
    }

    compression_result(CompressionResultInput {
        output,
        before_tokens,
        canonical_tool,
        family,
        mode: config.mode,
        redacted,
        capped,
        deduped,
        strategies,
    })
}

/// Return Forge's canonical compression tool family for a client tool name.
pub fn canonical_tool_name(tool_name: &str) -> String {
    let normalized = tool_name.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "bash" | "shell" | "sh" | "run" | "run_command" | "execute" | "exec" | "terminal" => {
            "bash".to_string()
        }
        "read" | "read_file" | "file_read" | "view" | "open_file" => "read".to_string(),
        "grep" | "rg" | "ripgrep" | "search" | "search_files" | "find_in_files" => {
            "grep".to_string()
        }
        "glob" | "find_files" | "list_files" => "glob".to_string(),
        _ => "generic".to_string(),
    }
}

/// Detect a shell command family for bash-style output routing.
pub fn detect_family(canonical_tool: &str, command: &str) -> String {
    if canonical_tool != "bash" {
        return canonical_tool.to_string();
    }
    let first = command
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_start_matches("env");
    match first {
        "git" => "git".to_string(),
        "npm" | "pnpm" | "yarn" | "bun" => "npm".to_string(),
        "cargo" | "rustc" => "cargo".to_string(),
        "pytest" | "go" | "jest" | "vitest" => "test".to_string(),
        "docker" | "podman" => "docker".to_string(),
        "pip" | "pip3" | "uv" => "pip".to_string(),
        "make" => "make".to_string(),
        "ls" | "find" | "tree" | "du" => "fs".to_string(),
        "grep" | "rg" | "ag" => "grep".to_string(),
        "cat" | "sed" | "awk" => "read".to_string(),
        _ => "generic".to_string(),
    }
}

/// Estimate tokens with the same lightweight heuristic used for proxy metrics.
pub fn estimate_tokens(value: &str) -> i64 {
    if value.is_empty() {
        0
    } else {
        value.chars().count().div_ceil(4) as i64
    }
}

#[derive(Debug)]
struct SafeFilterResult {
    output: String,
    redacted: bool,
    capped: bool,
    binary_suppressed: bool,
    strategies: Vec<String>,
}

fn apply_safe_filters(raw_output: &str, config: &ToolOutputCompressionConfig) -> SafeFilterResult {
    let mut output = raw_output.to_string();
    let mut redacted = false;
    let mut capped = false;
    let mut binary_suppressed = false;
    let mut strategies = Vec::new();

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

    let stripped = strip_ansi(&output);
    if stripped != output {
        output = stripped;
        strategies.push("strip_ansi".to_string());
    }

    let stripped = strip_thinking_blocks(&output);
    if stripped != output {
        output = stripped;
        strategies.push("strip_thinking".to_string());
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

fn apply_standard_filters(
    canonical_tool: &str,
    family: &str,
    command: &str,
    output: &str,
    max_output_bytes: usize,
    strategies: &mut Vec<String>,
) -> String {
    let mut current = output.to_string();

    current = apply_if_smaller(current, "json_minify", strategies, minify_json);

    let routed = match canonical_tool {
        "bash" => filter_bash_output(family, command, &current, max_output_bytes),
        "read" => filter_read_output(command, &current),
        "grep" => filter_grep_output(&current),
        "glob" => filter_glob_output(&current),
        _ => filter_generic_output(&current, max_output_bytes),
    };
    current = apply_candidate_if_smaller(current, "tool_family_filter", strategies, routed);

    current = apply_if_smaller(
        current,
        "fold_repeated_lines",
        strategies,
        fold_repeated_lines,
    );
    apply_if_smaller(
        current,
        "normalize_whitespace",
        strategies,
        normalize_whitespace,
    )
}

fn apply_aggressive_filters(output: &str, strategies: &mut Vec<String>) -> String {
    let current = apply_if_smaller(
        output.to_string(),
        "normalize_dynamic_log_noise",
        strategies,
        normalize_dynamic_log_noise,
    );
    apply_if_smaller(current, "toon_table", strategies, json_array_to_table)
}

fn apply_if_smaller<F>(
    current: String,
    strategy: &str,
    strategies: &mut Vec<String>,
    transform: F,
) -> String
where
    F: FnOnce(&str) -> Option<String>,
{
    let Some(candidate) = transform(&current) else {
        return current;
    };
    apply_candidate_if_smaller(current, strategy, strategies, candidate)
}

fn apply_candidate_if_smaller(
    current: String,
    strategy: &str,
    strategies: &mut Vec<String>,
    candidate: String,
) -> String {
    if candidate.len() < current.len() {
        strategies.push(strategy.to_string());
        candidate
    } else {
        current
    }
}

struct CompressionResultInput {
    output: String,
    before_tokens: i64,
    canonical_tool: String,
    family: String,
    mode: ToolOutputCompressionMode,
    redacted: bool,
    capped: bool,
    deduped: bool,
    strategies: Vec<String>,
}

fn compression_result(input: CompressionResultInput) -> ToolOutputCompressionResult {
    let after_tokens = estimate_tokens(&input.output);
    let saved_tokens = input.before_tokens.saturating_sub(after_tokens);
    let saved_pct = if input.before_tokens > 0 {
        saved_tokens.saturating_mul(100) / input.before_tokens
    } else {
        0
    };
    ToolOutputCompressionResult {
        output: input.output,
        before_tokens: input.before_tokens,
        after_tokens,
        saved_tokens,
        saved_pct,
        canonical_tool: input.canonical_tool,
        family: input.family,
        mode: input.mode,
        redacted: input.redacted,
        capped: input.capped,
        deduped: input.deduped,
        strategies: input.strategies,
    }
}

fn command_from_args(args: Option<&IndexMap<String, Value>>) -> Option<String> {
    let args = args?;
    for key in [
        "command",
        "cmd",
        "shell_command",
        "query",
        "pattern",
        "path",
        "file",
    ] {
        if let Some(value) = args.get(key).and_then(Value::as_str) {
            return Some(value.to_string());
        }
    }
    None
}

fn hash_output(output: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    output.hash(&mut hasher);
    hasher.finish()
}

static SECRET_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r#"sk-[A-Za-z0-9_-]{20,}"#,
        r#"sk-ant-[A-Za-z0-9_-]{20,}"#,
        r#"gh[pousr]_[A-Za-z0-9_]{20,}"#,
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

static TIMESTAMP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\b\d{4}-\d{2}-\d{2}[T ][0-9:.]+Z?\b"#).expect("valid timestamp regex")
});

static HASH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\b[0-9a-f]{32,64}\b"#).expect("valid hash regex"));

fn redact_secrets(output: &str) -> String {
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
            in_private_key = true;
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

fn strip_thinking_blocks(output: &str) -> String {
    let mut result = Vec::new();
    let mut in_thinking = false;
    for line in output.lines() {
        let lower = line.trim().to_ascii_lowercase();
        if lower.starts_with("<think") || lower.starts_with("<thinking") {
            in_thinking = true;
            continue;
        }
        if in_thinking {
            if lower.contains("</think>") || lower.contains("</thinking>") {
                in_thinking = false;
            }
            continue;
        }
        result.push(line.to_string());
    }
    preserve_trailing_newline(output, result.join("\n"))
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

fn minify_json(output: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(output.trim()).ok()?;
    serde_json::to_string(&parsed).ok()
}

fn fold_repeated_lines(output: &str) -> Option<String> {
    let mut result = Vec::new();
    let mut lines = output.lines().peekable();
    let mut changed = false;
    while let Some(line) = lines.next() {
        let mut count = 1usize;
        while lines.peek().is_some_and(|next| *next == line) {
            lines.next();
            count += 1;
        }
        result.push(line.to_string());
        if count > 3 {
            result.push(format!("[repeated {} more times]", count - 1));
            changed = true;
        } else {
            for _ in 1..count {
                result.push(line.to_string());
            }
        }
    }
    changed.then(|| preserve_trailing_newline(output, result.join("\n")))
}

fn normalize_whitespace(output: &str) -> Option<String> {
    let mut result = Vec::new();
    let mut blank_count = 0usize;
    let mut changed = false;
    for line in output.lines() {
        let trimmed = line.trim_end();
        if trimmed.len() != line.len() {
            changed = true;
        }
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                result.push(String::new());
            } else {
                changed = true;
            }
        } else {
            blank_count = 0;
            result.push(trimmed.to_string());
        }
    }
    changed.then(|| preserve_trailing_newline(output, result.join("\n")))
}

fn filter_bash_output(
    family: &str,
    command: &str,
    output: &str,
    max_output_bytes: usize,
) -> String {
    if command.contains("git diff") {
        return filter_git_diff(output);
    }
    match family {
        "git" => filter_signal_lines(
            output,
            &[
                "modified:",
                "new file:",
                "deleted:",
                "renamed:",
                "fatal:",
                "error:",
                "warning:",
            ],
            240,
        ),
        "cargo" => filter_signal_lines(
            output,
            &[
                "error",
                "warning",
                "failed",
                "panicked",
                "test result",
                "running",
            ],
            260,
        ),
        "npm" | "test" => filter_signal_lines(
            output,
            &[
                "error", "warning", "failed", "failure", "passed", "tests", "summary",
            ],
            260,
        ),
        "docker" | "pip" | "make" => filter_signal_lines(
            output,
            &[
                "error",
                "warning",
                "failed",
                "success",
                "built",
                "installed",
            ],
            220,
        ),
        "fs" | "glob" => filter_glob_output(output),
        "grep" => filter_grep_output(output),
        "read" => filter_read_output(command, output),
        _ => filter_generic_output(output, max_output_bytes),
    }
}

fn filter_git_diff(output: &str) -> String {
    let mut kept = Vec::new();
    for line in output.lines() {
        if line.starts_with("diff --git")
            || line.starts_with("@@")
            || line.starts_with("+++")
            || line.starts_with("---")
            || (line.starts_with('+') && !line.starts_with("+++"))
            || (line.starts_with('-') && !line.starts_with("---"))
        {
            kept.push(line.to_string());
        }
        if kept.len() >= 400 {
            kept.push("[git diff output truncated]".to_string());
            break;
        }
    }
    if kept.is_empty() {
        output.to_string()
    } else {
        preserve_trailing_newline(output, kept.join("\n"))
    }
}

fn filter_signal_lines(output: &str, needles: &[&str], max_lines: usize) -> String {
    let mut kept = Vec::new();
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if needles.iter().any(|needle| lower.contains(needle)) {
            kept.push(line.to_string());
        }
        if kept.len() >= max_lines {
            kept.push("[output truncated to signal lines]".to_string());
            break;
        }
    }
    if kept.is_empty() {
        output.to_string()
    } else {
        preserve_trailing_newline(output, kept.join("\n"))
    }
}

fn filter_read_output(command: &str, output: &str) -> String {
    if output.len() <= 4096 || looks_like_config_path(command) {
        return output.to_string();
    }
    let mut kept = Vec::new();
    for (idx, line) in output.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("use ")
            || trimmed.starts_with("pub ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("export ")
        {
            kept.push(format!("{}: {}", idx + 1, trimmed));
        }
        if kept.len() >= 200 {
            kept.push("[source outline truncated]".to_string());
            break;
        }
    }
    if kept.is_empty() {
        output.to_string()
    } else {
        format!("[Source outline for {command}]\n{}", kept.join("\n"))
    }
}

fn looks_like_config_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [".json", ".toml", ".yaml", ".yml", ".lock", ".env"]
        .iter()
        .any(|suffix| lower.ends_with(suffix))
}

fn filter_grep_output(output: &str) -> String {
    let mut by_file: IndexMap<String, Vec<String>> = IndexMap::new();
    for line in output.lines() {
        if is_noise_path(line) {
            continue;
        }
        let mut parts = line.splitn(3, ':');
        let Some(path) = parts.next() else {
            continue;
        };
        let Some(line_no) = parts.next() else {
            continue;
        };
        let Some(match_text) = parts.next() else {
            continue;
        };
        if line_no.parse::<usize>().is_err() {
            continue;
        }
        let entries = by_file.entry(path.to_string()).or_default();
        if entries.len() < 5 {
            entries.push(format!("{line_no}:{match_text}"));
        }
    }
    if by_file.is_empty() {
        return output.to_string();
    }
    let mut result = Vec::new();
    for (path, matches) in by_file.iter().take(80) {
        result.push(format!("{path}:"));
        result.extend(matches.iter().map(|line| format!("  {line}")));
    }
    if by_file.len() > 80 {
        result.push(format!("[{} more files omitted]", by_file.len() - 80));
    }
    result.join("\n")
}

fn filter_glob_output(output: &str) -> String {
    let mut by_dir: IndexMap<String, usize> = IndexMap::new();
    let mut kept = Vec::new();
    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        let path = line.trim();
        if is_noise_path(path) {
            continue;
        }
        if kept.len() < 200 {
            kept.push(path.to_string());
        }
        let dir = path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or(".");
        *by_dir.entry(dir.to_string()).or_insert(0) += 1;
    }
    if kept.is_empty() {
        return output.to_string();
    }
    if kept.len() < output.lines().count() / 2 {
        let mut result = kept;
        result.push("[noise paths omitted]".to_string());
        return result.join("\n");
    }
    if by_dir.len() > 12 {
        let mut result = Vec::new();
        for (dir, count) in by_dir.iter().take(80) {
            result.push(format!("{dir}/ ({count} files)"));
        }
        if by_dir.len() > 80 {
            result.push(format!("[{} more directories omitted]", by_dir.len() - 80));
        }
        return result.join("\n");
    }
    output.to_string()
}

fn filter_generic_output(output: &str, max_output_bytes: usize) -> String {
    if output.len() <= max_output_bytes / 2 {
        return output.to_string();
    }
    cap_output(output, max_output_bytes / 2)
}

fn is_noise_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [
        "/node_modules/",
        "/.git/",
        "/target/",
        "/dist/",
        "/build/",
        "/.next/",
        "/vendor/",
    ]
    .iter()
    .any(|part| lower.contains(part))
        || lower.starts_with("node_modules/")
        || lower.starts_with("target/")
        || lower.starts_with(".git/")
}

fn normalize_dynamic_log_noise(output: &str) -> Option<String> {
    let timestamps = TIMESTAMP_RE.replace_all(output, "[timestamp]").to_string();
    let hashes = HASH_RE.replace_all(&timestamps, "[hash]").to_string();
    (hashes != output).then_some(hashes)
}

fn json_array_to_table(output: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(output.trim()).ok()?;
    let Value::Array(items) = parsed else {
        return None;
    };
    if items.len() < 2 {
        return None;
    }
    let first = items.first()?.as_object()?;
    let keys: Vec<&String> = first.keys().collect();
    if keys.is_empty() {
        return None;
    }
    let mut rows = Vec::new();
    for item in &items {
        let obj = item.as_object()?;
        if obj.len() != keys.len() || !keys.iter().all(|key| obj.contains_key(*key)) {
            return None;
        }
        let mut row = Vec::new();
        for key in &keys {
            let value = obj.get(*key)?;
            if matches!(value, Value::Array(_) | Value::Object(_)) {
                return None;
            }
            row.push(match value {
                Value::String(value) => value.clone(),
                _ => value.to_string(),
            });
        }
        rows.push(row.join("\t"));
    }
    let header = keys
        .iter()
        .map(|key| key.as_str())
        .collect::<Vec<_>>()
        .join("\t");
    Some(format!(
        "[{} rows]\n{}\n{}",
        items.len(),
        header,
        rows.join("\n")
    ))
}

fn preserve_trailing_newline(original: &str, mut value: String) -> String {
    if original.ends_with('\n') && !value.ends_with('\n') {
        value.push('\n');
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use serde_json::json;

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
    fn disabled_returns_original_output() {
        let result = compress_tool_output(
            "bash",
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
            "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
            &safe_config(),
            None,
        );
        assert!(result.redacted);
        assert!(result.output.contains("[REDACTED_SECRET]"));
        assert!(!result.output.contains("sk-abcdefghijklmnopqrstuvwxyz"));
    }

    #[test]
    fn safe_strips_ansi_sequences() {
        let result = compress_tool_output(
            "bash",
            None,
            "\u{1b}[31merror\u{1b}[0m",
            &safe_config(),
            None,
        );
        assert_eq!(result.output, "error");
        assert!(result.strategies.contains(&"strip_ansi".to_string()));
    }

    #[test]
    fn safe_suppresses_binary_output() {
        let result = compress_tool_output("bash", None, "abc\0def", &safe_config(), None);
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
        let result = compress_tool_output("bash", None, "a".repeat(200).as_str(), &config, None);
        assert!(result.capped);
        assert!(result.output.contains("Tool output capped"));
    }

    #[test]
    fn standard_minifies_json_when_smaller() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let result =
            compress_tool_output("read", None, "{\n  \"a\": 1,\n  \"b\": 2\n}", &config, None);
        assert_eq!(result.output, "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn standard_routes_grep_output_by_file() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let result = compress_tool_output(
            "search",
            None,
            "src/a.rs:10:fn alpha()\nsrc/a.rs:20:fn beta()\ntarget/x.rs:1:noise\n",
            &config,
            None,
        );
        assert!(result.output.contains("src/a.rs:"));
        assert!(!result.output.contains("target/x.rs"));
    }

    #[test]
    fn standard_detects_bash_family_from_command_args() {
        let mut args = IndexMap::new();
        args.insert("command".to_string(), json!("cargo test"));
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let result = compress_tool_output(
            "shell",
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
    fn standard_filter_does_not_grow_output() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = "short";
        let result = compress_tool_output("bash", None, raw, &config, None);
        assert_eq!(result.output, raw);
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
        let first = compress_tool_output("bash", None, &raw, &config, Some(&state));
        let second = compress_tool_output("bash", None, &raw, &config, Some(&state));
        assert!(!first.deduped);
        assert!(second.deduped);
        assert!(second.output.contains("Duplicate of call #"));
    }
}
