//! Tool-output compression for proxy-forwarded tool result messages.
//!
//! This module intentionally compresses tool results, not tool calls. Tool-call
//! names and arguments remain API contracts owned by the client and guardrails.

use indexmap::IndexMap;
use serde_json::Value;
use std::fmt;
use std::str::FromStr;

mod families;
mod filters;
mod lzw;
mod postcall;
mod repair;
mod safety;
mod state;

use families::{filter_bash_output, filter_generic_output};
use filters::{filter_glob_output, filter_grep_output, filter_read_output};
use lzw::compress_lzw_dictionary;
use postcall::{
    fold_repeated_lines, json_array_to_table, minify_json, minimize_table_whitespace,
    normalize_dynamic_log_noise, normalize_whitespace,
};
use repair::compress_repair_dictionary;
use safety::apply_safe_filters;
pub use state::ToolOutputCompressionState;

/// Default maximum tool-output bytes retained before safe capping.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
/// Default maximum dedup records kept per session.
pub const DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION: usize = 128;
/// Default maximum sessions kept in dedup memory.
pub const DEFAULT_MAX_DEDUP_SESSIONS: usize = 64;

const LZW_DICTIONARY_HEADER: &str = "[Forge LZW Dictionary]";
const REPAIR_DICTIONARY_HEADER: &str = "[Forge RePair Dictionary]";
const DICTIONARY_MAX_DICT_SIZE: usize = 20;
const DICTIONARY_MAX_INPUT_BYTES: usize = 50_000;
const DICTIONARY_MIN_OCCURRENCES: usize = 3;
const DICTIONARY_MIN_NET_SAVINGS_BYTES: usize = 32;
const DICTIONARY_MIN_NET_SAVINGS_PERCENT: usize = 3;

fn is_dictionary_compressed_output(output: &str) -> bool {
    output.starts_with(LZW_DICTIONARY_HEADER) || output.starts_with(REPAIR_DICTIONARY_HEADER)
}

fn dictionary_has_meaningful_savings(original_len: usize, savings: usize) -> bool {
    savings >= DICTIONARY_MIN_NET_SAVINGS_BYTES
        && savings.saturating_mul(100) / original_len.max(1) >= DICTIONARY_MIN_NET_SAVINGS_PERCENT
}

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

/// Dictionary algorithm used by aggressive tool-output compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolOutputCompressionMethod {
    /// LZW-style repeated substring dictionary compression.
    #[default]
    Lzw,
    /// RePair-style repeated adjacent token-pair grammar compression.
    Repair,
    /// Run bounded dictionary methods and keep the smallest valid result.
    Auto,
}

impl ToolOutputCompressionMethod {
    /// Return the stable lowercase method name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lzw => "lzw",
            Self::Repair => "repair",
            Self::Auto => "auto",
        }
    }
}

impl fmt::Display for ToolOutputCompressionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ToolOutputCompressionMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "lzw" => Ok(Self::Lzw),
            "repair" | "re-pair" => Ok(Self::Repair),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "tool output compression method must be lzw, repair, or auto, got '{other}'"
            )),
        }
    }
}

/// Configuration for one tool-output compression pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutputCompressionConfig {
    /// Compression mode.
    pub mode: ToolOutputCompressionMode,
    /// Dictionary method used by aggressive compression.
    pub method: ToolOutputCompressionMethod,
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
            method: ToolOutputCompressionMethod::Lzw,
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
        output = apply_aggressive_filters(&output, config.method, &mut strategies);
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
    let tokens = command_tokens(command);
    let Some(first) = tokens.first() else {
        return "generic".to_string();
    };
    let cmd = basename(first);

    if matches!(cmd.as_str(), "yarn" | "bun" | "pnpm") {
        return "npm".to_string();
    }
    if cmd == "npx" {
        if let Some(inner) = wrapper_inner_command(&tokens[1..]) {
            return family_for_command(&inner);
        }
    }
    if matches!(cmd.as_str(), "poetry" | "uv") {
        if let Some(run_idx) = tokens.iter().position(|token| token == "run") {
            if let Some(inner) = tokens.get(run_idx + 1) {
                return family_for_command(&basename(inner));
            }
        }
        if cmd == "uv" && tokens.get(1).is_some_and(|token| token == "pip") {
            return "pip".to_string();
        }
    }

    family_for_command(&cmd)
}

fn command_tokens(command: &str) -> Vec<String> {
    let mut tokens = command.split_whitespace();
    let mut result = Vec::new();
    while let Some(token) = tokens.next() {
        if token == "env" {
            continue;
        }
        if token.contains('=') && !token.starts_with('-') && result.is_empty() {
            continue;
        }
        result.push(token.to_string());
        result.extend(tokens.map(str::to_string));
        break;
    }
    result
}

fn basename(command: &str) -> String {
    command
        .rsplit('/')
        .next()
        .unwrap_or(command)
        .to_ascii_lowercase()
}

fn wrapper_inner_command(tokens: &[String]) -> Option<String> {
    let first = tokens.first()?;
    let inner = if matches!(first.as_str(), "run" | "exec") {
        tokens.get(1)?
    } else {
        first
    };
    Some(basename(inner))
}

fn family_for_command(command: &str) -> String {
    match command {
        "git" => "git",
        "npm" | "pnpm" | "yarn" | "bun" => "npm",
        "cargo" | "rustc" | "rustup" => "cargo",
        "pytest" | "py.test" | "jest" | "mocha" | "vitest" | "go" | "python" | "python3" => "test",
        "docker" | "podman" => "docker",
        "pip" | "pip3" | "pipx" => "pip",
        "make" | "cmake" => "make",
        "ls" | "find" | "tree" | "du" | "df" | "wc" | "sort" | "uniq" | "diff" => "fs",
        "grep" | "rg" | "ag" | "ack" => "grep",
        "cat" | "head" | "tail" | "sed" | "awk" => "read",
        _ => "generic",
    }
    .to_string()
}

/// Estimate tokens with the same lightweight heuristic used for proxy metrics.
pub fn estimate_tokens(value: &str) -> i64 {
    if value.is_empty() {
        0
    } else {
        value.chars().count().div_ceil(4) as i64
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
    current = apply_if_smaller(
        current,
        "minimize_table_whitespace",
        strategies,
        minimize_table_whitespace,
    );

    let routed = match canonical_tool {
        "bash" => filter_bash_output(family, command, &current, max_output_bytes),
        "read" => filter_read_output(command, &current),
        "grep" => filter_grep_output(&current),
        "glob" => filter_glob_output(&current),
        _ => filter_generic_output(&current),
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

fn apply_aggressive_filters(
    output: &str,
    method: ToolOutputCompressionMethod,
    strategies: &mut Vec<String>,
) -> String {
    let current = apply_if_smaller(
        output.to_string(),
        "normalize_dynamic_log_noise",
        strategies,
        normalize_dynamic_log_noise,
    );
    let current = apply_if_smaller(current, "toon_table", strategies, json_array_to_table);
    apply_dictionary_filter(current, method, strategies)
}

fn apply_dictionary_filter(
    current: String,
    method: ToolOutputCompressionMethod,
    strategies: &mut Vec<String>,
) -> String {
    match method {
        ToolOutputCompressionMethod::Lzw => apply_if_smaller(
            current,
            "lzw_dictionary",
            strategies,
            compress_lzw_dictionary,
        ),
        ToolOutputCompressionMethod::Repair => apply_if_smaller(
            current,
            "repair_dictionary",
            strategies,
            compress_repair_dictionary,
        ),
        ToolOutputCompressionMethod::Auto => {
            let lzw = compress_lzw_dictionary(&current);
            let repair = compress_repair_dictionary(&current);
            let best = [lzw, repair]
                .into_iter()
                .flatten()
                .filter(|candidate| candidate.len() < current.len())
                .min_by_key(String::len);

            match best {
                Some(candidate) => {
                    strategies.push("auto_dictionary".to_string());
                    candidate
                }
                None => current,
            }
        }
    }
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
    fn safe_redaction_can_be_disabled() {
        let config = ToolOutputCompressionConfig {
            mode: ToolOutputCompressionMode::Safe,
            redact_secrets: false,
            ..ToolOutputCompressionConfig::default()
        };
        let result = compress_tool_output(
            "bash",
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
            "\u{1b}[31merror\u{1b}[0m",
            &safe_config(),
            None,
        );
        assert_eq!(result.output, "error");
        assert!(result.strategies.contains(&"strip_ansi".to_string()));
    }

    #[test]
    fn safe_preserves_thinking_blocks_from_tool_output() {
        let raw = "visible before\n<thinking>\nprivate chain\n</thinking>\nvisible after\n";
        let result = compress_tool_output("bash", None, raw, &safe_config(), None);
        assert_eq!(result.output, raw);
        assert!(!result.strategies.contains(&"strip_thinking".to_string()));
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
    fn standard_grep_unknown_output_is_preserved() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = "rg: regex parse error:\n    (?:\n    ^\nerror: unclosed group\n";

        let result = compress_tool_output("grep", None, raw, &config, None);

        assert_eq!(result.output, raw);
        assert!(!result.output.contains("(no matches)"));
    }

    #[test]
    fn standard_glob_all_noise_paths_are_preserved() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = "target/debug/build.log\nnode_modules/pkg/index.js\n";

        let result = compress_tool_output("glob", None, raw, &config, None);

        assert_eq!(result.output, raw);
        assert!(!result.output.contains("(no matches)"));
    }

    #[test]
    fn standard_cargo_unknown_success_output_is_preserved() {
        let mut args = IndexMap::new();
        args.insert("command".to_string(), json!("cargo build"));
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = "Compiling demo v0.1.0\nFinished dev [unoptimized] target(s) in 0.1s\n";

        let result = compress_tool_output("bash", Some(&args), raw, &config, None);

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

        let result = compress_tool_output("bash", Some(&args), raw, &config, None);

        assert!(result.output.starts_with("Errors (1):\nerror[E0425]"));
        assert!(!result.output.contains("\"reason\""));
    }

    #[test]
    fn standard_test_unknown_success_output_is_preserved() {
        let mut args = IndexMap::new();
        args.insert("command".to_string(), json!("python custom_harness.py"));
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = "custom harness completed without standard summary\n";

        let result = compress_tool_output("bash", Some(&args), raw, &config, None);

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

        let result = compress_tool_output("read_file", Some(&args), &source, &config, None);

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

        let result = compress_tool_output("read_file", Some(&args), raw, &config, None);

        assert_eq!(result.output, raw);
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

        let result = compress_tool_output("bash", Some(&args), raw, &config, None);

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

        let result = compress_tool_output("glob", None, &raw, &config, None);

        assert!(result.output.contains("src/main.rs"));
        assert!(result.output.contains("src/lib.rs"));
        assert!(!result.output.contains("node_modules"));
        assert!(!result.output.contains("target/debug"));
    }

    #[test]
    fn opentoken_filter_fixture_cases_match_expected_outputs() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/parity/fixtures/opentoken_tool_output_filters.json"
        ))
        .expect("valid OpenToken tool-output fixture");
        let cases = fixture["cases"].as_array().expect("fixture cases");
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);

        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let tool = case["tool"].as_str().expect("case tool");
            let input = fixture_input(case);
            let args = fixture_args(case);
            let result = compress_tool_output(tool, args.as_ref(), &input, &config, None);
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
    fn standard_filter_does_not_grow_output() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = "short";
        let result = compress_tool_output("bash", None, raw, &config, None);
        assert_eq!(result.output, raw);
    }

    #[test]
    fn aggressive_normalizes_timestamps_and_hashes() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
        let raw = "2026-05-28T12:34:56Z completed artifact 0123456789abcdef0123456789abcdef\n";

        let result = compress_tool_output("bash", None, raw, &config, None);

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

        let result = compress_tool_output("bash", None, raw, &config, None);

        assert!(result.output.starts_with("[3 rows]\n"));
        assert!(result
            .output
            .contains("long_status_name\tlong_duration_ms\tlong_file_path"));
        assert!(result.output.contains("failed\t20\tsrc/b.rs"));
        assert!(result.strategies.contains(&"toon_table".to_string()));
    }

    #[test]
    fn standard_does_not_apply_lzw() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard);
        let raw = repeated_lzw_output();

        let result = compress_tool_output("custom_tool", None, &raw, &config, None);

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

        let result = compress_tool_output("custom_tool", None, &raw, &config, None);

        assert_eq!(result.output, raw);
        assert!(!result.output.starts_with("[Forge RePair Dictionary]"));
        assert!(!result.strategies.contains(&"repair_dictionary".to_string()));
    }

    #[test]
    fn aggressive_lzw_records_strategy() {
        let config = ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive);
        let raw = repeated_lzw_output();

        let result = compress_tool_output("custom_tool", None, &raw, &config, None);

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

        let result = compress_tool_output("custom_tool", None, &raw, &config, None);

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

        let result = compress_tool_output("custom_tool", None, &raw, &config, None);

        assert_eq!(result.output.len(), lzw.len().min(repair.len()));
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
        let first = compress_tool_output("bash", None, &raw, &config, Some(&state));
        let second = compress_tool_output("bash", None, &raw, &config, Some(&state));
        assert!(!first.deduped);
        assert!(second.deduped);
        assert!(second.output.contains("Duplicate of call #"));
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
}
