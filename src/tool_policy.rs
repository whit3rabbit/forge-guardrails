//! Opt-in tool-call policy nudges for proxy-intercepted tool calls.

use crate::clients::base::ToolCall;
use crate::core::tool_spec::ToolSpec;
use indexmap::IndexSet;
use serde_json::Value;
use std::str::FromStr;

/// Default maximum string payload size for write/edit policy nudges.
pub const DEFAULT_MAX_WRITE_PAYLOAD_BYTES: usize = 64 * 1024;

/// Process/request-level policy preset for tool-call nudges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallPolicyMode {
    /// Do not apply tool-call policy nudges.
    Disabled,
    /// Apply all currently supported opt-in tool-call policy nudges.
    Standard,
}

impl ToolCallPolicyMode {
    /// Returns the stable string representation of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Standard => "standard",
        }
    }
}

impl std::fmt::Display for ToolCallPolicyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ToolCallPolicyMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "disabled" | "off" | "false" | "0" => Ok(Self::Disabled),
            "standard" | "on" | "true" | "1" => Ok(Self::Standard),
            other => Err(format!(
                "tool call policy mode must be disabled or standard, got '{other}'"
            )),
        }
    }
}

/// Opt-in controls for tool-call policy nudges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallPolicyConfig {
    /// Preset used to initialize the individual controls.
    pub mode: ToolCallPolicyMode,
    /// Nudge grep/glob/shell-grep symbol searches toward available LSP tools.
    pub lsp_first: bool,
    /// Nudge noisy shell commands toward quieter equivalents.
    pub quiet_commands: bool,
    /// Nudge oversized write/edit payloads.
    pub write_payload_caps: bool,
    /// Maximum payload size before write/edit policy nudges fire.
    pub max_write_payload_bytes: usize,
}

impl Default for ToolCallPolicyConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl ToolCallPolicyConfig {
    /// Returns a disabled configuration.
    pub fn disabled() -> Self {
        Self {
            mode: ToolCallPolicyMode::Disabled,
            lsp_first: false,
            quiet_commands: false,
            write_payload_caps: false,
            max_write_payload_bytes: DEFAULT_MAX_WRITE_PAYLOAD_BYTES,
        }
    }

    /// Returns a standard configuration.
    pub fn standard() -> Self {
        Self {
            mode: ToolCallPolicyMode::Standard,
            lsp_first: true,
            quiet_commands: true,
            write_payload_caps: true,
            max_write_payload_bytes: DEFAULT_MAX_WRITE_PAYLOAD_BYTES,
        }
    }

    /// Builds a configuration from a preset mode.
    pub fn from_mode(mode: ToolCallPolicyMode) -> Self {
        match mode {
            ToolCallPolicyMode::Disabled => Self::disabled(),
            ToolCallPolicyMode::Standard => Self::standard(),
        }
    }

    /// Returns true when at least one policy nudge is enabled.
    pub fn enabled(&self) -> bool {
        self.lsp_first || self.quiet_commands || self.write_payload_caps
    }
}

/// Per-request state used to suppress repeat advisory nudges.
#[derive(Debug, Default)]
pub struct ToolCallPolicyRequestState {
    quiet_command_fingerprints: IndexSet<String>,
}

impl ToolCallPolicyRequestState {
    /// Creates empty per-request policy state.
    pub fn new() -> Self {
        Self::default()
    }

    fn should_nudge_quiet_command(&mut self, fingerprint: String) -> bool {
        self.quiet_command_fingerprints.insert(fingerprint)
    }
}

/// Tool-call policy nudge returned to the model as a synthetic tool result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallPolicyNudge {
    /// Stable policy kind for metrics and tests.
    pub kind: &'static str,
    /// Human-readable retry instruction.
    pub content: String,
    /// Stable per-request fingerprint for this policy decision.
    pub fingerprint: String,
}

/// Evaluates one batch of model-produced tool calls against opt-in proxy policy.
pub fn evaluate_tool_call_policy(
    tool_calls: &[ToolCall],
    tool_specs: &[ToolSpec],
    config: &ToolCallPolicyConfig,
    state: &mut ToolCallPolicyRequestState,
) -> Option<ToolCallPolicyNudge> {
    if !config.enabled() {
        return None;
    }

    if config.lsp_first {
        let lsp_tools = available_lsp_tools(tool_specs);
        if !lsp_tools.is_empty() {
            for call in tool_calls {
                if let Some(nudge) = lsp_first_nudge(call, &lsp_tools) {
                    return Some(nudge);
                }
            }
        }
    }

    if config.write_payload_caps {
        for call in tool_calls {
            if let Some(nudge) = write_payload_cap_nudge(call, config.max_write_payload_bytes) {
                return Some(nudge);
            }
        }
    }

    if config.quiet_commands {
        for call in tool_calls {
            if let Some(nudge) = quiet_command_nudge(call, state) {
                return Some(nudge);
            }
        }
    }

    None
}

fn available_lsp_tools(tool_specs: &[ToolSpec]) -> Vec<String> {
    let supported = [
        "find_definition",
        "find_references",
        "get_hover",
        "document_symbols",
        "workspace_symbols",
    ];
    tool_specs
        .iter()
        .filter(|tool| supported.contains(&tool.name.as_str()))
        .map(|tool| tool.name.clone())
        .collect()
}

fn lsp_first_nudge(call: &ToolCall, lsp_tools: &[String]) -> Option<ToolCallPolicyNudge> {
    let tool_name = call.tool.to_ascii_lowercase();
    let symbol = if is_shell_tool(&tool_name) {
        shell_grep_symbol(command_arg(&call.args)?)?
    } else if is_grep_tool(&tool_name) {
        string_arg(
            &call.args,
            &["symbol", "name", "pattern", "query", "regex", "needle"],
        )
        .and_then(symbol_from_search_value)?
    } else if is_glob_tool(&tool_name) {
        string_arg(&call.args, &["pattern", "query", "glob"]).and_then(symbol_from_search_value)?
    } else {
        return None;
    };

    let tools = lsp_tools.join(", ");
    let fingerprint = format!("lsp_first:{}:{symbol}", call.tool);
    Some(ToolCallPolicyNudge {
        kind: "lsp_first",
        content: format!(
            "Use available LSP tools for symbol lookup instead of grep/glob/shell search. Available LSP tools: {tools}. Retry with the best matching LSP tool for `{symbol}`."
        ),
        fingerprint,
    })
}

fn quiet_command_nudge(
    call: &ToolCall,
    state: &mut ToolCallPolicyRequestState,
) -> Option<ToolCallPolicyNudge> {
    let tool_name = call.tool.to_ascii_lowercase();
    if !is_shell_tool(&tool_name) {
        return None;
    }
    let command = command_arg(&call.args)?.trim();
    let suggestion = quiet_command_suggestion(command)?;
    let fingerprint = format!("quiet:{}:{command}:{suggestion}", call.tool);
    if !state.should_nudge_quiet_command(fingerprint.clone()) {
        return None;
    }
    Some(ToolCallPolicyNudge {
        kind: "quiet_command",
        content: format!(
            "The requested shell command is likely to produce noisy output. Prefer `{suggestion}`. Repeat the original command only if verbose output is required."
        ),
        fingerprint,
    })
}

fn write_payload_cap_nudge(call: &ToolCall, max_bytes: usize) -> Option<ToolCallPolicyNudge> {
    let tool_name = call.tool.to_ascii_lowercase();
    if !is_write_or_edit_tool(&tool_name) {
        return None;
    }
    let bytes = write_payload_bytes(&call.args);
    if bytes <= max_bytes {
        return None;
    }
    Some(ToolCallPolicyNudge {
        kind: "write_payload_cap",
        content: format!(
            "The requested write/edit payload is too large for this proxy policy ({bytes} bytes > {max_bytes} bytes). Retry with a smaller targeted edit or split the change."
        ),
        fingerprint: format!("write_payload_cap:{}:{bytes}:{max_bytes}", call.tool),
    })
}

fn is_shell_tool(name: &str) -> bool {
    matches!(
        name,
        "bash" | "shell" | "run_command" | "execute_command" | "terminal" | "exec"
    )
}

fn is_grep_tool(name: &str) -> bool {
    matches!(name, "grep" | "rg" | "ripgrep")
}

fn is_glob_tool(name: &str) -> bool {
    matches!(name, "glob" | "find_files" | "file_glob")
}

fn is_write_or_edit_tool(name: &str) -> bool {
    matches!(
        name,
        "write"
            | "write_file"
            | "edit"
            | "edit_file"
            | "replace"
            | "apply_patch"
            | "create_file"
            | "update_file"
    )
}

fn command_arg(args: &indexmap::IndexMap<String, Value>) -> Option<&str> {
    string_arg(args, &["command", "cmd", "shell_command", "input"])
}

fn string_arg<'a>(args: &'a indexmap::IndexMap<String, Value>, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(value) = args.get(*key).and_then(Value::as_str) {
            return Some(value);
        }
    }
    args.values().find_map(Value::as_str)
}

fn shell_grep_symbol(command: &str) -> Option<String> {
    let mut parts = command.split_whitespace();
    let binary = strip_shell_quotes(parts.next()?)
        .rsplit('/')
        .next()?
        .to_string();
    if !matches!(binary.as_str(), "rg" | "ripgrep" | "grep") {
        return None;
    }

    let mut previous_took_value = false;
    for part in parts {
        let token = strip_shell_quotes(part);
        if previous_took_value {
            previous_took_value = false;
            continue;
        }
        if token.starts_with("--") {
            previous_took_value = matches!(
                token.as_str(),
                "--glob" | "--type" | "--context" | "--after-context" | "--before-context"
            );
            continue;
        }
        if token.starts_with('-') {
            continue;
        }
        return symbol_from_search_value(&token);
    }
    None
}

fn symbol_from_search_value(value: &str) -> Option<String> {
    let trimmed = strip_shell_quotes(value)
        .trim_matches('/')
        .replace("\\b", "")
        .replace(['^', '$'], "");
    for token in trimmed.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_')) {
        if looks_like_symbol_token(token) {
            return Some(token.to_string());
        }
    }
    None
}

fn looks_like_symbol_token(token: &str) -> bool {
    if token.len() < 3 {
        return false;
    }
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_') {
        return false;
    }

    let lower = token.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "todo" | "fixme" | "error" | "warning" | "debug" | "test" | "src" | "main"
    ) {
        return false;
    }
    token.contains('_') || token.chars().any(|ch| ch.is_ascii_uppercase())
}

fn strip_shell_quotes(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn quiet_command_suggestion(command: &str) -> Option<String> {
    if command_has_prefix(command, "git log") && !contains_word(command, "--oneline") {
        return Some(insert_after_prefix(command, "git log", "--oneline"));
    }
    for prefix in ["cargo build", "cargo check", "cargo clippy", "cargo test"] {
        if command_has_prefix(command, prefix) && !contains_word(command, "--quiet") {
            return Some(insert_after_prefix(command, prefix, "--quiet"));
        }
    }
    if command_has_prefix(command, "pytest") && !contains_word(command, "-q") {
        return Some(insert_after_prefix(command, "pytest", "-q"));
    }
    if command_has_prefix(command, "npm install") && !contains_word(command, "--silent") {
        return Some(insert_after_prefix(command, "npm install", "--silent"));
    }
    if command_has_prefix(command, "pip install") && !contains_word(command, "--quiet") {
        return Some(insert_after_prefix(command, "pip install", "--quiet"));
    }
    if command_has_prefix(command, "docker build") && !contains_word(command, "--progress=quiet") {
        return Some(insert_after_prefix(
            command,
            "docker build",
            "--progress=quiet",
        ));
    }
    if command_has_prefix(command, "curl") && !contains_word(command, "-s") {
        return Some(insert_after_prefix(command, "curl", "-s"));
    }
    if command_has_prefix(command, "make") && !contains_word(command, "-s") {
        return Some(insert_after_prefix(command, "make", "-s"));
    }
    if command_has_prefix(command, "tree") && !contains_word(command, "-I") {
        return Some(insert_after_prefix(
            command,
            "tree",
            "-I \"node_modules|.git|target|dist|build\"",
        ));
    }
    None
}

fn command_has_prefix(command: &str, prefix: &str) -> bool {
    command == prefix || command.starts_with(&format!("{prefix} "))
}

fn contains_word(command: &str, word: &str) -> bool {
    command.split_whitespace().any(|part| part == word)
}

fn insert_after_prefix(command: &str, prefix: &str, insertion: &str) -> String {
    let rest = command[prefix.len()..].trim_start();
    if rest.is_empty() {
        format!("{prefix} {insertion}")
    } else {
        format!("{prefix} {insertion} {rest}")
    }
}

fn write_payload_bytes(args: &indexmap::IndexMap<String, Value>) -> usize {
    let payload_keys = [
        "content",
        "text",
        "new_content",
        "patch",
        "diff",
        "replacement",
        "data",
    ];
    args.iter()
        .filter(|(key, _)| payload_keys.contains(&key.as_str()))
        .map(|(_, value)| value_payload_bytes(value))
        .sum()
}

fn value_payload_bytes(value: &Value) -> usize {
    match value {
        Value::String(value) => value.len(),
        Value::Array(values) => values.iter().map(value_payload_bytes).sum(),
        Value::Object(values) => values.values().map(value_payload_bytes).sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use serde_json::json;

    fn tool_spec(name: &str) -> ToolSpec {
        ToolSpec::from_json_schema(
            name,
            "",
            &json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        )
        .expect("tool spec")
    }

    fn call(name: &str, key: &str, value: &str) -> ToolCall {
        let mut args = IndexMap::new();
        args.insert(key.to_string(), json!(value));
        ToolCall::new(name, args)
    }

    #[test]
    fn lsp_nudge_requires_replacement_tool() {
        let mut state = ToolCallPolicyRequestState::new();
        let config = ToolCallPolicyConfig {
            lsp_first: true,
            ..ToolCallPolicyConfig::disabled()
        };
        let calls = vec![call("grep", "pattern", "UserService")];

        assert!(
            evaluate_tool_call_policy(&calls, &[tool_spec("grep")], &config, &mut state).is_none()
        );

        let nudge = evaluate_tool_call_policy(
            &calls,
            &[tool_spec("grep"), tool_spec("find_definition")],
            &config,
            &mut state,
        )
        .expect("lsp nudge");
        assert_eq!(nudge.kind, "lsp_first");
        assert!(nudge.content.contains("find_definition"));
        assert!(nudge.content.contains("UserService"));
    }

    #[test]
    fn quiet_command_nudges_once() {
        let mut state = ToolCallPolicyRequestState::new();
        let config = ToolCallPolicyConfig {
            quiet_commands: true,
            ..ToolCallPolicyConfig::disabled()
        };
        let calls = vec![call("bash", "command", "cargo test")];
        let first = evaluate_tool_call_policy(&calls, &[], &config, &mut state).expect("nudge");
        assert_eq!(first.kind, "quiet_command");
        assert!(first.content.contains("cargo test --quiet"));
        assert!(evaluate_tool_call_policy(&calls, &[], &config, &mut state).is_none());
    }

    #[test]
    fn write_payload_cap_detects_oversized_payload() {
        let mut state = ToolCallPolicyRequestState::new();
        let config = ToolCallPolicyConfig {
            write_payload_caps: true,
            max_write_payload_bytes: 4,
            ..ToolCallPolicyConfig::disabled()
        };
        let calls = vec![call("write_file", "content", "12345")];
        let nudge = evaluate_tool_call_policy(&calls, &[], &config, &mut state).expect("nudge");
        assert_eq!(nudge.kind, "write_payload_cap");
        assert!(nudge.content.contains("5 bytes > 4 bytes"));
    }
}
