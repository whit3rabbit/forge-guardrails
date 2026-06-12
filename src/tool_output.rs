//! Tool-output compression for proxy-forwarded tool result messages.
//!
//! This module intentionally compresses tool results, not tool calls. Tool-call
//! names and arguments remain API contracts owned by the client and guardrails.

use indexmap::IndexMap;
use serde_json::Value;

mod config;
mod families;
mod filters;
mod lzw;
mod postcall;
mod repair;
mod safety;
mod state;

#[cfg(test)]
mod tests;

use families::{filter_bash_output, filter_generic_output};
use filters::{filter_glob_output, filter_grep_output, filter_read_output};
use lzw::compress_lzw_dictionary;
use postcall::{
    fold_repeated_lines, json_array_to_table, minify_json, minimize_table_whitespace,
    normalize_dynamic_log_noise, normalize_whitespace,
};
use repair::compress_repair_dictionary;
use safety::apply_safe_filters;
pub(crate) use safety::redact_secrets;
pub use state::ToolOutputCompressionState;
use state::{config_fingerprint, MemoLookup, MemoRecord};

pub use config::{
    dictionary_has_meaningful_savings, is_dictionary_compressed_output,
    ToolOutputCompressionConfig, ToolOutputCompressionMethod, ToolOutputCompressionMode,
    ToolOutputCompressionResult, DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION, DEFAULT_MAX_DEDUP_SESSIONS,
    DEFAULT_MAX_OUTPUT_BYTES, DICTIONARY_MAX_DICT_SIZE, DICTIONARY_MAX_INPUT_BYTES,
    DICTIONARY_MIN_ENTRY_SAVINGS_BYTES, DICTIONARY_MIN_NET_SAVINGS_BYTES,
    DICTIONARY_MIN_NET_SAVINGS_PERCENT, DICTIONARY_MIN_OCCURRENCES, LZW_DICTIONARY_HEADER,
    REPAIR_DICTIONARY_HEADER,
};

/// Compress one tool output using the requested mode and optional dedup state.
///
/// Dedup additionally requires `tool_call_id` so a tool result re-sent in a
/// later request under the same call id keeps its content instead of being
/// treated as a duplicate of itself.
pub fn compress_tool_output(
    tool_name: &str,
    tool_call_id: Option<&str>,
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
            memo_reused: false,
            memo_changed: false,
            strategies: Vec::new(),
        });
    }

    // Memo lookup: if the same call id was compressed before with identical
    // input and config, reuse the stored bytes to keep prompt-cache prefixes
    // byte-stable across full-history resends.
    let memo_key = if config.enable_memo {
        if let (Some(state), Some(session_id), Some(tool_call_id)) =
            (state, config.session_id.as_deref(), tool_call_id)
        {
            let input_hash = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                raw_output.hash(&mut h);
                h.finish()
            };
            let input_len = raw_output.len();
            let config_fp = config_fingerprint(config);
            match state.lookup_memo(session_id, tool_call_id, input_hash, input_len, config_fp) {
                MemoLookup::Hit(cached) => {
                    return compression_result(CompressionResultInput {
                        output: cached,
                        before_tokens,
                        canonical_tool,
                        family,
                        mode: config.mode,
                        redacted: false,
                        capped: false,
                        deduped: false,
                        memo_reused: true,
                        memo_changed: false,
                        strategies: vec!["memo_reuse".to_string()],
                    });
                }
                MemoLookup::Changed => Some((
                    state,
                    session_id,
                    tool_call_id,
                    input_hash,
                    input_len,
                    config_fp,
                    true,
                )),
                MemoLookup::Miss => Some((
                    state,
                    session_id,
                    tool_call_id,
                    input_hash,
                    input_len,
                    config_fp,
                    false,
                )),
            }
        } else {
            None
        }
    } else {
        None
    };
    let memo_changed = memo_key.as_ref().is_some_and(|k| k.6);

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
        if let (Some(state), Some(session_id), Some(tool_call_id)) =
            (state, config.session_id.as_deref(), tool_call_id)
        {
            if let Some(marker) = state.deduplicate(
                session_id,
                tool_call_id,
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

    // Store the final post-dedup output in the memo so full-history resends
    // return byte-identical bytes (preserving upstream prompt-cache prefixes).
    if let Some((state, session_id, tool_call_id, input_hash, input_len, config_fp, _)) = memo_key {
        state.store_memo(
            session_id,
            tool_call_id,
            MemoRecord {
                input_hash,
                input_len,
                config_fingerprint: config_fp,
                output: output.clone(),
            },
            config.max_dedup_sessions,
        );
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
        memo_reused: false,
        memo_changed,
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
        .rsplit(['/', '\\'])
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

/// Extra alphabetic chars per additional token beyond a run's first token.
const ALPHA_CHARS_PER_EXTRA_TOKEN: usize = 7;
/// Digit chars per token; BPE vocabularies group digits in short chunks.
const DIGIT_CHARS_PER_TOKEN: usize = 3;
/// Punctuation/symbol chars per token; JSON and logs tokenize densely.
const PUNCT_CHARS_PER_TOKEN: usize = 2;
/// Space/tab chars per token within an indentation run after the first.
const SPACE_RUN_CHARS_PER_TOKEN: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenRunClass {
    Alpha,
    Digit,
    Space,
    Newline,
    Punct,
}

fn token_run_class(c: char) -> TokenRunClass {
    if c.is_alphabetic() || c == '_' {
        TokenRunClass::Alpha
    } else if c.is_ascii_digit() {
        TokenRunClass::Digit
    } else if c == ' ' || c == '\t' {
        TokenRunClass::Space
    } else if c == '\n' {
        TokenRunClass::Newline
    } else {
        TokenRunClass::Punct
    }
}

fn run_token_cost(class: TokenRunClass, len: usize, next: Option<TokenRunClass>) -> usize {
    match class {
        TokenRunClass::Alpha => 1 + (len - 1) / ALPHA_CHARS_PER_EXTRA_TOKEN,
        TokenRunClass::Digit => len.div_ceil(DIGIT_CHARS_PER_TOKEN),
        TokenRunClass::Punct => len.div_ceil(PUNCT_CHARS_PER_TOKEN),
        // A single space merges into a following word token for free; spaces
        // before punctuation, newlines, or end-of-text are their own tokens,
        // and longer runs are indentation that BPE splits into chunks.
        TokenRunClass::Space => {
            let merges_into_word = matches!(
                next,
                Some(TokenRunClass::Alpha) | Some(TokenRunClass::Digit)
            );
            if merges_into_word {
                (len - 1).div_ceil(SPACE_RUN_CHARS_PER_TOKEN)
            } else {
                len.div_ceil(SPACE_RUN_CHARS_PER_TOKEN).max(1)
            }
        }
        TokenRunClass::Newline => len,
    }
}

/// Estimate tokens with the same lightweight heuristic used for proxy metrics.
///
/// Run-classifier heuristic: word, digit, punctuation, and whitespace runs
/// are weighted separately so punctuation-heavy output (JSON, logs) estimates
/// closer to real BPE behavior than a flat chars/4. Deterministic; transform
/// acceptance additionally keeps a byte-size gate.
pub fn estimate_tokens(value: &str) -> i64 {
    let mut runs: Vec<(TokenRunClass, usize)> = Vec::new();
    for c in value.chars() {
        let class = token_run_class(c);
        match runs.last_mut() {
            Some((current, len)) if *current == class => *len += 1,
            _ => runs.push((class, 1)),
        }
    }
    let mut total = 0usize;
    for (idx, &(class, len)) in runs.iter().enumerate() {
        let next = runs.get(idx + 1).map(|&(next_class, _)| next_class);
        total += run_token_cost(class, len, next);
    }
    total as i64
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
                .filter(|candidate| candidate_has_token_savings(&current, candidate))
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
    if candidate_has_token_savings(&current, &candidate) {
        strategies.push(strategy.to_string());
        candidate
    } else {
        current
    }
}

fn candidate_has_token_savings(current: &str, candidate: &str) -> bool {
    candidate.len() <= current.len() && estimate_tokens(candidate) < estimate_tokens(current)
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
    memo_reused: bool,
    memo_changed: bool,
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
        memo_reused: input.memo_reused,
        memo_changed: input.memo_changed,
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
