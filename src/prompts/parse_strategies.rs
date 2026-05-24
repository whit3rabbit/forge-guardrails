//! Rescue parsing strategies for tool call extraction.
//!
//! Each strategy handles a different model output format.
//! Strategies are tried in priority order by rescue_tool_call.

use crate::clients::base::ToolCall;
use indexmap::IndexMap;
use serde_json::Value;

/// Strip thinking-block tags (bracket-style and Unicode-style).
pub fn strip_think_tags(text: &str) -> String {
    static BRACKET: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"(?is)\u{200B}?\[think\].*?\[/think\]\u{200B}?\s*").expect("regex")
    });
    static XML: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"(?is)<think(?:\s[^>]*)?>.*?</think\s*>\s*").expect("regex")
    });
    let text = BRACKET.replace_all(text, "").to_string();
    XML.replace_all(&text, "").to_string()
}

/// Extract the trailing sequence of word characters from a string.
fn extract_trailing_word(text: &str) -> Option<String> {
    let mut end = text.len();
    let bytes = text.as_bytes();
    while end > 0 {
        let ch = bytes[end - 1];
        if ch.is_ascii_alphanumeric() || ch == b'_' {
            end -= 1;
        } else {
            break;
        }
    }
    if end < text.len() {
        Some(text[end..].to_string())
    } else {
        None
    }
}

/// Find the end index of a balanced brace sequence starting at position 0.
fn find_balanced_brace(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes[0] != b'{' {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i];
        if in_string {
            if ch == b'\\' {
                i += 2;
                continue;
            }
            if ch == b'"' {
                in_string = false;
            }
        } else {
            match ch {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Parse rehearsal syntax: tool_name[ARGS]{json_args}
pub fn parse_rehearsal(text: &str, available_tools: &[&str]) -> Vec<ToolCall> {
    let mut results = Vec::new();
    let mut search_from = 0;
    while search_from < text.len() {
        let marker = "[ARGS]";
        let marker_pos = match text[search_from..].find(marker) {
            Some(pos) => search_from + pos,
            None => break,
        };
        let before = &text[..marker_pos];
        let tool_name = match extract_trailing_word(before) {
            Some(name) => name,
            None => {
                search_from = marker_pos + marker.len();
                continue;
            }
        };
        if !available_tools.contains(&tool_name.as_str()) {
            search_from = marker_pos + marker.len();
            continue;
        }
        let after_marker = marker_pos + marker.len();
        let rest = &text[after_marker..];
        let mut brace_start = None;
        for (i, ch) in rest.char_indices() {
            if ch == '{' {
                brace_start = Some(after_marker + i);
                break;
            }
            if !ch.is_whitespace() {
                break;
            }
        }
        let brace_start = match brace_start {
            Some(pos) => pos,
            None => {
                search_from = after_marker;
                continue;
            }
        };
        let brace_text = &text[brace_start..];
        let brace_end = match find_balanced_brace(brace_text) {
            Some(end) => brace_start + end,
            None => {
                search_from = after_marker;
                continue;
            }
        };
        let json_str = &text[brace_start..=brace_end];
        match serde_json::from_str::<Value>(json_str) {
            Ok(Value::Object(map)) => {
                let args: IndexMap<String, Value> =
                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                results.push(ToolCall::new(tool_name, args));
                search_from = brace_end + 1;
            }
            _ => {
                search_from = after_marker;
            }
        }
    }
    results
}

/// Strip exactly one leading newline and one trailing newline.
fn strip_first_last_newline(s: &str) -> String {
    let mut result = s;
    if result.starts_with('\n') {
        result = &result[1..];
    }
    if result.ends_with('\n') {
        result = &result[..result.len() - 1];
    }
    result.to_string()
}

/// Parse Qwen XML format: <function=name>...</function> with <parameter=key>value</parameter>
pub fn parse_qwen_xml(text: &str, available_tools: &[&str]) -> Vec<ToolCall> {
    let mut results = Vec::new();
    let mut search_from = 0;
    while search_from < text.len() {
        let open_tag = match text[search_from..].find("<function=") {
            Some(pos) => search_from + pos,
            None => break,
        };
        let after_eq = open_tag + "<function=".len();
        let close_bracket = match text[after_eq..].find('>') {
            Some(pos) => after_eq + pos,
            None => break,
        };
        let func_name = &text[after_eq..close_bracket];
        let close_tag_marker = "</function>";
        let body_start = close_bracket + 1;
        let close_tag_pos = match text[body_start..].find(close_tag_marker) {
            Some(pos) => body_start + pos,
            None => break, // unclosed function tag: no match
        };
        let body = &text[body_start..close_tag_pos];
        if !available_tools.contains(&func_name) {
            search_from = close_tag_pos + close_tag_marker.len();
            continue;
        }
        let mut args: IndexMap<String, Value> = IndexMap::new();
        let mut param_search = 0;
        while param_search < body.len() {
            let param_open = match body[param_search..].find("<parameter=") {
                Some(pos) => param_search + pos,
                None => break,
            };
            let param_after_eq = param_open + "<parameter=".len();
            let param_close_bracket = match body[param_after_eq..].find('>') {
                Some(pos) => param_after_eq + pos,
                None => break,
            };
            let param_name = body[param_after_eq..param_close_bracket].to_string();
            let param_body_start = param_close_bracket + 1;
            let param_close_marker = "</parameter>";
            let remaining = &body[param_body_start..];
            let close_rel = remaining.find(param_close_marker);
            let next_param_rel = remaining.find("<parameter=");
            let (param_close_pos, next_search) = match (close_rel, next_param_rel) {
                (Some(close), Some(next_param)) if next_param < close => {
                    let pos = param_body_start + next_param;
                    (pos, pos)
                }
                (Some(close), _) => {
                    let pos = param_body_start + close;
                    (pos, pos + param_close_marker.len())
                }
                (None, Some(next_param)) => {
                    let pos = param_body_start + next_param;
                    (pos, pos)
                }
                (None, None) => (body.len(), body.len()),
            };
            let raw_value = &body[param_body_start..param_close_pos];
            let stripped = strip_first_last_newline(raw_value);
            args.insert(param_name, Value::String(stripped));
            param_search = next_search;
        }
        results.push(ToolCall::new(func_name.to_string(), args));
        search_from = close_tag_pos + close_tag_marker.len();
    }
    results
}

/// Parse Mistral bracket-tag format: [TOOL_CALLS]name{...}
pub fn parse_mistral_bracket(text: &str, available_tools: &[&str]) -> Vec<ToolCall> {
    let mut results = Vec::new();
    let marker = "[TOOL_CALLS]";
    let mut search_from = 0;
    while search_from < text.len() {
        let marker_pos = match text[search_from..].find(marker) {
            Some(pos) => search_from + pos,
            None => break,
        };
        let after_marker = marker_pos + marker.len();
        let rest = &text[after_marker..];
        let name_end = rest
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        let tool_name = &rest[..name_end];
        if tool_name.is_empty() || !available_tools.contains(&tool_name) {
            search_from = after_marker;
            continue;
        }
        let after_name = &rest[name_end..];
        let mut brace_offset = None;
        for (i, ch) in after_name.char_indices() {
            if ch == '{' {
                brace_offset = Some(name_end + i);
                break;
            }
            if !ch.is_whitespace() {
                break;
            }
        }
        let brace_offset = match brace_offset {
            Some(pos) => pos,
            None => {
                search_from = after_marker;
                continue;
            }
        };
        let brace_text = &rest[brace_offset..];
        let brace_end = match find_balanced_brace(brace_text) {
            Some(end) => brace_offset + end,
            None => {
                search_from = after_marker;
                continue;
            }
        };
        let json_str = &rest[brace_offset..=brace_end];
        match serde_json::from_str::<Value>(json_str) {
            Ok(Value::Object(map)) => {
                let args: IndexMap<String, Value> =
                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                results.push(ToolCall::new(tool_name.to_string(), args));
                search_from = after_marker + brace_end + 1;
            }
            _ => {
                search_from = after_marker;
            }
        }
    }
    results
}
