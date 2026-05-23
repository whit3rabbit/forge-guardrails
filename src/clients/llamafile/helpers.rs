//! Helper functions for Llamafile client.
//!
//! Message merging, reasoning tag extraction, shard suffix stripping,
//! and prompt-mode message downgrade.

use serde_json::{json, Map, Value};

use crate::clients::base::SamplingParams;

/// Merge consecutive same-role messages for server template compatibility.
///
/// Messages with tool_calls or tool role are invisible to the checker and
/// do not trigger merges. Visible messages separated by invisible messages
/// of the same role are merged together.
pub fn merge_messages(messages: &[Value]) -> Vec<Value> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Value> = Vec::new();
    let mut pending_texts: Vec<String> = Vec::new();
    let mut pending_role: Option<&str> = None;
    let mut pending_invisible: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let has_tool_calls = msg.get("tool_calls").is_some();
        let is_invisible = has_tool_calls || role == "tool";

        if is_invisible {
            pending_invisible.push(msg.clone());
        } else {
            let text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            match pending_role {
                Some(r) if r == role => {
                    pending_texts.push(text.to_string());
                }
                Some(r) => {
                    result.push(json!({"role": r, "content": pending_texts.join("\n")}));
                    result.append(&mut pending_invisible);
                    pending_texts = vec![text.to_string()];
                    pending_role = Some(role);
                }
                None => {
                    pending_texts.push(text.to_string());
                    pending_role = Some(role);
                }
            }
        }
    }

    if let Some(r) = pending_role {
        result.push(json!({"role": r, "content": pending_texts.join("\n")}));
        result.extend(pending_invisible);
    } else {
        result.extend(pending_invisible);
    }

    result
}

/// Extract model identity from a GGUF file path.
///
/// Strips file suffix and shard index pattern from the stem.
pub fn extract_model_identity(path: &std::path::Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    static RE: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"-000\d+-of-000\d+").expect("valid regex")
    });
    RE.replace(stem, "").to_string()
}

/// Apply sampling parameters to the request body.
///
/// Per-call overrides win over instance fields. Instance is not mutated.
#[allow(clippy::too_many_arguments)]
pub fn apply_sampling(
    instance_temp: Option<f64>,
    instance_top_p: Option<f64>,
    instance_top_k: Option<i64>,
    instance_min_p: Option<f64>,
    instance_repeat_penalty: Option<f64>,
    instance_presence_penalty: Option<f64>,
    instance_chat_kwargs: &Option<Map<String, Value>>,
    recommended_defaults: &Option<Map<String, Value>>,
    per_call: Option<&SamplingParams>,
    body: &mut Value,
) {
    let mut params = serde_json::Map::new();

    if let Some(ref defaults) = recommended_defaults {
        for (k, v) in defaults {
            params.insert(k.clone(), v.clone());
        }
    }

    if let Some(t) = instance_temp {
        params.insert("temperature".into(), json!(t));
    }
    if let Some(t) = instance_top_p {
        params.insert("top_p".into(), json!(t));
    }
    if let Some(k) = instance_top_k {
        params.insert("top_k".into(), json!(k));
    }
    if let Some(m) = instance_min_p {
        params.insert("min_p".into(), json!(m));
    }
    if let Some(r) = instance_repeat_penalty {
        params.insert("repeat_penalty".into(), json!(r));
    }
    if let Some(p) = instance_presence_penalty {
        params.insert("presence_penalty".into(), json!(p));
    }

    if let Some(sp) = per_call {
        for (k, v) in sp {
            if matches!(
                k.as_str(),
                "temperature"
                    | "top_p"
                    | "top_k"
                    | "min_p"
                    | "repeat_penalty"
                    | "presence_penalty"
                    | "seed"
            ) {
                params.insert(k.clone(), v.clone());
            }
        }
    }

    if let Some(obj) = body.as_object_mut() {
        for (k, v) in params {
            obj.insert(k, v);
        }
        if let Some(kwargs) = per_call.and_then(|sp| sp.get("chat_template_kwargs")) {
            obj.insert("chat_template_kwargs".into(), kwargs.clone());
        } else if let Some(ref kwargs) = instance_chat_kwargs {
            obj.insert("chat_template_kwargs".into(), json!(kwargs));
        }
    }
}

/// Downgrade messages for prompt mode: tool role -> user, tool_calls -> JSON text.
pub fn downgrade_messages_for_prompt(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|msg| {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").cloned().unwrap_or(Value::Null);

            if role == "tool" {
                json!({"role": "user", "content": content})
            } else if let Some(tool_calls) = msg.get("tool_calls") {
                let texts: Vec<String> = tool_calls
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|tc| {
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("");
                                let args = tc.get("function").and_then(|f| f.get("arguments"));
                                let args_str = match args {
                                    Some(Value::String(s)) => s.clone(),
                                    Some(v) => serde_json::to_string(v).unwrap_or_default(),
                                    None => "{}".to_string(),
                                };
                                format!("{{\"tool\": \"{}\", \"args\": {}}}", name, args_str)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                json!({"role": "assistant", "content": texts.join("\n")})
            } else {
                msg.clone()
            }
        })
        .collect()
}

/// Extract reasoning from think tags (bracket and XML style).
pub fn extract_reasoning_tags(content: &str) -> String {
    let bracket = extract_bracket_think(content);
    let xml = extract_xml_think(content);
    let mut parts = Vec::new();
    if !bracket.is_empty() {
        parts.push(bracket);
    }
    if !xml.is_empty() {
        parts.push(xml);
    }
    parts.join("\n\n")
}

fn extract_bracket_think(content: &str) -> String {
    static RE: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"(?is)\[think\](.*?)\[/think\]").expect("regex")
    });
    RE.captures_iter(content)
        .filter_map(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn extract_xml_think(content: &str) -> String {
    static RE: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"(?is)<think(?:\s[^>]*)?>(.*?)</think\s*>").expect("regex")
    });
    RE.captures_iter(content)
        .filter_map(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Strip reasoning tags from text content.
pub fn strip_reasoning_tags(content: &str) -> String {
    static BRACKET: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"(?is)\[think\].*?\[/think\]\s*").expect("regex")
    });
    static XML: std::sync::LazyLock<regex_lite::Regex> = std::sync::LazyLock::new(|| {
        regex_lite::Regex::new(r"(?is)<think(?:\s[^>]*)?>.*?</think\s*>\s*").expect("regex")
    });
    let text = BRACKET.replace_all(content, "").to_string();
    XML.replace_all(&text, "").to_string()
}

/// Resolve reasoning from a response.
///
/// Priority: server-parsed reasoning field > think tags in content > content fallback.
pub fn resolve_reasoning(think: bool, response: &Value) -> Option<String> {
    if !think {
        return None;
    }
    if let Some(r) = response.get("reasoning").and_then(|r| r.as_str()) {
        if !r.is_empty() {
            return Some(r.to_string());
        }
    }
    let content = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let tags = extract_reasoning_tags(content);
    if !tags.is_empty() {
        return Some(tags);
    }
    let has_tool_calls = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .is_some();
    if !has_tool_calls && !content.is_empty() {
        Some(content.to_string())
    } else {
        None
    }
}

/// Resolve reasoning from streaming-accumulated fields.
///
/// Used when streaming has separately accumulated `reasoning_content` (from
/// server-side reasoning field) and `content` (plain text accumulation).
/// Priority matches Python's `_resolve_reasoning`:
///   1. accumulated_reasoning (server reasoning_content field)
///   2. think tags extracted from accumulated_content
///   3. accumulated_content itself as fallback
pub fn resolve_full_reasoning(
    accumulated_reasoning: &str,
    accumulated_content: &str,
) -> Option<String> {
    if !accumulated_reasoning.is_empty() {
        return Some(accumulated_reasoning.to_string());
    }
    if !accumulated_content.is_empty() {
        let tag_reasoning = extract_reasoning_tags(accumulated_content);
        if !tag_reasoning.is_empty() {
            return Some(tag_reasoning);
        }
        // Content fallback (model narrating before tool call)
        return Some(accumulated_content.to_string());
    }
    None
}

/// Extract think tags from content, returning (reasoning, cleaned_content).
///
/// Matches Python's `_extract_think_tags(text)` return of `(reasoning, remaining)`.
/// Supports [THINK]...[/THINK] (Mistral) and <think>...</think> (Qwen/DeepSeek).
pub fn extract_think_tags(content: &str) -> (String, String) {
    let reasoning = extract_reasoning_tags(content);
    let cleaned = if reasoning.is_empty() {
        content.to_string()
    } else {
        strip_reasoning_tags(content)
    };
    (reasoning, cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn merge_consecutive_same_role() {
        let msgs = vec![
            json!({"role": "user", "content": "A"}),
            json!({"role": "user", "content": "B"}),
        ];
        assert_eq!(merge_messages(&msgs).len(), 1);
    }

    #[test]
    fn merge_preserves_tool_calls() {
        let msgs = vec![
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "a", "arguments": "{}"}}]}),
            json!({"role": "assistant", "content": "more"}),
        ];
        assert_eq!(merge_messages(&msgs).len(), 2);
    }

    #[test]
    fn merge_invisible_tool_role() {
        let msgs = vec![
            json!({"role": "user", "content": "A"}),
            json!({"role": "tool", "tool_call_id": "x", "content": "result"}),
            json!({"role": "user", "content": "B"}),
        ];
        assert_eq!(merge_messages(&msgs).len(), 2);
    }

    #[test]
    fn merge_empty_list() {
        assert!(merge_messages(&[]).is_empty());
    }

    #[test]
    fn extract_bracket_think() {
        assert_eq!(
            extract_reasoning_tags("Some [think]deep[/think] text"),
            "deep"
        );
    }

    #[test]
    fn extract_bracket_think_uppercase() {
        assert_eq!(
            extract_reasoning_tags("Some [THINK]deep[/THINK] text"),
            "deep"
        );
    }

    #[test]
    fn extract_xml_think() {
        assert_eq!(
            extract_reasoning_tags("Some <think type=\"r\">deep</think > text"),
            "deep"
        );
    }

    #[test]
    fn extract_multiple_blocks() {
        assert_eq!(
            extract_reasoning_tags("[think]first[/think] middle <think >second</think >"),
            "first\n\nsecond"
        );
    }

    #[test]
    fn extract_multiline() {
        assert_eq!(
            extract_reasoning_tags("[think]\nline1\nline2\n[/think]"),
            "line1\nline2"
        );
    }

    #[test]
    fn extract_empty() {
        assert!(extract_reasoning_tags("[think][/think] text").is_empty());
    }

    #[test]
    fn strip_bracket() {
        assert_eq!(
            strip_reasoning_tags("before [think]r[/think] after"),
            "before after"
        );
    }

    #[test]
    fn strip_xml() {
        assert_eq!(
            strip_reasoning_tags("before <think >r</think > after"),
            "before after"
        );
    }

    #[test]
    fn reasoning_server_field() {
        let r = json!({"reasoning": "server", "choices": [{"message": {"content": "[think]tag[/think]"}}]});
        assert_eq!(resolve_reasoning(true, &r), Some("server".to_string()));
    }

    #[test]
    fn reasoning_disabled() {
        assert!(resolve_reasoning(false, &json!({"reasoning": "x"})).is_none());
    }

    #[test]
    fn reasoning_content_fallback() {
        assert_eq!(
            resolve_reasoning(
                true,
                &json!({"choices": [{"message": {"content": "text"}}]})
            ),
            Some("text".to_string())
        );
    }

    #[test]
    fn model_identity_strips_shard() {
        let path = Path::new("/models/qwen3-00001-of-00005-Q4_K_M.gguf");
        assert_eq!(extract_model_identity(path), "qwen3-Q4_K_M");
    }

    #[test]
    fn model_identity_no_shard() {
        assert_eq!(
            extract_model_identity(Path::new("/models/m-Q4.gguf")),
            "m-Q4"
        );
    }

    #[test]
    fn downgrade_tool_role() {
        let result = downgrade_messages_for_prompt(&[json!({"role": "tool", "content": "r"})]);
        assert_eq!(result[0]["role"], "user");
    }

    #[test]
    fn downgrade_tool_calls() {
        let msgs = vec![json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "run", "arguments": "{\"x\": 1}"}}],
        })];
        let result = downgrade_messages_for_prompt(&msgs);
        assert!(result[0]["content"]
            .as_str()
            .expect("str")
            .contains("\"tool\": \"run\""));
    }
}
