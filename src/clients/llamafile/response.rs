use indexmap::IndexMap;
use serde_json::Value;

use super::{helpers, LlamafileClient};
use crate::clients::base::{LLMResponse, TextResponse, ToolCall};
use crate::core::tool_spec::ToolSpec;
use crate::prompts::extract_tool_call;

impl LlamafileClient {
    pub(super) fn parse_native_response(&self, response: &Value) -> LLMResponse {
        let choice = response.get("choices").and_then(|c| c.get(0));
        let message = choice.and_then(|c| c.get("message"));
        let reasoning = helpers::resolve_reasoning(self.think, response);
        let content = message
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str());

        if let Some(tcs) = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
        {
            if !tcs.is_empty() {
                let mut calls = Vec::new();
                for (i, tc) in tcs.iter().enumerate() {
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    let id = tc.get("id").and_then(|i| i.as_str()).map(str::to_string);
                    let args_raw = tc.get("function").and_then(|f| f.get("arguments"));
                    let args = match parse_args(args_raw) {
                        Ok(args) => args,
                        Err(raw_args) => {
                            let text = content.map(str::to_string).unwrap_or(raw_args);
                            return LLMResponse::Text(TextResponse::new(text));
                        }
                    };
                    let mut call = ToolCall::new(name, args);
                    if let Some(id_val) = id {
                        call = call.with_id(id_val);
                    }
                    if i == 0 {
                        if let Some(r) = reasoning.as_ref() {
                            call = call.with_reasoning(r);
                        }
                    }
                    calls.push(call);
                }
                return LLMResponse::ToolCalls(calls);
            }
        }

        let cleaned = if self.think {
            helpers::strip_reasoning_tags(content.unwrap_or(""))
        } else {
            content.unwrap_or("").to_string()
        };
        LLMResponse::Text(TextResponse::new(cleaned))
    }

    pub(super) fn parse_prompt_response(
        &self,
        response: &Value,
        tools: &[ToolSpec],
    ) -> LLMResponse {
        let content = response
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        let (reasoning, cleaned) = if self.think {
            helpers::extract_think_tags(content)
        } else {
            (String::new(), content.to_string())
        };
        let mut calls = extract_tool_call(&cleaned, &names);
        if calls.is_empty() {
            LLMResponse::Text(TextResponse::new(cleaned))
        } else {
            if let Some(first) = calls.first_mut() {
                if !reasoning.is_empty() {
                    *first = first.clone().with_reasoning(reasoning);
                }
            }
            LLMResponse::ToolCalls(calls)
        }
    }
}

pub(super) fn parse_args(args_raw: Option<&Value>) -> Result<IndexMap<String, Value>, String> {
    match args_raw {
        Some(Value::String(s)) => parse_args_json(s),
        Some(Value::Object(obj)) => Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        _ => Ok(IndexMap::new()),
    }
}

pub(super) fn parse_args_json(args_json: &str) -> Result<IndexMap<String, Value>, String> {
    match serde_json::from_str::<Value>(args_json) {
        Ok(Value::Object(obj)) => Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        _ => Err(args_json.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use super::*;

    #[test]
    fn parse_native_tool_call() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read", "arguments": "{\"path\": \"/x\"}"}},
        ]}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "read");
                assert_eq!(calls[0].args["path"], "/x");
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_native_text() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"content": "Hello"}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, "Hello"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_native_args_as_dict() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "run", "arguments": {"x": 1}}},
        ]}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::ToolCalls(calls) => assert_eq!(calls[0].args["x"], 1),
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_native_null_content() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"content": null}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, ""),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn malformed_native_args_fall_back_to_text() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"content": "fallback", "tool_calls": [
            {"function": {"name": "run", "arguments": "{broken"}},
        ]}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, "fallback"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_native_tool_id_and_reasoning() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {
            "content": "<think>reason</think>",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "run", "arguments": "{}"}
            }]
        }}]});
        match c.parse_native_response(&resp) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].id, Some("call_1".to_string()));
                assert_eq!(calls[0].reasoning, Some("reason".to_string()));
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_prompt_strips_think_tags_and_attaches_reasoning() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_mode("prompt");
        let tool =
            ToolSpec::from_json_schema("run", "Run", &json!({"type": "object", "properties": {}}))
                .expect("ok");
        let resp = json!({"choices": [{"message": {"content": "<think >reason</think >{\"tool\":\"run\",\"args\":{}}"}}]});

        match c.parse_prompt_response(&resp, &[tool]) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "run");
                assert_eq!(calls[0].reasoning, Some("reason".to_string()));
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_prompt_text_fallback() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_mode("prompt");
        let tool =
            ToolSpec::from_json_schema("run", "Run", &json!({"type": "object", "properties": {}}))
                .expect("ok");
        let resp = json!({"choices": [{"message": {"content": "<think>reason</think>Hello"}}]});

        match c.parse_prompt_response(&resp, &[tool]) {
            LLMResponse::Text(text) => assert_eq!(text.content, "Hello"),
            _ => panic!("expected text"),
        }
    }
}
