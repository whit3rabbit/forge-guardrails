use indexmap::IndexMap;
use serde_json::Value;

use super::OllamaClient;
use crate::clients::base::{LLMResponse, TextResponse, TokenUsage, ToolCall};

impl OllamaClient {
    /// Resolves reasoning/thinking content from Ollama response, if enabled.
    pub fn resolve_reasoning(think: bool, response: &Value) -> Option<String> {
        if !think {
            return None;
        }
        let message = response.get("message");
        if let Some(r) = message
            .and_then(|m| m.get("thinking"))
            .and_then(|r| r.as_str())
        {
            if !r.is_empty() {
                return Some(r.to_string());
            }
        }
        message
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }

    pub(super) fn record_usage(&self, response: &Value) {
        let p = response
            .get("prompt_eval_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let c = response
            .get("eval_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if let Ok(mut g) = self.last_usage.lock() {
            *g = Some(TokenUsage::new(p, c, p + c));
        }
    }

    /// Returns the last token usage recorded by this client.
    pub fn get_last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|g| g.clone())
    }

    pub(super) fn is_think_unsupported_error(response: &Value) -> bool {
        response
            .get("error")
            .and_then(|e| e.as_str())
            .map(|s| {
                let l = s.to_lowercase();
                l.contains("think") && l.contains("support")
                    || l.contains("thinking") && l.contains("not")
            })
            .unwrap_or(false)
    }

    pub(super) fn parse_send_response(&self, response: &Value, think: bool) -> LLMResponse {
        let content = response
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        match self.parse_tool_calls(response, think) {
            Ok(Some(calls)) => return LLMResponse::ToolCalls(calls),
            Ok(None) => {}
            Err(raw_args) => {
                let text = if content.is_empty() {
                    raw_args
                } else {
                    content.to_string()
                };
                return LLMResponse::Text(TextResponse::new(text));
            }
        }
        LLMResponse::Text(TextResponse::new(content))
    }

    pub(super) fn parse_tool_calls(
        &self,
        response: &Value,
        think: bool,
    ) -> Result<Option<Vec<ToolCall>>, String> {
        let Some(tcs) = response
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
        else {
            return Ok(None);
        };
        if tcs.is_empty() {
            return Ok(None);
        }
        let reasoning = Self::resolve_reasoning(think, response);
        let mut calls = Vec::new();
        for (i, tc) in tcs.iter().enumerate() {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let args_val = tc.get("function").and_then(|f| f.get("arguments"));
            let args = parse_tool_args_value(args_val)?;
            let mut call = ToolCall::new(name, args);
            if i == 0 {
                if let Some(r) = reasoning.as_ref() {
                    call = call.with_reasoning(r);
                }
            }
            calls.push(call);
        }
        Ok(Some(calls))
    }
}

pub(super) fn parse_tool_args_value(
    args_val: Option<&Value>,
) -> Result<IndexMap<String, Value>, String> {
    match args_val {
        Some(Value::Object(obj)) => Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(obj)) => Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            _ => Err(s.clone()),
        },
        _ => Ok(IndexMap::new()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn reasoning_message_thinking_preferred() {
        assert_eq!(
            OllamaClient::resolve_reasoning(
                true,
                &json!({"message": {"thinking": "s", "content": "c"}})
            ),
            Some("s".into())
        );
    }

    #[test]
    fn reasoning_content_fallback() {
        assert_eq!(
            OllamaClient::resolve_reasoning(true, &json!({"message": {"content": "c"}})),
            Some("c".into())
        );
    }

    #[test]
    fn reasoning_disabled() {
        assert!(
            OllamaClient::resolve_reasoning(false, &json!({"message": {"thinking": "s"}}))
                .is_none()
        );
    }

    #[test]
    fn reasoning_empty() {
        assert!(
            OllamaClient::resolve_reasoning(true, &json!({"message": {"content": ""}})).is_none()
        );
    }

    #[test]
    fn parse_tool_call() {
        let c = OllamaClient::new("llama3");
        let r = json!({"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read", "arguments": {"path": "/tmp/x"}}},
        ]}});
        match c.parse_send_response(&r, true) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "read");
                assert_eq!(calls[0].args["path"], "/tmp/x");
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_text() {
        let c = OllamaClient::new("llama3");
        match c.parse_send_response(&json!({"message": {"content": "Hi"}}), true) {
            LLMResponse::Text(t) => assert_eq!(t.content, "Hi"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_empty_tool_calls() {
        let c = OllamaClient::new("llama3");
        match c.parse_send_response(
            &json!({"message": {"content": "No tools", "tool_calls": []}}),
            true,
        ) {
            LLMResponse::Text(t) => assert_eq!(t.content, "No tools"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn usage_from_response() {
        let c = OllamaClient::new("llama3");
        c.record_usage(&json!({"prompt_eval_count": 100, "eval_count": 50}));
        let u = c.get_last_usage().expect("set");
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.total_tokens, 150);
    }

    #[test]
    fn reasoning_first_tool_call() {
        let c = OllamaClient::new("llama3").with_think(Some(true));
        let r = json!({"message": {"thinking": "thinking", "content": "", "tool_calls": [
            {"function": {"name": "run", "arguments": {}}}
        ]}});
        match c.parse_send_response(&r, true) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].reasoning, Some("thinking".into()))
            }
            _ => panic!("expected tool calls"),
        }
    }
}
