use serde_json::{json, Value};

use super::io::normalize_chat_completions_url;
use super::types::ProviderConfig;
use crate::schema::parse_json_object_from_text;

pub(crate) struct JsonLlmClient {
    pub(crate) provider: String,
    pub(crate) chat_url: String,
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) http_client: reqwest::Client,
}

impl JsonLlmClient {
    pub(crate) fn new(config: ProviderConfig) -> Self {
        Self {
            provider: config.provider,
            chat_url: normalize_chat_completions_url(&config.chat_url),
            model: config.model,
            api_key: config.api_key,
            http_client: reqwest::Client::new(),
        }
    }

    pub(crate) async fn complete_json(
        &self,
        system: &str,
        user: &str,
        response_schema: Option<Value>,
    ) -> Result<Value, String> {
        let strict_schema = self.provider == "openrouter";
        let mut body = self.request_body(system, user);
        if strict_schema {
            if let Some(schema) = response_schema.clone() {
                attach_openrouter_strict_schema(&mut body, schema);
            }
        }
        match self.post_chat(&body).await {
            Ok(value) => Ok(value),
            Err(err) if strict_schema && openrouter_strict_schema_unavailable(&err) => {
                eprintln!(
                    "api fallback api=openrouter model={} reason=strict_json_schema_unavailable",
                    self.model
                );
                self.post_chat(&self.request_body(system, user)).await
            }
            Err(err) => Err(err),
        }
    }

    pub(crate) fn request_body(&self, system: &str, user: &str) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0,
        });
        if let Some(obj) = body.as_object_mut() {
            let token_key = if self.provider == "openrouter" {
                "max_tokens"
            } else {
                "max_completion_tokens"
            };
            obj.insert(token_key.to_string(), json!(1200));
        }
        body
    }

    async fn post_chat(&self, body: &Value) -> Result<Value, String> {
        let mut request = self
            .http_client
            .post(&self.chat_url)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(api_key) = self.api_key.as_deref() {
            request = request.bearer_auth(api_key);
        }
        let response = request
            .send()
            .await
            .map_err(|err| format!("failed to call reviewer/verifier LLM: {err}"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| format!("failed to read reviewer/verifier response: {err}"))?;
        if !status.is_success() {
            return Err(format!("reviewer/verifier returned HTTP {status}: {text}"));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|err| format!("failed to parse reviewer/verifier HTTP JSON: {err}"))?;
        let content = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| "reviewer/verifier response missing message.content".to_string())?;
        parse_json_object_from_text(content)
    }
}

pub(crate) fn attach_openrouter_strict_schema(body: &mut Value, schema: Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    obj.insert(
        "response_format".to_string(),
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "forge_dataset_review",
                "strict": true,
                "schema": schema,
            }
        }),
    );
    obj.insert("provider".to_string(), json!({"require_parameters": true}));
}

pub(crate) fn openrouter_strict_schema_unavailable(error: &str) -> bool {
    error.contains("HTTP 404")
        && error.contains("No endpoints found that can handle the requested parameters")
}

pub(crate) fn reviewer_response_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "label": {"type": "string", "enum": [
                "valid",
                "wrong_tool_semantic",
                "wrong_arguments_semantic",
                "tool_not_needed",
                "needs_clarification"
            ]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
            "corrected_candidate_call": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"}
                        },
                        "required": ["name", "arguments"]
                    }
                ]
            }
        },
        "required": ["label", "confidence", "rationale", "corrected_candidate_call"]
    })
}

pub(crate) fn verifier_response_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "accepted": {"type": "boolean"},
            "rationale": {"type": "string"}
        },
        "required": ["accepted", "rationale"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_request_uses_supported_token_parameter() {
        let client = JsonLlmClient::new(ProviderConfig {
            provider: "openrouter".to_string(),
            chat_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            model: "openrouter/free".to_string(),
            api_key: None,
        });
        let body = client.request_body("system", "user");
        assert_eq!(body["max_tokens"], 1200);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn minimax_request_keeps_openai_compatible_token_parameter() {
        let client = JsonLlmClient::new(ProviderConfig {
            provider: "minimax".to_string(),
            chat_url: "https://api.minimax.io/v1/chat/completions".to_string(),
            model: "MiniMax-M2.7".to_string(),
            api_key: None,
        });
        let body = client.request_body("system", "user");
        assert_eq!(body["max_completion_tokens"], 1200);
        assert!(body.get("max_tokens").is_none());
    }
}
