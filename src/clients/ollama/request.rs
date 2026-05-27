use serde_json::{json, Value};

use super::{streaming, OllamaClient};
use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMClient, LLMResponse, SamplingParams, StreamChunk,
    TextResponse,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

impl OllamaClient {
    pub(super) fn build_request_body(
        &self,
        messages: Vec<Value>,
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
        think: bool,
    ) -> Value {
        let mut body = json!({"model": self.model, "messages": messages, "stream": false});
        if think {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("think".into(), json!(true));
            }
        }
        if let Some(tl) = tools {
            if !tl.is_empty() {
                let fmt: Vec<Value> = tl
                    .iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.get_json_schema()
                            },
                        })
                    })
                    .collect();
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tools".into(), json!(fmt));
                }
            }
        }
        let opts = self.build_options(sampling);
        if !opts.is_empty() {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("options".into(), Value::Object(opts));
            }
        }
        body
    }
}

impl LLMClient for OllamaClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::Ollama
    }

    fn last_usage(&self) -> Option<crate::clients::base::TokenUsage> {
        self.last_usage.lock().ok().and_then(|g| g.clone())
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        let think = self.think.lock().map(|g| *g).unwrap_or(false);
        let think_resolved = self.think_resolved.lock().map(|g| *g).unwrap_or(false);
        let body =
            self.build_request_body(messages.clone(), tools.as_deref(), sampling.as_ref(), think);
        let resp = match self
            .http_client
            .post(format!("{}/api/chat", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return Err(if e.is_timeout() {
                    BackendError::new(408, e.to_string())
                } else {
                    BackendError::new(0, e.to_string())
                });
            }
        };
        let status = resp.status().as_u16() as i64;
        if status == 500 {
            return Ok(LLMResponse::Text(TextResponse::new(
                resp.text().await.unwrap_or_default(),
            )));
        }
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            // Detect think-unsupported on non-200 non-500 — may be a 400.
            if status == 400 {
                if let Ok(ej) = serde_json::from_str::<Value>(&text) {
                    if Self::is_think_unsupported_error(&ej) {
                        if !think_resolved {
                            // Persist: disable think for future calls (Python parity).
                            if let Ok(mut g) = self.think.lock() {
                                *g = false;
                            }
                            if let Ok(mut g) = self.think_resolved.lock() {
                                *g = true;
                            }
                            let retry_body = self.build_request_body(
                                messages,
                                tools.as_deref(),
                                sampling.as_ref(),
                                false,
                            );
                            let retry_resp = self
                                .http_client
                                .post(format!("{}/api/chat", self.base_url))
                                .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
                                .json(&retry_body)
                                .send()
                                .await
                                .map_err(|e| BackendError::new(0, e.to_string()))?;
                            let rs = retry_resp.status().as_u16() as i64;
                            if rs == 500 {
                                return Ok(LLMResponse::Text(TextResponse::new(
                                    retry_resp.text().await.unwrap_or_default(),
                                )));
                            }
                            if !retry_resp.status().is_success() {
                                return Err(BackendError::new(
                                    rs,
                                    retry_resp.text().await.unwrap_or_default(),
                                ));
                            }
                            let rj2: Value = retry_resp
                                .json()
                                .await
                                .map_err(|e| BackendError::new(rs, e.to_string()))?;
                            self.record_usage(&rj2);
                            return Ok(self.parse_send_response(&rj2, false));
                        } else {
                            return Err(BackendError::ThinkingNotSupported {
                                model: self.model.clone(),
                                status_code: status,
                                body: text,
                            });
                        }
                    }
                }
            }
            return Err(BackendError::new(status, text));
        }
        let rj: Value = resp
            .json()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        // Mark resolved after first successful call.
        if !think_resolved {
            if let Ok(mut g) = self.think_resolved.lock() {
                *g = true;
            }
        }
        self.record_usage(&rj);
        Ok(self.parse_send_response(&rj, think))
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        let think = self.think.lock().map(|g| *g).unwrap_or(false);
        let think_resolved = self.think_resolved.lock().map(|g| *g).unwrap_or(false);
        let mut body =
            self.build_request_body(messages.clone(), tools.as_deref(), sampling.as_ref(), think);
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".to_string(), Value::Bool(true));
        }
        let resp = match self
            .http_client
            .post(format!("{}/api/chat", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return Err(if e.is_timeout() {
                    StreamError::new(format!("Backend error (status 408): {}", e))
                } else {
                    StreamError::new(e.to_string())
                });
            }
        };
        let status = resp.status().as_u16() as i64;
        if status == 500 {
            let text = resp.text().await.unwrap_or_default();
            let chunk = StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new(text)));
            return Ok(Box::pin(futures_util::stream::once(
                async move { Ok(chunk) },
            )));
        }
        if status == 400 {
            let bt = resp.text().await.unwrap_or_default();
            if let Ok(ej) = serde_json::from_str::<Value>(&bt) {
                if Self::is_think_unsupported_error(&ej) {
                    if !think_resolved {
                        // Persist: disable think for future calls (Python parity).
                        if let Ok(mut g) = self.think.lock() {
                            *g = false;
                        }
                        if let Ok(mut g) = self.think_resolved.lock() {
                            *g = true;
                        }
                        let rb = self.build_request_body(
                            messages,
                            tools.as_deref(),
                            sampling.as_ref(),
                            false,
                        );
                        let mut rb_obj = rb;
                        if let Some(obj) = rb_obj.as_object_mut() {
                            obj.insert("stream".to_string(), Value::Bool(true));
                        }
                        let rr = self
                            .http_client
                            .post(format!("{}/api/chat", self.base_url))
                            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
                            .json(&rb_obj)
                            .send()
                            .await
                            .map_err(|e| StreamError::new(e.to_string()))?;
                        return Ok(Box::pin(streaming::parse_ollama_ndjson(
                            rr,
                            false,
                            self.last_usage.clone(),
                        )));
                    } else {
                        return Err(StreamError::new(format!(
                            "Thinking mode not supported for model '{}'",
                            self.model
                        )));
                    }
                }
            }
            return Err(StreamError::new(format!(
                "Backend error (status 400): {}",
                bt
            )));
        }
        if !resp.status().is_success() {
            return Err(StreamError::new(format!(
                "Backend error (status {}): {}",
                status,
                resp.text().await.unwrap_or_default()
            )));
        }
        // Mark resolved after first successful call.
        if !think_resolved {
            if let Ok(mut g) = self.think_resolved.lock() {
                *g = true;
            }
        }
        Ok(Box::pin(streaming::parse_ollama_ndjson(
            resp,
            think,
            self.last_usage.clone(),
        )))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        let guard = self
            .num_ctx
            .lock()
            .map_err(|e| ContextDiscoveryError::new(e.to_string()))?;
        Ok(*guard)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_body_structure() {
        let c = OllamaClient::new("llama3").with_temperature(0.7);
        let b = c.build_request_body(
            vec![json!({"role": "user", "content": "Hi"})],
            None,
            None,
            false,
        );
        assert_eq!(b["model"], "llama3");
        assert_eq!(b["stream"], false);
        assert_eq!(b["options"]["temperature"], 0.7);
    }

    #[test]
    fn request_body_think() {
        let c = OllamaClient::new("llama3").with_think(Some(true));
        assert_eq!(
            c.build_request_body(vec![], None, None, true)["think"],
            true
        );
    }

    #[test]
    fn request_body_tools() {
        let s = ToolSpec::from_json_schema(
            "run",
            "Run",
            &json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        )
        .expect("ok");
        let c = OllamaClient::new("llama3");
        let b = c.build_request_body(
            vec![json!({"role": "user", "content": "Go"})],
            Some(&[s]),
            None,
            false,
        );
        assert_eq!(b["tools"].as_array().expect("a").len(), 1);
    }

    #[test]
    fn tool_role_passthrough() {
        let c = OllamaClient::new("llama3");
        let msgs = vec![json!({"role": "tool", "content": "data"})];
        let b = c.build_request_body(msgs.clone(), None, None, false);
        assert_eq!(b["messages"][0]["role"], "tool");
    }

    #[test]
    fn api_format_ollama() {
        assert_eq!(OllamaClient::new("llama3").api_format(), ApiFormat::Ollama);
    }

    #[test]
    fn tool_format_basic() {
        let s = ToolSpec::from_json_schema(
            "t",
            "T",
            &json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        )
        .expect("ok");
        let c = OllamaClient::new("llama3");
        let b = c.build_request_body(vec![], Some(&[s]), None, false);
        assert_eq!(b["tools"][0]["type"], "function");
    }

    #[test]
    fn tool_format_enum() {
        let s = ToolSpec::from_json_schema(
            "e",
            "E",
            &json!({"type": "object", "properties": {"m": {"type": "string", "enum": ["a","b"]}}}),
        )
        .expect("ok");
        let c = OllamaClient::new("llama3");
        let b = c.build_request_body(vec![], Some(&[s]), None, false);
        assert!(b["tools"][0]["function"]["parameters"]
            .get("properties")
            .is_some());
    }
}
