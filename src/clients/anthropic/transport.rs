use serde_json::Value;

use super::streaming::{process_anthropic_sse_line, AnthropicStreamState};
use super::usage::usage_from_response;
use super::{convert, AnthropicClient};
use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMClient, LLMRequestOptions, LLMResponse,
    LLMResponseEnvelope, LLMUsageDetails, SamplingParams,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

impl LLMClient for AnthropicClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<crate::clients::base::TokenUsage> {
        self.last_usage.lock().ok().and_then(|g| g.clone())
    }

    fn last_usage_details(&self) -> Option<LLMUsageDetails> {
        self.last_usage_details.lock().ok().and_then(|g| g.clone())
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        self.send_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, BackendError> {
        Ok(self
            .send_envelope_with_options(messages, tools, options)
            .await?
            .response)
    }

    async fn send_envelope_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponseEnvelope, BackendError> {
        let sampling = options.sampling.clone();
        if let Some(sp) = &sampling {
            log::debug!(
                "AnthropicClient: ignoring sampling keys: {:?}",
                sp.keys().collect::<Vec<_>>()
            );
        }

        let body = self.build_body_with_options(messages, tools, options, false);

        let mut req = self
            .http_client
            .post(format!("{}/messages", self.base_url))
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("x-api-key", key.as_str());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        let status = resp.status().as_u16() as i64;

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(BackendError::new(status, body_text));
        }

        let response_json: Value = resp
            .json()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;

        self.record_usage(&response_json);
        let (usage, usage_details) = usage_from_response(&response_json);
        Ok(LLMResponseEnvelope {
            response: convert::parse_response(&response_json),
            usage: Some(usage),
            usage_details,
            call_info: None,
        })
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        self.send_stream_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_stream_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<ChunkStream, StreamError> {
        let sampling = options.sampling.clone();
        if let Some(sp) = &sampling {
            log::debug!(
                "AnthropicClient: ignoring sampling keys: {:?}",
                sp.keys().collect::<Vec<_>>()
            );
        }

        let body = self.build_body_with_options(messages, tools, options, true);

        let mut req = self
            .http_client
            .post(format!("{}/messages", self.base_url))
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("x-api-key", key.as_str());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| StreamError::new(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16() as i64;
            let body_text = resp.text().await.unwrap_or_default();
            return Err(StreamError::new(format!(
                "Backend error (status {}): {}",
                status, body_text
            )));
        }

        // Incremental SSE streaming aligned with Python AnthropicClient.send_stream().
        // Tracks:
        //   content_block_start  → detect tool_use blocks by index
        //   content_block_delta  → text_delta (TEXT_DELTA) or input_json_delta (TOOL_CALL_DELTA)
        //   content_block_stop   → reset current tool index
        //   message_stop         → emit FINAL chunk built from accumulated state
        //   message_delta        → capture usage (input/output_tokens)
        let byte_stream = resp.bytes_stream();
        let last_usage = self.last_usage.clone();
        let last_usage_details = self.last_usage_details.clone();
        let stream = async_stream::stream! {
            use futures_util::StreamExt;
            // SSE line buffer for partial-line data across byte chunks.
            let mut line_buf = String::new();
            let mut inner = Box::pin(byte_stream);

            let mut state = AnthropicStreamState::default();

            loop {
                match inner.next().await {
                    Some(Ok(bytes)) => {
                        line_buf.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Some(Err(e)) => {
                        yield Err(StreamError::new(e.to_string()));
                        return;
                    }
                    None => break,
                }
                // Process complete lines from the buffer.
                while let Some(newline_pos) = line_buf.find('\n') {
                    let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                    line_buf = line_buf[newline_pos + 1..].to_string();

                    match process_anthropic_sse_line(
                        &line,
                        &mut state,
                        &last_usage,
                        &last_usage_details,
                    ) {
                        Ok(Some(chunk)) => {
                            let is_final = chunk.chunk_type == ChunkType::Final;
                            yield Ok(chunk);
                            if is_final {
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    }
                }
            }

            let final_line = line_buf.trim_end_matches('\r').to_string();
            if !final_line.is_empty() {
                match process_anthropic_sse_line(
                    &final_line,
                    &mut state,
                    &last_usage,
                    &last_usage_details,
                ) {
                    Ok(Some(chunk)) => {
                        yield Ok(chunk);
                        return;
                    }
                    Ok(None) => {}
                    Err(err) => {
                        yield Err(err);
                        return;
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(Some(200_000))
    }
}
