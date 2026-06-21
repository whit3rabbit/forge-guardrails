use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde_json::Value;

use super::{helpers, response};
use crate::clients::base::{
    ChunkType, LLMResponse, StreamChunk, TextResponse, TokenUsage, ToolCall,
};
use crate::clients::openai_compat;
use crate::error::StreamError;
use crate::prompts::{extract_tool_call, rescue_tool_call};

pub(super) fn parse_openai_sse(
    resp: reqwest::Response,
    think: bool,
    tool_names: Vec<String>,
    is_prompt: bool,
    last_usage: Arc<Mutex<HashMap<i64, TokenUsage>>>,
    slot_id: i64,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    let byte_stream = resp.bytes_stream();
    let stream = async_stream::stream! {
        use futures_util::StreamExt;
        let mut inner = Box::pin(byte_stream);
        let mut line_buf = String::new();

        let mut acc_content = String::new();
        let mut acc_reasoning = String::new();
        let mut acc_tools: Vec<(String, String, Option<String>)> = Vec::new();
        let mut final_response: Option<LLMResponse> = None;
        let mut stream_usage = None;

        loop {
            while let Some(newline_pos) = line_buf.find('\n') {
                let raw = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();
                let Some(data) = raw.strip_prefix("data: ") else { continue; };
                if data == "[DONE]" {
                    match final_response.take() {
                        Some(response) => {
                            yield Ok(StreamChunk::new(ChunkType::Final)
                                .with_response(response)
                                .with_metadata(stream_usage.clone(), None, None));
                        }
                        None => {
                            yield Err(StreamError::default());
                        }
                    }
                    return;
                }
                let evt: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(usage) = evt.get("usage") {
                    let prompt = usage.get("prompt_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                    let completion = usage.get("completion_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                    let usage = TokenUsage::new(prompt, completion, prompt + completion);
                    if let Ok(mut guard) = last_usage.lock() {
                        guard.insert(slot_id, usage.clone());
                    }
                    stream_usage = Some(usage);
                }

                if !evt.get("choices").is_some_and(|c| c.as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
                    continue;
                }

                let choice = &evt["choices"][0];
                let delta = choice.get("delta");

                if let Some(d) = delta {
                    if let Some(rc) = d.get("reasoning_content").and_then(|r| r.as_str()) {
                        if !rc.is_empty() {
                            acc_reasoning.push_str(rc);
                        }
                    }

                    if let Some(text) = d.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            acc_content.push_str(text);
                            yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(text));
                        }
                    }

                    if let Some(tcs) = d.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                            while acc_tools.len() <= idx {
                                acc_tools.push((String::new(), String::new(), None));
                            }
                            if let Some(name) = openai_compat::tool_call_name(tc) {
                                acc_tools[idx].0 = name.to_string();
                            }
                            if let Some(args) = openai_compat::tool_call_arguments(tc) {
                                let delta = match args {
                                    Value::String(text) => text.clone(),
                                    other => other.to_string(),
                                };
                                acc_tools[idx].1.push_str(&delta);
                                yield Ok(StreamChunk::new(ChunkType::ToolCallDelta).with_content(delta));
                            }
                            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                acc_tools[idx].2 = Some(id.to_string());
                            }
                        }
                    }
                }

                if choice.get("finish_reason").and_then(|r| r.as_str()).is_some() {
                    final_response = Some(final_response_from_parts(
                        think,
                        is_prompt,
                        &tool_names,
                        &acc_content,
                        &acc_reasoning,
                        &acc_tools,
                    ));
                }
            }

            match inner.next().await {
                Some(Ok(bytes)) => line_buf.push_str(&String::from_utf8_lossy(&bytes)),
                Some(Err(e)) => { yield Err(StreamError::new(e.to_string())); return; }
                None => {
                    match final_response.take() {
                        Some(response) => {
                            yield Ok(StreamChunk::new(ChunkType::Final)
                                .with_response(response)
                                .with_metadata(stream_usage.clone(), None, None));
                        }
                        None => {
                            yield Err(StreamError::default());
                        }
                    }
                    return;
                }
            }
        }
    };
    stream
}

fn final_response_from_parts(
    think: bool,
    is_prompt: bool,
    tool_names: &[String],
    acc_content: &str,
    acc_reasoning: &str,
    acc_tools: &[(String, String, Option<String>)],
) -> LLMResponse {
    if !acc_tools.is_empty() {
        return native_tool_response(think, acc_content, acc_reasoning, acc_tools);
    }
    if is_prompt {
        return prompt_response(think, tool_names, acc_content, acc_reasoning);
    }

    let cleaned = if think {
        helpers::strip_reasoning_tags(acc_content)
    } else {
        acc_content.to_string()
    };
    LLMResponse::Text(TextResponse::new(cleaned))
}

fn native_tool_response(
    think: bool,
    acc_content: &str,
    acc_reasoning: &str,
    acc_tools: &[(String, String, Option<String>)],
) -> LLMResponse {
    let reasoning = if think {
        helpers::resolve_full_reasoning(acc_reasoning, acc_content)
    } else {
        None
    };
    let mut calls = Vec::new();
    let mut bad_args = false;
    for (i, (name, args_json, id)) in acc_tools.iter().enumerate() {
        let args = if args_json.is_empty() {
            IndexMap::new()
        } else {
            match response::parse_args_json(args_json) {
                Ok(args) => args,
                Err(_) => {
                    bad_args = true;
                    break;
                }
            }
        };
        let mut call = ToolCall::new(name, args);
        if let Some(id_val) = id {
            call = call.with_id(id_val);
        }
        if i == 0 {
            if let Some(r) = &reasoning {
                call = call.with_reasoning(r);
            }
        }
        calls.push(call);
    }

    if bad_args {
        let content = if acc_content.is_empty() {
            acc_tools
                .first()
                .map(|(_, args, _)| args.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            acc_content.to_string()
        };
        LLMResponse::Text(TextResponse::new(content))
    } else {
        LLMResponse::ToolCalls(calls)
    }
}

fn prompt_response(
    think: bool,
    tool_names: &[String],
    acc_content: &str,
    acc_reasoning: &str,
) -> LLMResponse {
    let (think_text, cleaned) = helpers::extract_think_tags(acc_content);
    let names: Vec<&str> = tool_names.iter().map(|n| n.as_str()).collect();
    let mut extracted = extract_tool_call(&cleaned, &names);
    if extracted.is_empty() {
        extracted = rescue_tool_call(&cleaned, &names);
    }
    if extracted.is_empty() {
        LLMResponse::Text(TextResponse::new(cleaned))
    } else {
        let mut result = extracted;
        if let Some(first) = result.first_mut() {
            let r = if think {
                helpers::resolve_full_reasoning(acc_reasoning, &think_text)
            } else {
                None
            };
            if let Some(r_val) = r {
                *first = first.clone().with_reasoning(&r_val);
            }
        }
        LLMResponse::ToolCalls(result)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[tokio::test]
    async fn parses_llama_cpp_top_level_stream_tool_call() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
            "\"name\":\"run\",\"arguments\":\"{\\\"x\\\"\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
            "\"arguments\":\":1}\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let response = sse_response(body).await;

        let mut stream = Box::pin(parse_openai_sse(
            response,
            true,
            vec!["run".to_string()],
            false,
            Arc::new(Mutex::new(HashMap::new())),
            0,
        ));

        let mut final_response = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("stream chunk");
            if chunk.chunk_type == ChunkType::Final {
                final_response = chunk.response;
            }
        }

        match final_response.expect("final response") {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].id.as_deref(), Some("call_1"));
                assert_eq!(calls[0].tool, "run");
                assert_eq!(calls[0].args["x"], 1);
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[test]
    fn prompt_stream_final_rescues_qwen_xml_tool_call() {
        let response = prompt_response(
            true,
            &["run".to_string()],
            "<think>reason</think><function=run><parameter=path>/tmp/a</parameter></function>",
            "",
        );

        match response {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "run");
                assert_eq!(calls[0].args["path"], "/tmp/a");
                assert_eq!(calls[0].reasoning, Some("reason".to_string()));
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[test]
    fn prompt_stream_final_rescues_mistral_bracket_tool_call() {
        let response = prompt_response(
            true,
            &["run".to_string()],
            "[TOOL_CALLS]run{\"path\":\"/tmp/a\"}",
            "",
        );

        match response {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "run");
                assert_eq!(calls[0].args["path"], "/tmp/a");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    async fn sse_response(body: String) -> reqwest::Response {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let body_len = body.len();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut request_buf = [0_u8; 1024];
            let _ = socket.read(&mut request_buf).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {body_len}\r\n\r\n{body}"
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });

        reqwest::Client::new()
            .get(format!("http://{addr}"))
            .send()
            .await
            .expect("response")
    }
}
