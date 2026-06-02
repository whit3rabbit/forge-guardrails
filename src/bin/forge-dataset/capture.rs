use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

use crate::cli::CaptureCli;
use crate::schema::{capture_candidate_call, CAPTURE_SCHEMA_VERSION};
use crate::stub_tools::{
    available_tools_for_training, content_as_message, execute_tool, registries_for_domains,
    request_tools, StubRegistry, StubScenario, ToolExecution,
};

pub(crate) async fn run(cli: CaptureCli) -> Result<(), String> {
    let registries = registries_for_domains(&cli.domains)?;
    ensure_parent_dir(&cli.output)?;
    let client = reqwest::Client::new();
    let chat_url = normalize_chat_completions_url(&cli.proxy_base_url);

    for registry in &registries {
        for scenario in &registry.scenarios {
            capture_scenario(&client, &chat_url, &cli, registry, scenario).await?;
        }
    }
    Ok(())
}

async fn capture_scenario(
    client: &reqwest::Client,
    chat_url: &str,
    cli: &CaptureCli,
    registry: &StubRegistry,
    scenario: &StubScenario,
) -> Result<(), String> {
    let example_group_id = format!(
        "forge-dataset-{}-{}-{}",
        registry.domain,
        scenario.id,
        unix_ms()
    );
    let request_tools = request_tools(registry);
    let available_tools = Value::Array(available_tools_for_training(registry));
    let mut messages = vec![
        json!({
            "role": "system",
            "content": "You are using a deterministic, harmless stub tool registry for dataset capture. Use tools only when they are needed for the user's request."
        }),
        json!({"role": "user", "content": scenario.user_request}),
    ];
    let mut completed_tools = Vec::new();
    let mut recent_errors = Vec::new();

    for turn in 0..cli.max_turns {
        let body = json!({
            "model": cli.model,
            "messages": messages,
            "tools": request_tools,
            "stream": false,
        });
        let response = post_json(client, chat_url, &body).await?;
        let choice = response
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .ok_or_else(|| "proxy response missing choices[0]".to_string())?;
        let message = choice
            .get("message")
            .ok_or_else(|| "proxy response missing choices[0].message".to_string())?;
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            break;
        }

        messages.push(assistant_message_from_response(message));
        for (call_index, raw_call) in tool_calls.iter().enumerate() {
            let parsed = parse_tool_call(raw_call, turn, call_index)?;
            let candidate_call = capture_candidate_call(&parsed.name, parsed.arguments.clone());
            let workflow_state = workflow_state(&completed_tools, &recent_errors);
            let execution = execute_tool(registry, &parsed.name, &parsed.arguments);
            let row = capture_row(
                &example_group_id,
                registry,
                scenario,
                turn,
                call_index,
                &parsed.call_id,
                &available_tools,
                workflow_state,
                candidate_call,
                &execution,
                &response,
                &cli.model,
            );
            append_jsonl(&cli.output, &row)?;

            if execution.status.as_str() == "ok" {
                completed_tools.push(parsed.name.clone());
            } else {
                recent_errors.push(content_as_message(&execution.content));
            }

            messages.push(json!({
                "role": "tool",
                "tool_call_id": parsed.call_id,
                "name": parsed.name,
                "content": content_as_message(&execution.content),
                "_forge": {"tool_status": execution.status.as_str()}
            }));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_row(
    example_group_id: &str,
    registry: &StubRegistry,
    scenario: &StubScenario,
    turn: usize,
    call_index: usize,
    tool_call_id: &str,
    available_tools: &Value,
    workflow_state: Value,
    candidate_call: Value,
    execution: &ToolExecution,
    proxy_response: &Value,
    model: &str,
) -> Value {
    json!({
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": "tool_call_candidate",
        "example_group_id": example_group_id,
        "source_bucket": "real_model_call",
        "user_request": scenario.user_request,
        "workflow_state": workflow_state,
        "available_tools": available_tools,
        "candidate_call": candidate_call,
        "tool_result": {
            "status": execution.status.as_str(),
            "content": execution.content,
        },
        "proxy_trace": {
            "domain": registry.domain,
            "scenario": scenario.id,
            "turn": turn,
            "call_index": call_index,
            "tool_call_id": tool_call_id,
            "response_id": proxy_response.get("id").cloned().unwrap_or(Value::Null),
            "finish_reason": proxy_response
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .cloned()
                .unwrap_or(Value::Null),
            "model": model,
        },
        "metadata": {
            "private_agent_log": true,
            "public_export_allowed": false,
            "provenance": {
                "source": "forge-dataset",
                "captured_at_unix_ms": unix_ms(),
                "domain": registry.domain,
                "scenario": scenario.id,
            }
        }
    })
}

fn workflow_state(completed_tools: &[String], recent_errors: &[String]) -> Value {
    json!({
        "required_steps": [],
        "completed_steps": completed_tools,
        "pending_steps": [],
        "terminal_tools": ["respond"],
        "recent_errors": recent_errors,
    })
}

async fn post_json(client: &reqwest::Client, url: &str, body: &Value) -> Result<Value, String> {
    let response = client
        .post(url)
        .header("content-type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|err| format!("failed to call proxy: {err}"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| format!("failed to read proxy response: {err}"))?;
    if !status.is_success() {
        return Err(format!("proxy returned HTTP {status}: {text}"));
    }
    serde_json::from_str(&text).map_err(|err| format!("failed to parse proxy JSON: {err}"))
}

fn assistant_message_from_response(message: &Value) -> Value {
    let mut out = Map::new();
    out.insert("role".to_string(), json!("assistant"));
    out.insert(
        "content".to_string(),
        message.get("content").cloned().unwrap_or(Value::Null),
    );
    if let Some(tool_calls) = message.get("tool_calls") {
        out.insert("tool_calls".to_string(), tool_calls.clone());
    }
    Value::Object(out)
}

struct ParsedToolCall {
    call_id: String,
    name: String,
    arguments: Value,
}

fn parse_tool_call(
    raw_call: &Value,
    turn: usize,
    call_index: usize,
) -> Result<ParsedToolCall, String> {
    let function = raw_call
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| "tool call missing function object".to_string())?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "tool call function.name must be a string".to_string())?
        .to_string();
    let arguments = match function.get("arguments") {
        Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::Object(obj)) => Value::Object(obj),
            Ok(_) => Value::Object(Map::new()),
            Err(err) => return Err(format!("tool call arguments are invalid JSON: {err}")),
        },
        Some(Value::Object(obj)) => Value::Object(obj.clone()),
        _ => Value::Object(Map::new()),
    };
    let call_id = raw_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("capture_call_{turn}_{call_index}"));
    Ok(ParsedToolCall {
        call_id,
        name,
        arguments,
    })
}

fn normalize_chat_completions_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

fn append_jsonl(path: &str, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    writeln!(file, "{line}").map_err(|err| format!("failed to write {path}: {err}"))
}

fn ensure_parent_dir(path: &str) -> Result<(), String> {
    let Some(parent) = Path::new(path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("failed to create {}: {err}", parent.display()))
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stub_tools::registries_for_domains;

    #[test]
    fn capture_row_includes_group_context_and_private_provenance() {
        let registry = registries_for_domains(&["shopping".to_string()])
            .expect("registry")
            .remove(0);
        let scenario = registry.scenarios.first().expect("scenario");
        let candidate = json!({
            "name": "compare_products",
            "arguments": {"product_ids": ["SKU-HEADPHONES-1", "SKU-HEADPHONES-2"]}
        });
        let execution = execute_tool(&registry, "compare_products", &candidate["arguments"]);
        let row = capture_row(
            "group-1",
            &registry,
            scenario,
            0,
            0,
            "call-1",
            &Value::Array(available_tools_for_training(&registry)),
            workflow_state(&[], &[]),
            candidate,
            &execution,
            &json!({"id": "chatcmpl-1", "choices": [{"finish_reason": "tool_calls"}]}),
            "test-model",
        );

        assert_eq!(row["schema_version"], CAPTURE_SCHEMA_VERSION);
        assert_eq!(row["example_group_id"], "group-1");
        assert_eq!(row["source_bucket"], "real_model_call");
        assert_eq!(row["proxy_trace"]["domain"], "shopping");
        assert_eq!(row["metadata"]["private_agent_log"], true);
        assert_eq!(row["metadata"]["public_export_allowed"], false);
    }

    #[test]
    fn normalizes_openai_compatible_base_urls() {
        assert_eq!(
            normalize_chat_completions_url("http://127.0.0.1:8081"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
        assert_eq!(
            normalize_chat_completions_url("http://127.0.0.1:8081/v1"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
    }
}
