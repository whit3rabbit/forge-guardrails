use std::collections::HashSet;
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
    touch_jsonl(&cli.output)?;
    let client = reqwest::Client::new();
    let chat_url = normalize_chat_completions_url(&cli.proxy_base_url);
    let scenarios_per_run = registries
        .iter()
        .map(|registry| registry.scenarios.len())
        .sum::<usize>();
    let total_scenarios = scenarios_per_run * cli.runs;
    let mut scenario_index = 0;
    let mut scenario_errors = 0usize;

    eprintln!(
        "capture start output={} model={} domains={} runs={} scenarios={} max_turns={} max_scenario_errors={}",
        cli.output,
        cli.model,
        cli.domains.join(","),
        cli.runs,
        total_scenarios,
        cli.max_turns,
        cli.max_scenario_errors
    );

    for run_index in 0..cli.runs {
        for registry in &registries {
            for scenario in &registry.scenarios {
                scenario_index += 1;
                eprintln!(
                    "capture scenario {}/{} run={} domain={} scenario={}",
                    scenario_index, total_scenarios, run_index, registry.domain, scenario.id
                );
                if let Err(err) =
                    capture_scenario(&client, &chat_url, &cli, registry, scenario, run_index).await
                {
                    scenario_errors += 1;
                    eprintln!(
                        "capture warning scenario {}/{} run={} domain={} scenario={} failed: {}",
                        scenario_index,
                        total_scenarios,
                        run_index,
                        registry.domain,
                        scenario.id,
                        err
                    );
                    if scenario_errors > cli.max_scenario_errors {
                        return Err(format!(
                            "capture exceeded --max-scenario-errors {} after {} failed scenarios",
                            cli.max_scenario_errors, scenario_errors
                        ));
                    }
                }
            }
        }
    }
    eprintln!(
        "capture complete output={} scenario_errors={}",
        cli.output, scenario_errors
    );
    Ok(())
}

async fn capture_scenario(
    client: &reqwest::Client,
    chat_url: &str,
    cli: &CaptureCli,
    registry: &StubRegistry,
    scenario: &StubScenario,
    run_index: usize,
) -> Result<(), String> {
    let example_group_id = format!(
        "forge-dataset-{}-{}-run-{:05}-{}",
        registry.domain,
        scenario.id,
        run_index,
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
    let mut seen_tool_call_ids = HashSet::new();

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
        let normalized_tool_calls =
            normalize_tool_call_ids(&tool_calls, turn, &mut seen_tool_call_ids);

        messages.push(assistant_message_from_response(
            message,
            &normalized_tool_calls,
        ));
        for (call_index, raw_call) in normalized_tool_calls.iter().enumerate() {
            let parsed = parse_tool_call(raw_call, turn, call_index)?;
            let candidate_call = capture_candidate_call(&parsed.name, parsed.arguments.clone());
            let workflow_state = workflow_state(&completed_tools, &recent_errors);
            let execution = execute_tool(registry, &parsed.name, &parsed.arguments);
            let row = capture_row(
                &example_group_id,
                registry,
                scenario,
                run_index,
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
            eprintln!(
                "capture row group={} turn={} call={} tool={} status={}",
                example_group_id,
                turn,
                call_index,
                parsed.name,
                execution.status.as_str()
            );

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
    run_index: usize,
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
            "run_index": run_index,
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
                "run_index": run_index,
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

fn assistant_message_from_response(message: &Value, tool_calls: &[Value]) -> Value {
    let mut out = Map::new();
    out.insert("role".to_string(), json!("assistant"));
    out.insert(
        "content".to_string(),
        message.get("content").cloned().unwrap_or(Value::Null),
    );
    if !tool_calls.is_empty() {
        out.insert("tool_calls".to_string(), Value::Array(tool_calls.to_vec()));
    }
    Value::Object(out)
}

fn normalize_tool_call_ids(
    tool_calls: &[Value],
    turn: usize,
    seen: &mut HashSet<String>,
) -> Vec<Value> {
    let mut normalized = Vec::with_capacity(tool_calls.len());
    for (call_index, raw_call) in tool_calls.iter().enumerate() {
        let mut call = raw_call.clone();
        let raw_id = raw_call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("capture_call_{turn}_{call_index}"));
        let mut id = raw_id.clone();
        if !seen.insert(id.clone()) {
            let mut suffix = 1;
            loop {
                id = format!("{raw_id}_dup_{suffix}");
                if seen.insert(id.clone()) {
                    break;
                }
                suffix += 1;
            }
            eprintln!(
                "capture warning turn={} call={} duplicate tool_call_id={} normalized_to={}",
                turn, call_index, raw_id, id
            );
        }
        if let Some(obj) = call.as_object_mut() {
            obj.insert("id".to_string(), Value::String(id));
        }
        normalized.push(call);
    }
    normalized
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
    let mut line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    line.push('\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    file.write_all(line.as_bytes())
        .map_err(|err| format!("failed to write {path}: {err}"))?;
    file.flush()
        .map_err(|err| format!("failed to flush {path}: {err}"))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {path}: {err}"))
}

fn touch_jsonl(path: &str) -> Result<(), String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {path}: {err}"))
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
        assert_eq!(row["proxy_trace"]["run_index"], 0);
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

    #[test]
    fn normalizes_duplicate_tool_call_ids_before_history_reuse() {
        let tool_calls = vec![
            json!({
                "id": "call_DUPLICATE",
                "type": "function",
                "function": {"name": "lookup_ticket", "arguments": "{}"}
            }),
            json!({
                "id": "call_DUPLICATE",
                "type": "function",
                "function": {"name": "lookup_ticket", "arguments": "{}"}
            }),
        ];

        let mut seen = HashSet::new();
        let normalized = normalize_tool_call_ids(&tool_calls, 2, &mut seen);
        assert_eq!(normalized[0]["id"], "call_DUPLICATE");
        assert_eq!(normalized[1]["id"], "call_DUPLICATE_dup_1");

        let assistant = assistant_message_from_response(&json!({"content": null}), &normalized);
        assert_eq!(
            assistant["tool_calls"]
                .as_array()
                .expect("tool calls")
                .iter()
                .filter_map(|call| call.get("id").and_then(Value::as_str))
                .collect::<Vec<_>>(),
            vec!["call_DUPLICATE", "call_DUPLICATE_dup_1"]
        );
        assert_eq!(
            parse_tool_call(&normalized[1], 2, 1)
                .expect("parsed")
                .call_id,
            "call_DUPLICATE_dup_1"
        );
    }

    #[test]
    fn normalizes_missing_tool_call_ids_before_history_reuse() {
        let tool_calls = vec![
            json!({
                "type": "function",
                "function": {"name": "lookup_ticket", "arguments": "{}"}
            }),
            json!({
                "id": "",
                "type": "function",
                "function": {"name": "lookup_ticket", "arguments": "{}"}
            }),
        ];

        let mut seen = HashSet::new();
        let normalized = normalize_tool_call_ids(&tool_calls, 3, &mut seen);
        assert_eq!(normalized[0]["id"], "capture_call_3_0");
        assert_eq!(normalized[1]["id"], "capture_call_3_1");
        assert_eq!(
            parse_tool_call(&normalized[0], 3, 0)
                .expect("parsed")
                .call_id,
            "capture_call_3_0"
        );
        assert_eq!(
            parse_tool_call(&normalized[1], 3, 1)
                .expect("parsed")
                .call_id,
            "capture_call_3_1"
        );
    }

    #[test]
    fn normalizes_tool_call_ids_across_scenario_history() {
        let first_turn = vec![json!({
            "id": "call_REUSED",
            "type": "function",
            "function": {"name": "search_kb", "arguments": "{}"}
        })];
        let second_turn = vec![json!({
            "id": "call_REUSED",
            "type": "function",
            "function": {"name": "search_kb", "arguments": "{}"}
        })];

        let mut seen = HashSet::new();
        let first_normalized = normalize_tool_call_ids(&first_turn, 0, &mut seen);
        let second_normalized = normalize_tool_call_ids(&second_turn, 1, &mut seen);

        assert_eq!(first_normalized[0]["id"], "call_REUSED");
        assert_eq!(second_normalized[0]["id"], "call_REUSED_dup_1");
        assert_eq!(
            parse_tool_call(&second_normalized[0], 1, 0)
                .expect("parsed")
                .call_id,
            "call_REUSED_dup_1"
        );
    }
}
