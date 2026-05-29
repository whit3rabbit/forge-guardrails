use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use forge_guardrails::{Message, MessageType, ToolSpec};
use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::scenarios::SmokeScenario;

pub(crate) fn write_hard_negatives(
    output: Option<&str>,
    row: &Value,
    scenario: &SmokeScenario,
    messages: &[Message],
) -> Result<(), String> {
    let Some(output) = output else {
        return Ok(());
    };
    let Some(corrected_positive) = scenario.corrected_positive.as_ref() else {
        return Ok(());
    };
    if row.get("success").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(());
    }

    let corrected_candidate_calls = corrected_candidate_calls(corrected_positive);
    let corrected_candidate_call = corrected_candidate_calls
        .as_array()
        .and_then(|calls| calls.first())
        .cloned()
        .unwrap_or(Value::Null);
    let corrected_final_response = corrected_positive
        .get("final_text")
        .cloned()
        .unwrap_or(Value::Null);
    let training_context = hard_negative_context(row, scenario, messages);
    let candidate_calls = candidate_calls_from_trace(row);
    let candidate_call = candidate_calls
        .as_array()
        .and_then(|calls| calls.last())
        .cloned()
        .unwrap_or(Value::Null);

    let outcome = json!({
        "scenario": scenario.name,
        "scenario_family": scenario.name,
        "user_request": scenario.user_message,
        "run": row.get("run").cloned().unwrap_or(Value::Null),
        "failure_kind": row.get("failure_kind").cloned().unwrap_or(Value::Null),
        "accuracy": row.get("accuracy").cloned().unwrap_or(Value::Null),
        "corrected_positive": corrected_positive,
        "corrected_candidate_call": corrected_candidate_call,
        "corrected_candidate_calls": corrected_candidate_calls,
        "corrected_final_response": corrected_final_response,
    });

    let tool_row = json!({
        "kind": "tool_call",
        "context": training_context,
        "candidate": {
            "tool_sequence": row.get("tool_sequence").cloned().unwrap_or(Value::Null),
            "tool_args": row.get("tool_args").cloned().unwrap_or(Value::Null),
            "candidate_call": candidate_call,
            "candidate_calls": candidate_calls,
        },
        "classifier_scores": row.get("classifier_scores").cloned().unwrap_or_else(|| json!([])),
        "outcome": outcome.clone(),
    });
    append_jsonl(
        &hard_negative_path(output, "tool_call_hard_negatives"),
        &tool_row,
    )?;

    let final_row = json!({
        "kind": "final_response",
        "context": training_context,
        "candidate": {
            "final_text": row.get("final_text").cloned().unwrap_or(Value::Null),
        },
        "final_response_classifier_scores": row
            .get("final_response_classifier_scores")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "outcome": outcome,
    });
    append_jsonl(
        &hard_negative_path(output, "final_response_hard_negatives"),
        &final_row,
    )
}

fn hard_negative_context(row: &Value, scenario: &SmokeScenario, messages: &[Message]) -> Value {
    json!({
        "user_request": scenario.user_message,
        "workflow_state": hard_negative_workflow_state(row, scenario, messages),
        "available_tools": available_tools(&scenario.workflow.tools),
        "required_facts": scenario.required_facts,
    })
}

fn hard_negative_workflow_state(
    row: &Value,
    scenario: &SmokeScenario,
    messages: &[Message],
) -> Value {
    let tool_sequence = row
        .get("tool_sequence")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let prior_tools = tool_sequence
        .iter()
        .take(tool_sequence.len().saturating_sub(1))
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let completed_steps = scenario
        .workflow
        .required_steps
        .iter()
        .filter(|step| prior_tools.iter().any(|name| name == &step.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let pending_steps = scenario
        .workflow
        .required_steps
        .iter()
        .filter(|step| !completed_steps.contains(step))
        .cloned()
        .collect::<Vec<_>>();
    let mut terminal_tools = scenario
        .workflow
        .terminal_tools
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    terminal_tools.sort();

    json!({
        "required_steps": scenario.workflow.required_steps,
        "completed_steps": completed_steps,
        "pending_steps": pending_steps,
        "terminal_tools": terminal_tools,
        "recent_errors": recent_errors(messages, row),
    })
}

fn available_tools(tools: &IndexMap<String, forge_guardrails::ToolDef>) -> Value {
    Value::Array(
        tools
            .values()
            .map(|tool| tool_spec_json(&tool.spec))
            .collect::<Vec<_>>(),
    )
}

fn tool_spec_json(spec: &ToolSpec) -> Value {
    json!({
        "name": spec.name,
        "description": spec.description,
        "parameters": spec
            .json_schema
            .clone()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    })
}

fn recent_errors(messages: &[Message], row: &Value) -> Value {
    let mut errors = messages
        .iter()
        .filter_map(|message| {
            let is_error = matches!(
                message.metadata.msg_type,
                MessageType::RetryNudge | MessageType::StepNudge | MessageType::PrerequisiteNudge
            ) || (message.metadata.msg_type == MessageType::ToolResult
                && message.content.contains("[ToolError]"));
            if is_error && !message.content.trim().is_empty() {
                Some(json!(message.content))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if let Some(error) = row.get("error_message").filter(|value| !value.is_null()) {
        errors.push(error.clone());
    }
    Value::Array(errors)
}

fn candidate_calls_from_trace(row: &Value) -> Value {
    let names = row
        .get("tool_sequence")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let args = row
        .get("tool_args")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Value::Array(
        names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let name = name.as_str()?;
                Some(json!({
                    "name": name,
                    "arguments": args.get(index).cloned().unwrap_or_else(|| json!({})),
                }))
            })
            .collect(),
    )
}

fn corrected_candidate_calls(corrected_positive: &Value) -> Value {
    if let Some(calls) = corrected_positive.get("candidate_calls") {
        return calls.clone();
    }
    if let Some(call) = corrected_positive.get("candidate_call") {
        return json!([call]);
    }
    json!([])
}

fn append_jsonl(path: &Path, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    writeln!(file, "{line}").map_err(|err| err.to_string())
}

fn hard_negative_path(output: &str, suffix: &str) -> PathBuf {
    let path = Path::new(output);
    let stem = path.file_stem().and_then(|value| value.to_str());
    let extension = path.extension().and_then(|value| value.to_str());
    let file_name = match (stem, extension) {
        (Some(stem), Some(extension)) => format!("{stem}.{suffix}.{extension}"),
        (Some(stem), None) => format!("{stem}.{suffix}"),
        _ => format!("{output}.{suffix}.jsonl"),
    };
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenarios::build_scenario;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn write_hard_negatives_creates_sibling_files_for_failed_rows() {
        let mut dir = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        dir.push(format!(
            "forge-eval-report-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("tempdir");
        let output = dir.join("rows.jsonl");
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        let row = json!({
            "scenario": "basic_2step",
            "run": 1,
            "success": false,
            "accuracy": false,
            "failure_kind": "accuracy_false",
            "tool_sequence": ["get_country_info", "summarize"],
            "tool_args": [{"country": "France"}, {"content": "bad"}],
            "final_text": "bad",
            "classifier_scores": [],
            "final_response_classifier_scores": [],
        });

        write_hard_negatives(output.to_str(), &row, &scenario, &[]).expect("write");

        let tool_path = dir.join("rows.tool_call_hard_negatives.jsonl");
        let final_path = dir.join("rows.final_response_hard_negatives.jsonl");
        let tool_text = fs::read_to_string(tool_path).expect("tool hard negatives");
        let final_text = fs::read_to_string(final_path).expect("final hard negatives");
        let tool_row: Value = serde_json::from_str(tool_text.trim()).expect("tool row");
        let final_row: Value = serde_json::from_str(final_text.trim()).expect("final row");
        assert_eq!(tool_row["kind"], json!("tool_call"));
        assert_eq!(
            tool_row["context"]["user_request"],
            json!("What is the capital of France?")
        );
        assert_eq!(
            tool_row["context"]["required_facts"],
            json!(["Paris", "capital"])
        );
        assert_eq!(
            tool_row["context"]["workflow_state"]["required_steps"],
            json!(["get_country_info"])
        );
        assert_eq!(
            tool_row["candidate"]["candidate_call"],
            json!({"name": "summarize", "arguments": {"content": "bad"}})
        );
        assert_eq!(
            tool_row["outcome"]["corrected_candidate_call"],
            json!({"name": "summarize", "arguments": {"content": "The capital of France is Paris."}})
        );
        assert_eq!(final_row["kind"], json!("final_response"));
        assert_eq!(
            final_row["outcome"]["corrected_final_response"],
            json!("The capital of France is Paris.")
        );
    }
}
