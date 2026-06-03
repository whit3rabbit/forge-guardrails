use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde_json::Value;

use crate::cli::ValidateCli;
use crate::schema::{
    is_training_label, validate_candidate_call, CAPTURE_SCHEMA_VERSION,
    TRAINING_INPUT_SCHEMA_VERSION, TRAINING_INPUT_SCHEMA_VERSION_V1, TRAINING_SCHEMA_VERSION,
};

const PROMPT_SCHEMA_VERSION: &str = "forge-dataset-tool-prompt/v1";
const PROXY_CAPTURE_SCHEMA_VERSION: &str = "forge-proxy-training-capture/v1";
const REJECT_SCHEMA_VERSION: &str = "forge-dataset-review-reject/v1";
const ASSEMBLE_CONFLICT_SCHEMA_VERSION: &str = "forge-dataset-assemble-conflict/v1";

pub(crate) fn run(cli: ValidateCli) -> Result<(), String> {
    let mut errors = Vec::new();
    for input in cli.inputs {
        match validate_file(&input) {
            Ok(summary) => {
                println!(
                    "validated {} rows={} schemas={}",
                    input,
                    summary.rows,
                    summary.schema_counts_text()
                );
            }
            Err(err) => errors.push(err),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[derive(Debug, Default)]
struct ValidationSummary {
    rows: usize,
    schema_counts: BTreeMap<String, usize>,
}

impl ValidationSummary {
    fn schema_counts_text(&self) -> String {
        self.schema_counts
            .iter()
            .map(|(schema, count)| format!("{schema}:{count}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn validate_file(path: &str) -> Result<ValidationSummary, String> {
    let file = File::open(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut summary = ValidationSummary::default();
    let mut errors = Vec::new();

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                errors.push(format!("{path}:{line_number} read error: {err}"));
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = match serde_json::from_str::<Value>(trimmed) {
            Ok(row) => row,
            Err(err) => {
                errors.push(format!("{path}:{line_number} invalid JSONL row: {err}"));
                continue;
            }
        };
        summary.rows += 1;
        match validate_row(&row) {
            Ok(schema) => {
                *summary.schema_counts.entry(schema.to_string()).or_insert(0) += 1;
            }
            Err(err) => errors.push(format!("{path}:{line_number} {err}")),
        }
    }

    if errors.is_empty() {
        Ok(summary)
    } else {
        Err(errors.join("\n"))
    }
}

fn validate_row(row: &Value) -> Result<&str, String> {
    let schema = required_str(row, "schema_version")?;
    match schema {
        PROMPT_SCHEMA_VERSION => validate_prompt_row(row)?,
        PROXY_CAPTURE_SCHEMA_VERSION => validate_proxy_capture_row(row)?,
        CAPTURE_SCHEMA_VERSION => validate_capture_row(row)?,
        TRAINING_SCHEMA_VERSION => validate_training_row(row)?,
        REJECT_SCHEMA_VERSION => validate_reject_row(row)?,
        ASSEMBLE_CONFLICT_SCHEMA_VERSION => validate_conflict_row(row)?,
        other => return Err(format!("unknown schema_version '{other}'")),
    }
    Ok(schema)
}

fn validate_prompt_row(row: &Value) -> Result<(), String> {
    required_str(row, "domain")?;
    required_str(row, "scenario")?;
    required_str(row, "user_request")?;
    let request = required_object(row, "request")?;
    required_str(request, "model")?;
    required_array(request, "messages")?;
    required_array(request, "tools")?;
    required_array(row, "available_tools")?;
    validate_private_metadata(row)?;
    Ok(())
}

fn validate_capture_row(row: &Value) -> Result<(), String> {
    required_str(row, "kind")?;
    required_str(row, "example_group_id")?;
    required_str(row, "user_request")?;
    required_object(row, "workflow_state")?;
    let available_tools = required_array_value(row, "available_tools")?;
    let candidate_call = required_object_value(row, "candidate_call")?;
    validate_candidate_call(available_tools, candidate_call)?;
    required_object(row, "tool_result")?;
    required_object(row, "proxy_trace")?;
    validate_private_metadata(row)?;
    Ok(())
}

fn validate_proxy_capture_row(row: &Value) -> Result<(), String> {
    required_str(row, "kind")?;
    required_str(row, "example_group_id")?;
    required_str(row, "user_request")?;
    required_object(row, "workflow_state")?;
    let available_tools = required_array_value(row, "available_tools")?;
    let candidate_call = required_object_value(row, "candidate_call")?;
    validate_candidate_call(available_tools, candidate_call)?;
    required_str(row, "deterministic_status")?;
    validate_private_metadata(row)?;
    Ok(())
}

fn validate_training_row(row: &Value) -> Result<(), String> {
    let input = required_object(row, "input")?;
    let input_schema = required_str(input, "schema_version")?;
    if input_schema != TRAINING_INPUT_SCHEMA_VERSION
        && input_schema != TRAINING_INPUT_SCHEMA_VERSION_V1
    {
        return Err(format!(
            "input.schema_version must be {TRAINING_INPUT_SCHEMA_VERSION} or {TRAINING_INPUT_SCHEMA_VERSION_V1}"
        ));
    }
    required_str(input, "user_request")?;
    required_object(input, "workflow_state")?;
    let available_tools = required_array_value(input, "available_tools")?;
    let candidate_call = required_object_value(input, "candidate_call")?;
    validate_candidate_call(available_tools, candidate_call)?;
    let label = required_str(row, "label")?;
    if !is_training_label(label) {
        return Err(format!("unsupported label '{label}'"));
    }
    let review = required_object(row, "review")?;
    required_str(review, "source")?;
    if review.get("example_group_id").is_none() && review.get("task_group_id").is_none() {
        return Err("review must include example_group_id or task_group_id".to_string());
    }
    if let Some(corrected) = row
        .get("corrected_positive")
        .and_then(|value| value.get("candidate_call"))
    {
        validate_candidate_call(available_tools, corrected)?;
    }
    Ok(())
}

fn validate_reject_row(row: &Value) -> Result<(), String> {
    required_str(row, "reason")?;
    required_str(row, "detail")?;
    if let Some(capture) = row.get("capture") {
        validate_capture_row(capture)?;
    }
    Ok(())
}

fn validate_conflict_row(row: &Value) -> Result<(), String> {
    required_str(row, "scorer_input")?;
    required_object(row, "kept")?;
    required_object(row, "conflict")?;
    Ok(())
}

fn validate_private_metadata(row: &Value) -> Result<(), String> {
    let metadata = required_object(row, "metadata")?;
    if metadata.get("private_agent_log").and_then(Value::as_bool) != Some(true) {
        return Err("metadata.private_agent_log must be true".to_string());
    }
    if metadata
        .get("public_export_allowed")
        .and_then(Value::as_bool)
        != Some(false)
    {
        return Err("metadata.public_export_allowed must be false".to_string());
    }
    Ok(())
}

fn required_str<'a>(row: &'a Value, key: &str) -> Result<&'a str, String> {
    row.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn required_object<'a>(row: &'a Value, key: &str) -> Result<&'a Value, String> {
    required_object_value(row, key)
}

fn required_array<'a>(row: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    row.get(key)
        .and_then(Value::as_array)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty array"))
}

fn required_array_value<'a>(row: &'a Value, key: &str) -> Result<&'a Value, String> {
    let value = row
        .get(key)
        .ok_or_else(|| format!("{key} must be a non-empty array"))?;
    if value.as_array().is_some_and(|items| !items.is_empty()) {
        Ok(value)
    } else {
        Err(format!("{key} must be a non-empty array"))
    }
}

fn required_object_value<'a>(row: &'a Value, key: &str) -> Result<&'a Value, String> {
    let value = row
        .get(key)
        .ok_or_else(|| format!("{key} must be an object"))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(format!("{key} must be an object"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn training_row() -> Value {
        json!({
            "schema_version": TRAINING_SCHEMA_VERSION,
            "input": {
                "schema_version": TRAINING_INPUT_SCHEMA_VERSION,
                "user_request": "Compare products.",
                "workflow_state": {
                    "required_steps": [],
                    "completed_steps": [],
                    "pending_steps": [],
                    "terminal_tools": ["respond"],
                    "recent_errors": []
                },
                "available_tools": [{
                    "name": "compare_products",
                    "description": "Compare products.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_ids": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["product_ids"]
                    }
                }],
                "candidate_call": {
                    "name": "compare_products",
                    "arguments": {"product_ids": ["SKU-1", "SKU-2"]}
                }
            },
            "label": "valid",
            "review": {
                "source": "forge-dataset",
                "source_bucket": "real_model_call",
                "example_group_id": "group-1"
            }
        })
    }

    #[test]
    fn validates_training_row_envelope() {
        validate_training_row(&training_row()).expect("valid row");
    }

    #[test]
    fn rejects_unknown_training_label() {
        let mut row = training_row();
        row["label"] = json!("synthetic_unrelated_tool");
        let err = validate_training_row(&row).expect_err("invalid label");
        assert!(err.contains("unsupported label"));
    }
}
