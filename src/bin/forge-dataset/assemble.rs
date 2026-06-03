use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::cli::AssembleCli;
use crate::schema::{
    is_training_label, validate_candidate_call, TRAINING_INPUT_SCHEMA_VERSION,
    TRAINING_INPUT_SCHEMA_VERSION_V1, TRAINING_SCHEMA_VERSION,
};

const NOTEBOOK_OUTPUT: &str = "agent_training.notebook.jsonl";
const MANIFEST_OUTPUT: &str = "dataset_manifest.json";
const QUARANTINE_OUTPUT: &str = "quarantine.jsonl";
const CONFLICTS_OUTPUT: &str = "conflicts.jsonl";
const QUARANTINE_SCHEMA_VERSION: &str = "forge-dataset-review-reject/v1";
const CONFLICT_SCHEMA_VERSION: &str = "forge-dataset-assemble-conflict/v1";

pub(crate) fn run(cli: AssembleCli) -> Result<(), String> {
    let summary = assemble(cli)?;
    println!(
        "assembled rows={} duplicates={} conflicts={} quarantine={} output={}",
        summary.accepted,
        summary.duplicates,
        summary.conflicts,
        summary.quarantine,
        summary.combined_path.display()
    );
    Ok(())
}

#[derive(Debug)]
struct AssembleSummary {
    accepted: usize,
    duplicates: usize,
    conflicts: usize,
    quarantine: usize,
    combined_path: PathBuf,
}

fn assemble(cli: AssembleCli) -> Result<AssembleSummary, String> {
    fs::create_dir_all(&cli.out_dir)
        .map_err(|err| format!("failed to create {}: {err}", cli.out_dir))?;
    let out_dir = PathBuf::from(&cli.out_dir);
    let combined_path = combined_path(&out_dir, &cli.combined_output);
    let notebook_path = out_dir.join(NOTEBOOK_OUTPUT);
    let manifest_path = out_dir.join(MANIFEST_OUTPUT);
    let quarantine_path = out_dir.join(QUARANTINE_OUTPUT);
    let conflicts_path = out_dir.join(CONFLICTS_OUTPUT);

    let mut state = AssembleState::default();
    for (input_index, input) in cli.inputs.iter().enumerate() {
        read_input(input, input_index, &mut state)?;
    }

    write_jsonl(&combined_path, &state.rows)?;
    let adapter_rows = state
        .rows
        .iter()
        .map(notebook_adapter_row)
        .collect::<Vec<_>>();
    write_jsonl(&notebook_path, &adapter_rows)?;
    write_jsonl(&quarantine_path, &state.quarantine)?;
    write_jsonl(&conflicts_path, &state.conflicts)?;
    write_manifest(&manifest_path, &cli, &state, &combined_path)?;

    Ok(AssembleSummary {
        accepted: state.rows.len(),
        duplicates: state.duplicates,
        conflicts: state.conflicts.len(),
        quarantine: state.quarantine.len(),
        combined_path,
    })
}

#[derive(Debug, Default)]
struct AssembleState {
    rows: Vec<Value>,
    quarantine: Vec<Value>,
    conflicts: Vec<Value>,
    seen_by_input: HashMap<String, SeenRow>,
    duplicates: usize,
    input_stats: Vec<InputStats>,
}

#[derive(Debug, Clone)]
struct SeenRow {
    label: String,
    row: Value,
    input_path: String,
    line_number: usize,
}

#[derive(Debug, Default)]
struct InputStats {
    path: String,
    rows_seen: usize,
    accepted: usize,
    duplicates: usize,
    conflicts: usize,
    quarantine: usize,
}

fn read_input(path: &str, input_index: usize, state: &mut AssembleState) -> Result<(), String> {
    let file = File::open(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut stats = InputStats {
        path: path.to_string(),
        ..InputStats::default()
    };

    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                stats.quarantine += 1;
                state.quarantine.push(quarantine_row(
                    path,
                    line_number,
                    "read_error",
                    &err.to_string(),
                    Value::Null,
                ));
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        stats.rows_seen += 1;
        let mut row = match serde_json::from_str::<Value>(trimmed) {
            Ok(row) => row,
            Err(err) => {
                stats.quarantine += 1;
                state.quarantine.push(quarantine_row(
                    path,
                    line_number,
                    "invalid_json",
                    &err.to_string(),
                    Value::String(trimmed.to_string()),
                ));
                continue;
            }
        };

        if let Err(err) =
            validate_assemble_training_row(&row).and_then(|_| enforce_private_provenance(&mut row))
        {
            stats.quarantine += 1;
            state.quarantine.push(quarantine_row(
                path,
                line_number,
                "invalid_training_row",
                &err,
                row,
            ));
            continue;
        }

        let input_key = serialize_model_input(&row["input"]);
        let label = row["label"].as_str().unwrap_or("").to_string();
        if let Some(seen) = state.seen_by_input.get(&input_key) {
            if seen.label == label {
                stats.duplicates += 1;
                state.duplicates += 1;
                continue;
            }
            stats.conflicts += 1;
            state
                .conflicts
                .push(conflict_row(&input_key, seen, path, line_number, &row));
            continue;
        }

        state.seen_by_input.insert(
            input_key,
            SeenRow {
                label,
                row: row.clone(),
                input_path: path.to_string(),
                line_number,
            },
        );
        stats.accepted += 1;
        state.rows.push(row);
    }

    if state.input_stats.len() == input_index {
        state.input_stats.push(stats);
    } else {
        state.input_stats.insert(input_index, stats);
    }
    Ok(())
}

fn validate_assemble_training_row(row: &Value) -> Result<(), String> {
    if row.get("schema_version").and_then(Value::as_str) != Some(TRAINING_SCHEMA_VERSION) {
        return Err(format!("schema_version must be {TRAINING_SCHEMA_VERSION}"));
    }
    let input = row
        .get("input")
        .filter(|value| value.is_object())
        .ok_or_else(|| "input must be an object".to_string())?;
    let input_schema = input
        .get("schema_version")
        .and_then(Value::as_str)
        .ok_or_else(|| "input.schema_version must be a string".to_string())?;
    if input_schema != TRAINING_INPUT_SCHEMA_VERSION
        && input_schema != TRAINING_INPUT_SCHEMA_VERSION_V1
    {
        return Err(format!(
            "input.schema_version must be {TRAINING_INPUT_SCHEMA_VERSION} or {TRAINING_INPUT_SCHEMA_VERSION_V1}"
        ));
    }
    input
        .get("user_request")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "input.user_request must be a non-empty string".to_string())?;
    input
        .get("workflow_state")
        .filter(|value| value.is_object())
        .ok_or_else(|| "input.workflow_state must be an object".to_string())?;
    let available_tools = input
        .get("available_tools")
        .filter(|value| value.as_array().is_some_and(|items| !items.is_empty()))
        .ok_or_else(|| "input.available_tools must be a non-empty array".to_string())?;
    let candidate_call = input
        .get("candidate_call")
        .filter(|value| value.is_object())
        .ok_or_else(|| "input.candidate_call must be an object".to_string())?;
    validate_candidate_call(available_tools, candidate_call)?;
    let label = row
        .get("label")
        .and_then(Value::as_str)
        .ok_or_else(|| "label must be a string".to_string())?;
    if !is_training_label(label) {
        return Err(format!("unsupported label '{label}'"));
    }
    let review = row
        .get("review")
        .filter(|value| value.is_object())
        .ok_or_else(|| "review must be an object".to_string())?;
    review
        .get("source")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "review.source must be a non-empty string".to_string())?;
    Ok(())
}

fn enforce_private_provenance(row: &mut Value) -> Result<(), String> {
    reject_explicit_public_export(row)?;
    reject_explicit_non_private(row)?;
    let review = row
        .get_mut("review")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "review must be an object".to_string())?;
    review.insert("private_agent_log".to_string(), Value::Bool(true));
    review.insert("public_export_allowed".to_string(), Value::Bool(false));
    Ok(())
}

fn reject_explicit_public_export(value: &Value) -> Result<(), String> {
    match value {
        Value::Object(map) => {
            if map.get("public_export_allowed").and_then(Value::as_bool) == Some(true) {
                return Err("public_export_allowed must not be true".to_string());
            }
            for child in map.values() {
                reject_explicit_public_export(child)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                reject_explicit_public_export(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn reject_explicit_non_private(value: &Value) -> Result<(), String> {
    match value {
        Value::Object(map) => {
            if map.get("private_agent_log").and_then(Value::as_bool) == Some(false) {
                return Err("private_agent_log must not be false".to_string());
            }
            for child in map.values() {
                reject_explicit_non_private(child)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                reject_explicit_non_private(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn serialize_model_input(input: &Value) -> String {
    let workflow_state = input.get("workflow_state").unwrap_or(&Value::Null);
    let tool_text = input
        .get("available_tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .map(serialize_tool)
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default();
    format!(
        "SCHEMA_VERSION:\n{}\n\nUSER_REQUEST:\n{}\n\nWORKFLOW_STATE:\nrequired_steps={}\ncompleted_steps={}\npending_steps={}\nterminal_tools={}\nrecent_errors={}\n\nAVAILABLE_TOOLS:\n{}\n\nCANDIDATE_CALL:\n{}",
        input.get("schema_version").and_then(Value::as_str).unwrap_or(""),
        input.get("user_request").and_then(Value::as_str).unwrap_or(""),
        json_compact(workflow_state.get("required_steps").unwrap_or(&Value::Null)),
        json_compact(workflow_state.get("completed_steps").unwrap_or(&Value::Null)),
        json_compact(workflow_state.get("pending_steps").unwrap_or(&Value::Null)),
        json_compact(workflow_state.get("terminal_tools").unwrap_or(&Value::Null)),
        json_compact(workflow_state.get("recent_errors").unwrap_or(&Value::Null)),
        tool_text,
        json_compact(input.get("candidate_call").unwrap_or(&Value::Null)),
    )
}

fn serialize_tool(tool: &Value) -> String {
    let function = tool.get("function").filter(|value| value.is_object());
    let normalized = function.unwrap_or(tool);
    let name = normalized
        .get("name")
        .or_else(|| normalized.get("tool_name"))
        .and_then(Value::as_str)
        .unwrap_or("unknown_tool");
    let description = normalized
        .get("description")
        .or_else(|| normalized.get("desc"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let parameters = normalized
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    format!(
        "{}: {}\nPARAMETERS: {}",
        name,
        description,
        json_compact(&parameters)
    )
}

fn json_compact(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn notebook_adapter_row(row: &Value) -> Value {
    let input = row.get("input").unwrap_or(&Value::Null);
    let review = row.get("review").unwrap_or(&Value::Null);
    let mut metadata = input
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    metadata.insert("generator".to_string(), json!("forge-dataset"));
    metadata.insert("private_agent_log".to_string(), json!(true));
    metadata.insert("public_export_allowed".to_string(), json!(false));
    metadata.insert(
        "example_group_id".to_string(),
        review
            .get("example_group_id")
            .or_else(|| review.get("task_group_id"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    metadata.insert(
        "source".to_string(),
        review.get("source").cloned().unwrap_or(Value::Null),
    );
    if let Some(source_bucket) = review.get("source_bucket") {
        metadata.insert("source_bucket".to_string(), source_bucket.clone());
    }

    json!({
        "kind": "tool_call",
        "label": row.get("label").cloned().unwrap_or(Value::Null),
        "user_request": input.get("user_request").cloned().unwrap_or(Value::Null),
        "workflow_state": input.get("workflow_state").cloned().unwrap_or(Value::Null),
        "available_tools": input.get("available_tools").cloned().unwrap_or_else(|| json!([])),
        "candidate_call": input.get("candidate_call").cloned().unwrap_or(Value::Null),
        "metadata": Value::Object(metadata),
        "rank_score": review_confidence(review),
    })
}

fn review_confidence(review: &Value) -> Value {
    review
        .get("confidence")
        .or_else(|| {
            review
                .get("reviewer")
                .and_then(|value| value.get("confidence"))
        })
        .cloned()
        .unwrap_or_else(|| json!(1.0))
}

fn conflict_row(
    input_key: &str,
    seen: &SeenRow,
    path: &str,
    line_number: usize,
    row: &Value,
) -> Value {
    json!({
        "schema_version": CONFLICT_SCHEMA_VERSION,
        "scorer_input": input_key,
        "kept": {
            "path": seen.input_path,
            "line_number": seen.line_number,
            "label": seen.label,
            "row": seen.row,
        },
        "conflict": {
            "path": path,
            "line_number": line_number,
            "label": row.get("label").cloned().unwrap_or(Value::Null),
            "row": row,
        }
    })
}

fn quarantine_row(path: &str, line_number: usize, reason: &str, detail: &str, row: Value) -> Value {
    json!({
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "reason": reason,
        "detail": detail,
        "path": path,
        "line_number": line_number,
        "row": row,
    })
}

fn write_manifest(
    path: &Path,
    cli: &AssembleCli,
    state: &AssembleState,
    combined_path: &Path,
) -> Result<(), String> {
    let labels = label_counts(&state.rows);
    let inputs = state
        .input_stats
        .iter()
        .map(|stats| {
            json!({
                "path": stats.path,
                "rows_seen": stats.rows_seen,
                "accepted": stats.accepted,
                "duplicates": stats.duplicates,
                "conflicts": stats.conflicts,
                "quarantine": stats.quarantine,
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema_version": "forge-dataset-assembled-manifest/v1",
        "created_unix": unix_secs(),
        "inputs": inputs,
        "outputs": {
            "training_toolcall": combined_path.display().to_string(),
            "notebook_adapter": PathBuf::from(&cli.out_dir).join(NOTEBOOK_OUTPUT).display().to_string(),
            "quarantine": PathBuf::from(&cli.out_dir).join(QUARANTINE_OUTPUT).display().to_string(),
            "conflicts": PathBuf::from(&cli.out_dir).join(CONFLICTS_OUTPUT).display().to_string(),
        },
        "counts": {
            "accepted": state.rows.len(),
            "duplicates": state.duplicates,
            "conflicts": state.conflicts.len(),
            "quarantine": state.quarantine.len(),
        },
        "labels": labels,
        "metadata": {
            "private_agent_log": true,
            "public_export_allowed": false,
            "combined_output": cli.combined_output,
        }
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())? + "\n",
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn label_counts(rows: &[Value]) -> Value {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for row in rows {
        if let Some(label) = row.get("label").and_then(Value::as_str) {
            *counts.entry(label.to_string()).or_insert(0) += 1;
        }
    }
    json!(counts)
}

fn combined_path(out_dir: &Path, combined_output: &str) -> PathBuf {
    let path = PathBuf::from(combined_output);
    if path.is_absolute() {
        path
    } else {
        out_dir.join(path)
    }
}

fn write_jsonl(path: &Path, rows: &[Value]) -> Result<(), String> {
    let mut file =
        File::create(path).map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    for row in rows {
        writeln!(
            file,
            "{}",
            serde_json::to_string(row).map_err(|err| err.to_string())?
        )
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    Ok(())
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("forge-dataset-assemble-{}-{}", name, unix_secs()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn tool_row(label: &str) -> Value {
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
                },
                "metadata": {
                    "scenario_family": "shopping"
                }
            },
            "label": label,
            "review": {
                "source": "forge-dataset",
                "source_bucket": "real_model_call",
                "example_group_id": "group-1"
            }
        })
    }

    fn write_rows(path: &Path, rows: &[Value]) {
        let text = rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("json") + "\n")
            .collect::<String>();
        fs::write(path, text).expect("write rows");
    }

    #[test]
    fn assemble_dedupes_conflicts_and_writes_notebook_adapter() {
        let dir = temp_dir("dedupe");
        let input_a = dir.join("a.jsonl");
        let input_b = dir.join("b.jsonl");
        write_rows(&input_a, &[tool_row("valid"), tool_row("valid")]);
        write_rows(&input_b, &[tool_row("wrong_tool_semantic")]);

        let out_dir = dir.join("out");
        let summary = assemble(AssembleCli {
            inputs: vec![input_a.display().to_string(), input_b.display().to_string()],
            out_dir: out_dir.display().to_string(),
            combined_output: "combined.jsonl".to_string(),
        })
        .expect("assemble");

        assert_eq!(summary.accepted, 1);
        assert_eq!(summary.duplicates, 1);
        assert_eq!(summary.conflicts, 1);
        let combined = fs::read_to_string(out_dir.join("combined.jsonl")).expect("combined");
        assert_eq!(combined.lines().count(), 1);
        let row: Value = serde_json::from_str(combined.lines().next().expect("row")).expect("row");
        assert_eq!(row["review"]["private_agent_log"], true);
        assert_eq!(row["review"]["public_export_allowed"], false);
        let adapter = fs::read_to_string(out_dir.join(NOTEBOOK_OUTPUT)).expect("adapter");
        let adapter_row: Value = serde_json::from_str(adapter.lines().next().expect("adapter row"))
            .expect("adapter row");
        assert_eq!(adapter_row["kind"], "tool_call");
        assert_eq!(adapter_row["metadata"]["generator"], "forge-dataset");
    }

    #[test]
    fn assemble_quarantines_public_export_rows() {
        let dir = temp_dir("privacy");
        let input = dir.join("rows.jsonl");
        let mut row = tool_row("valid");
        row["review"]["public_export_allowed"] = json!(true);
        write_rows(&input, &[row]);

        let out_dir = dir.join("out");
        let summary = assemble(AssembleCli {
            inputs: vec![input.display().to_string()],
            out_dir: out_dir.display().to_string(),
            combined_output: "combined.jsonl".to_string(),
        })
        .expect("assemble");

        assert_eq!(summary.accepted, 0);
        assert_eq!(summary.quarantine, 1);
        let quarantine = fs::read_to_string(out_dir.join(QUARANTINE_OUTPUT)).expect("quarantine");
        assert!(quarantine.contains("public_export_allowed"));
    }
}
