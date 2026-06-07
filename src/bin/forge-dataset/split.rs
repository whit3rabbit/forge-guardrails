use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::cli::SplitCli;
use crate::schema::{
    is_training_label, validate_candidate_call, TRAINING_INPUT_SCHEMA_VERSION,
    TRAINING_INPUT_SCHEMA_VERSION_V1, TRAINING_SCHEMA_VERSION,
};

const MANIFEST_OUTPUT: &str = "split_manifest.json";
const MANIFEST_SCHEMA_VERSION: &str = "forge-dataset-split-manifest/v1";

pub(crate) fn run(cli: SplitCli) -> Result<(), String> {
    let summary = split(cli)?;
    println!(
        "split rows={} train={} validation={} groups={} validation_groups={} train_output={} validation_output={}",
        summary.rows,
        summary.train_rows,
        summary.validation_rows,
        summary.groups,
        summary.validation_groups,
        summary.train_path.display(),
        summary.validation_path.display()
    );
    Ok(())
}

#[derive(Debug)]
struct SplitSummary {
    rows: usize,
    train_rows: usize,
    validation_rows: usize,
    groups: usize,
    validation_groups: usize,
    train_path: PathBuf,
    validation_path: PathBuf,
}

#[derive(Debug, Clone)]
struct SplitRow {
    row: Value,
    group_key: String,
}

fn split(cli: SplitCli) -> Result<SplitSummary, String> {
    fs::create_dir_all(&cli.out_dir)
        .map_err(|err| format!("failed to create {}: {err}", cli.out_dir))?;
    let out_dir = PathBuf::from(&cli.out_dir);
    let train_path = out_dir.join(&cli.train_output);
    let validation_path = out_dir.join(&cli.validation_output);
    let manifest_path = out_dir.join(MANIFEST_OUTPUT);

    let rows = read_training_rows(&cli.input)?;
    let validation_groups = choose_validation_groups(&rows, cli.validation_ratio, &cli.seed);

    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for row in &rows {
        if validation_groups.contains(&row.group_key) {
            validation_rows.push(row.row.clone());
        } else {
            train_rows.push(row.row.clone());
        }
    }

    write_jsonl_atomic(&train_path, &train_rows)?;
    write_jsonl_atomic(&validation_path, &validation_rows)?;
    write_json_atomic(
        &manifest_path,
        &manifest(
            &cli,
            &rows,
            &train_rows,
            &validation_rows,
            &validation_groups,
        ),
    )?;

    Ok(SplitSummary {
        rows: rows.len(),
        train_rows: train_rows.len(),
        validation_rows: validation_rows.len(),
        groups: group_count(&rows),
        validation_groups: validation_groups.len(),
        train_path,
        validation_path,
    })
}

fn read_training_rows(path: &str) -> Result<Vec<SplitRow>, String> {
    let file = File::open(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|err| format!("{path}:{line_number} read error: {err}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Value>(trimmed)
            .map_err(|err| format!("{path}:{line_number} invalid JSONL row: {err}"))?;
        validate_training_row(&row).map_err(|err| format!("{path}:{line_number} {err}"))?;
        rows.push(SplitRow {
            group_key: group_key(&row),
            row,
        });
    }
    Ok(rows)
}

fn validate_training_row(row: &Value) -> Result<(), String> {
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
    row.get("review")
        .filter(|value| value.is_object())
        .and_then(|review| review.get("source"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "review.source must be a non-empty string".to_string())?;
    Ok(())
}

fn choose_validation_groups(rows: &[SplitRow], ratio: f64, seed: &str) -> HashSet<String> {
    let mut group_sizes = BTreeMap::<String, usize>::new();
    for row in rows {
        *group_sizes.entry(row.group_key.clone()).or_insert(0) += 1;
    }
    if ratio <= 0.0 || group_sizes.is_empty() {
        return HashSet::new();
    }
    if ratio >= 1.0 {
        return group_sizes.into_keys().collect();
    }

    let target_rows = ((rows.len() as f64) * ratio).ceil() as usize;
    let mut groups = group_sizes
        .into_iter()
        .map(|(group, size)| (stable_hash(seed, &group), group, size))
        .collect::<Vec<_>>();
    groups.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let mut selected = HashSet::new();
    let mut selected_rows = 0;
    for (_, group, size) in groups {
        if selected_rows >= target_rows {
            break;
        }
        selected_rows += size;
        selected.insert(group);
    }
    selected
}

fn group_key(row: &Value) -> String {
    let review = row.get("review").unwrap_or(&Value::Null);
    for key in [
        "example_group_id",
        "task_group_id",
        "capture_key",
        "row_key",
    ] {
        if let Some(value) = review.get(key).and_then(stable_value_key) {
            return format!("review.{key}:{value}");
        }
    }
    let input = row.get("input").unwrap_or(&Value::Null);
    format!("input:{}", json_compact(input))
}

fn stable_value_key(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.to_string()),
        Value::Null => None,
        other => Some(json_compact(other)),
    }
}

fn stable_hash(seed: &str, key: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in seed.bytes().chain(std::iter::once(0)).chain(key.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn manifest(
    cli: &SplitCli,
    rows: &[SplitRow],
    train_rows: &[Value],
    validation_rows: &[Value],
    validation_groups: &HashSet<String>,
) -> Value {
    let all_rows = rows.iter().map(|row| row.row.clone()).collect::<Vec<_>>();
    json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_unix": unix_secs(),
        "input": cli.input,
        "outputs": {
            "train": PathBuf::from(&cli.out_dir).join(&cli.train_output).display().to_string(),
            "validation": PathBuf::from(&cli.out_dir).join(&cli.validation_output).display().to_string(),
        },
        "seed": cli.seed,
        "validation_ratio": cli.validation_ratio,
        "counts": {
            "rows": rows.len(),
            "train_rows": train_rows.len(),
            "validation_rows": validation_rows.len(),
            "groups": group_count(rows),
            "train_groups": train_group_count(rows, validation_groups),
            "validation_groups": validation_groups.len(),
        },
        "all": counts_for_rows(&all_rows),
        "train": counts_for_rows(train_rows),
        "validation": counts_for_rows(validation_rows),
        "metadata": {
            "private_agent_log": true,
            "public_export_allowed": false,
            "group_aware": true,
        }
    })
}

fn counts_for_rows(rows: &[Value]) -> Value {
    let mut labels = BTreeMap::<String, usize>::new();
    let mut source_buckets = BTreeMap::<String, usize>::new();
    let mut row_schemas = BTreeMap::<String, usize>::new();
    let mut input_schemas = BTreeMap::<String, usize>::new();
    for row in rows {
        count_value(&mut row_schemas, row.get("schema_version"));
        count_value(
            &mut input_schemas,
            row.get("input")
                .and_then(|input| input.get("schema_version")),
        );
        count_value(&mut labels, row.get("label"));
        count_value(
            &mut source_buckets,
            row.get("review")
                .and_then(|review| review.get("source_bucket")),
        );
    }
    json!({
        "rows": rows.len(),
        "labels": labels,
        "source_buckets": source_buckets,
        "row_schemas": row_schemas,
        "input_schemas": input_schemas,
    })
}

fn count_value(counts: &mut BTreeMap<String, usize>, value: Option<&Value>) {
    let key = value
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .unwrap_or("unknown");
    *counts.entry(key.to_string()).or_insert(0) += 1;
}

fn group_count(rows: &[SplitRow]) -> usize {
    rows.iter()
        .map(|row| row.group_key.clone())
        .collect::<BTreeSet<_>>()
        .len()
}

fn train_group_count(rows: &[SplitRow], validation_groups: &HashSet<String>) -> usize {
    rows.iter()
        .filter(|row| !validation_groups.contains(&row.group_key))
        .map(|row| row.group_key.clone())
        .collect::<BTreeSet<_>>()
        .len()
}

fn write_jsonl_atomic(path: &Path, rows: &[Value]) -> Result<(), String> {
    write_atomic(path, |file| {
        for row in rows {
            let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
            writeln!(file, "{line}")
                .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        }
        Ok(())
    })
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<(), String> {
    write_atomic(path, |file| {
        let text = serde_json::to_string_pretty(value).map_err(|err| err.to_string())?;
        writeln!(file, "{text}").map_err(|err| format!("failed to write {}: {err}", path.display()))
    })
}

fn write_atomic<F>(path: &Path, write: F) -> Result<(), String>
where
    F: FnOnce(&mut File) -> Result<(), String>,
{
    ensure_parent_dir(path)?;
    let tmp = temp_path(path);
    let mut file =
        File::create(&tmp).map_err(|err| format!("failed to create {}: {err}", tmp.display()))?;
    if let Err(err) = write(&mut file).and_then(|_| {
        file.sync_all()
            .map_err(|sync_err| format!("failed to sync {}: {sync_err}", tmp.display()))
    }) {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    fs::rename(&tmp, path).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        format!(
            "failed to move {} to {}: {err}",
            tmp.display(),
            path.display()
        )
    })
}

fn temp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("split-output");
    path.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        unix_secs()
    ))
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    fs::create_dir_all(parent)
        .map_err(|err| format!("failed to create {}: {err}", parent.display()))
}

fn json_compact(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
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
        let dir = std::env::temp_dir().join(format!(
            "forge-dataset-split-{}-{}-{}",
            name,
            std::process::id(),
            unix_secs()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn tool() -> Value {
        json!({
            "name": "compare_products",
            "description": "Compare products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["product_ids"]
            }
        })
    }

    fn training_row(group: &str, label: &str, source_bucket: &str, input_schema: &str) -> Value {
        json!({
            "schema_version": TRAINING_SCHEMA_VERSION,
            "input": {
                "schema_version": input_schema,
                "user_request": format!("Compare products for {group}."),
                "workflow_state": {
                    "required_steps": [],
                    "completed_steps": [],
                    "pending_steps": [],
                    "terminal_tools": ["respond"],
                    "recent_errors": []
                },
                "available_tools": [tool()],
                "candidate_call": {
                    "name": "compare_products",
                    "arguments": {"product_ids": ["SKU-1", "SKU-2"]}
                }
            },
            "label": label,
            "review": {
                "source": "forge-dataset",
                "source_bucket": source_bucket,
                "example_group_id": group,
                "row_key": format!("{group}:{source_bucket}:{label}")
            }
        })
    }

    fn write_jsonl(path: &Path, rows: &[Value]) {
        let text = rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("json"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(path, text).expect("write input");
    }

    fn read_jsonl(path: &Path) -> Vec<Value> {
        let text = fs::read_to_string(path).expect("read jsonl");
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("row"))
            .collect()
    }

    fn groups(rows: &[Value]) -> HashSet<String> {
        rows.iter()
            .filter_map(|row| {
                row.get("review")
                    .and_then(|review| review.get("example_group_id"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    #[test]
    fn split_keeps_groups_on_one_side_and_is_deterministic() {
        let dir = temp_dir("deterministic");
        let input = dir.join("combined.jsonl");
        let rows = vec![
            training_row(
                "group-a",
                "valid",
                "real_model_call",
                TRAINING_INPUT_SCHEMA_VERSION,
            ),
            training_row(
                "group-a",
                "wrong_arguments_semantic",
                "targeted_alternative",
                TRAINING_INPUT_SCHEMA_VERSION,
            ),
            training_row(
                "group-b",
                "valid",
                "real_model_call",
                TRAINING_INPUT_SCHEMA_VERSION,
            ),
            training_row(
                "group-c",
                "tool_not_needed",
                "targeted_alternative",
                TRAINING_INPUT_SCHEMA_VERSION,
            ),
        ];
        write_jsonl(&input, &rows);

        split(SplitCli {
            input: input.display().to_string(),
            out_dir: dir.display().to_string(),
            train_output: "train.jsonl".to_string(),
            validation_output: "validation.jsonl".to_string(),
            validation_ratio: 0.5,
            seed: "seed-1".to_string(),
        })
        .expect("split");

        let train = read_jsonl(&dir.join("train.jsonl"));
        let validation = read_jsonl(&dir.join("validation.jsonl"));
        let train_groups = groups(&train);
        let validation_groups = groups(&validation);
        assert!(train_groups.is_disjoint(&validation_groups));
        assert!(!train.is_empty());
        assert!(!validation.is_empty());

        let first_validation = fs::read_to_string(dir.join("validation.jsonl")).expect("read");
        split(SplitCli {
            input: input.display().to_string(),
            out_dir: dir.display().to_string(),
            train_output: "train.jsonl".to_string(),
            validation_output: "validation.jsonl".to_string(),
            validation_ratio: 0.5,
            seed: "seed-1".to_string(),
        })
        .expect("split again");
        let second_validation = fs::read_to_string(dir.join("validation.jsonl")).expect("read");
        assert_eq!(first_validation, second_validation);
    }

    #[test]
    fn split_manifest_counts_labels_sources_and_input_schemas() {
        let dir = temp_dir("manifest");
        let input = dir.join("combined.jsonl");
        let rows = vec![
            training_row(
                "group-a",
                "valid",
                "real_model_call",
                TRAINING_INPUT_SCHEMA_VERSION_V1,
            ),
            training_row(
                "group-b",
                "wrong_arguments_semantic",
                "targeted_alternative",
                TRAINING_INPUT_SCHEMA_VERSION,
            ),
        ];
        write_jsonl(&input, &rows);

        split(SplitCli {
            input: input.display().to_string(),
            out_dir: dir.display().to_string(),
            train_output: "train.jsonl".to_string(),
            validation_output: "validation.jsonl".to_string(),
            validation_ratio: 0.5,
            seed: "seed-2".to_string(),
        })
        .expect("split");

        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST_OUTPUT)).expect("manifest"))
                .expect("manifest json");
        assert_eq!(manifest["schema_version"], MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest["all"]["labels"]["valid"], 1);
        assert_eq!(manifest["all"]["labels"]["wrong_arguments_semantic"], 1);
        assert_eq!(manifest["all"]["source_buckets"]["real_model_call"], 1);
        assert_eq!(manifest["all"]["source_buckets"]["targeted_alternative"], 1);
        assert_eq!(
            manifest["all"]["input_schemas"][TRAINING_INPUT_SCHEMA_VERSION_V1],
            1
        );
        assert_eq!(
            manifest["all"]["input_schemas"][TRAINING_INPUT_SCHEMA_VERSION],
            1
        );
    }

    #[test]
    fn split_fails_invalid_rows_before_writing_outputs() {
        let dir = temp_dir("invalid");
        let input = dir.join("combined.jsonl");
        let mut row = training_row(
            "group-a",
            "valid",
            "real_model_call",
            TRAINING_INPUT_SCHEMA_VERSION,
        );
        row["label"] = json!("synthetic_unrelated_tool");
        write_jsonl(&input, &[row]);

        let err = split(SplitCli {
            input: input.display().to_string(),
            out_dir: dir.display().to_string(),
            train_output: "train.jsonl".to_string(),
            validation_output: "validation.jsonl".to_string(),
            validation_ratio: 0.5,
            seed: "seed-3".to_string(),
        })
        .expect_err("invalid row");

        assert!(err.contains("unsupported label"));
        assert!(!dir.join("train.jsonl").exists());
        assert!(!dir.join("validation.jsonl").exists());
    }
}
