use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::types::RejectRecord;

pub(crate) fn read_jsonl(path: &str) -> Result<Vec<Value>, String> {
    read_jsonl_path(Path::new(path))
}

pub(crate) fn count_jsonl_rows(path: &str) -> Result<usize, String> {
    let file = File::open(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut count = 0;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| format!("{path}:{} read error: {err}", index + 1))?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

pub(crate) fn read_jsonl_path(path: &Path) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Value>(trimmed)
            .map_err(|err| format!("{}:{} invalid JSONL row: {err}", path.display(), index + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

pub(crate) fn append_jsonl(path: &str, row: &Value) -> Result<(), String> {
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

pub(crate) fn append_reject(
    path: &Path,
    reason: &str,
    detail: &str,
    capture: &Value,
    raw_response: Option<Value>,
) -> Result<(), String> {
    let cap_key = capture_key(capture);
    append_jsonl_path(
        path,
        &json!({
            "schema_version": "forge-dataset-review-reject/v1",
            "reason": reason,
            "detail": detail,
            "example_group_id": capture.get("example_group_id").cloned().unwrap_or(Value::Null),
            "capture_key": cap_key,
            "capture": capture,
            "raw_response": raw_response.unwrap_or(Value::Null),
        }),
    )
}

pub(crate) fn append_reject_record(path: &Path, record: &RejectRecord) -> Result<(), String> {
    append_reject(
        path,
        &record.reason,
        &record.detail,
        &record.capture,
        record.raw_response.clone(),
    )
}

pub(crate) fn append_jsonl_path(path: &Path, row: &Value) -> Result<(), String> {
    let mut line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    line.push('\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    file.write_all(line.as_bytes())
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    file.flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}

pub(crate) fn touch_jsonl(path: &str) -> Result<(), String> {
    ensure_parent_dir(path)?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {path}: {err}"))
}

pub(crate) fn touch_jsonl_path(path: &Path) -> Result<(), String> {
    ensure_parent_dir_path(path)?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}

pub(crate) fn rejects_path(output: &str) -> PathBuf {
    let path = Path::new(output);
    let stem = path.file_stem().and_then(|value| value.to_str());
    let extension = path.extension().and_then(|value| value.to_str());
    let file_name = match (stem, extension) {
        (Some(stem), Some(extension)) => format!("{stem}.rejects.{extension}"),
        (Some(stem), None) => format!("{stem}.rejects"),
        _ => format!("{output}.rejects.jsonl"),
    };
    path.with_file_name(file_name)
}

pub(crate) fn capture_key(capture: &Value) -> String {
    let group = capture
        .get("example_group_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-group");
    let trace = capture.get("proxy_trace").unwrap_or(&Value::Null);
    let turn = trace
        .get("turn")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown-turn".to_string());
    let call_index = trace
        .get("call_index")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown-call".to_string());
    let tool_call_id = trace
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-tool-call");
    format!("{group}:turn-{turn}:call-{call_index}:tool-call-{tool_call_id}")
}

pub(crate) fn row_key(
    example_group_id: &str,
    source_bucket: &str,
    candidate_call: &Value,
) -> String {
    let candidate = serde_json::to_string(candidate_call).unwrap_or_else(|_| "null".to_string());
    format!("{example_group_id}:{source_bucket}:{candidate}")
}

pub(crate) fn ensure_parent_dir(path: &str) -> Result<(), String> {
    ensure_parent_dir_path(Path::new(path))
}

pub(crate) fn ensure_parent_dir_path(path: &Path) -> Result<(), String> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("failed to create {}: {err}", parent.display()))
}

pub(crate) fn normalize_chat_completions_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "forge-dataset-io-{name}-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    #[test]
    fn rejects_path_is_sibling_jsonl() {
        assert_eq!(
            rejects_path("target/dataset/training.toolcall.jsonl"),
            PathBuf::from("target/dataset/training.toolcall.rejects.jsonl")
        );
    }

    #[test]
    fn touch_jsonl_creates_empty_file() {
        let path = temp_path("touch");
        touch_jsonl(path.to_str().expect("path")).expect("touch");

        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "");
        std::fs::remove_file(path).expect("remove");
    }

    #[test]
    fn append_jsonl_is_readable_after_return() {
        let path = temp_path("append");
        append_jsonl(path.to_str().expect("path"), &json!({"ok": true})).expect("append");

        let rows = read_jsonl_path(&path).expect("read rows");
        assert_eq!(rows, vec![json!({"ok": true})]);
        std::fs::remove_file(path).expect("remove");
    }
}
