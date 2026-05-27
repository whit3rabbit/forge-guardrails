use crate::clients::base::ToolCall;
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const FORGE_CLASSIFIER_LOG_ENV: &str = "FORGE_CLASSIFIER_LOG";

static CLASSIFIER_LOG_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn emit_proxy_classifier_jsonl(event: Value) {
    let Some(path) = std::env::var_os(FORGE_CLASSIFIER_LOG_ENV) else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let path = Path::new(&path);
    if let Err(err) = append_proxy_classifier_jsonl(path, &event) {
        tracing::warn!(
            target: "forge.classifier",
            error = %err,
            path = %path.display(),
            "failed to write proxy classifier JSONL event"
        );
    }
}

pub(super) fn append_proxy_classifier_jsonl(path: &Path, event: &Value) -> Result<(), String> {
    let _guard = CLASSIFIER_LOG_LOCK
        .lock()
        .map_err(|err| format!("classifier log lock poisoned: {err}"))?;
    let line = serde_json::to_string(event).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    writeln!(file, "{line}").map_err(|err| err.to_string())
}

pub(super) fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

pub(super) fn proxy_tool_call_for_json(call: &ToolCall) -> Value {
    Value::Object(
        [
            ("name".to_string(), Value::String(call.tool.clone())),
            (
                "arguments".to_string(),
                Value::Object(
                    call.args
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                ),
            ),
        ]
        .into_iter()
        .collect(),
    )
}
