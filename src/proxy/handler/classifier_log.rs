use crate::clients::base::ToolCall;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const FORGE_CLASSIFIER_LOG_ENV: &str = "FORGE_CLASSIFIER_LOG";
const FORGE_CLASSIFIER_LOG_QUEUE_CAPACITY_ENV: &str = "FORGE_CLASSIFIER_LOG_QUEUE_CAPACITY";
const FORGE_CLASSIFIER_LOG_MAX_EVENT_BYTES_ENV: &str = "FORGE_CLASSIFIER_LOG_MAX_EVENT_BYTES";
const FORGE_CLASSIFIER_LOG_REDACT_ENV: &str = "FORGE_CLASSIFIER_LOG_REDACT";

const DEFAULT_QUEUE_CAPACITY: usize = 256;
const DEFAULT_MAX_EVENT_BYTES: usize = 256 * 1024;
const REDACTED: &str = "[redacted]";

static CLASSIFIER_LOG_SINK: LazyLock<Mutex<Option<Arc<ClassifierLogSink>>>> =
    LazyLock::new(|| Mutex::new(None));
static CLASSIFIER_LOG_INIT_WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub(super) struct ClassifierLogConfig {
    path: PathBuf,
    queue_capacity: usize,
    max_event_bytes: usize,
    redact: bool,
}

pub(super) struct ClassifierLogSink {
    sender: tokio::sync::mpsc::Sender<ClassifierLogCommand>,
    dropped_events: AtomicU64,
    redact: bool,
}

enum ClassifierLogCommand {
    Event(Value),
    #[cfg(test)]
    Flush(tokio::sync::oneshot::Sender<()>),
}

impl ClassifierLogConfig {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
            redact: false,
        }
    }

    pub(super) fn with_queue_capacity(mut self, queue_capacity: usize) -> Self {
        self.queue_capacity = queue_capacity.max(1);
        self
    }

    pub(super) fn with_max_event_bytes(mut self, max_event_bytes: usize) -> Self {
        self.max_event_bytes = max_event_bytes;
        self
    }

    pub(super) fn with_redaction(mut self, redact: bool) -> Self {
        self.redact = redact;
        self
    }
}

impl ClassifierLogSink {
    pub(super) fn spawn(config: ClassifierLogConfig) -> Result<Arc<Self>, String> {
        tokio::runtime::Handle::try_current()
            .map_err(|err| format!("classifier log sink requires a Tokio runtime: {err}"))?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.path)
            .map_err(|err| format!("failed to open {}: {err}", config.path.display()))?;
        let (sender, receiver) = tokio::sync::mpsc::channel(config.queue_capacity.max(1));
        let sink = Arc::new(Self {
            sender,
            dropped_events: AtomicU64::new(0),
            redact: config.redact,
        });
        spawn_writer_task(file, receiver, config.max_event_bytes, config.path);
        Ok(sink)
    }

    pub(super) fn emit(&self, event: Value) {
        let event = if self.redact {
            redact_event(event)
        } else {
            event
        };
        match self.sender.try_send(ClassifierLogCommand::Event(event)) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                self.record_drop("queue_full");
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                self.record_drop("writer_closed");
            }
        }
    }

    #[cfg(test)]
    pub(super) async fn flush(&self) -> Result<(), String> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.sender
            .send(ClassifierLogCommand::Flush(sender))
            .await
            .map_err(|_| "classifier log writer closed".to_string())?;
        receiver
            .await
            .map_err(|_| "classifier log flush failed".to_string())
    }

    fn record_drop(&self, reason: &'static str) {
        let dropped = self.dropped_events.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_power_of_two() {
            tracing::warn!(
                target: "forge.classifier",
                reason,
                dropped,
                "dropped proxy classifier JSONL event"
            );
        }
    }
}

pub(super) fn init_proxy_classifier_log_sink_from_env() {
    let _ = classifier_log_sink_from_env();
}

pub(super) fn emit_proxy_classifier_jsonl(event: Value) {
    let Some(sink) = classifier_log_sink_from_env() else {
        return;
    };
    sink.emit(event);
}

fn classifier_log_sink_from_env() -> Option<Arc<ClassifierLogSink>> {
    let mut guard = match CLASSIFIER_LOG_SINK.lock() {
        Ok(guard) => guard,
        Err(err) => {
            warn_classifier_log_init_once(format!("classifier log lock poisoned: {err}"));
            return None;
        }
    };
    if let Some(sink) = guard.as_ref() {
        return Some(sink.clone());
    }

    let config = config_from_env()?;
    match ClassifierLogSink::spawn(config) {
        Ok(sink) => {
            *guard = Some(sink.clone());
            Some(sink)
        }
        Err(err) => {
            warn_classifier_log_init_once(err);
            None
        }
    }
}

fn config_from_env() -> Option<ClassifierLogConfig> {
    let path = std::env::var_os(FORGE_CLASSIFIER_LOG_ENV)?;
    if path.as_os_str().is_empty() {
        return None;
    }
    Some(
        ClassifierLogConfig::new(PathBuf::from(path))
            .with_queue_capacity(usize_env_or_default(
                FORGE_CLASSIFIER_LOG_QUEUE_CAPACITY_ENV,
                DEFAULT_QUEUE_CAPACITY,
            ))
            .with_max_event_bytes(usize_env_or_default(
                FORGE_CLASSIFIER_LOG_MAX_EVENT_BYTES_ENV,
                DEFAULT_MAX_EVENT_BYTES,
            ))
            .with_redaction(bool_env_or_default(FORGE_CLASSIFIER_LOG_REDACT_ENV, false)),
    )
}

fn spawn_writer_task(
    file: File,
    receiver: tokio::sync::mpsc::Receiver<ClassifierLogCommand>,
    max_event_bytes: usize,
    path: PathBuf,
) {
    tokio::task::spawn_blocking(move || writer_loop(file, receiver, max_event_bytes, path));
}

fn writer_loop(
    mut file: File,
    mut receiver: tokio::sync::mpsc::Receiver<ClassifierLogCommand>,
    max_event_bytes: usize,
    path: PathBuf,
) {
    while let Some(command) = receiver.blocking_recv() {
        match command {
            ClassifierLogCommand::Event(event) => {
                let line = match serde_json::to_string(&event) {
                    Ok(line) => line,
                    Err(err) => {
                        tracing::warn!(
                            target: "forge.classifier",
                            error = %err,
                            "failed to serialize proxy classifier JSONL event"
                        );
                        continue;
                    }
                };
                if line.len() > max_event_bytes {
                    tracing::warn!(
                        target: "forge.classifier",
                        event_bytes = line.len(),
                        max_event_bytes,
                        "dropped oversized proxy classifier JSONL event"
                    );
                    continue;
                }
                if let Err(err) = writeln!(file, "{line}") {
                    tracing::warn!(
                        target: "forge.classifier",
                        error = %err,
                        path = %path.display(),
                        "failed to write proxy classifier JSONL event"
                    );
                }
            }
            #[cfg(test)]
            ClassifierLogCommand::Flush(done) => {
                if let Err(err) = file.flush() {
                    tracing::warn!(
                        target: "forge.classifier",
                        error = %err,
                        path = %path.display(),
                        "failed to flush proxy classifier JSONL"
                    );
                }
                let _ = done.send(());
            }
        }
    }
    let _ = file.flush();
}

fn redact_event(mut event: Value) -> Value {
    let Some(obj) = event.as_object_mut() else {
        return event;
    };
    for key in [
        "user_request",
        "initial_user_request",
        "workflow_state",
        "required_facts",
        "candidate_final_response",
    ] {
        if obj.contains_key(key) {
            obj.insert(key.to_string(), Value::String(REDACTED.to_string()));
        }
    }

    if let Some(Value::Object(candidate)) = obj.get_mut("candidate_call") {
        if candidate.contains_key("arguments") {
            candidate.insert("arguments".to_string(), Value::String(REDACTED.to_string()));
        }
    }

    if let Some(Value::Array(results)) = obj.get_mut("tool_results") {
        for result in results {
            if let Some(result) = result.as_object_mut() {
                if result.contains_key("content") {
                    result.insert("content".to_string(), Value::String(REDACTED.to_string()));
                }
            }
        }
    }

    event
}

fn usize_env_or_default(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) if raw.trim().is_empty() => default,
        Ok(raw) => match raw.parse::<usize>() {
            Ok(value) => value,
            Err(err) => {
                tracing::warn!(
                    target: "forge.classifier",
                    name,
                    value = raw,
                    error = %err,
                    "invalid classifier log numeric environment value; using default"
                );
                default
            }
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(err) => {
            tracing::warn!(
                target: "forge.classifier",
                name,
                error = %err,
                "failed to read classifier log environment value; using default"
            );
            default
        }
    }
}

fn bool_env_or_default(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) if raw.trim().is_empty() => default,
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => {
                tracing::warn!(
                    target: "forge.classifier",
                    name,
                    value = raw,
                    "invalid classifier log boolean environment value; using default"
                );
                default
            }
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(err) => {
            tracing::warn!(
                target: "forge.classifier",
                name,
                error = %err,
                "failed to read classifier log environment value; using default"
            );
            default
        }
    }
}

fn warn_classifier_log_init_once(error: String) {
    if !CLASSIFIER_LOG_INIT_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            target: "forge.classifier",
            error,
            "failed to initialize proxy classifier JSONL sink"
        );
    }
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
