use crate::clients::base::ToolCall;
use crate::core::message::{Message, MessageRole};
use crate::core::tool_spec::ToolSpec;
use crate::guardrails::{recent_errors_from_messages, StepEnforcer};
use crate::tools::respond::RESPOND_TOOL_NAME;
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const FORGE_TRAINING_CAPTURE_LOG_ENV: &str = "FORGE_TRAINING_CAPTURE_LOG";
const FORGE_TRAINING_CAPTURE_LOG_QUEUE_CAPACITY_ENV: &str =
    "FORGE_TRAINING_CAPTURE_LOG_QUEUE_CAPACITY";
const FORGE_TRAINING_CAPTURE_LOG_MAX_EVENT_BYTES_ENV: &str =
    "FORGE_TRAINING_CAPTURE_LOG_MAX_EVENT_BYTES";
const FORGE_TRAINING_CAPTURE_LOG_REDACT_ENV: &str = "FORGE_TRAINING_CAPTURE_LOG_REDACT";

const DEFAULT_QUEUE_CAPACITY: usize = 256;
const DEFAULT_MAX_EVENT_BYTES: usize = 256 * 1024;
const REDACTED: &str = "[redacted]";

static TRAINING_CAPTURE_SINK: LazyLock<Mutex<Option<Arc<TrainingCaptureSink>>>> =
    LazyLock::new(|| Mutex::new(None));
static TRAINING_CAPTURE_INIT_WARNED: AtomicBool = AtomicBool::new(false);
static EXAMPLE_GROUP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(super) struct TrainingCaptureConfig {
    path: PathBuf,
    queue_capacity: usize,
    max_event_bytes: usize,
    redact: bool,
}

pub(super) struct TrainingCaptureSink {
    sender: tokio::sync::mpsc::Sender<TrainingCaptureCommand>,
    dropped_events: AtomicU64,
    redact: bool,
}

enum TrainingCaptureCommand {
    Event(Value),
    #[cfg(test)]
    Flush(tokio::sync::oneshot::Sender<()>),
}

impl TrainingCaptureConfig {
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

impl TrainingCaptureSink {
    pub(super) fn spawn(config: TrainingCaptureConfig) -> Result<Arc<Self>, String> {
        tokio::runtime::Handle::try_current()
            .map_err(|err| format!("training capture sink requires a Tokio runtime: {err}"))?;
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
        match self.sender.try_send(TrainingCaptureCommand::Event(event)) {
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
            .send(TrainingCaptureCommand::Flush(sender))
            .await
            .map_err(|_| "training capture writer closed".to_string())?;
        receiver
            .await
            .map_err(|_| "training capture flush failed".to_string())
    }

    fn record_drop(&self, reason: &'static str) {
        let dropped = self.dropped_events.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_power_of_two() {
            tracing::warn!(
                target: "forge.training_capture",
                reason,
                dropped,
                "dropped proxy training capture JSONL event"
            );
        }
    }
}

pub(super) fn init_proxy_training_capture_sink_from_env() {
    let _ = training_capture_sink_from_env();
}

pub(super) fn shutdown_proxy_training_capture_sink() {
    match TRAINING_CAPTURE_SINK.lock() {
        Ok(mut guard) => {
            *guard = None;
        }
        Err(err) => {
            warn_training_capture_init_once(format!("training capture lock poisoned: {err}"));
        }
    }
}

pub(super) fn emit_proxy_training_capture_jsonl(event: Value) {
    let Some(sink) = training_capture_sink_from_env() else {
        return;
    };
    sink.emit(event);
}

pub(super) fn emit_proxy_training_tool_call_candidates(
    messages: &[Message],
    tool_calls: &[ToolCall],
    step_enforcer: Option<&StepEnforcer>,
    tool_specs: &[ToolSpec],
) {
    if tool_calls.is_empty() || !training_capture_config_present() {
        return;
    }

    let user_request = latest_proxy_user_request(messages).unwrap_or_default();
    let workflow_state = workflow_state_for_capture(messages, step_enforcer, tool_specs);
    let available_tools = available_tools_for_json(tool_specs);
    let example_group_id = next_example_group_id();

    for (candidate_index, call) in tool_calls.iter().enumerate() {
        emit_proxy_training_capture_jsonl(training_capture_event_for_candidate(
            &example_group_id,
            candidate_index,
            user_request,
            initial_proxy_user_request(messages).unwrap_or_default(),
            &workflow_state,
            &available_tools,
            call,
        ));
    }
}

pub(super) fn training_capture_event_for_candidate(
    example_group_id: &str,
    candidate_index: usize,
    user_request: &str,
    initial_user_request: &str,
    workflow_state: &Value,
    available_tools: &Value,
    call: &ToolCall,
) -> Value {
    json!({
        "schema_version": "forge-proxy-training-capture/v1",
        "kind": "tool_call_candidate",
        "unix_ms": unix_ms(),
        "example_group_id": example_group_id,
        "candidate_index": candidate_index,
        "user_request": user_request,
        "initial_user_request": initial_user_request,
        "workflow_state": workflow_state,
        "available_tools": available_tools,
        "candidate_call": proxy_tool_call_for_json(call),
        "deterministic_status": "accepted",
        "later_tool_result": Value::Null,
        "metadata": {
            "private_agent_log": true,
            "public_export_allowed": false,
            "source": "forge_proxy"
        }
    })
}

fn training_capture_sink_from_env() -> Option<Arc<TrainingCaptureSink>> {
    let mut guard = match TRAINING_CAPTURE_SINK.lock() {
        Ok(guard) => guard,
        Err(err) => {
            warn_training_capture_init_once(format!("training capture lock poisoned: {err}"));
            return None;
        }
    };
    if let Some(sink) = guard.as_ref() {
        return Some(sink.clone());
    }

    let config = config_from_env()?;
    match TrainingCaptureSink::spawn(config) {
        Ok(sink) => {
            *guard = Some(sink.clone());
            Some(sink)
        }
        Err(err) => {
            warn_training_capture_init_once(err);
            None
        }
    }
}

fn training_capture_config_present() -> bool {
    std::env::var_os(FORGE_TRAINING_CAPTURE_LOG_ENV)
        .is_some_and(|path| !path.as_os_str().is_empty())
}

fn config_from_env() -> Option<TrainingCaptureConfig> {
    let path = std::env::var_os(FORGE_TRAINING_CAPTURE_LOG_ENV)?;
    if path.as_os_str().is_empty() {
        return None;
    }
    Some(
        TrainingCaptureConfig::new(PathBuf::from(path))
            .with_queue_capacity(usize_env_or_default(
                FORGE_TRAINING_CAPTURE_LOG_QUEUE_CAPACITY_ENV,
                DEFAULT_QUEUE_CAPACITY,
            ))
            .with_max_event_bytes(usize_env_or_default(
                FORGE_TRAINING_CAPTURE_LOG_MAX_EVENT_BYTES_ENV,
                DEFAULT_MAX_EVENT_BYTES,
            ))
            .with_redaction(bool_env_or_default(
                FORGE_TRAINING_CAPTURE_LOG_REDACT_ENV,
                false,
            )),
    )
}

fn spawn_writer_task(
    file: File,
    receiver: tokio::sync::mpsc::Receiver<TrainingCaptureCommand>,
    max_event_bytes: usize,
    path: PathBuf,
) {
    tokio::task::spawn_blocking(move || writer_loop(file, receiver, max_event_bytes, path));
}

fn writer_loop(
    mut file: File,
    mut receiver: tokio::sync::mpsc::Receiver<TrainingCaptureCommand>,
    max_event_bytes: usize,
    path: PathBuf,
) {
    while let Some(command) = receiver.blocking_recv() {
        match command {
            TrainingCaptureCommand::Event(event) => {
                let line = match serde_json::to_string(&event) {
                    Ok(line) => line,
                    Err(err) => {
                        tracing::warn!(
                            target: "forge.training_capture",
                            error = %err,
                            "failed to serialize proxy training capture JSONL event"
                        );
                        continue;
                    }
                };
                if line.len() > max_event_bytes {
                    tracing::warn!(
                        target: "forge.training_capture",
                        event_bytes = line.len(),
                        max_event_bytes,
                        "dropped oversized proxy training capture JSONL event"
                    );
                    continue;
                }
                if let Err(err) = writeln!(file, "{line}") {
                    tracing::warn!(
                        target: "forge.training_capture",
                        error = %err,
                        path = %path.display(),
                        "failed to write proxy training capture JSONL event"
                    );
                }
            }
            #[cfg(test)]
            TrainingCaptureCommand::Flush(done) => {
                if let Err(err) = file.flush() {
                    tracing::warn!(
                        target: "forge.training_capture",
                        error = %err,
                        path = %path.display(),
                        "failed to flush proxy training capture JSONL"
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
        "available_tools",
        "later_tool_result",
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

    event
}

fn workflow_state_for_capture(
    messages: &[Message],
    step_enforcer: Option<&StepEnforcer>,
    tool_specs: &[ToolSpec],
) -> Value {
    let recent_errors = recent_errors_from_messages(messages, 8);
    match step_enforcer {
        Some(enforcer) => json!({
            "required_steps": enforcer.tracker.required_steps(),
            "completed_steps": enforcer.completed_steps().keys().cloned().collect::<Vec<_>>(),
            "pending_steps": enforcer.pending(),
            "terminal_tools": enforcer.terminal_tools.iter().cloned().collect::<Vec<_>>(),
            "recent_errors": recent_errors,
        }),
        None => json!({
            "required_steps": [],
            "completed_steps": [],
            "pending_steps": [],
            "terminal_tools": proxy_terminal_tools_for_capture(tool_specs),
            "recent_errors": recent_errors,
        }),
    }
}

fn proxy_terminal_tools_for_capture(tool_specs: &[ToolSpec]) -> Vec<String> {
    tool_specs
        .iter()
        .filter(|spec| spec.name == RESPOND_TOOL_NAME)
        .map(|spec| spec.name.clone())
        .collect()
}

fn latest_proxy_user_request(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::User)
        .map(|message| message.content.as_str())
}

fn initial_proxy_user_request(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .find(|message| message.role == MessageRole::User)
        .map(|message| message.content.as_str())
}

fn available_tools_for_json(tool_specs: &[ToolSpec]) -> Value {
    Value::Array(tool_specs.iter().map(tool_spec_for_json).collect())
}

fn tool_spec_for_json(spec: &ToolSpec) -> Value {
    json!({
        "name": spec.name,
        "description": spec.description,
        "parameters": spec
            .json_schema
            .clone()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    })
}

fn proxy_tool_call_for_json(call: &ToolCall) -> Value {
    json!({
        "name": call.tool,
        "arguments": Value::Object(
            call.args
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        )
    })
}

fn next_example_group_id() -> String {
    let seq = EXAMPLE_GROUP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("proxy-{}-{seq}", unix_ms())
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn usize_env_or_default(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) if raw.trim().is_empty() => default,
        Ok(raw) => match raw.parse::<usize>() {
            Ok(value) => value,
            Err(err) => {
                tracing::warn!(
                    target: "forge.training_capture",
                    name,
                    value = raw,
                    error = %err,
                    "invalid training capture numeric environment value; using default"
                );
                default
            }
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(err) => {
            tracing::warn!(
                target: "forge.training_capture",
                name,
                error = %err,
                "failed to read training capture environment value; using default"
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
                    target: "forge.training_capture",
                    name,
                    value = raw,
                    "invalid training capture boolean environment value; using default"
                );
                default
            }
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(err) => {
            tracing::warn!(
                target: "forge.training_capture",
                name,
                error = %err,
                "failed to read training capture environment value; using default"
            );
            default
        }
    }
}

fn warn_training_capture_init_once(error: String) {
    if !TRAINING_CAPTURE_INIT_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            target: "forge.training_capture",
            error,
            "failed to initialize proxy training capture JSONL sink"
        );
    }
}
