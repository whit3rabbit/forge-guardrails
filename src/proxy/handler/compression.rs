use crate::core::message::{Message, MessageRole, MessageType, ToolCallInfo};
use crate::tool_output::{
    compress_tool_output, ToolOutputCompressionConfig, ToolOutputCompressionResult,
    ToolOutputCompressionState, LZW_DICTIONARY_HEADER, REPAIR_DICTIONARY_HEADER,
};
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const FORGE_TOOL_OUTPUT_COMPRESSION_LOG_ENV: &str = "FORGE_TOOL_OUTPUT_COMPRESSION_LOG";
const FORGE_TOOL_OUTPUT_COMPRESSION_LOG_QUEUE_CAPACITY_ENV: &str =
    "FORGE_TOOL_OUTPUT_COMPRESSION_LOG_QUEUE_CAPACITY";
const FORGE_TOOL_OUTPUT_COMPRESSION_LOG_MAX_EVENT_BYTES_ENV: &str =
    "FORGE_TOOL_OUTPUT_COMPRESSION_LOG_MAX_EVENT_BYTES";

const DEFAULT_QUEUE_CAPACITY: usize = 256;
const DEFAULT_MAX_EVENT_BYTES: usize = 64 * 1024;

static COMPRESSION_LOG_SINK: LazyLock<Mutex<Option<Arc<ToolOutputCompressionLogSink>>>> =
    LazyLock::new(|| Mutex::new(None));
static COMPRESSION_LOG_INIT_WARNED: AtomicBool = AtomicBool::new(false);

/// An update to a tool call output due to compression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutputCompressionUpdate {
    pub tool_call_id: Option<String>,
    pub output: String,
}

/// Compresses the tool result message content in place using the given config and state.
pub fn compress_proxy_tool_results(
    messages: &mut [Message],
    config: &ToolOutputCompressionConfig,
    state: Option<&ToolOutputCompressionState>,
    request_debug: Option<&Value>,
) -> Vec<ToolOutputCompressionUpdate> {
    if !config.enabled() {
        return Vec::new();
    }

    let mut pending_tool_calls: IndexMap<String, ToolCallInfo> = IndexMap::new();
    let mut updates = Vec::new();
    let mut tool_result_index = 0usize;
    for (message_index, message) in messages.iter_mut().enumerate() {
        match message.role {
            MessageRole::Assistant => {
                let Some(tool_calls) = &message.tool_calls else {
                    continue;
                };
                for call in tool_calls {
                    pending_tool_calls.insert(call.call_id.clone(), call.clone());
                }
            }
            MessageRole::Tool => {
                if message.metadata.msg_type != MessageType::ToolResult {
                    continue;
                }
                tool_result_index += 1;
                let call = message
                    .tool_call_id
                    .as_deref()
                    .and_then(|call_id| pending_tool_calls.get(call_id));
                let tool_name = call
                    .map(|call| call.name.as_str())
                    .or(message.tool_name.as_deref())
                    .unwrap_or("generic");
                let args = call.and_then(|call| call.args.as_ref());
                let result = compress_tool_output(
                    tool_name,
                    message.tool_call_id.as_deref(),
                    args,
                    &message.content,
                    config,
                    state,
                );
                if result.output != message.content {
                    tracing::info!(
                        target: "forge.tool_output",
                        tool = %result.canonical_tool,
                        family = %result.family,
                        mode = %result.mode,
                        before_tokens = result.before_tokens,
                        after_tokens = result.after_tokens,
                        saved_tokens = result.saved_tokens,
                        saved_pct = result.saved_pct,
                        redacted = result.redacted,
                        capped = result.capped,
                        deduped = result.deduped,
                        strategies = %result.strategies.join(","),
                        "compressed proxy tool output"
                    );
                    emit_proxy_tool_output_compression_jsonl(compression_event(
                        CompressionEventInput {
                            tool_call_id: message.tool_call_id.as_deref(),
                            tool_name,
                            message_index,
                            tool_result_index,
                            args,
                            input_output: &message.content,
                            request_debug,
                        },
                        &result,
                    ));
                    updates.push(ToolOutputCompressionUpdate {
                        tool_call_id: message.tool_call_id.clone(),
                        output: result.output.clone(),
                    });
                    message.content = result.output;
                }
            }
            _ => {}
        }
    }
    updates
}

pub(super) struct CompressionEventInput<'a> {
    pub(super) tool_call_id: Option<&'a str>,
    pub(super) tool_name: &'a str,
    pub(super) message_index: usize,
    pub(super) tool_result_index: usize,
    pub(super) args: Option<&'a IndexMap<String, Value>>,
    pub(super) input_output: &'a str,
    pub(super) request_debug: Option<&'a Value>,
}

pub(super) fn compression_event(
    input: CompressionEventInput<'_>,
    result: &ToolOutputCompressionResult,
) -> Value {
    let CompressionEventInput {
        tool_call_id,
        tool_name,
        message_index,
        tool_result_index,
        args,
        input_output,
        request_debug,
    } = input;
    let mut event = json!({
        "kind": "tool_output_compression",
        "event_version": 2,
        "timestamp_ms": unix_ms(),
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "message_index": message_index,
        "tool_result_index": tool_result_index,
        "canonical_tool": result.canonical_tool.clone(),
        "family": result.family.clone(),
        "mode": result.mode.as_str(),
        "strategies": result.strategies.clone(),
        "input_bytes": input_output.len(),
        "input_chars": input_output.chars().count(),
        "input_lines": line_count(input_output),
        "input_fingerprint64": fingerprint64(input_output),
        "output_bytes": result.output.len(),
        "output_chars": result.output.chars().count(),
        "output_lines": line_count(&result.output),
        "output_fingerprint64": fingerprint64(&result.output),
        "before_tokens": result.before_tokens,
        "after_tokens": result.after_tokens,
        "saved_tokens": result.saved_tokens,
        "saved_pct": result.saved_pct,
        "redacted": result.redacted,
        "capped": result.capped,
        "deduped": result.deduped,
    });
    if let Some(args) = args {
        // Fingerprint the redacted serialization so secret-bearing argument
        // values cannot be dictionary-correlated from telemetry hashes.
        event["args_fingerprint64"] = json!(serde_json::to_string(args)
            .ok()
            .map(|args| fingerprint64(&crate::tool_output::redact_secrets(&args))));
    }
    if let Some(method) = dictionary_method(&result.output) {
        event["dictionary_method"] = json!(method);
    }
    if let Some(request_debug) = request_debug {
        event["request"] = request_debug.clone();
    }
    event
}

fn dictionary_method(output: &str) -> Option<&'static str> {
    if output.starts_with(LZW_DICTIONARY_HEADER) {
        Some("lzw")
    } else if output.starts_with(REPAIR_DICTIONARY_HEADER) {
        Some("repair")
    } else {
        None
    }
}

/// Patches the Anthropic request JSON body with the compressed tool outputs.
pub fn patch_anthropic_tool_results(
    body: &mut Value,
    updates: &[ToolOutputCompressionUpdate],
) -> bool {
    let mut pending = IndexMap::new();
    for update in updates {
        let Some(tool_call_id) = update
            .tool_call_id
            .as_deref()
            .filter(|tool_call_id| !tool_call_id.is_empty())
        else {
            return false;
        };
        pending.insert(tool_call_id.to_string(), update.output.clone());
    }

    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    for message in messages {
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        if !patch_anthropic_content_blocks(content, &mut pending) {
            return false;
        }
    }

    pending.is_empty()
}

fn patch_anthropic_content_blocks(
    content: &mut Value,
    pending: &mut IndexMap<String, String>,
) -> bool {
    let Value::Array(blocks) = content else {
        return true;
    };

    for block in blocks {
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }
        let Some(tool_use_id) = obj
            .get("tool_use_id")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            continue;
        };
        let Some(output) = pending.get(&tool_use_id).cloned() else {
            continue;
        };
        if !patch_anthropic_tool_result_content(obj, &output) {
            return false;
        }
        pending.shift_remove(&tool_use_id);
    }

    true
}

fn patch_anthropic_tool_result_content(
    obj: &mut serde_json::Map<String, Value>,
    output: &str,
) -> bool {
    let output = raw_anthropic_tool_result_output(obj, output);
    match obj.get_mut("content") {
        Some(Value::String(content)) => {
            *content = output;
            true
        }
        Some(Value::Null) | None => {
            obj.insert("content".to_string(), Value::String(output));
            true
        }
        Some(Value::Array(blocks)) => patch_single_anthropic_tool_result_text_block(blocks, output),
        _ => false,
    }
}

fn raw_anthropic_tool_result_output(obj: &serde_json::Map<String, Value>, output: &str) -> String {
    if obj.get("is_error").and_then(Value::as_bool) == Some(true) {
        output.strip_prefix("Error: ").unwrap_or(output).to_string()
    } else {
        output.to_string()
    }
}

fn patch_single_anthropic_tool_result_text_block(blocks: &mut [Value], output: String) -> bool {
    let [block] = blocks else {
        return false;
    };
    let Some(obj) = block.as_object_mut() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some("text") {
        return false;
    }
    if !obj.get("text").is_some_and(Value::is_string) {
        return false;
    }
    obj.insert("text".to_string(), Value::String(output));
    true
}

#[derive(Debug, Clone)]
pub(super) struct ToolOutputCompressionLogConfig {
    path: PathBuf,
    queue_capacity: usize,
    max_event_bytes: usize,
}

pub(super) struct ToolOutputCompressionLogSink {
    sender: tokio::sync::mpsc::Sender<ToolOutputCompressionLogCommand>,
    dropped_events: AtomicU64,
}

enum ToolOutputCompressionLogCommand {
    Event(Value),
    #[cfg(test)]
    Flush(tokio::sync::oneshot::Sender<()>),
}

impl ToolOutputCompressionLogConfig {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
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
}

impl ToolOutputCompressionLogSink {
    pub(super) fn spawn(config: ToolOutputCompressionLogConfig) -> Result<Arc<Self>, String> {
        tokio::runtime::Handle::try_current().map_err(|err| {
            format!("tool-output compression log sink requires a Tokio runtime: {err}")
        })?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.path)
            .map_err(|err| format!("failed to open {}: {err}", config.path.display()))?;
        let (sender, receiver) = tokio::sync::mpsc::channel(config.queue_capacity.max(1));
        let sink = Arc::new(Self {
            sender,
            dropped_events: AtomicU64::new(0),
        });
        spawn_writer_task(file, receiver, config.max_event_bytes, config.path);
        Ok(sink)
    }

    pub(super) fn emit(&self, event: Value) {
        match self
            .sender
            .try_send(ToolOutputCompressionLogCommand::Event(event))
        {
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
            .send(ToolOutputCompressionLogCommand::Flush(sender))
            .await
            .map_err(|_| "tool-output compression log writer closed".to_string())?;
        receiver
            .await
            .map_err(|_| "tool-output compression log flush failed".to_string())
    }

    fn record_drop(&self, reason: &'static str) {
        let dropped = self.dropped_events.fetch_add(1, Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_power_of_two() {
            tracing::warn!(
                target: "forge.tool_output",
                reason,
                dropped,
                "dropped proxy tool-output compression JSONL event"
            );
        }
    }
}

pub(super) fn init_proxy_tool_output_compression_log_sink_from_env() {
    let _ = compression_log_sink_from_env();
}

pub(super) fn shutdown_proxy_tool_output_compression_log_sink() {
    match COMPRESSION_LOG_SINK.lock() {
        Ok(mut guard) => {
            *guard = None;
        }
        Err(err) => {
            warn_compression_log_init_once(format!(
                "tool-output compression log lock poisoned: {err}"
            ));
        }
    }
}

fn emit_proxy_tool_output_compression_jsonl(event: Value) {
    let Some(sink) = compression_log_sink_from_env() else {
        return;
    };
    sink.emit(event);
}

fn compression_log_sink_from_env() -> Option<Arc<ToolOutputCompressionLogSink>> {
    let mut guard = match COMPRESSION_LOG_SINK.lock() {
        Ok(guard) => guard,
        Err(err) => {
            warn_compression_log_init_once(format!(
                "tool-output compression log lock poisoned: {err}"
            ));
            return None;
        }
    };
    if let Some(sink) = guard.as_ref() {
        return Some(sink.clone());
    }

    let config = config_from_env()?;
    match ToolOutputCompressionLogSink::spawn(config) {
        Ok(sink) => {
            *guard = Some(sink.clone());
            Some(sink)
        }
        Err(err) => {
            warn_compression_log_init_once(err);
            None
        }
    }
}

fn config_from_env() -> Option<ToolOutputCompressionLogConfig> {
    let path = std::env::var_os(FORGE_TOOL_OUTPUT_COMPRESSION_LOG_ENV)?;
    if path.as_os_str().is_empty() {
        return None;
    }
    Some(
        ToolOutputCompressionLogConfig::new(PathBuf::from(path))
            .with_queue_capacity(usize_env_or_default(
                FORGE_TOOL_OUTPUT_COMPRESSION_LOG_QUEUE_CAPACITY_ENV,
                DEFAULT_QUEUE_CAPACITY,
            ))
            .with_max_event_bytes(usize_env_or_default(
                FORGE_TOOL_OUTPUT_COMPRESSION_LOG_MAX_EVENT_BYTES_ENV,
                DEFAULT_MAX_EVENT_BYTES,
            )),
    )
}

fn spawn_writer_task(
    file: File,
    receiver: tokio::sync::mpsc::Receiver<ToolOutputCompressionLogCommand>,
    max_event_bytes: usize,
    path: PathBuf,
) {
    tokio::task::spawn_blocking(move || writer_loop(file, receiver, max_event_bytes, path));
}

fn writer_loop(
    mut file: File,
    mut receiver: tokio::sync::mpsc::Receiver<ToolOutputCompressionLogCommand>,
    max_event_bytes: usize,
    path: PathBuf,
) {
    while let Some(command) = receiver.blocking_recv() {
        match command {
            ToolOutputCompressionLogCommand::Event(event) => {
                let line = match serde_json::to_string(&event) {
                    Ok(line) => line,
                    Err(err) => {
                        tracing::warn!(
                            target: "forge.tool_output",
                            error = %err,
                            "failed to serialize proxy tool-output compression JSONL event"
                        );
                        continue;
                    }
                };
                if line.len() > max_event_bytes {
                    tracing::warn!(
                        target: "forge.tool_output",
                        event_bytes = line.len(),
                        max_event_bytes,
                        "dropped oversized proxy tool-output compression JSONL event"
                    );
                    continue;
                }
                if let Err(err) = writeln!(file, "{line}") {
                    tracing::warn!(
                        target: "forge.tool_output",
                        error = %err,
                        path = %path.display(),
                        "failed to write proxy tool-output compression JSONL event"
                    );
                }
            }
            #[cfg(test)]
            ToolOutputCompressionLogCommand::Flush(done) => {
                if let Err(err) = file.flush() {
                    tracing::warn!(
                        target: "forge.tool_output",
                        error = %err,
                        path = %path.display(),
                        "failed to flush proxy tool-output compression JSONL"
                    );
                }
                let _ = done.send(());
            }
        }
    }
    let _ = file.flush();
}

fn usize_env_or_default(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn warn_compression_log_init_once(error: String) {
    if COMPRESSION_LOG_INIT_WARNED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        tracing::warn!(
            target: "forge.tool_output",
            error,
            "proxy tool-output compression JSONL logging disabled"
        );
    }
}

fn line_count(value: &str) -> usize {
    if value.is_empty() {
        0
    } else {
        value.lines().count()
    }
}

fn fingerprint64(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}
