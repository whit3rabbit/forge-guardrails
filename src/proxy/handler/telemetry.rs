use sentry::protocol::Event;

use crate::clients::base::ToolCall;
use crate::guardrails::{ClassifierAction, FinalResponseScore, ToolCallScore};

const FORGE_SENTRY_ENABLED: &str = "FORGE_SENTRY_ENABLED";
const MAX_TAG_VALUE_CHARS: usize = 128;
const MAX_TAG_ITEMS: usize = 8;

pub(super) fn capture_tool_call_classifier_non_allow(call: &ToolCall, score: &ToolCallScore) {
    if score.action == ClassifierAction::Allow || !sentry_enabled_from_env() {
        return;
    }
    sentry::capture_event(tool_call_classifier_event(call, score));
}

pub(super) fn capture_final_response_classifier_non_allow(
    terminal_tool: &str,
    score: &FinalResponseScore,
) {
    if score.action == ClassifierAction::Allow || !sentry_enabled_from_env() {
        return;
    }
    sentry::capture_event(final_response_classifier_event(terminal_tool, score));
}

pub(super) fn capture_guardrail_exhausted(
    reason: &str,
    tool_calls: &[ToolCall],
    pending_steps: &[String],
    retries: Option<i32>,
    max_retries: Option<i32>,
    stream: Option<bool>,
) {
    if !sentry_enabled_from_env() {
        return;
    }
    sentry::capture_event(guardrail_exhausted_event(
        reason,
        tool_calls,
        pending_steps,
        retries,
        max_retries,
        stream,
    ));
}

pub(super) fn tool_call_classifier_event(call: &ToolCall, score: &ToolCallScore) -> Event<'static> {
    let mut event = base_event("classifier_tool_call_non_allow", sentry::Level::Warning);
    insert_tag(&mut event, "tool", &call.tool);
    insert_tag(&mut event, "label", score.label.as_label().as_ref());
    insert_tag(&mut event, "action", score.action.as_str());
    insert_tag(&mut event, "confidence", format!("{:.3}", score.confidence));
    insert_tag(&mut event, "latency_ms", format!("{:.1}", score.latency_ms));
    insert_tag(&mut event, "model_version", &score.model_version);
    event
}

pub(super) fn final_response_classifier_event(
    terminal_tool: &str,
    score: &FinalResponseScore,
) -> Event<'static> {
    let mut event = base_event(
        "classifier_final_response_non_allow",
        sentry::Level::Warning,
    );
    insert_tag(&mut event, "terminal_tool", terminal_tool);
    insert_tag(&mut event, "label", score.label.as_label().as_ref());
    insert_tag(&mut event, "action", score.action.as_str());
    insert_tag(&mut event, "confidence", format!("{:.3}", score.confidence));
    insert_tag(&mut event, "latency_ms", format!("{:.1}", score.latency_ms));
    insert_tag(&mut event, "model_version", &score.model_version);
    event
}

pub(super) fn guardrail_exhausted_event(
    reason: &str,
    tool_calls: &[ToolCall],
    pending_steps: &[String],
    retries: Option<i32>,
    max_retries: Option<i32>,
    stream: Option<bool>,
) -> Event<'static> {
    let mut event = base_event("guardrail_exhausted", sentry::Level::Error);
    insert_tag(&mut event, "reason", reason);
    insert_tag(&mut event, "tool_count", tool_calls.len().to_string());
    insert_tag(
        &mut event,
        "tool_names",
        safe_list(tool_calls.iter().map(|call| call.tool.as_str())),
    );
    insert_tag(
        &mut event,
        "pending_step_count",
        pending_steps.len().to_string(),
    );
    insert_tag(
        &mut event,
        "pending_steps",
        safe_list(pending_steps.iter().map(String::as_str)),
    );
    if let Some(retries) = retries {
        insert_tag(&mut event, "retries", retries.to_string());
    }
    if let Some(max_retries) = max_retries {
        insert_tag(&mut event, "max_retries", max_retries.to_string());
    }
    if let Some(stream) = stream {
        insert_tag(&mut event, "stream", stream.to_string());
    }
    event
}

fn base_event(kind: &str, level: sentry::Level) -> Event<'static> {
    let mut event = Event::new();
    event.level = level;
    event.message = Some("forge proxy aggregate telemetry".to_string());
    insert_tag(&mut event, "forge.event", kind);
    insert_tag(&mut event, "component", "proxy.guardrails");
    event
}

fn insert_tag(event: &mut Event<'static>, key: &str, value: impl AsRef<str>) {
    event
        .tags
        .insert(key.to_string(), safe_tag_value(value.as_ref()));
}

fn safe_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let values = items
        .take(MAX_TAG_ITEMS)
        .map(safe_tag_value)
        .collect::<Vec<_>>();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn safe_tag_value(value: &str) -> String {
    let sanitized = value
        .chars()
        .filter(|ch| !ch.is_control())
        .take(MAX_TAG_VALUE_CHARS)
        .collect::<String>()
        .trim()
        .to_string();
    if sanitized.is_empty() {
        "none".to_string()
    } else {
        sanitized
    }
}

fn sentry_enabled_from_env() -> bool {
    match std::env::var(FORGE_SENTRY_ENABLED) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}
