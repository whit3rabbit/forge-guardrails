use serde_json::{json, Value};

use crate::clients::base::ToolCall;
use crate::core::message::{Message, MessageRole, MessageType};
use crate::core::tool_spec::ToolSpec;
use crate::guardrails::{
    recent_errors_from_messages, ClassifierAction, FinalResponseContext, FinalResponseScorer,
    FinalResponseToolResult, ScoringContext, StepEnforcer, ToolCallScorer, WorkflowStateForScoring,
};
use crate::tools::respond::RESPOND_TOOL_NAME;

use super::classifier_log::{emit_proxy_classifier_jsonl, proxy_tool_call_for_json, unix_ms};

pub(super) fn score_proxy_tool_calls(
    scorer: Option<&dyn ToolCallScorer>,
    messages: &[Message],
    tool_calls: &[ToolCall],
    step_enforcer: Option<&StepEnforcer>,
    tool_specs: &[ToolSpec],
) -> Option<String> {
    let scorer = scorer?;
    let user_request = latest_proxy_user_request(messages).unwrap_or_default();
    let recent_errors = recent_errors_from_messages(messages, 8);
    let ctx = match step_enforcer {
        Some(enforcer) => ScoringContext::from_step_enforcer(
            user_request,
            enforcer,
            &enforcer.terminal_tools,
            recent_errors,
            tool_specs,
        ),
        None => ScoringContext::new(
            user_request,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            proxy_terminal_tools_for_scoring(tool_specs),
            recent_errors,
            tool_specs,
        ),
    };

    let mut nudge: Option<String> = None;
    for call in tool_calls {
        match scorer.score(&ctx, call) {
            Ok(score) => {
                tracing::info!(
                    target: "forge.classifier",
                    label = ?score.label,
                    confidence = score.confidence,
                    action = ?score.action,
                    latency_ms = score.latency_ms,
                    tool = %call.tool,
                    "tool-call classifier score"
                );
                emit_proxy_classifier_jsonl(json!({
                    "kind": "tool_call",
                    "unix_ms": unix_ms(),
                    "user_request": ctx.user_request.as_str(),
                    "initial_user_request": initial_proxy_user_request(messages).unwrap_or_default(),
                    "workflow_state": &ctx.workflow_state,
                    "candidate_call": proxy_tool_call_for_json(call),
                    "tool": call.tool.as_str(),
                    "label": score.label.as_label().as_ref(),
                    "confidence": score.confidence,
                    "action": score.action.as_str(),
                    "latency_ms": score.latency_ms,
                    "model_version": score.model_version.as_str(),
                }));
                if matches!(
                    score.action,
                    ClassifierAction::AdvisoryNudge | ClassifierAction::Block
                ) {
                    let content = crate::prompts::classifier_nudge(score.label.as_label().as_ref());
                    if score.action == ClassifierAction::Block || nudge.is_none() {
                        nudge = Some(content);
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    target: "forge.classifier",
                    error = %err,
                    tool = %call.tool,
                    "classifier scoring failed; allowing deterministic path"
                );
            }
        }
    }
    nudge
}

pub(super) fn score_proxy_final_tool_calls(
    scorer: Option<&dyn FinalResponseScorer>,
    messages: &[Message],
    tool_calls: &[ToolCall],
    step_enforcer: Option<&StepEnforcer>,
    tool_specs: &[ToolSpec],
) -> Option<String> {
    let terminal_tools = proxy_terminal_tool_set(step_enforcer, tool_specs);
    let mut nudge = None;
    for call in tool_calls
        .iter()
        .filter(|call| terminal_tools.contains(call.tool.as_str()))
    {
        let candidate = proxy_candidate_final_response_from_call(call);
        let mut trace = proxy_tool_trace_from_messages(messages);
        trace.push(call.tool.clone());
        if let Some(content) = score_proxy_final_candidate(
            scorer,
            messages,
            &candidate,
            trace,
            step_enforcer,
            tool_specs,
            Some(call.tool.as_str()),
        ) {
            nudge = Some(content);
            break;
        }
    }
    nudge
}

pub(super) fn score_proxy_final_text(
    scorer: Option<&dyn FinalResponseScorer>,
    messages: &[Message],
    candidate: &str,
    step_enforcer: Option<&StepEnforcer>,
    tool_specs: &[ToolSpec],
) -> Option<String> {
    score_proxy_final_candidate(
        scorer,
        messages,
        candidate,
        proxy_tool_trace_from_messages(messages),
        step_enforcer,
        tool_specs,
        None,
    )
}

fn score_proxy_final_candidate(
    scorer: Option<&dyn FinalResponseScorer>,
    messages: &[Message],
    candidate: &str,
    tool_trace: Vec<String>,
    step_enforcer: Option<&StepEnforcer>,
    tool_specs: &[ToolSpec],
    terminal_tool: Option<&str>,
) -> Option<String> {
    let scorer = scorer?;
    let user_request = latest_proxy_user_request(messages).unwrap_or_default();
    let workflow_state = match step_enforcer {
        Some(enforcer) => WorkflowStateForScoring {
            required_steps: enforcer.tracker.required_steps().to_vec(),
            completed_steps: enforcer.completed_steps().keys().cloned().collect(),
            pending_steps: enforcer.pending(),
            terminal_tools: enforcer.terminal_tools.iter().cloned().collect(),
            recent_errors: recent_errors_from_messages(messages, 8),
        },
        None => WorkflowStateForScoring {
            required_steps: Vec::new(),
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            terminal_tools: proxy_terminal_tools_for_scoring(tool_specs),
            recent_errors: recent_errors_from_messages(messages, 8),
        },
    };
    let ctx = FinalResponseContext {
        schema_version: "final-response-verifier-input/v1".to_string(),
        user_request: user_request.to_string(),
        workflow_state,
        required_facts: Vec::new(),
        tool_trace,
        tool_results: proxy_tool_results_from_messages(messages),
        candidate_final_response: candidate.to_string(),
        metadata: None,
    };
    match scorer.score(&ctx) {
        Ok(score) => {
            tracing::info!(
                target: "forge.classifier",
                label = %score.label.as_label(),
                confidence = score.confidence,
                action = %score.action.as_str(),
                latency_ms = score.latency_ms,
                terminal_tool = terminal_tool.unwrap_or("text"),
                "final-response classifier score"
            );
            emit_proxy_classifier_jsonl(json!({
                "kind": "final_response",
                "unix_ms": unix_ms(),
                "user_request": ctx.user_request.as_str(),
                "initial_user_request": initial_proxy_user_request(messages).unwrap_or_default(),
                "workflow_state": &ctx.workflow_state,
                "required_facts": &ctx.required_facts,
                "tool_trace": &ctx.tool_trace,
                "tool_results": ctx.tool_results.iter().map(|result| {
                    json!({"tool_name": result.tool_name.as_str(), "content": result.content.as_str()})
                }).collect::<Vec<_>>(),
                "candidate_final_response": ctx.candidate_final_response.as_str(),
                "terminal_tool": terminal_tool.unwrap_or("text"),
                "label": score.label.as_label().as_ref(),
                "confidence": score.confidence,
                "action": score.action.as_str(),
                "latency_ms": score.latency_ms,
                "model_version": score.model_version.as_str(),
            }));
            if matches!(
                score.action,
                ClassifierAction::AdvisoryNudge | ClassifierAction::Block
            ) {
                Some(crate::prompts::classifier_nudge(
                    score.label.as_label().as_ref(),
                ))
            } else {
                None
            }
        }
        Err(err) => {
            tracing::warn!(
                target: "forge.classifier",
                error = %err,
                terminal_tool = terminal_tool.unwrap_or("text"),
                "final-response classifier scoring failed; allowing deterministic path"
            );
            None
        }
    }
}

fn proxy_terminal_tool_set<'a>(
    step_enforcer: Option<&'a StepEnforcer>,
    tool_specs: &'a [ToolSpec],
) -> std::collections::HashSet<&'a str> {
    match step_enforcer {
        Some(enforcer) => enforcer.terminal_tools.iter().map(String::as_str).collect(),
        None => tool_specs
            .iter()
            .filter(|spec| spec.name == RESPOND_TOOL_NAME)
            .map(|spec| spec.name.as_str())
            .collect(),
    }
}

fn proxy_candidate_final_response_from_call(call: &ToolCall) -> String {
    for key in ["message", "answer", "content", "report", "summary"] {
        if let Some(value) = call.args.get(key) {
            return value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
        }
    }
    Value::Object(
        call.args
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
    .to_string()
}

fn proxy_tool_trace_from_messages(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| message.tool_calls.as_ref())
        .flat_map(|calls| calls.iter().map(|call| call.name.clone()))
        .collect()
}

fn proxy_tool_results_from_messages(messages: &[Message]) -> Vec<FinalResponseToolResult> {
    messages
        .iter()
        .filter(|message| message.metadata.msg_type == MessageType::ToolResult)
        .filter_map(|message| {
            Some(FinalResponseToolResult {
                tool_name: message.tool_name.clone()?,
                content: message.content.clone(),
            })
        })
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
        .find(|message| {
            message.role == MessageRole::User
                && message.metadata.msg_type == MessageType::UserInput
                && !message.content.trim().is_empty()
        })
        .map(|message| message.content.as_str())
}

fn proxy_terminal_tools_for_scoring(tool_specs: &[ToolSpec]) -> Vec<String> {
    tool_specs
        .iter()
        .filter(|spec| spec.name == RESPOND_TOOL_NAME)
        .map(|spec| spec.name.clone())
        .collect()
}
