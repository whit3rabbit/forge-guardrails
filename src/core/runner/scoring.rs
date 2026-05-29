use crate::clients::base::{LLMClient, ToolCall};
use crate::core::message::{Message, MessageType};
use crate::core::tool_spec::ToolSpec;
use crate::guardrails::{
    recent_errors_from_messages, ClassifierAction, FinalResponseContext, FinalResponseToolResult,
    ScoringContext, ScoringPipeline, StepEnforcer, WorkflowStateForScoring,
};
use serde_json::Value;
use std::sync::Arc;

use super::{latest_user_request, WorkflowRunner};

impl<C: LLMClient> WorkflowRunner<C> {
    pub(super) async fn score_tool_calls(
        &self,
        fallback_user_message: &str,
        messages: &[Message],
        tool_calls: &[ToolCall],
        step_enforcer: &StepEnforcer,
        tool_specs: &[ToolSpec],
    ) -> Option<String> {
        let scorer = self.scorer.clone()?;
        let pipeline = ScoringPipeline::new(Some(scorer), None);
        let user_request = latest_user_request(messages).unwrap_or(fallback_user_message);
        let ctx = Arc::new(ScoringContext::from_step_enforcer(
            user_request,
            step_enforcer,
            &step_enforcer.terminal_tools,
            recent_errors_from_messages(messages, 8),
            tool_specs,
        ));
        pipeline
            .score_tool_calls(
                ctx,
                tool_calls,
                |call, score| {
                    tracing::info!(
                        target: "forge.classifier",
                        label = ?score.label,
                        confidence = score.confidence,
                        action = ?score.action,
                        latency_ms = score.latency_ms,
                        tool = %call.tool,
                        "tool-call classifier score"
                    );
                    if let Some(callback) = &self.on_tool_call_score {
                        callback(call, score);
                    }
                },
                |call, err| {
                    tracing::warn!(
                        target: "forge.classifier",
                        error = %err,
                        tool = %call.tool,
                        "classifier scoring failed; allowing deterministic path"
                    );
                },
            )
            .await
    }

    pub(super) async fn score_candidate_final_responses(
        &self,
        fallback_user_message: &str,
        messages: &[Message],
        tool_calls: &[ToolCall],
        step_enforcer: &StepEnforcer,
    ) -> Option<String> {
        let scorer = self.final_response_scorer.clone()?;
        let pipeline = ScoringPipeline::new(None, Some(scorer));
        let user_request = latest_user_request(messages).unwrap_or(fallback_user_message);
        let base_trace = Self::tool_trace_from_messages(messages);
        let tool_results = Self::tool_results_from_messages(messages);
        let mut nudge: Option<String> = None;
        for call in tool_calls
            .iter()
            .filter(|call| step_enforcer.terminal_tools.contains(&call.tool))
        {
            let mut tool_trace = base_trace.clone();
            tool_trace.push(call.tool.clone());
            let ctx = Arc::new(FinalResponseContext {
                schema_version: "final-response-verifier-input/v1".to_string(),
                user_request: user_request.to_string(),
                workflow_state: WorkflowStateForScoring {
                    required_steps: step_enforcer.tracker.required_steps().to_vec(),
                    completed_steps: step_enforcer.completed_steps().keys().cloned().collect(),
                    pending_steps: step_enforcer.pending(),
                    terminal_tools: step_enforcer.terminal_tools.iter().cloned().collect(),
                    recent_errors: recent_errors_from_messages(messages, 8),
                },
                required_facts: Vec::new(),
                tool_trace,
                tool_results: tool_results.clone(),
                candidate_final_response: Self::candidate_final_response_from_call(call),
                metadata: None,
            });
            let mut action = None;
            if let Some(content) = pipeline
                .score_final_response(
                    ctx,
                    |score| {
                        action = Some(score.action);
                        tracing::info!(
                            target: "forge.classifier",
                            label = %score.label.as_label(),
                            confidence = score.confidence,
                            action = %score.action.as_str(),
                            latency_ms = score.latency_ms,
                            terminal_tool = %call.tool,
                            "final-response classifier score"
                        );
                        if let Some(callback) = &self.on_final_response_score {
                            callback(score);
                        }
                    },
                    |err| {
                        tracing::warn!(
                            target: "forge.classifier",
                            error = %err,
                            terminal_tool = %call.tool,
                            "final-response classifier scoring failed; allowing deterministic path"
                        );
                    },
                )
                .await
            {
                if action == Some(ClassifierAction::Block) || nudge.is_none() {
                    nudge = Some(content);
                }
            }
        }
        nudge
    }

    fn candidate_final_response_from_call(call: &ToolCall) -> String {
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

    fn tool_trace_from_messages(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .filter_map(|message| message.tool_calls.as_ref())
            .flat_map(|calls| calls.iter().map(|call| call.name.clone()))
            .collect()
    }

    fn tool_results_from_messages(messages: &[Message]) -> Vec<FinalResponseToolResult> {
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
}
