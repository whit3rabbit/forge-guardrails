use crate::clients::base::{LLMClient, ToolCall};
use crate::core::inference;
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::core::workflow::Workflow;
use crate::error::{ForgeError, ToolError, ToolExecutionError};
use crate::guardrails::{ErrorTracker, StepEnforcer};
use crate::prompts::nudges;
use serde_json::Value;

use super::WorkflowRunner;

pub(super) struct ToolResultRecord {
    pub(super) tool_name: String,
    pub(super) call_id: String,
    pub(super) content: String,
}

impl<C: LLMClient> WorkflowRunner<C> {
    /// Execute a batch of tool calls, returning the terminal tool result if found.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool_batch(
        &self,
        calls: &[ToolCall],
        messages: &mut Vec<Message>,
        workflow: &Workflow,
        step_enforcer: &mut StepEnforcer,
        error_tracker: &mut ErrorTracker,
        tool_call_counter: &mut i64,
        iteration: i32,
    ) -> Result<Option<Value>, ForgeError> {
        let mut terminal_result: Option<Value> = None;
        let mut batch_had_error = false;
        let mut last_error: Option<ForgeError> = None;

        // Execute each tool sequentially.
        let mut results: Vec<ToolResultRecord> = Vec::new();
        for tc in calls {
            let call_id = if let Some(ref id) = tc.id {
                id.clone()
            } else {
                let id = inference::format_tool_call_id(*tool_call_counter);
                *tool_call_counter += 1;
                id
            };

            let callable = match workflow.get_callable(&tc.tool) {
                Ok(c) => c,
                Err(_) => {
                    results.push(ToolResultRecord {
                        tool_name: tc.tool.clone(),
                        call_id,
                        content: "[TOOL_ERROR] Tool not found".to_string(),
                    });
                    continue;
                }
            };

            match callable(tc.args.clone()).await {
                Ok(output) => {
                    step_enforcer.record(&tc.tool, Some(&tc.args));

                    let is_terminal = workflow.terminal_tools.contains(&tc.tool);
                    if is_terminal {
                        terminal_result = Some(Self::terminal_output_value(&output));
                    }

                    results.push(ToolResultRecord {
                        tool_name: tc.tool.clone(),
                        call_id,
                        content: Self::tool_result_content(&output),
                    });
                }
                Err(ToolError::Resolution(e)) => {
                    // Soft error: feed back as tool result, don't count toward tool_errors.
                    error_tracker.record_result(false, true);
                    results.push(ToolResultRecord {
                        tool_name: tc.tool.clone(),
                        call_id,
                        content: format!("[ToolResolutionError] {}", e),
                    });
                }
                Err(ToolError::Execution(e)) => {
                    // Hard error: increment consecutive tool errors count.
                    batch_had_error = true;
                    error_tracker.record_result(false, false);
                    last_error = Some(ForgeError::ToolExecution(ToolExecutionError::new(
                        tc.tool.clone(),
                        e.to_string(),
                    )));
                    results.push(ToolResultRecord {
                        tool_name: tc.tool.clone(),
                        call_id,
                        content: format!("[ToolError] ToolExecutionError: {}", e),
                    });
                }
            }
        }

        self.emit_tool_result_records(&results, messages, iteration);

        // Post-batch bookkeeping — matches current Rust behavior.
        if batch_had_error {
            if error_tracker.tool_errors_exhausted() {
                if let Some(err) = last_error {
                    return Err(err);
                }
            }
        } else {
            error_tracker.reset_errors();
            step_enforcer.reset_premature();
            step_enforcer.reset_prereq_violations();
        }

        // Check for terminal result.
        if let Some(val) = terminal_result {
            return Ok(Some(val));
        }

        Ok(None)
    }

    pub(super) fn is_mixed_terminal_batch(tool_calls: &[ToolCall], workflow: &Workflow) -> bool {
        let mut has_terminal = false;
        let mut has_nonterminal = false;
        for tc in tool_calls {
            if workflow.terminal_tools.contains(&tc.tool) {
                has_terminal = true;
            } else {
                has_nonterminal = true;
            }
            if has_terminal && has_nonterminal {
                return true;
            }
        }
        false
    }

    pub(super) fn mixed_terminal_batch_nudge(
        workflow: &Workflow,
        step_enforcer: &StepEnforcer,
    ) -> String {
        let pending = step_enforcer.pending();
        let allowed_owned: Vec<String> = if pending.is_empty() {
            workflow
                .tools
                .keys()
                .filter(|name| !workflow.terminal_tools.contains(*name))
                .cloned()
                .collect()
        } else {
            pending
        };
        let allowed: Vec<&str> = allowed_owned.iter().map(String::as_str).collect();
        let blocked: Vec<&str> = workflow.terminal_tools.iter().map(String::as_str).collect();
        nudges::unsafe_batch_nudge(&allowed, &blocked)
    }

    /// Record the assistant's successful tool-call turn after guardrail checks.
    pub(super) fn emit_assistant_tool_calls(
        &self,
        mut calls: Vec<ToolCall>,
        messages: &mut Vec<Message>,
        tool_call_counter: &mut i64,
        step_index: i64,
    ) -> Vec<ToolCall> {
        if let Some(reasoning) = calls.first().and_then(|tc| tc.reasoning.as_ref()) {
            let reasoning_msg = Message::new(
                MessageRole::Assistant,
                reasoning.as_str(),
                MessageMeta::new(MessageType::Reasoning).with_step_index(step_index),
            );
            self.fire_message(&reasoning_msg);
            messages.push(reasoning_msg);
        }

        let mut infos = Vec::with_capacity(calls.len());
        for tc in &mut calls {
            let call_id = inference::format_tool_call_id(*tool_call_counter);
            *tool_call_counter += 1;
            tc.id = Some(call_id.clone());
            infos.push(ToolCallInfo::new(&tc.tool, Some(tc.args.clone()), call_id));
        }

        let tool_call_msg = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall).with_step_index(step_index),
        )
        .with_tool_calls(infos);
        self.fire_message(&tool_call_msg);
        messages.push(tool_call_msg);

        calls
    }

    pub(super) fn emit_guardrail_nudge_results(
        &self,
        calls: &[ToolCall],
        messages: &mut Vec<Message>,
        step_index: i64,
        msg_type: MessageType,
        prefix: &str,
        nudge_content: &str,
    ) {
        let error_content = format!("{} {}", prefix, nudge_content);
        for tc in calls {
            let call_id = tc.id.clone().unwrap_or_default();
            let result_msg = Message::new(
                MessageRole::Tool,
                error_content.as_str(),
                MessageMeta::new(msg_type).with_step_index(step_index),
            )
            .with_tool_name(&tc.tool)
            .with_tool_call_id(&call_id);
            self.fire_message(&result_msg);
            messages.push(result_msg);
        }
    }

    pub(super) fn emit_tool_result_records(
        &self,
        records: &[ToolResultRecord],
        messages: &mut Vec<Message>,
        iteration: i32,
    ) {
        for record in records {
            let result_msg = Message::new(
                MessageRole::Tool,
                record.content.as_str(),
                MessageMeta::new(MessageType::ToolResult).with_step_index(iteration as i64),
            )
            .with_tool_name(record.tool_name.as_str())
            .with_tool_call_id(record.call_id.as_str());
            self.fire_message(&result_msg);
            messages.push(result_msg);
        }
    }

    fn terminal_output_value(output: &Value) -> Value {
        if output.is_null() {
            Value::Null
        } else if let Some(s) = output.as_str() {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()))
        } else {
            output.clone()
        }
    }

    fn tool_result_content(output: &Value) -> String {
        if let Some(s) = output.as_str() {
            s.to_string()
        } else {
            output.to_string()
        }
    }
}
