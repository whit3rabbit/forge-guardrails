//! Workflow runner: orchestrates the multi-turn agentic tool-calling loop.
//!
//! WorkflowRunner drives the iterative loop: inference, guardrails check,
//! tool execution, error tracking, and termination on terminal tool success.

use crate::client::LLMClient;
use crate::context::ContextManager;
use crate::error::{
    ForgeError, MaxIterationsError, StepEnforcementError, ToolCallError, WorkflowCancelledError,
};
use crate::guardrails::{GuardAction, Guardrails, RetryNudgeFn, TerminalTool};
use crate::inference::{self, OnChunkFn};
use crate::message::{Message, MessageMeta, MessageRole, MessageType};
use crate::streaming::LLMResponse;
use crate::tool_spec::ToolSpec;
use crate::workflow::Workflow;
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::watch;

/// Callback type for message events during a run.
pub type OnMessageFn = Box<dyn Fn(&Message) + Send + Sync>;

/// Workflow runner orchestrating multi-turn LLM tool-calling loops.
///
/// Generic over the LLM client type because the LLMClient trait uses
/// async methods and is not dyn-compatible.
pub struct WorkflowRunner<C: LLMClient> {
    client: Arc<C>,
    context_manager: Arc<tokio::sync::Mutex<ContextManager>>,
    max_iterations: i32,
    max_retries_per_step: i32,
    max_tool_errors: i32,
    stream: bool,
    on_chunk: Option<Arc<OnChunkFn>>,
    on_message: Option<Arc<OnMessageFn>>,
    rescue_enabled: bool,
    retry_nudge_fn: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
}

impl<C: LLMClient> WorkflowRunner<C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Arc<C>,
        context_manager: Arc<tokio::sync::Mutex<ContextManager>>,
        max_iterations: i32,
        max_retries_per_step: i32,
        max_tool_errors: i32,
        stream: bool,
        on_chunk: Option<OnChunkFn>,
        on_message: Option<OnMessageFn>,
        rescue_enabled: bool,
        retry_nudge: Option<String>,
    ) -> Self {
        let retry_nudge_fn = retry_nudge.map(|s| {
            Arc::new(move |_raw: &str| s.clone()) as Arc<dyn Fn(&str) -> String + Send + Sync>
        });
        Self {
            client,
            context_manager,
            max_iterations,
            max_retries_per_step,
            max_tool_errors,
            stream,
            on_chunk: on_chunk.map(Arc::new),
            on_message: on_message.map(Arc::new),
            rescue_enabled,
            retry_nudge_fn,
        }
    }

    /// Create a builder-like constructor that accepts a callable retry nudge.
    #[allow(clippy::too_many_arguments)]
    pub fn with_retry_nudge_fn(
        client: Arc<C>,
        context_manager: Arc<tokio::sync::Mutex<ContextManager>>,
        max_iterations: i32,
        max_retries_per_step: i32,
        max_tool_errors: i32,
        stream: bool,
        on_chunk: Option<OnChunkFn>,
        on_message: Option<OnMessageFn>,
        rescue_enabled: bool,
        retry_nudge_fn: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
    ) -> Self {
        Self {
            client,
            context_manager,
            max_iterations,
            max_retries_per_step,
            max_tool_errors,
            stream,
            on_chunk: on_chunk.map(Arc::new),
            on_message: on_message.map(Arc::new),
            rescue_enabled,
            retry_nudge_fn,
        }
    }

    /// Run a workflow to completion.
    ///
    /// Takes a workflow definition, user message, optional prompt variables,
    /// optional initial messages to seed conversation, and optional cancellation
    /// channel. Returns the terminal tool's result value on success.
    pub async fn run(
        &self,
        workflow: &Workflow,
        user_message: &str,
        prompt_vars: Option<&IndexMap<String, String>>,
        initial_messages: Option<Vec<Message>>,
        cancel: Option<watch::Receiver<bool>>,
    ) -> Result<Value, ForgeError> {
        let tool_names: Vec<String> = workflow.tools.keys().cloned().collect();
        let tool_specs: Vec<ToolSpec> = workflow.tools.values().map(|d| d.spec.clone()).collect();

        let terminal_set: IndexSet<String> = workflow.terminal_tools.iter().cloned().collect();
        let terminal_tool = if terminal_set.len() == 1 {
            TerminalTool::Single(
                terminal_set
                    .first()
                    .expect("terminal set has one element")
                    .clone(),
            )
        } else {
            TerminalTool::Multiple(terminal_set)
        };

        let retry_nudge_for_guardrails: Option<RetryNudgeFn> = self
            .retry_nudge_fn
            .clone()
            .map(|f| Box::new(move |raw: &str| f(raw)) as RetryNudgeFn);

        let mut guardrails = Guardrails::new(
            tool_names.clone(),
            terminal_tool,
            Some(workflow.required_steps.clone()),
            self.max_retries_per_step,
            self.max_tool_errors,
            self.rescue_enabled,
            3, // max premature attempts
            retry_nudge_for_guardrails,
        );

        let mut messages: Vec<Message> = Vec::new();
        let mut tool_call_counter: i64 = 0;
        let mut iteration: i32 = 0;
        let mut last_raw_response: Option<String> = None;

        // Build initial messages or use provided seed.
        let _seed_offset: usize;
        if let Some(seed) = initial_messages {
            for msg in &seed {
                messages.push(msg.clone());
            }
            _seed_offset = seed.len();
        } else {
            let system_content =
                workflow.build_system_prompt(prompt_vars.unwrap_or(&IndexMap::new()));
            let system_msg = Message::new(
                MessageRole::System,
                &system_content,
                MessageMeta::new(MessageType::SystemPrompt),
            );
            self.fire_message(&system_msg);
            messages.push(system_msg);

            let user_msg = Message::new(
                MessageRole::User,
                user_message,
                MessageMeta::new(MessageType::UserInput),
            );
            self.fire_message(&user_msg);
            messages.push(user_msg);
            _seed_offset = 0;
        }

        loop {
            // Check cancellation.
            if let Some(ref rx) = cancel {
                if *rx.borrow() {
                    let completed = guardrails.completed_steps();
                    let msgs: Vec<String> = messages.iter().map(|m| m.content.clone()).collect();
                    return Err(ForgeError::WorkflowCancelled(WorkflowCancelledError::new(
                        msgs,
                        completed,
                        iteration as i64,
                    )));
                }
            }

            iteration += 1;
            if iteration > self.max_iterations {
                let completed = guardrails.completed_steps();
                let pending = workflow.required_steps.clone();
                return Err(ForgeError::MaxIterations(MaxIterationsError::new(
                    iteration as i64,
                    completed,
                    pending,
                )));
            }

            // Remaining budget for attempts in this iteration.
            let remaining = self.max_retries_per_step + 1;
            let step_hint = workflow.required_steps.join(", ");

            let mut ctx = self.context_manager.lock().await;

            let retry_nudge_for_validator: Option<RetryNudgeFn> = self
                .retry_nudge_fn
                .clone()
                .map(|f| Box::new(move |raw: &str| f(raw)) as RetryNudgeFn);

            let validator = crate::guardrails::ResponseValidator::new(
                tool_names.clone(),
                self.rescue_enabled,
                retry_nudge_for_validator,
            );
            let mut error_tracker = crate::guardrails::ErrorTracker::new(
                self.max_retries_per_step,
                self.max_tool_errors,
            );

            let inference_result = inference::run_inference(
                &mut messages,
                self.client.as_ref(),
                &mut ctx,
                &validator,
                &mut error_tracker,
                &tool_specs,
                &mut tool_call_counter,
                iteration as i64,
                &step_hint,
                Some(remaining),
                self.stream,
                self.on_chunk.as_ref().map(|f| f.as_ref()),
                None,
            )
            .await;

            drop(ctx);

            let result = match inference_result {
                Some(r) => r,
                None => {
                    let raw = last_raw_response.as_deref().unwrap_or("no response");
                    return Err(ForgeError::ToolCall(
                        ToolCallError::new("Max attempts exhausted within iteration")
                            .with_raw_response(raw),
                    ));
                }
            };

            // Fire callbacks for new messages from inference.
            for msg in &result.new_messages {
                self.fire_message(msg);
            }

            // Guardrails check.
            let check = guardrails.check(&result.response);
            match check.action {
                GuardAction::Execute => {
                    let calls = check.tool_calls.expect("execute requires tool_calls");
                    let result_val = self
                        .execute_tool_batch(
                            &calls,
                            &mut messages,
                            workflow,
                            &mut guardrails,
                            &mut tool_call_counter,
                            iteration,
                            &mut last_raw_response,
                        )
                        .await?;
                    if let Some(val) = result_val {
                        return Ok(val);
                    }
                    // No terminal result yet; continue loop.
                }
                GuardAction::Retry => {
                    last_raw_response = inference::response_to_raw_string(&result.response);
                    continue;
                }
                GuardAction::StepBlocked => {
                    let nudge = check.nudge.expect("step_blocked requires nudge");

                    if let LLMResponse::ToolCalls(ref calls) = result.response {
                        for tc in calls {
                            let call_id = tc.id.clone().unwrap_or_default();
                            let error_content =
                                format!("[TOOL_ERROR] Step blocked: {}", nudge.content);
                            let result_msg = Message::new(
                                MessageRole::Tool,
                                &error_content,
                                MessageMeta::new(MessageType::ToolResult)
                                    .with_step_index(iteration as i64),
                            )
                            .with_tool_name(&tc.tool)
                            .with_tool_call_id(&call_id);
                            self.fire_message(&result_msg);
                            messages.push(result_msg);
                        }
                    }

                    let nudge_msg = Message::new(
                        MessageRole::User,
                        &nudge.content,
                        MessageMeta::new(MessageType::StepNudge).with_step_index(iteration as i64),
                    );
                    self.fire_message(&nudge_msg);
                    messages.push(nudge_msg);
                    last_raw_response = inference::response_to_raw_string(&result.response);
                    continue;
                }
                GuardAction::Fatal => {
                    let reason = check.reason.unwrap_or_default();
                    return Err(self.fatal_to_error(&reason, workflow));
                }
            }
        }
    }

    /// Execute a batch of tool calls, returning the terminal tool result if found.
    async fn execute_tool_batch(
        &self,
        calls: &[crate::streaming::ToolCall],
        messages: &mut Vec<Message>,
        workflow: &Workflow,
        guardrails: &mut Guardrails,
        tool_call_counter: &mut i64,
        iteration: i32,
        last_raw_response: &mut Option<String>,
    ) -> Result<Option<Value>, ForgeError> {
        let mut terminal_result: Option<Value> = None;
        let mut any_error = false;
        let mut first_error_tool = String::new();

        // Execute each tool sequentially.
        let mut results: Vec<(String, String, String, bool, bool)> = Vec::new();
        for tc in calls {
            let call_id = if let Some(ref id) = tc.id {
                id.clone()
            } else {
                *tool_call_counter += 1;
                inference::format_tool_call_id(*tool_call_counter)
            };

            let callable = match workflow.get_callable(&tc.tool) {
                Ok(c) => c,
                Err(_) => {
                    let result_str = "[TOOL_ERROR] Tool not found".to_string();
                    results.push((tc.tool.clone(), call_id, result_str, false, false));
                    continue;
                }
            };

            let args_vec: Vec<String> = tc
                .args
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();

            match callable(args_vec) {
                Ok(output) => {
                    let is_terminal = workflow.terminal_tools.contains(&tc.tool);

                    let executed_names: Vec<&str> = calls.iter().map(|c| c.tool.as_str()).collect();

                    if is_terminal {
                        let all_satisfied = guardrails.record(&executed_names);
                        if all_satisfied {
                            let val = if output.is_empty() {
                                Value::Null
                            } else {
                                serde_json::from_str(&output)
                                    .unwrap_or(Value::String(output.clone()))
                            };
                            terminal_result = Some(val);
                        }
                    } else {
                        guardrails.record(&executed_names);
                    }

                    results.push((tc.tool.clone(), call_id, output, true, false));
                }
                Err(e) => {
                    // All callable errors are ToolResolutionError (soft).
                    any_error = true;
                    if first_error_tool.is_empty() {
                        first_error_tool = tc.tool.clone();
                    }
                    let result_str = format!("[RESOLUTION_ERROR] {}", e);
                    results.push((tc.tool.clone(), call_id, result_str, false, true));
                }
            }
        }

        // Emit tool result messages.
        for (name, call_id, result_str, _success, _is_soft) in &results {
            let result_msg = Message::new(
                MessageRole::Tool,
                result_str.as_str(),
                MessageMeta::new(MessageType::ToolResult).with_step_index(iteration as i64),
            )
            .with_tool_name(name.as_str())
            .with_tool_call_id(call_id.as_str());
            self.fire_message(&result_msg);
            messages.push(result_msg);
        }

        // Check for terminal result.
        if let Some(val) = terminal_result {
            return Ok(Some(val));
        }

        // No terminal result. Signal continuation.
        // If there were errors, the tool result messages already feed back
        // to the model. The loop continues in the caller.
        // We return a "continue" signal via a non-fatal error variant.
        // The run() loop handles this by continuing iteration.
        if any_error {
            *last_raw_response = Some(first_error_tool.clone());
        }

        Ok(None)
    }

    fn fire_message(&self, msg: &Message) {
        if let Some(ref cb) = self.on_message {
            cb(msg);
        }
    }

    fn fatal_to_error(&self, reason: &str, workflow: &Workflow) -> ForgeError {
        if reason.contains("Too many bad responses") {
            ForgeError::ToolCall(ToolCallError::new(reason))
        } else if reason.contains("Too many skipped required steps") {
            let pending = workflow.required_steps.clone();
            let terminal_name = workflow
                .terminal_tools
                .iter()
                .next()
                .cloned()
                .unwrap_or_default();
            ForgeError::StepEnforcement(StepEnforcementError::new(terminal_name, 4, pending))
        } else {
            ForgeError::ToolCall(ToolCallError::new(reason))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn workflow_runner_type_exists() {
        // Type-level verification that WorkflowRunner<SomeClient> compiles.
    }
}
