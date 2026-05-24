//! Workflow runner: orchestrates the multi-turn agentic tool-calling loop.
//!
//! WorkflowRunner drives the iterative loop: inference, guardrails check,
//! tool execution, error tracking, and termination on terminal tool success.

use super::inference::{self, OnChunkFn};
use super::message::{Message, MessageMeta, MessageRole, MessageType};
use super::tool_spec::ToolSpec;
use super::workflow::Workflow;
use crate::clients::base::LLMClient;
use crate::clients::base::{LLMResponse, ToolCall};
use crate::context::manager::ContextManager;
use crate::error::{
    ForgeError, MaxIterationsError, PrerequisiteError, StepEnforcementError, WorkflowCancelledError,
};
use crate::guardrails::{RetryNudgeFn, StepEnforcer};
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::watch;

/// Callback type for message events during a run.
pub type OnMessageFn = Box<dyn Fn(&Message) + Send + Sync>;

/// Type alias for the runner-level dynamic nudge function.
pub type NudgeCallbackFn = dyn Fn(&str) -> String + Send + Sync;

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
    retry_nudge_fn: Option<Arc<NudgeCallbackFn>>,
}

impl<C: LLMClient> WorkflowRunner<C> {
    /// Creates a new `WorkflowRunner`.
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
        let retry_nudge_fn =
            retry_nudge.map(|s| Arc::new(move |_raw: &str| s.clone()) as Arc<NudgeCallbackFn>);
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
        retry_nudge_fn: Option<Arc<NudgeCallbackFn>>,
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

        let mut tool_prerequisites = indexmap::IndexMap::new();
        for (name, tool_def) in &workflow.tools {
            if !tool_def.prerequisites.is_empty() {
                let mapped: Vec<crate::guardrails::StepPrerequisite> = tool_def
                    .prerequisites
                    .iter()
                    .map(|p| match p {
                        super::workflow::PrerequisiteSpec::NameOnly(n) => {
                            crate::guardrails::StepPrerequisite::NameOnly(n.clone())
                        }
                        super::workflow::PrerequisiteSpec::ArgMatched { tool, match_arg } => {
                            crate::guardrails::StepPrerequisite::ArgMatched {
                                tool: tool.clone(),
                                match_arg: match_arg.clone(),
                            }
                        }
                    })
                    .collect();
                tool_prerequisites.insert(name.clone(), mapped);
            }
        }

        // Match Python: keep validator, step enforcer, and error tracker as
        // separate stateful components owned by the runner.
        let retry_nudge_for_validator: Option<RetryNudgeFn> = self
            .retry_nudge_fn
            .clone()
            .map(|f| Box::new(move |raw: &str| f(raw)) as RetryNudgeFn);
        let validator = crate::guardrails::ResponseValidator::new(
            tool_names.clone(),
            self.rescue_enabled,
            retry_nudge_for_validator,
        );
        let mut error_tracker =
            crate::guardrails::ErrorTracker::new(self.max_retries_per_step, self.max_tool_errors);
        let mut step_enforcer = StepEnforcer::new(
            workflow.required_steps.clone(),
            terminal_set,
            Some(tool_prerequisites),
            3, // max premature attempts
            2, // max prerequisite violations
        );

        let mut messages: Vec<Message> = Vec::new();
        let mut tool_call_counter: i64 = 0;
        // iteration tracks consumed budget; starts at 0, incremented by result.attempts.
        let mut iteration: i32 = 0;

        // Build initial messages or use provided seed.
        // Note: on_message is NOT fired for seed messages — only for new messages
        // produced during this run. This matches Python's `initial_messages` path.
        if let Some(seed) = initial_messages {
            for msg in seed {
                messages.push(msg);
            }
        } else {
            let system_content =
                workflow.build_system_prompt(prompt_vars.unwrap_or(&IndexMap::new()));
            let system_msg = Message::new(
                MessageRole::System,
                &system_content,
                MessageMeta::new(MessageType::SystemPrompt),
            );
            messages.push(system_msg);

            let user_msg = Message::new(
                MessageRole::User,
                user_message,
                MessageMeta::new(MessageType::UserInput),
            );
            messages.push(user_msg);
        }

        while iteration < self.max_iterations {
            // Check cancellation.
            if let Some(ref rx) = cancel {
                if *rx.borrow() {
                    let completed = step_enforcer.completed_steps();
                    let msgs: Vec<String> = messages.iter().map(|m| m.content.clone()).collect();
                    return Err(ForgeError::WorkflowCancelled(WorkflowCancelledError::new(
                        msgs,
                        completed,
                        iteration as i64,
                    )));
                }
            }

            // Remaining inference budget: how many LLM calls can still be made.
            // Python: max_attempts = self.max_iterations - iteration
            let remaining = self.max_iterations - iteration;
            let step_hint = step_enforcer.summary_hint();

            let mut ctx = self.context_manager.lock().await;

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
            .await?;

            drop(ctx);

            // None means max_attempts exhausted — break to raise MaxIterationsError.
            // Python: `if result is None: break`
            let result = match inference_result {
                Some(r) => r,
                None => break,
            };

            // Retries within inference consume iteration budget.
            // Python: `iteration += result.attempts`
            iteration += result.attempts;

            // Fire callbacks for new messages from inference.
            for msg in &result.new_messages {
                self.fire_message(msg);
            }
            tool_call_counter = result.tool_call_counter;

            if let LLMResponse::Text(ref text) = result.response {
                let text_msg = Message::new(
                    MessageRole::Assistant,
                    &text.content,
                    MessageMeta::new(MessageType::TextResponse).with_step_index(iteration as i64),
                );
                self.fire_message(&text_msg);
                messages.push(text_msg);
                continue;
            }

            let tool_calls = match result.response {
                LLMResponse::ToolCalls(calls) => calls,
                LLMResponse::Text(_) => unreachable!("text response handled above"),
            };

            let step_check = step_enforcer.check(&tool_calls);
            if step_check.needs_nudge {
                if step_enforcer.premature_exhausted() {
                    let attempted = tool_calls
                        .iter()
                        .find(|tc| workflow.terminal_tools.contains(&tc.tool))
                        .map(|tc| tc.tool.clone())
                        .unwrap_or_default();
                    return Err(ForgeError::StepEnforcement(StepEnforcementError::new(
                        attempted,
                        step_enforcer.premature_attempts() as i64,
                        step_enforcer.pending(),
                    )));
                }
                let nudge = step_check.nudge.expect("step nudge required");
                let calls = self.emit_assistant_tool_calls(
                    tool_calls,
                    &mut messages,
                    &mut tool_call_counter,
                    iteration as i64,
                );
                for tc in &calls {
                    let call_id = tc.id.clone().unwrap_or_default();
                    let error_content = format!("[StepEnforcementError] {}", nudge.content);
                    let result_msg = Message::new(
                        MessageRole::Tool,
                        &error_content,
                        MessageMeta::new(MessageType::StepNudge).with_step_index(iteration as i64),
                    )
                    .with_tool_name(&tc.tool)
                    .with_tool_call_id(&call_id);
                    self.fire_message(&result_msg);
                    messages.push(result_msg);
                }
                continue;
            }

            let prereq_check = step_enforcer.check_prerequisites(&tool_calls);
            if prereq_check.needs_nudge {
                if step_enforcer.prereq_exhausted() {
                    for tc in &tool_calls {
                        if let Some(prereqs) = step_enforcer.tool_prerequisites.get(&tc.tool) {
                            let rust_prereqs: Vec<super::steps::Prerequisite> = prereqs
                                .iter()
                                .map(|p| match p {
                                    crate::guardrails::StepPrerequisite::NameOnly(n) => {
                                        super::steps::Prerequisite::NameOnly(n.clone())
                                    }
                                    crate::guardrails::StepPrerequisite::ArgMatched {
                                        tool,
                                        match_arg,
                                    } => super::steps::Prerequisite::ArgMatched {
                                        tool: tool.clone(),
                                        match_arg: match_arg.clone(),
                                    },
                                })
                                .collect();
                            let check_res = step_enforcer.tracker.check_prerequisites(
                                &tc.tool,
                                &tc.args,
                                &rust_prereqs,
                            );
                            if !check_res.satisfied {
                                return Err(ForgeError::Prerequisite(PrerequisiteError::new(
                                    tc.tool.clone(),
                                    step_enforcer.prereq_violations() as i64,
                                    check_res.missing,
                                )));
                            }
                        }
                    }
                    return Err(ForgeError::Prerequisite(PrerequisiteError::new(
                        "",
                        step_enforcer.prereq_violations() as i64,
                        Vec::new(),
                    )));
                }
                let nudge = prereq_check.nudge.expect("prerequisite nudge required");
                let calls = self.emit_assistant_tool_calls(
                    tool_calls,
                    &mut messages,
                    &mut tool_call_counter,
                    iteration as i64,
                );
                for tc in &calls {
                    let call_id = tc.id.clone().unwrap_or_default();
                    let error_content = format!("[PrerequisiteError] {}", nudge.content);
                    let result_msg = Message::new(
                        MessageRole::Tool,
                        &error_content,
                        MessageMeta::new(MessageType::PrerequisiteNudge)
                            .with_step_index(iteration as i64),
                    )
                    .with_tool_name(&tc.tool)
                    .with_tool_call_id(&call_id);
                    self.fire_message(&result_msg);
                    messages.push(result_msg);
                }
                continue;
            }

            let calls = self.emit_assistant_tool_calls(
                tool_calls,
                &mut messages,
                &mut tool_call_counter,
                iteration as i64,
            );
            let result_val = self
                .execute_tool_batch(
                    &calls,
                    &mut messages,
                    workflow,
                    &mut step_enforcer,
                    &mut error_tracker,
                    &mut tool_call_counter,
                    iteration,
                )
                .await?;
            if let Some(val) = result_val {
                return Ok(val);
            }
        }

        // Step 4 — Max iterations exceeded (loop exited)
        let completed = step_enforcer.completed_steps();
        let pending = step_enforcer.pending();
        Err(ForgeError::MaxIterations(MaxIterationsError::new(
            self.max_iterations as i64,
            completed,
            pending,
        )))
    }

    /// Execute a batch of tool calls, returning the terminal tool result if found.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_batch(
        &self,
        calls: &[crate::clients::base::ToolCall],
        messages: &mut Vec<Message>,
        workflow: &Workflow,
        step_enforcer: &mut StepEnforcer,
        error_tracker: &mut crate::guardrails::ErrorTracker,
        tool_call_counter: &mut i64,
        iteration: i32,
    ) -> Result<Option<Value>, ForgeError> {
        let mut terminal_result: Option<Value> = None;
        let mut batch_had_error = false;
        let mut last_error: Option<(String, ForgeError)> = None;

        // Execute each tool sequentially.
        let mut results: Vec<(String, String, String, bool, bool)> = Vec::new();
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
                    let result_str = "[TOOL_ERROR] Tool not found".to_string();
                    results.push((tc.tool.clone(), call_id, result_str, false, false));
                    continue;
                }
            };

            match callable(tc.args.clone()).await {
                Ok(output) => {
                    step_enforcer.record(&tc.tool, Some(&tc.args));

                    let is_terminal = workflow.terminal_tools.contains(&tc.tool);
                    if is_terminal {
                        let val = if output.is_null() {
                            Value::Null
                        } else if output.is_string() {
                            let s = output.as_str().unwrap().to_string();
                            serde_json::from_str(&s).unwrap_or(Value::String(s))
                        } else {
                            output.clone()
                        };
                        terminal_result = Some(val);
                    }

                    let output_str = if output.is_string() {
                        output.as_str().unwrap().to_string()
                    } else {
                        output.to_string()
                    };

                    results.push((tc.tool.clone(), call_id, output_str, true, false));
                }
                Err(crate::error::ToolError::Resolution(e)) => {
                    // Soft error: feed back as tool result, don't count toward tool_errors.
                    error_tracker.record_result(false, true);
                    let result_str = format!("[ToolResolutionError] {}", e);
                    results.push((tc.tool.clone(), call_id, result_str, false, true));
                }
                Err(crate::error::ToolError::Execution(e)) => {
                    // Hard error: increment consecutive tool errors count.
                    batch_had_error = true;
                    error_tracker.record_result(false, false);
                    last_error = Some((
                        tc.tool.clone(),
                        ForgeError::ToolExecution(crate::error::ToolExecutionError::new(
                            tc.tool.clone(),
                            e.to_string(),
                        )),
                    ));
                    let result_str = format!("[ToolError] ToolExecutionError: {}", e);
                    results.push((tc.tool.clone(), call_id, result_str, false, false));
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

        // Post-batch bookkeeping — matches Python's error_tracker.record_result / reset_errors
        if batch_had_error {
            if error_tracker.tool_errors_exhausted() {
                if let Some((_, err)) = last_error {
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

    /// Record the assistant's successful tool-call turn after guardrail checks.
    fn emit_assistant_tool_calls(
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

        let mut infos = Vec::new();
        for tc in &mut calls {
            let call_id = inference::format_tool_call_id(*tool_call_counter);
            *tool_call_counter += 1;
            tc.id = Some(call_id.clone());
            infos.push(crate::core::message::ToolCallInfo::new(
                &tc.tool,
                Some(tc.args.clone()),
                call_id,
            ));
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

    fn fire_message(&self, msg: &Message) {
        if let Some(ref cb) = self.on_message {
            cb(msg);
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
