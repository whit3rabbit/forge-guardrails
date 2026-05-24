//! Workflow runner: orchestrates the multi-turn agentic tool-calling loop.
//!
//! WorkflowRunner drives the iterative loop: inference, guardrails check,
//! tool execution, error tracking, and termination on terminal tool success.

use super::inference::{self, OnChunkFn};
use super::message::{Message, MessageMeta, MessageRole, MessageType};
use super::steps::Prerequisite;
use super::tool_spec::ToolSpec;
use super::workflow::{PrerequisiteSpec, Workflow};
use crate::clients::base::LLMClient;
use crate::clients::base::{LLMResponse, ToolCall};
use crate::context::manager::ContextManager;
use crate::error::{
    ForgeError, MaxIterationsError, PrerequisiteError, StepEnforcementError, ToolError,
    ToolExecutionError, WorkflowCancelledError,
};
use crate::guardrails::{
    ErrorTracker, ResponseValidator, RetryNudgeFn, StepEnforcer, StepPrerequisite,
};
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

struct RunnerGuardrails {
    validator: ResponseValidator,
    error_tracker: ErrorTracker,
    step_enforcer: StepEnforcer,
}

struct ToolResultRecord {
    tool_name: String,
    call_id: String,
    content: String,
}

fn map_tool_prerequisites(workflow: &Workflow) -> IndexMap<String, Vec<StepPrerequisite>> {
    let mut tool_prerequisites = IndexMap::new();
    for (name, tool_def) in &workflow.tools {
        if !tool_def.prerequisites.is_empty() {
            let mapped = tool_def
                .prerequisites
                .iter()
                .map(map_prerequisite_spec)
                .collect();
            tool_prerequisites.insert(name.clone(), mapped);
        }
    }
    tool_prerequisites
}

fn map_prerequisite_spec(prereq: &PrerequisiteSpec) -> StepPrerequisite {
    match prereq {
        PrerequisiteSpec::NameOnly(name) => StepPrerequisite::NameOnly(name.clone()),
        PrerequisiteSpec::ArgMatched { tool, match_arg } => StepPrerequisite::ArgMatched {
            tool: tool.clone(),
            match_arg: match_arg.clone(),
        },
    }
}

fn map_step_prerequisite(prereq: &StepPrerequisite) -> Prerequisite {
    match prereq {
        StepPrerequisite::NameOnly(name) => Prerequisite::NameOnly(name.clone()),
        StepPrerequisite::ArgMatched { tool, match_arg } => Prerequisite::ArgMatched {
            tool: tool.clone(),
            match_arg: match_arg.clone(),
        },
    }
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

    fn build_guardrails(&self, workflow: &Workflow) -> RunnerGuardrails {
        let tool_names: Vec<String> = workflow.tools.keys().cloned().collect();
        let terminal_set: IndexSet<String> = workflow.terminal_tools.iter().cloned().collect();
        let retry_nudge_for_validator: Option<RetryNudgeFn> = self
            .retry_nudge_fn
            .clone()
            .map(|f| Box::new(move |raw: &str| f(raw)) as RetryNudgeFn);

        RunnerGuardrails {
            validator: ResponseValidator::new(
                tool_names,
                self.rescue_enabled,
                retry_nudge_for_validator,
            ),
            error_tracker: ErrorTracker::new(self.max_retries_per_step, self.max_tool_errors),
            step_enforcer: StepEnforcer::new(
                workflow.required_steps.clone(),
                terminal_set,
                Some(map_tool_prerequisites(workflow)),
                3, // max premature attempts
                2, // max prerequisite violations
            ),
        }
    }

    fn build_initial_messages(
        workflow: &Workflow,
        user_message: &str,
        prompt_vars: Option<&IndexMap<String, String>>,
        initial_messages: Option<Vec<Message>>,
    ) -> Vec<Message> {
        // Note: on_message is NOT fired for seed messages — only for new messages
        // produced during this run. This matches the current Rust behavior.
        if let Some(seed) = initial_messages {
            return seed;
        }

        let system_content = workflow.build_system_prompt(prompt_vars.unwrap_or(&IndexMap::new()));
        vec![
            Message::new(
                MessageRole::System,
                &system_content,
                MessageMeta::new(MessageType::SystemPrompt),
            ),
            Message::new(
                MessageRole::User,
                user_message,
                MessageMeta::new(MessageType::UserInput),
            ),
        ]
    }

    fn prerequisite_error(step_enforcer: &StepEnforcer, tool_calls: &[ToolCall]) -> ForgeError {
        for tc in tool_calls {
            if let Some(prereqs) = step_enforcer.tool_prerequisites.get(&tc.tool) {
                let rust_prereqs: Vec<Prerequisite> =
                    prereqs.iter().map(map_step_prerequisite).collect();
                let check_res =
                    step_enforcer
                        .tracker
                        .check_prerequisites(&tc.tool, &tc.args, &rust_prereqs);
                if !check_res.satisfied {
                    return ForgeError::Prerequisite(PrerequisiteError::new(
                        tc.tool.clone(),
                        step_enforcer.prereq_violations() as i64,
                        check_res.missing,
                    ));
                }
            }
        }

        ForgeError::Prerequisite(PrerequisiteError::new(
            "",
            step_enforcer.prereq_violations() as i64,
            Vec::new(),
        ))
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
        let tool_specs: Vec<ToolSpec> = workflow.tools.values().map(|d| d.spec.clone()).collect();

        // Match Python: keep validator, step enforcer, and error tracker as
        // separate stateful components owned by the runner.
        let mut guardrails = self.build_guardrails(workflow);

        let mut messages =
            Self::build_initial_messages(workflow, user_message, prompt_vars, initial_messages);
        let mut tool_call_counter: i64 = 0;
        // iteration tracks consumed budget; starts at 0, incremented by result.attempts.
        let mut iteration: i32 = 0;

        while iteration < self.max_iterations {
            // Check cancellation.
            if let Some(ref rx) = cancel {
                if *rx.borrow() {
                    let completed = guardrails.step_enforcer.completed_steps();
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
            let step_hint = guardrails.step_enforcer.summary_hint();

            let mut ctx = self.context_manager.lock().await;

            let inference_result = inference::run_inference(
                &mut messages,
                self.client.as_ref(),
                &mut ctx,
                &guardrails.validator,
                &mut guardrails.error_tracker,
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

            let step_check = guardrails.step_enforcer.check(&tool_calls);
            if step_check.needs_nudge {
                if guardrails.step_enforcer.premature_exhausted() {
                    let attempted = tool_calls
                        .iter()
                        .find(|tc| workflow.terminal_tools.contains(&tc.tool))
                        .map(|tc| tc.tool.clone())
                        .unwrap_or_default();
                    return Err(ForgeError::StepEnforcement(StepEnforcementError::new(
                        attempted,
                        guardrails.step_enforcer.premature_attempts() as i64,
                        guardrails.step_enforcer.pending(),
                    )));
                }
                let nudge = step_check.nudge.expect("step nudge required");
                let calls = self.emit_assistant_tool_calls(
                    tool_calls,
                    &mut messages,
                    &mut tool_call_counter,
                    iteration as i64,
                );
                self.emit_guardrail_nudge_results(
                    &calls,
                    &mut messages,
                    iteration as i64,
                    MessageType::StepNudge,
                    "[StepEnforcementError]",
                    &nudge.content,
                );
                continue;
            }

            let prereq_check = guardrails.step_enforcer.check_prerequisites(&tool_calls);
            if prereq_check.needs_nudge {
                if guardrails.step_enforcer.prereq_exhausted() {
                    return Err(Self::prerequisite_error(
                        &guardrails.step_enforcer,
                        &tool_calls,
                    ));
                }
                let nudge = prereq_check.nudge.expect("prerequisite nudge required");
                let calls = self.emit_assistant_tool_calls(
                    tool_calls,
                    &mut messages,
                    &mut tool_call_counter,
                    iteration as i64,
                );
                self.emit_guardrail_nudge_results(
                    &calls,
                    &mut messages,
                    iteration as i64,
                    MessageType::PrerequisiteNudge,
                    "[PrerequisiteError]",
                    &nudge.content,
                );
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
                    &mut guardrails.step_enforcer,
                    &mut guardrails.error_tracker,
                    &mut tool_call_counter,
                    iteration,
                )
                .await?;
            if let Some(val) = result_val {
                return Ok(val);
            }
        }

        // Step 4 — Max iterations exceeded (loop exited)
        let completed = guardrails.step_enforcer.completed_steps();
        let pending = guardrails.step_enforcer.pending();
        Err(ForgeError::MaxIterations(MaxIterationsError::new(
            self.max_iterations as i64,
            completed,
            pending,
        )))
    }

    fn emit_guardrail_nudge_results(
        &self,
        calls: &[ToolCall],
        messages: &mut Vec<Message>,
        step_index: i64,
        msg_type: MessageType,
        prefix: &str,
        nudge_content: &str,
    ) {
        for tc in calls {
            let call_id = tc.id.clone().unwrap_or_default();
            let error_content = format!("{} {}", prefix, nudge_content);
            let result_msg = Message::new(
                MessageRole::Tool,
                &error_content,
                MessageMeta::new(msg_type).with_step_index(step_index),
            )
            .with_tool_name(&tc.tool)
            .with_tool_call_id(&call_id);
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

    fn emit_tool_result_records(
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

    /// Execute a batch of tool calls, returning the terminal tool result if found.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_batch(
        &self,
        calls: &[crate::clients::base::ToolCall],
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
