//! Workflow runner: orchestrates the multi-turn agentic tool-calling loop.
//!
//! WorkflowRunner drives the iterative loop: inference, guardrails check,
//! tool execution, error tracking, and termination on terminal tool success.

use super::inference::{self, OnChunkFn};
use super::message::{Message, MessageMeta, MessageRole, MessageType};
use super::tool_spec::ToolSpec;
use super::workflow::Workflow;
use crate::clients::base::LLMClient;
use crate::clients::base::LLMResponse;
use crate::context::manager::ContextManager;
use crate::error::{
    ForgeError, MaxIterationsError, StepEnforcementError, ToolCallError, WorkflowCancelledError,
};
use crate::guardrails::{GuardAction, Guardrails, RetryNudgeFn, TerminalTool};
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

        // Build guardrails once before the loop — matches Python which constructs
        // validator, step_enforcer, and error_tracker once outside the while loop.
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
        let mut guardrails = Guardrails::new(
            tool_names.clone(),
            terminal_tool,
            Some(workflow.required_steps.clone()),
            Some(tool_prerequisites),
            self.max_retries_per_step,
            self.max_tool_errors,
            self.rescue_enabled,
            3, // max premature attempts
            retry_nudge_for_guardrails,
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
                    let completed = guardrails.completed_steps();
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
                            &mut error_tracker,
                            &mut tool_call_counter,
                            iteration,
                        )
                        .await?;
                    if let Some(val) = result_val {
                        return Ok(val);
                    }
                    // No terminal result yet; continue loop.
                }
                GuardAction::Retry => {
                    continue;
                }
                GuardAction::StepBlocked => {
                    let nudge = check.nudge.expect("step_blocked requires nudge");

                    if let LLMResponse::ToolCalls(ref calls) = result.response {
                        for tc in calls {
                            let call_id = tc.id.clone().unwrap_or_default();
                            let prefix = if nudge.kind == "prerequisite" {
                                "[PrerequisiteError]"
                            } else {
                                "[StepEnforcementError]"
                            };
                            let msg_type = if nudge.kind == "prerequisite" {
                                MessageType::PrerequisiteNudge
                            } else {
                                MessageType::StepNudge
                            };
                            let error_content = format!("{} {}", prefix, nudge.content);
                            let result_msg = Message::new(
                                MessageRole::Tool,
                                &error_content,
                                MessageMeta::new(MessageType::ToolResult)
                                    .with_original_type(msg_type)
                                    .with_step_index(iteration as i64),
                            )
                            .with_tool_name(&tc.tool)
                            .with_tool_call_id(&call_id);
                            self.fire_message(&result_msg);
                            messages.push(result_msg);
                        }
                    }

                    continue;
                }
                GuardAction::Fatal => {
                    let reason = check.reason.unwrap_or_default();
                    return Err(self.fatal_to_error_with_guardrails(
                        &reason,
                        workflow,
                        &guardrails,
                    ));
                }
            }
        }

        // Step 4 — Max iterations exceeded (loop exited)
        let completed = guardrails.completed_steps();
        let pending = workflow.required_steps.clone();
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
        guardrails: &mut Guardrails,
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

            match callable(tc.args.clone()).await {
                Ok(output) => {
                    guardrails.step_enforcer.record(&tc.tool, Some(&tc.args));

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
            guardrails.step_enforcer.reset_premature();
            guardrails.step_enforcer.reset_prereq_violations();
        }

        // Check for terminal result.
        if let Some(val) = terminal_result {
            return Ok(Some(val));
        }

        Ok(None)
    }

    fn fire_message(&self, msg: &Message) {
        if let Some(ref cb) = self.on_message {
            cb(msg);
        }
    }

    fn fatal_to_error_with_guardrails(
        &self,
        reason: &str,
        workflow: &Workflow,
        guardrails: &Guardrails,
    ) -> ForgeError {
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
        } else if reason.contains("Too many prerequisite violations") {
            let mut tool_name = String::new();
            let mut missing = Vec::new();
            for (tname, def) in &workflow.tools {
                if !def.prerequisites.is_empty() {
                    let sps: Vec<super::steps::Prerequisite> = def
                        .prerequisites
                        .iter()
                        .map(|p| match p {
                            super::workflow::PrerequisiteSpec::NameOnly(n) => {
                                super::steps::Prerequisite::NameOnly(n.clone())
                            }
                            super::workflow::PrerequisiteSpec::ArgMatched { tool, match_arg } => {
                                super::steps::Prerequisite::ArgMatched {
                                    tool: tool.clone(),
                                    match_arg: match_arg.clone(),
                                }
                            }
                        })
                        .collect();
                    let check_res = guardrails.step_enforcer.tracker.check_prerequisites(
                        tname,
                        &IndexMap::new(),
                        &sps,
                    );
                    if !check_res.satisfied {
                        tool_name = tname.clone();
                        missing = check_res.missing;
                        break;
                    }
                }
            }
            ForgeError::Prerequisite(crate::error::PrerequisiteError::new(
                tool_name,
                guardrails.step_enforcer.prereq_violations() as i64,
                missing,
            ))
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
