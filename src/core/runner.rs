//! Workflow runner: orchestrates the multi-turn agentic tool-calling loop.
//!
//! WorkflowRunner drives the iterative loop: inference, guardrails check,
//! tool execution, error tracking, and termination on terminal tool success.

pub(crate) mod execution;
pub(crate) mod guardrails;
pub(crate) mod scoring;

use super::inference::{self, OnChunkFn};
use super::message::{Message, MessageMeta, MessageRole, MessageType};
use super::steps::Prerequisite;
use super::tool_spec::ToolSpec;
use super::workflow::Workflow;
use crate::clients::base::LLMClient;
use crate::clients::base::{LLMResponse, ToolCall};
use crate::context::manager::ContextManager;
use crate::error::{
    ForgeError, MaxIterationsError, PrerequisiteError, StepEnforcementError, ToolCallError,
    WorkflowCancelledError,
};
use crate::guardrails::{
    ErrorTracker, FinalResponseScore, FinalResponseScorer, ResponseValidator, RetryNudgeFn,
    StepEnforcer, ToolCallScore, ToolCallScorer,
};
use indexmap::IndexMap;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::watch;

/// Callback type for message events during a run.
pub type OnMessageFn = Box<dyn Fn(&Message) + Send + Sync>;

/// Type alias for the runner-level dynamic nudge function.
pub type NudgeCallbackFn = dyn Fn(&str) -> String + Send + Sync;

/// Callback type for classifier score events during a run.
pub type ToolCallScoreFn = dyn Fn(&ToolCall, &ToolCallScore) + Send + Sync;
/// Callback type for final-response classifier score events during a run.
pub type FinalResponseScoreFn = dyn Fn(&FinalResponseScore) + Send + Sync;

/// Workflow runner orchestrating multi-turn LLM tool-calling loops.
///
/// Generic over the LLM client type because the LLMClient trait uses
/// async methods and is not dyn-compatible.
pub struct WorkflowRunner<C: LLMClient> {
    pub(super) client: Arc<C>,
    pub(super) context_manager: Arc<tokio::sync::Mutex<ContextManager>>,
    pub(super) max_iterations: i32,
    pub(super) max_retries_per_step: i32,
    pub(super) max_tool_errors: i32,
    pub(super) stream: bool,
    pub(super) on_chunk: Option<Arc<OnChunkFn>>,
    pub(super) on_message: Option<Arc<OnMessageFn>>,
    pub(super) rescue_enabled: bool,
    pub(super) retry_nudge_fn: Option<Arc<NudgeCallbackFn>>,
    pub(super) scorer: Option<Arc<dyn ToolCallScorer>>,
    pub(super) on_tool_call_score: Option<Arc<ToolCallScoreFn>>,
    pub(super) final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    pub(super) on_final_response_score: Option<Arc<FinalResponseScoreFn>>,
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
            scorer: None,
            on_tool_call_score: None,
            final_response_scorer: None,
            on_final_response_score: None,
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
            scorer: None,
            on_tool_call_score: None,
            final_response_scorer: None,
            on_final_response_score: None,
        }
    }

    /// Attach a tool-call scorer that runs after deterministic checks pass.
    pub fn with_tool_call_scorer(
        mut self,
        scorer: Arc<dyn ToolCallScorer>,
        on_tool_call_score: Option<Arc<ToolCallScoreFn>>,
    ) -> Self {
        self.scorer = Some(scorer);
        self.on_tool_call_score = on_tool_call_score;
        self
    }

    /// Attach a final-response scorer that runs before terminal answers are accepted.
    pub fn with_final_response_scorer(
        mut self,
        scorer: Arc<dyn FinalResponseScorer>,
        on_final_response_score: Option<Arc<FinalResponseScoreFn>>,
    ) -> Self {
        self.final_response_scorer = Some(scorer);
        self.on_final_response_score = on_final_response_score;
        self
    }

    fn build_guardrails(&self, workflow: &Workflow) -> guardrails::RunnerGuardrails {
        let tool_specs: Vec<ToolSpec> = workflow.tools.values().map(|d| d.spec.clone()).collect();
        let terminal_set: indexmap::IndexSet<String> =
            workflow.terminal_tools.iter().cloned().collect();
        let retry_nudge_for_validator: Option<RetryNudgeFn> = self
            .retry_nudge_fn
            .clone()
            .map(|f| Box::new(move |raw: &str| f(raw)) as RetryNudgeFn);

        guardrails::RunnerGuardrails {
            validator: ResponseValidator::from_tool_specs(
                tool_specs,
                self.rescue_enabled,
                retry_nudge_for_validator,
            ),
            error_tracker: ErrorTracker::new(self.max_retries_per_step, self.max_tool_errors),
            step_enforcer: StepEnforcer::new(
                workflow.required_steps.clone(),
                terminal_set,
                Some(guardrails::map_tool_prerequisites(workflow)),
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
                let rust_prereqs: Vec<Prerequisite> = prereqs
                    .iter()
                    .map(guardrails::map_step_prerequisite)
                    .collect();
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

            let inference_result = inference::run_inference_shared_context(
                &mut messages,
                self.client.as_ref(),
                self.context_manager.as_ref(),
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

            if Self::is_mixed_terminal_batch(&tool_calls, workflow) {
                guardrails.error_tracker.record_retry();
                if guardrails.error_tracker.retries_exhausted() {
                    let raw =
                        inference::response_to_raw_string(&LLMResponse::ToolCalls(tool_calls))
                            .unwrap_or_default();
                    return Err(ForgeError::ToolCall(
                        ToolCallError::new(format!(
                            "Retries exhausted after {} consecutive failed attempts",
                            guardrails.error_tracker.max_retries()
                        ))
                        .with_raw_response(raw),
                    ));
                }

                let nudge_content =
                    Self::mixed_terminal_batch_nudge(workflow, &guardrails.step_enforcer);
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
                    MessageType::RetryNudge,
                    "[MixedTerminalBatch]",
                    &nudge_content,
                );
                continue;
            }

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

            if let Some(nudge_content) = self
                .score_tool_calls(
                    user_message,
                    &messages,
                    &tool_calls,
                    &guardrails.step_enforcer,
                    &tool_specs,
                )
                .await
            {
                guardrails.error_tracker.record_retry();
                if guardrails.error_tracker.retries_exhausted() {
                    let raw =
                        inference::response_to_raw_string(&LLMResponse::ToolCalls(tool_calls))
                            .unwrap_or_default();
                    return Err(ForgeError::ToolCall(
                        ToolCallError::new(format!(
                            "Retries exhausted after {} consecutive classifier objections",
                            guardrails.error_tracker.max_retries()
                        ))
                        .with_raw_response(raw),
                    ));
                }
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
                    MessageType::RetryNudge,
                    "[ClassifierNudge]",
                    &nudge_content,
                );
                continue;
            }

            if let Some(nudge_content) = self
                .score_candidate_final_responses(
                    user_message,
                    &messages,
                    &tool_calls,
                    &guardrails.step_enforcer,
                )
                .await
            {
                guardrails.error_tracker.record_retry();
                if guardrails.error_tracker.retries_exhausted() {
                    let raw =
                        inference::response_to_raw_string(&LLMResponse::ToolCalls(tool_calls))
                            .unwrap_or_default();
                    return Err(ForgeError::ToolCall(
                        ToolCallError::new(format!(
                            "Retries exhausted after {} consecutive final-response objections",
                            guardrails.error_tracker.max_retries()
                        ))
                        .with_raw_response(raw),
                    ));
                }
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
                    MessageType::RetryNudge,
                    "[FinalResponseNudge]",
                    &nudge_content,
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

    pub(super) fn fire_message(&self, msg: &Message) {
        if let Some(ref cb) = self.on_message {
            cb(msg);
        }
    }
}

pub(super) fn latest_user_request(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::User)
        .map(|message| message.content.as_str())
}

#[cfg(test)]
mod tests {
    #[test]
    fn workflow_runner_type_exists() {
        // Type-level verification that WorkflowRunner<SomeClient> compiles.
    }
}
