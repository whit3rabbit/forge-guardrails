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
    ForgeError, MaxIterationsError, PrerequisiteError, StepEnforcementError, ToolCallError,
    ToolError, ToolExecutionError, WorkflowCancelledError,
};
use crate::guardrails::{
    recent_errors_from_messages, ClassifierAction, ErrorTracker, FinalResponseContext,
    FinalResponseScore, FinalResponseScorer, FinalResponseToolResult, ResponseValidator,
    RetryNudgeFn, ScoringContext, StepEnforcer, StepPrerequisite, ToolCallScore, ToolCallScorer,
    WorkflowStateForScoring,
};
use crate::prompts::nudges;
use indexmap::{IndexMap, IndexSet};
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
    scorer: Option<Arc<dyn ToolCallScorer>>,
    on_tool_call_score: Option<Arc<ToolCallScoreFn>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    on_final_response_score: Option<Arc<FinalResponseScoreFn>>,
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

    fn build_guardrails(&self, workflow: &Workflow) -> RunnerGuardrails {
        let tool_specs: Vec<ToolSpec> = workflow.tools.values().map(|d| d.spec.clone()).collect();
        let terminal_set: IndexSet<String> = workflow.terminal_tools.iter().cloned().collect();
        let retry_nudge_for_validator: Option<RetryNudgeFn> = self
            .retry_nudge_fn
            .clone()
            .map(|f| Box::new(move |raw: &str| f(raw)) as RetryNudgeFn);

        RunnerGuardrails {
            validator: ResponseValidator::from_tool_specs(
                tool_specs,
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

            if let Some(nudge_content) = self.score_tool_calls(
                user_message,
                &messages,
                &tool_calls,
                &guardrails.step_enforcer,
                &tool_specs,
            ) {
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

            if let Some(nudge_content) = self.score_candidate_final_responses(
                user_message,
                &messages,
                &tool_calls,
                &guardrails.step_enforcer,
            ) {
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

    fn score_tool_calls(
        &self,
        fallback_user_message: &str,
        messages: &[Message],
        tool_calls: &[ToolCall],
        step_enforcer: &StepEnforcer,
        tool_specs: &[ToolSpec],
    ) -> Option<String> {
        let scorer = self.scorer.as_ref()?;
        let user_request = latest_user_request(messages).unwrap_or(fallback_user_message);
        let ctx = ScoringContext::from_step_enforcer(
            user_request,
            step_enforcer,
            &step_enforcer.terminal_tools,
            recent_errors_from_messages(messages, 8),
            tool_specs,
        );
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
                    if let Some(callback) = &self.on_tool_call_score {
                        callback(call, &score);
                    }
                    if matches!(
                        score.action,
                        ClassifierAction::AdvisoryNudge | ClassifierAction::Block
                    ) {
                        let content = nudges::classifier_nudge(score.label.as_label().as_ref());
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

    fn score_candidate_final_responses(
        &self,
        fallback_user_message: &str,
        messages: &[Message],
        tool_calls: &[ToolCall],
        step_enforcer: &StepEnforcer,
    ) -> Option<String> {
        let scorer = self.final_response_scorer.as_ref()?;
        let terminal_calls = tool_calls
            .iter()
            .filter(|call| step_enforcer.terminal_tools.contains(&call.tool))
            .collect::<Vec<_>>();
        if terminal_calls.is_empty() {
            return None;
        }

        let user_request = latest_user_request(messages).unwrap_or(fallback_user_message);
        let base_trace = Self::tool_trace_from_messages(messages);
        let tool_results = Self::tool_results_from_messages(messages);
        let mut nudge: Option<String> = None;
        for call in terminal_calls {
            let mut tool_trace = base_trace.clone();
            tool_trace.push(call.tool.clone());
            let ctx = FinalResponseContext {
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
            };
            match scorer.score(&ctx) {
                Ok(score) => {
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
                        callback(&score);
                    }
                    if matches!(
                        score.action,
                        ClassifierAction::AdvisoryNudge | ClassifierAction::Block
                    ) {
                        let content =
                            crate::prompts::classifier_nudge(score.label.as_label().as_ref());
                        if score.action == ClassifierAction::Block || nudge.is_none() {
                            nudge = Some(content);
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        target: "forge.classifier",
                        error = %err,
                        terminal_tool = %call.tool,
                        "final-response classifier scoring failed; allowing deterministic path"
                    );
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

    fn is_mixed_terminal_batch(tool_calls: &[ToolCall], workflow: &Workflow) -> bool {
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

    fn mixed_terminal_batch_nudge(workflow: &Workflow, step_enforcer: &StepEnforcer) -> String {
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

        let mut infos = Vec::with_capacity(calls.len());
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

fn latest_user_request(messages: &[Message]) -> Option<&str> {
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
