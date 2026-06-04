use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use futures_util::future::join_all;
use serde_json::{json, Value};

use crate::cli::{default_minimax_model, default_openrouter_model, ReviewCli};
use crate::schema::{
    is_allowed_label, mutated_arguments_for_tool, parse_json_object_from_text, tool_by_name,
    validate_candidate_call, TRAINING_INPUT_SCHEMA_VERSION, TRAINING_SCHEMA_VERSION,
};

pub(crate) async fn run(cli: ReviewCli) -> Result<(), String> {
    ensure_parent_dir(&cli.output)?;
    let reject_path = rejects_path(&cli.output);
    ensure_parent_dir_path(&reject_path)?;
    let env_file = EnvFile::load(&cli.env_file);
    let reviewer_config = resolve_provider_config(&cli, &env_file, ReviewRole::Reviewer)?;
    let verifier_config = if cli.verifier_provider == "same"
        && cli.verifier_base_url.trim().is_empty()
        && cli.verifier_model.trim().is_empty()
        && cli.verifier_api_key.is_none()
    {
        reviewer_config.clone()
    } else {
        let verifier_provider = if cli.verifier_provider == "same" {
            reviewer_config.provider.clone()
        } else {
            cli.verifier_provider.clone()
        };
        resolve_provider_config_for(&cli, &env_file, ReviewRole::Verifier, &verifier_provider)?
    };
    let reviewer_label = format!("{}:{}", reviewer_config.provider, reviewer_config.model);
    let verifier_label = format!("{}:{}", verifier_config.provider, verifier_config.model);
    let reviewer = JsonLlmClient::new(reviewer_config);
    let verifier = JsonLlmClient::new(verifier_config);

    let mut resume_state = if cli.resume {
        ResumeState::load(&cli.output, &reject_path)?
    } else {
        ResumeState::default()
    };
    let mut alternative_proposals = Vec::new();
    let total_captures = count_jsonl_rows(&cli.input)?;
    let mut progress = ReviewProgress::new(total_captures);
    eprintln!(
        "review start input={} output={} rejects={} reviewer={} verifier={} resume={} concurrency={} captures={}",
        cli.input,
        cli.output,
        reject_path.display(),
        reviewer_label,
        verifier_label,
        cli.resume,
        cli.concurrency,
        total_captures
    );
    if cli.resume {
        eprintln!(
            "review resume state processed_captures={} accepted_real={} accepted_alternatives={}",
            resume_state.processed_capture_keys.len(),
            resume_state.accepted_real_count,
            resume_state.accepted_alternative_count
        );
    }

    let file =
        File::open(&cli.input).map_err(|err| format!("failed to read {}: {err}", cli.input))?;
    let mut batch = Vec::with_capacity(cli.concurrency);
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        progress.seen += 1;
        let line =
            line.map_err(|err| format!("{}:{} read error: {err}", cli.input, line_index + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let capture = serde_json::from_str::<Value>(trimmed)
            .map_err(|err| format!("{}:{} invalid JSONL row: {err}", cli.input, line_index + 1))?;
        let capture_key = capture_key(&capture);
        if resume_state.processed_capture_keys.contains(&capture_key) {
            progress.skipped += 1;
            progress.log_skip_if_needed();
            if resume_state.valid_real_capture_keys.contains(&capture_key) {
                alternative_proposals.extend(propose_targeted_alternatives(&capture));
            }
            continue;
        }

        progress.log_reviewing(&capture);
        batch.push(CaptureJob {
            row_index: progress.seen,
            capture,
        });
        if batch.len() >= cli.concurrency {
            flush_capture_batch(CaptureBatchRun {
                batch: &mut batch,
                reviewer: &reviewer,
                verifier: &verifier,
                reject_path: &reject_path,
                output_path: &cli.output,
                resume_state: &mut resume_state,
                progress: &mut progress,
                alternative_proposals: &mut alternative_proposals,
            })
            .await?;
        }
    }
    flush_capture_batch(CaptureBatchRun {
        batch: &mut batch,
        reviewer: &reviewer,
        verifier: &verifier,
        reject_path: &reject_path,
        output_path: &cli.output,
        resume_state: &mut resume_state,
        progress: &mut progress,
        alternative_proposals: &mut alternative_proposals,
    })
    .await?;

    eprintln!(
        "review capture pass complete seen={} skipped={} accepted_real={} accepted_corrected={} rejected={} proposed_alternatives={}",
        progress.seen,
        progress.skipped,
        progress.accepted_real,
        progress.accepted_corrected,
        progress.rejected,
        alternative_proposals.len()
    );
    let alternative_cap =
        alternative_cap(resume_state.accepted_real_count, cli.max_alternative_ratio);
    let remaining_alternative_cap =
        alternative_cap.saturating_sub(resume_state.accepted_alternative_count);
    verify_alternatives(AlternativeReviewRun {
        reviewer: &reviewer,
        verifier: &verifier,
        proposals: alternative_proposals,
        alternative_cap: remaining_alternative_cap,
        max_per_group: cli.max_alternatives_per_group,
        reject_path: &reject_path,
        output_path: &cli.output,
        resume_state: &mut resume_state,
        progress: &mut progress,
    })
    .await?;
    eprintln!(
        "review complete seen={} skipped={} accepted_real={} accepted_corrected={} accepted_alternatives={} rejected={} output={}",
        progress.seen,
        progress.skipped,
        progress.accepted_real,
        progress.accepted_corrected,
        progress.accepted_alternatives,
        progress.rejected,
        cli.output
    );

    Ok(())
}

#[derive(Debug)]
struct CaptureJob {
    row_index: usize,
    capture: Value,
}

struct CaptureBatchRun<'a> {
    batch: &'a mut Vec<CaptureJob>,
    reviewer: &'a JsonLlmClient,
    verifier: &'a JsonLlmClient,
    reject_path: &'a Path,
    output_path: &'a str,
    resume_state: &'a mut ResumeState,
    progress: &'a mut ReviewProgress,
    alternative_proposals: &'a mut Vec<AlternativeProposal>,
}

#[derive(Debug, Default)]
struct CaptureReviewOutcome {
    writes: Vec<CaptureWrite>,
    alternative_proposals: Vec<AlternativeProposal>,
}

impl CaptureReviewOutcome {
    fn training(&mut self, row: Value) {
        self.writes.push(CaptureWrite::Training(row));
    }

    fn reject(
        &mut self,
        reason: &str,
        detail: impl Into<String>,
        capture: &Value,
        raw_response: Option<Value>,
    ) {
        self.writes.push(CaptureWrite::Reject(RejectRecord {
            reason: reason.to_string(),
            detail: detail.into(),
            capture: capture.clone(),
            raw_response,
        }));
    }
}

#[derive(Debug)]
enum CaptureWrite {
    Training(Value),
    Reject(RejectRecord),
}

#[derive(Debug)]
struct RejectRecord {
    reason: String,
    detail: String,
    capture: Value,
    raw_response: Option<Value>,
}

async fn flush_capture_batch(run: CaptureBatchRun<'_>) -> Result<(), String> {
    let CaptureBatchRun {
        batch,
        reviewer,
        verifier,
        reject_path,
        output_path,
        resume_state,
        progress,
        alternative_proposals,
    } = run;
    if batch.is_empty() {
        return Ok(());
    }

    let jobs = std::mem::take(batch);
    let outcomes = join_all(jobs.into_iter().map(|job| async move {
        let row_index = job.row_index;
        let outcome = review_capture_job(reviewer, verifier, job.capture).await;
        (row_index, outcome)
    }))
    .await;

    for (row_index, outcome) in outcomes {
        apply_capture_outcome(
            row_index,
            outcome,
            reject_path,
            output_path,
            resume_state,
            progress,
            alternative_proposals,
        )?;
    }
    Ok(())
}

async fn review_capture_job(
    reviewer: &JsonLlmClient,
    verifier: &JsonLlmClient,
    capture: Value,
) -> CaptureReviewOutcome {
    let mut outcome = CaptureReviewOutcome::default();
    let decision = match review_capture(reviewer, &capture).await {
        Ok(decision) => decision,
        Err(err) => {
            outcome.reject("reviewer_rejected", err, &capture, None);
            return outcome;
        }
    };

    if let Some(corrected) = decision.corrected_candidate_call.as_ref() {
        if let Err(err) = validate_candidate_call(&capture["available_tools"], corrected) {
            outcome.reject(
                "invalid_corrected_candidate_call",
                err,
                &capture,
                Some(decision.raw.clone()),
            );
            return outcome;
        }
    }

    let candidate_call = capture["candidate_call"].clone();
    let verifier_decision = match verify_label(
        verifier,
        &capture,
        &candidate_call,
        &decision.label,
        &decision.rationale,
    )
    .await
    {
        Ok(decision) => decision,
        Err(err) => {
            outcome.reject("verifier_invalid_response", err, &capture, None);
            return outcome;
        }
    };
    if !verifier_decision.accepted {
        outcome.reject(
            "verifier_rejected",
            verifier_decision.rationale,
            &capture,
            Some(verifier_decision.raw),
        );
        return outcome;
    }
    if let Some(reason) = post_review_quality_reject(
        &capture,
        &candidate_call,
        &decision.label,
        "real_model_call",
    ) {
        outcome.reject(
            "post_review_quality_rejected",
            reason,
            &capture,
            Some(decision.raw.clone()),
        );
        return outcome;
    }

    let corrected_positive = decision
        .corrected_candidate_call
        .as_ref()
        .filter(|_| decision.label != "valid")
        .map(|call| json!({"candidate_call": call}));
    let real_row = training_row(
        &capture,
        candidate_call,
        &decision.label,
        "real_model_call",
        &decision,
        &verifier_decision,
        corrected_positive,
    );
    outcome.training(real_row);
    if decision.label == "valid" {
        outcome
            .alternative_proposals
            .extend(propose_targeted_alternatives(&capture));
    }

    if decision.label != "valid" {
        if let Some(corrected) = decision.corrected_candidate_call.as_ref() {
            if let Some(reason) = post_review_quality_reject(
                &capture,
                corrected,
                "valid",
                "reviewer_corrected_positive",
            ) {
                outcome.reject(
                    "post_review_corrected_positive_quality_rejected",
                    reason,
                    &capture,
                    Some(decision.raw.clone()),
                );
                return outcome;
            }
            let corrected_verifier = match verify_label(
                verifier,
                &capture,
                corrected,
                "valid",
                "Reviewer supplied this corrected candidate call as the positive target.",
            )
            .await
            {
                Ok(decision) => decision,
                Err(err) => {
                    outcome.reject(
                        "corrected_positive_verifier_invalid_response",
                        err,
                        &capture,
                        None,
                    );
                    return outcome;
                }
            };
            if corrected_verifier.accepted {
                let corrected_decision = ReviewDecision {
                    label: "valid".to_string(),
                    confidence: decision.confidence,
                    rationale: "Reviewer-corrected positive.".to_string(),
                    corrected_candidate_call: None,
                    raw: decision.raw.clone(),
                };
                let corrected_row = training_row(
                    &capture,
                    corrected.clone(),
                    "valid",
                    "reviewer_corrected_positive",
                    &corrected_decision,
                    &corrected_verifier,
                    None,
                );
                outcome.training(corrected_row);
            } else {
                outcome.reject(
                    "corrected_positive_verifier_rejected",
                    corrected_verifier.rationale,
                    &capture,
                    Some(corrected_verifier.raw),
                );
            }
        }
    }

    outcome
}

fn apply_capture_outcome(
    row_index: usize,
    outcome: CaptureReviewOutcome,
    reject_path: &Path,
    output_path: &str,
    resume_state: &mut ResumeState,
    progress: &mut ReviewProgress,
    alternative_proposals: &mut Vec<AlternativeProposal>,
) -> Result<(), String> {
    for write in outcome.writes {
        match write {
            CaptureWrite::Training(row) => {
                let label = row
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let source_bucket = row
                    .get("review")
                    .and_then(|review| review.get("source_bucket"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                append_jsonl(output_path, &row)?;
                resume_state.record_training_row(&row);
                match source_bucket.as_str() {
                    "real_model_call" => progress.accepted_real += 1,
                    "reviewer_corrected_positive" => progress.accepted_corrected += 1,
                    _ => {}
                }
                progress.log_accepted(row_index, &label, &source_bucket);
            }
            CaptureWrite::Reject(reject) => {
                append_reject_record(reject_path, &reject)?;
                progress.rejected += 1;
                progress.log_rejected(row_index, &reject.reason);
            }
        }
    }
    alternative_proposals.extend(outcome.alternative_proposals);
    Ok(())
}

async fn review_capture(client: &JsonLlmClient, capture: &Value) -> Result<ReviewDecision, String> {
    let system = concat!(
        "You label Forge tool-call verifier examples. Return only JSON. ",
        "Allowed labels are valid, wrong_tool_semantic, wrong_arguments_semantic, ",
        "tool_not_needed, needs_clarification. Do not use classifier telemetry as truth. ",
        "Judge the candidate call in the current workflow state, not from a generic ideal plan. ",
        "A successful tool result is evidence that the call is plausible. Do not mark a tool wrong ",
        "just because another read/inspect tool could also have been used. Terminal tools indicate ",
        "how to answer after tool work is done; they do not forbid a still-useful non-terminal tool. ",
        "If the same non-terminal tool has already completed and there are no pending steps, another ",
        "call to that tool is usually tool_not_needed, not valid. For shopping compare workflows, ",
        "search_products followed by compare_products with product IDs from the search result is valid; ",
        "do not invent a required get_product step unless the user specifically asked for product details."
    );
    let user = format!(
        "Review this captured candidate call and return JSON with \
         label, confidence, rationale, and optional corrected_candidate_call.\n\n{}",
        serde_json::to_string_pretty(capture).map_err(|err| err.to_string())?
    );
    parse_reviewer_decision(
        client
            .complete_json(system, &user, Some(reviewer_response_schema()))
            .await?,
    )
}

async fn verify_label(
    client: &JsonLlmClient,
    capture: &Value,
    candidate_call: &Value,
    label: &str,
    rationale: &str,
) -> Result<VerifierDecision, String> {
    let system = concat!(
        "You verify a proposed Forge tool-call verifier label. ",
        "Return only JSON with accepted boolean and rationale string. ",
        "Do not invent a different label. Verify the proposed label against the actual ",
        "user request, workflow state, available tools, candidate call, and tool result. ",
        "Reject labels that over-penalize normal multi-step tool use. A successful ",
        "search_products then compare_products sequence is valid for a comparison request. ",
        "If a candidate repeats an already completed non-terminal tool with no pending work, ",
        "the valid label should be rejected."
    );
    let user = json!({
        "task": "Accept or reject the proposed label. Do not relabel.",
        "proposed_label": label,
        "reviewer_rationale": rationale,
        "capture_context": {
            "user_request": capture.get("user_request").cloned().unwrap_or(Value::Null),
            "workflow_state": capture.get("workflow_state").cloned().unwrap_or(Value::Null),
            "available_tools": capture.get("available_tools").cloned().unwrap_or(Value::Null),
            "candidate_call": candidate_call,
            "tool_result": capture.get("tool_result").cloned().unwrap_or(Value::Null),
        }
    });
    let raw = client
        .complete_json(
            system,
            &serde_json::to_string_pretty(&user).map_err(|err| err.to_string())?,
            Some(verifier_response_schema()),
        )
        .await?;
    parse_verifier_decision(raw, label)
}

async fn review_generated_alternative(
    client: &JsonLlmClient,
    proposal: &AlternativeProposal,
) -> Result<ReviewDecision, String> {
    let system = concat!(
        "You review generated Forge tool-call verifier alternatives. Return only JSON. ",
        "Allowed labels are valid, wrong_tool_semantic, wrong_arguments_semantic, ",
        "tool_not_needed, needs_clarification. Generated wrong-tool rows must use ",
        "a real available competing tool, be schema-valid for that wrong tool, and ",
        "be semantically wrong for the request. If the proposed label is not clearly ",
        "correct, return the better label or needs_clarification."
    );
    let user = json!({
        "task": "Review this generated non-valid training alternative.",
        "proposed_label": proposal.label,
        "capture_context": {
            "user_request": proposal.capture.get("user_request").cloned().unwrap_or(Value::Null),
            "workflow_state": proposal.capture.get("workflow_state").cloned().unwrap_or(Value::Null),
            "available_tools": proposal.capture.get("available_tools").cloned().unwrap_or(Value::Null),
            "original_candidate_call": proposal.capture.get("candidate_call").cloned().unwrap_or(Value::Null),
            "generated_candidate_call": proposal.candidate_call,
        }
    });
    parse_reviewer_decision(
        client
            .complete_json(
                system,
                &serde_json::to_string_pretty(&user).map_err(|err| err.to_string())?,
                Some(reviewer_response_schema()),
            )
            .await?,
    )
}

async fn verify_alternatives(run: AlternativeReviewRun<'_>) -> Result<usize, String> {
    let AlternativeReviewRun {
        reviewer,
        verifier,
        proposals,
        alternative_cap,
        max_per_group,
        reject_path,
        output_path,
        resume_state,
        progress,
    } = run;
    let mut accepted = 0;
    eprintln!(
        "review alternatives start proposed={} remaining_cap={} max_per_group={}",
        proposals.len(),
        alternative_cap,
        max_per_group
    );
    for proposal in proposals {
        if accepted >= alternative_cap {
            break;
        }
        progress.alternatives_seen += 1;
        let row_key = row_key(
            &proposal.example_group_id,
            "targeted_alternative",
            &proposal.candidate_call,
        );
        if resume_state.processed_row_keys.contains(&row_key) {
            progress.alternatives_skipped += 1;
            progress.log_alternative_skip_if_needed();
            continue;
        }
        let group_count = *resume_state
            .alternatives_per_group
            .get(&proposal.example_group_id)
            .unwrap_or(&0);
        if group_count >= max_per_group {
            progress.alternatives_skipped += 1;
            progress.log_alternative_skip_if_needed();
            continue;
        }
        if let Some(reason) = post_review_quality_reject(
            &proposal.capture,
            &proposal.candidate_call,
            &proposal.label,
            "targeted_alternative",
        ) {
            append_reject(
                reject_path,
                "targeted_alternative_quality_rejected",
                &reason,
                &proposal.capture,
                None,
            )?;
            progress.rejected += 1;
            progress.log_alternative_rejected("targeted_alternative_quality_rejected");
            continue;
        }
        progress.log_alternative_reviewing(&proposal);
        let reviewer_decision = match review_generated_alternative(reviewer, &proposal).await {
            Ok(decision) => decision,
            Err(err) => {
                append_reject(
                    reject_path,
                    "targeted_alternative_reviewer_invalid_response",
                    &err,
                    &proposal.capture,
                    None,
                )?;
                progress.rejected += 1;
                progress.log_alternative_rejected("targeted_alternative_reviewer_invalid_response");
                continue;
            }
        };
        if reviewer_decision.label != proposal.label {
            append_reject(
                reject_path,
                "targeted_alternative_reviewer_rejected_label",
                &format!(
                    "reviewer returned '{}' for proposed '{}'",
                    reviewer_decision.label, proposal.label
                ),
                &proposal.capture,
                Some(reviewer_decision.raw),
            )?;
            progress.rejected += 1;
            progress.log_alternative_rejected("targeted_alternative_reviewer_rejected_label");
            continue;
        }
        if reviewer_decision.corrected_candidate_call.is_some() {
            append_reject(
                reject_path,
                "targeted_alternative_reviewer_returned_correction",
                "generated non-valid alternatives must not carry corrected positives",
                &proposal.capture,
                Some(reviewer_decision.raw),
            )?;
            progress.rejected += 1;
            progress.log_alternative_rejected("targeted_alternative_reviewer_returned_correction");
            continue;
        }
        let verifier_decision = match verify_label(
            verifier,
            &proposal.capture,
            &proposal.candidate_call,
            &proposal.label,
            &reviewer_decision.rationale,
        )
        .await
        {
            Ok(decision) => decision,
            Err(err) => {
                append_reject(
                    reject_path,
                    "targeted_alternative_verifier_invalid_response",
                    &err,
                    &proposal.capture,
                    None,
                )?;
                progress.rejected += 1;
                progress.log_alternative_rejected("targeted_alternative_verifier_invalid_response");
                continue;
            }
        };
        if verifier_decision.accepted {
            let row = training_row(
                &proposal.capture,
                proposal.candidate_call,
                &proposal.label,
                "targeted_alternative",
                &reviewer_decision,
                &verifier_decision,
                None,
            );
            append_jsonl(output_path, &row)?;
            resume_state.record_training_row(&row);
            accepted += 1;
            progress.accepted_alternatives += 1;
            progress.log_alternative_accepted(&proposal.label);
        } else {
            append_reject(
                reject_path,
                "targeted_alternative_verifier_rejected",
                &verifier_decision.rationale,
                &proposal.capture,
                Some(verifier_decision.raw),
            )?;
            progress.rejected += 1;
            progress.log_alternative_rejected("targeted_alternative_verifier_rejected");
        }
    }
    eprintln!(
        "review alternatives complete seen={} skipped={} accepted={} rejected_total={}",
        progress.alternatives_seen, progress.alternatives_skipped, accepted, progress.rejected
    );
    Ok(accepted)
}

#[derive(Debug, Clone)]
struct ReviewDecision {
    label: String,
    confidence: f64,
    rationale: String,
    corrected_candidate_call: Option<Value>,
    raw: Value,
}

#[derive(Debug, Clone)]
struct VerifierDecision {
    accepted: bool,
    rationale: String,
    raw: Value,
}

#[derive(Debug, Clone)]
struct AlternativeProposal {
    capture: Value,
    example_group_id: String,
    candidate_call: Value,
    label: String,
}

struct AlternativeReviewRun<'a> {
    reviewer: &'a JsonLlmClient,
    verifier: &'a JsonLlmClient,
    proposals: Vec<AlternativeProposal>,
    alternative_cap: usize,
    max_per_group: usize,
    reject_path: &'a Path,
    output_path: &'a str,
    resume_state: &'a mut ResumeState,
    progress: &'a mut ReviewProgress,
}

#[derive(Debug)]
struct ReviewProgress {
    total: usize,
    seen: usize,
    skipped: usize,
    accepted_real: usize,
    accepted_corrected: usize,
    accepted_alternatives: usize,
    rejected: usize,
    alternatives_seen: usize,
    alternatives_skipped: usize,
}

impl ReviewProgress {
    fn new(total: usize) -> Self {
        Self {
            total,
            seen: 0,
            skipped: 0,
            accepted_real: 0,
            accepted_corrected: 0,
            accepted_alternatives: 0,
            rejected: 0,
            alternatives_seen: 0,
            alternatives_skipped: 0,
        }
    }

    fn log_skip_if_needed(&self) {
        if self.skipped == 1 || self.skipped.is_multiple_of(250) {
            eprintln!(
                "review resume skipped={} seen={}/{}",
                self.skipped, self.seen, self.total
            );
        }
    }

    fn log_reviewing(&self, capture: &Value) {
        eprintln!(
            "review row {}/{} group={} domain={} scenario={} tool={}",
            self.seen,
            self.total,
            value_str(capture.get("example_group_id")),
            trace_str(capture, "domain"),
            trace_str(capture, "scenario"),
            candidate_tool(capture.get("candidate_call"))
        );
    }

    fn log_accepted(&self, row_index: usize, label: &str, source_bucket: &str) {
        eprintln!(
            "review row {}/{} accepted label={} source_bucket={} accepted_real={} accepted_corrected={} rejected={}",
            row_index,
            self.total,
            label,
            source_bucket,
            self.accepted_real,
            self.accepted_corrected,
            self.rejected
        );
    }

    fn log_rejected(&self, row_index: usize, reason: &str) {
        eprintln!(
            "review row {}/{} rejected reason={} accepted_real={} accepted_corrected={} rejected={}",
            row_index,
            self.total,
            reason,
            self.accepted_real,
            self.accepted_corrected,
            self.rejected
        );
    }

    fn log_alternative_skip_if_needed(&self) {
        if self.alternatives_skipped == 1 || self.alternatives_skipped.is_multiple_of(250) {
            eprintln!(
                "review alternative skipped={} seen={}",
                self.alternatives_skipped, self.alternatives_seen
            );
        }
    }

    fn log_alternative_reviewing(&self, proposal: &AlternativeProposal) {
        eprintln!(
            "review alternative {} group={} label={} tool={}",
            self.alternatives_seen,
            proposal.example_group_id,
            proposal.label,
            candidate_tool(Some(&proposal.candidate_call))
        );
    }

    fn log_alternative_accepted(&self, label: &str) {
        eprintln!(
            "review alternative {} accepted label={} accepted_alternatives={} rejected={}",
            self.alternatives_seen, label, self.accepted_alternatives, self.rejected
        );
    }

    fn log_alternative_rejected(&self, reason: &str) {
        eprintln!(
            "review alternative {} rejected reason={} accepted_alternatives={} rejected={}",
            self.alternatives_seen, reason, self.accepted_alternatives, self.rejected
        );
    }
}

fn value_str(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("unknown")
}

fn trace_str<'a>(capture: &'a Value, key: &str) -> &'a str {
    capture
        .get("proxy_trace")
        .and_then(|trace| trace.get(key))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

fn candidate_tool(candidate: Option<&Value>) -> &str {
    candidate
        .and_then(|candidate| candidate.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

fn post_review_quality_reject(
    capture: &Value,
    candidate_call: &Value,
    label: &str,
    source_bucket: &str,
) -> Option<String> {
    if label == "valid" && repeats_completed_non_terminal_tool(capture, candidate_call) {
        return Some(format!(
            "{source_bucket} proposed valid for an already-completed non-terminal tool with no pending workflow steps"
        ));
    }

    if label == "wrong_tool_semantic" && looks_like_wrong_tool_overreach(capture, candidate_call) {
        return Some(
            "candidate uses a semantically matching tool for the request; if arguments are bad, this should be wrong_arguments_semantic, not wrong_tool_semantic"
                .to_string(),
        );
    }

    None
}

fn repeats_completed_non_terminal_tool(capture: &Value, candidate_call: &Value) -> bool {
    let Some(name) = candidate_call.get("name").and_then(Value::as_str) else {
        return false;
    };
    if name == "respond" {
        return false;
    }
    let Some(workflow_state) = capture.get("workflow_state") else {
        return false;
    };
    let pending_empty = workflow_state
        .get("pending_steps")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty);
    if !pending_empty {
        return false;
    }
    workflow_state
        .get("completed_steps")
        .and_then(Value::as_array)
        .is_some_and(|completed| completed.iter().any(|step| step.as_str() == Some(name)))
}

fn looks_like_wrong_tool_overreach(capture: &Value, candidate_call: &Value) -> bool {
    let Some(name) = candidate_call.get("name").and_then(Value::as_str) else {
        return false;
    };
    let request = capture
        .get("user_request")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    if request.contains("compare") && name == "compare_products" {
        return true;
    }
    if request.contains("list") && name == "list_files" {
        return true;
    }
    if request.contains("look up") && name == "lookup_ticket" {
        return true;
    }
    false
}

fn parse_reviewer_decision(raw: Value) -> Result<ReviewDecision, String> {
    let label = raw
        .get("label")
        .and_then(Value::as_str)
        .ok_or_else(|| "reviewer JSON missing label".to_string())?;
    if !is_allowed_label(label) {
        return Err(format!("reviewer returned unsupported label '{label}'"));
    }
    let confidence = raw
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| "reviewer JSON missing numeric confidence".to_string())?;
    if !(0.0..=1.0).contains(&confidence) {
        return Err("reviewer confidence must be between 0.0 and 1.0".to_string());
    }
    let rationale = raw
        .get("rationale")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let corrected_candidate_call = raw
        .get("corrected_candidate_call")
        .filter(|value| !value.is_null())
        .cloned();
    Ok(ReviewDecision {
        label: label.to_string(),
        confidence,
        rationale,
        corrected_candidate_call,
        raw,
    })
}

fn parse_verifier_decision(raw: Value, expected_label: &str) -> Result<VerifierDecision, String> {
    if let Some(label) = raw.get("label").and_then(Value::as_str) {
        if label != expected_label {
            return Err(format!(
                "verifier invented label '{label}' instead of accepting/rejecting '{expected_label}'"
            ));
        }
    }
    let accepted = match raw.get("accepted").and_then(Value::as_bool) {
        Some(value) => value,
        None => match raw.get("decision").and_then(Value::as_str) {
            Some("accept") | Some("accepted") => true,
            Some("reject") | Some("rejected") => false,
            _ => return Err("verifier JSON missing accepted boolean".to_string()),
        },
    };
    let rationale = raw
        .get("rationale")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok(VerifierDecision {
        accepted,
        rationale,
        raw,
    })
}

fn training_row(
    capture: &Value,
    candidate_call: Value,
    label: &str,
    source_bucket: &str,
    review: &ReviewDecision,
    verifier: &VerifierDecision,
    corrected_positive: Option<Value>,
) -> Value {
    let example_group_id = capture
        .get("example_group_id")
        .cloned()
        .unwrap_or(Value::Null);
    let example_group_id_str = example_group_id.as_str().unwrap_or("unknown-group");
    let capture_key = capture_key(capture);
    let row_key = row_key(example_group_id_str, source_bucket, &candidate_call);
    let mut row = json!({
        "schema_version": TRAINING_SCHEMA_VERSION,
        "input": {
            "schema_version": TRAINING_INPUT_SCHEMA_VERSION,
            "user_request": capture.get("user_request").cloned().unwrap_or(Value::Null),
            "workflow_state": capture.get("workflow_state").cloned().unwrap_or(Value::Null),
            "available_tools": capture.get("available_tools").cloned().unwrap_or_else(|| json!([])),
            "candidate_call": candidate_call,
            "metadata": training_metadata(capture)
        },
        "label": label,
        "review": {
            "source": "forge-dataset",
            "source_bucket": source_bucket,
            "example_group_id": example_group_id,
            "capture_key": capture_key,
            "row_key": row_key,
            "capture_schema_version": capture.get("schema_version").cloned().unwrap_or(Value::Null),
            "reviewer": {
                "label": review.label,
                "confidence": review.confidence,
                "rationale": review.rationale
            },
            "verifier": {
                "accepted": verifier.accepted,
                "rationale": verifier.rationale
            },
            "capture_provenance": capture
                .get("metadata")
                .and_then(|metadata| metadata.get("provenance"))
                .cloned()
                .unwrap_or(Value::Null)
        }
    });
    if let Some(corrected_positive) = corrected_positive {
        if let Some(obj) = row.as_object_mut() {
            obj.insert("corrected_positive".to_string(), corrected_positive);
        }
    }
    row
}

fn training_metadata(capture: &Value) -> Value {
    json!({
        "scenario_family": capture
            .get("proxy_trace")
            .and_then(|trace| trace.get("domain"))
            .and_then(Value::as_str)
            .unwrap_or("forge_dataset"),
        "requires_transform": false,
        "requires_synthesis": false,
        "requires_all_tool_facts": false,
        "must_acknowledge_missing_data": false
    })
}

fn propose_targeted_alternatives(capture: &Value) -> Vec<AlternativeProposal> {
    let mut proposals = Vec::new();
    let available_tools = &capture["available_tools"];
    let original = &capture["candidate_call"];
    let Some(current_name) = original.get("name").and_then(Value::as_str) else {
        return proposals;
    };
    let example_group_id = capture
        .get("example_group_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-group")
        .to_string();

    if let Some(tool) = tool_by_name(available_tools, current_name) {
        if let Some(mutated_args) =
            mutated_arguments_for_tool(tool, original.get("arguments").unwrap_or(&Value::Null))
        {
            let candidate_call = json!({"name": current_name, "arguments": mutated_args});
            if validate_candidate_call(available_tools, &candidate_call).is_ok()
                && candidate_call != *original
            {
                proposals.push(AlternativeProposal {
                    capture: capture.clone(),
                    example_group_id: example_group_id.clone(),
                    candidate_call,
                    label: "wrong_arguments_semantic".to_string(),
                });
            }
        }
    }

    if let Some(candidate_call) = curated_wrong_tool_candidate(capture, current_name) {
        if validate_candidate_call(available_tools, &candidate_call).is_ok() {
            proposals.push(AlternativeProposal {
                capture: capture.clone(),
                example_group_id,
                candidate_call,
                label: "wrong_tool_semantic".to_string(),
            });
        }
    }
    proposals
}

fn curated_wrong_tool_candidate(capture: &Value, current_name: &str) -> Option<Value> {
    let domain = capture
        .get("proxy_trace")
        .and_then(|trace| trace.get("domain"))
        .and_then(Value::as_str)?;
    let scenario = capture
        .get("proxy_trace")
        .and_then(|trace| trace.get("scenario"))
        .and_then(Value::as_str)?;

    let candidate = match (domain, scenario) {
        ("shopping", "compare_headphones") => {
            json!({"name": "add_to_cart", "arguments": {"product_id": "SKU-HEADPHONES-1", "quantity": 1}})
        }
        ("shopping", "add_dock") => {
            json!({"name": "compare_products", "arguments": {"product_ids": ["SKU-DOCK-1", "SKU-HEADPHONES-1"]}})
        }
        ("shopping", "product_details") => {
            json!({"name": "add_to_cart", "arguments": {"product_id": "SKU-HEADPHONES-1", "quantity": 1}})
        }
        ("calendar", "find_and_hold") => {
            json!({"name": "list_events", "arguments": {"date": "2026-06-05"}})
        }
        ("calendar", "list_events") => {
            json!({"name": "create_calendar_hold", "arguments": {"slot_id": "slot-001"}})
        }
        ("calendar", "cancel_hold") => {
            json!({"name": "list_events", "arguments": {"date": "2026-06-05"}})
        }
        ("support", "lookup_ticket") => {
            json!({"name": "update_ticket", "arguments": {"ticket_id": "TCK-1001", "note": "Reviewed by dataset stub."}})
        }
        ("support", "search_kb") => {
            json!({"name": "lookup_ticket", "arguments": {"ticket_id": "TCK-1001"}})
        }
        ("support", "escalate_ticket") => {
            json!({"name": "update_ticket", "arguments": {"ticket_id": "TCK-2002", "note": "Customer is blocked from billing."}})
        }
        ("forge_eval", "smoke_eval") => {
            json!({"name": "run_release_eval", "arguments": {"scenario": "basic_2step", "runs": 1}})
        }
        ("forge_eval", "release_eval") => {
            json!({"name": "run_smoke_eval", "arguments": {"scenario": "basic_2step", "runs": 1}})
        }
        ("forge_eval", "diagnose_failure") => {
            json!({"name": "report_result", "arguments": {"summary": "Failure diagnosed."}})
        }
        ("forge_eval", "inspect_workflow_state") => {
            json!({"name": "report_result", "arguments": {"summary": "Workflow state checked."}})
        }
        ("forge_eval", "fetch_zero_padded") => {
            json!({"name": "summarize_records", "arguments": {"content": "Fetched 0010 records."}})
        }
        _ => return None,
    };

    if candidate.get("name").and_then(Value::as_str) == Some(current_name) {
        None
    } else {
        Some(candidate)
    }
}

fn alternative_cap(real_row_count: usize, ratio: f64) -> usize {
    ((real_row_count as f64 * ratio) + 1e-9).floor() as usize
}

#[derive(Debug, Default)]
struct ResumeState {
    processed_capture_keys: HashSet<String>,
    processed_row_keys: HashSet<String>,
    valid_real_capture_keys: HashSet<String>,
    alternatives_per_group: HashMap<String, usize>,
    accepted_real_count: usize,
    accepted_alternative_count: usize,
}

impl ResumeState {
    fn load(output_path: &str, reject_path: &Path) -> Result<Self, String> {
        let mut state = Self::default();
        state.load_training_output(output_path)?;
        state.load_rejects(reject_path)?;
        Ok(state)
    }

    fn load_training_output(&mut self, path: &str) -> Result<(), String> {
        if !Path::new(path).exists() {
            return Ok(());
        }
        for row in read_jsonl(path)? {
            self.record_training_row(&row);
        }
        Ok(())
    }

    fn load_rejects(&mut self, path: &Path) -> Result<(), String> {
        if !path.exists() {
            return Ok(());
        }
        for row in read_jsonl_path(path)? {
            if let Some(capture_key) = row.get("capture_key").and_then(Value::as_str) {
                self.processed_capture_keys.insert(capture_key.to_string());
            } else if let Some(capture) = row.get("capture") {
                self.processed_capture_keys.insert(capture_key(capture));
            }
        }
        Ok(())
    }

    fn record_training_row(&mut self, row: &Value) {
        let review = row.get("review").unwrap_or(&Value::Null);
        let source_bucket = review
            .get("source_bucket")
            .and_then(Value::as_str)
            .unwrap_or("");
        if let Some(row_key) = review.get("row_key").and_then(Value::as_str) {
            self.processed_row_keys.insert(row_key.to_string());
        } else if let (Some(example_group_id), Some(candidate_call)) = (
            review.get("example_group_id").and_then(Value::as_str),
            row.get("input")
                .and_then(|input| input.get("candidate_call")),
        ) {
            self.processed_row_keys.insert(row_key(
                example_group_id,
                source_bucket,
                candidate_call,
            ));
        }
        if let Some(capture_key) = review.get("capture_key").and_then(Value::as_str) {
            if source_bucket == "real_model_call" {
                self.processed_capture_keys.insert(capture_key.to_string());
                self.accepted_real_count += 1;
                if row.get("label").and_then(Value::as_str) == Some("valid") {
                    self.valid_real_capture_keys.insert(capture_key.to_string());
                }
            }
        }
        if source_bucket == "targeted_alternative" {
            self.accepted_alternative_count += 1;
            if let Some(group_id) = review.get("example_group_id").and_then(Value::as_str) {
                *self
                    .alternatives_per_group
                    .entry(group_id.to_string())
                    .or_insert(0) += 1;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReviewRole {
    Reviewer,
    Verifier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProviderConfig {
    provider: String,
    chat_url: String,
    model: String,
    api_key: Option<String>,
}

struct JsonLlmClient {
    provider: String,
    chat_url: String,
    model: String,
    api_key: Option<String>,
    http_client: reqwest::Client,
}

impl JsonLlmClient {
    fn new(config: ProviderConfig) -> Self {
        Self {
            provider: config.provider,
            chat_url: normalize_chat_completions_url(&config.chat_url),
            model: config.model,
            api_key: config.api_key,
            http_client: reqwest::Client::new(),
        }
    }

    async fn complete_json(
        &self,
        system: &str,
        user: &str,
        response_schema: Option<Value>,
    ) -> Result<Value, String> {
        let strict_schema = self.provider == "openrouter";
        let mut body = self.request_body(system, user);
        if strict_schema {
            if let Some(schema) = response_schema.clone() {
                attach_openrouter_strict_schema(&mut body, schema);
            }
        }
        match self.post_chat(&body).await {
            Ok(value) => Ok(value),
            Err(err) if strict_schema && openrouter_strict_schema_unavailable(&err) => {
                eprintln!(
                    "api fallback api=openrouter model={} reason=strict_json_schema_unavailable",
                    self.model
                );
                self.post_chat(&self.request_body(system, user)).await
            }
            Err(err) => Err(err),
        }
    }

    fn request_body(&self, system: &str, user: &str) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0,
        });
        if let Some(obj) = body.as_object_mut() {
            let token_key = if self.provider == "openrouter" {
                "max_tokens"
            } else {
                "max_completion_tokens"
            };
            obj.insert(token_key.to_string(), json!(1200));
        }
        body
    }

    async fn post_chat(&self, body: &Value) -> Result<Value, String> {
        let mut request = self
            .http_client
            .post(&self.chat_url)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(api_key) = self.api_key.as_deref() {
            request = request.bearer_auth(api_key);
        }
        let response = request
            .send()
            .await
            .map_err(|err| format!("failed to call reviewer/verifier LLM: {err}"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| format!("failed to read reviewer/verifier response: {err}"))?;
        if !status.is_success() {
            return Err(format!("reviewer/verifier returned HTTP {status}: {text}"));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|err| format!("failed to parse reviewer/verifier HTTP JSON: {err}"))?;
        let content = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| "reviewer/verifier response missing message.content".to_string())?;
        parse_json_object_from_text(content)
    }
}

fn attach_openrouter_strict_schema(body: &mut Value, schema: Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    obj.insert(
        "response_format".to_string(),
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "forge_dataset_review",
                "strict": true,
                "schema": schema,
            }
        }),
    );
    obj.insert("provider".to_string(), json!({"require_parameters": true}));
}

fn openrouter_strict_schema_unavailable(error: &str) -> bool {
    error.contains("HTTP 404")
        && error.contains("No endpoints found that can handle the requested parameters")
}

fn reviewer_response_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "label": {"type": "string", "enum": [
                "valid",
                "wrong_tool_semantic",
                "wrong_arguments_semantic",
                "tool_not_needed",
                "needs_clarification"
            ]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
            "corrected_candidate_call": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"}
                        },
                        "required": ["name", "arguments"]
                    }
                ]
            }
        },
        "required": ["label", "confidence", "rationale", "corrected_candidate_call"]
    })
}

fn verifier_response_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "accepted": {"type": "boolean"},
            "rationale": {"type": "string"}
        },
        "required": ["accepted", "rationale"]
    })
}

fn resolve_provider_config(
    cli: &ReviewCli,
    env_file: &EnvFile,
    role: ReviewRole,
) -> Result<ProviderConfig, String> {
    let provider = cli.provider.clone();
    resolve_provider_config_for(cli, env_file, role, &provider)
}

fn resolve_provider_config_for(
    cli: &ReviewCli,
    env_file: &EnvFile,
    role: ReviewRole,
    requested_provider: &str,
) -> Result<ProviderConfig, String> {
    let manual_base_url = match role {
        ReviewRole::Reviewer => cli.reviewer_base_url.as_str(),
        ReviewRole::Verifier => cli.verifier_base_url.as_str(),
    };
    let manual_model = match role {
        ReviewRole::Reviewer => cli.reviewer_model.as_str(),
        ReviewRole::Verifier => cli.verifier_model.as_str(),
    };
    let manual_key = match role {
        ReviewRole::Reviewer => cli.reviewer_api_key.clone(),
        ReviewRole::Verifier => cli.verifier_api_key.clone(),
    };

    if !manual_base_url.trim().is_empty() && !manual_model.trim().is_empty() {
        return Ok(ProviderConfig {
            provider: requested_provider.to_string(),
            chat_url: manual_base_url.to_string(),
            model: manual_model.to_string(),
            api_key: manual_key.or_else(|| lookup_env(env_file, &role_api_key_names(&role))),
        });
    }

    let provider = if requested_provider == "auto" {
        if lookup_env(env_file, &["MINIMAX_API_KEY"]).is_some() {
            "minimax"
        } else if lookup_env(env_file, &["OPENROUTER_API_KEY"]).is_some() {
            "openrouter"
        } else {
            return Err(format!(
                "{} review requires MINIMAX_API_KEY, OPENROUTER_API_KEY, or manual base URL/model",
                role_name(&role)
            ));
        }
    } else {
        requested_provider
    };

    match provider {
        "minimax" => {
            let api_key = manual_key
                .or_else(|| lookup_env(env_file, &["MINIMAX_API_KEY"]))
                .ok_or_else(|| "MINIMAX_API_KEY is required for provider minimax".to_string())?;
            Ok(ProviderConfig {
                provider: "minimax".to_string(),
                chat_url: "https://api.minimax.io/v1/chat/completions".to_string(),
                model: if manual_model.trim().is_empty() {
                    if cli.minimax_model.trim().is_empty() {
                        lookup_env(
                            env_file,
                            &[
                                "FORGE_DATASET_MINIMAX_MODEL",
                                "GENERATETD_MINIMAX_MODEL",
                                "MINIMAX_MODEL",
                            ],
                        )
                        .unwrap_or_else(|| default_minimax_model().to_string())
                    } else {
                        cli.minimax_model.clone()
                    }
                } else {
                    manual_model.to_string()
                },
                api_key: Some(api_key),
            })
        }
        "openrouter" => {
            let api_key = manual_key
                .or_else(|| lookup_env(env_file, &["OPENROUTER_API_KEY"]))
                .ok_or_else(|| {
                    "OPENROUTER_API_KEY is required for provider openrouter".to_string()
                })?;
            Ok(ProviderConfig {
                provider: "openrouter".to_string(),
                chat_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
                model: if manual_model.trim().is_empty() {
                    if cli.openrouter_model.trim().is_empty() {
                        lookup_env(
                            env_file,
                            &[
                                "FORGE_DATASET_OPENROUTER_MODEL",
                                "GENERATETD_OPENROUTER_MODEL",
                                "OPENROUTER_MODEL",
                            ],
                        )
                        .unwrap_or_else(|| default_openrouter_model().to_string())
                    } else {
                        cli.openrouter_model.clone()
                    }
                } else {
                    manual_model.to_string()
                },
                api_key: Some(api_key),
            })
        }
        other => Err(format!("unknown provider: {other}")),
    }
}

fn role_api_key_names(role: &ReviewRole) -> [&'static str; 3] {
    match role {
        ReviewRole::Reviewer => [
            "FORGE_DATASET_REVIEWER_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ],
        ReviewRole::Verifier => [
            "FORGE_DATASET_VERIFIER_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ],
    }
}

fn role_name(role: &ReviewRole) -> &'static str {
    match role {
        ReviewRole::Reviewer => "reviewer",
        ReviewRole::Verifier => "verifier",
    }
}

#[derive(Debug, Clone, Default)]
struct EnvFile {
    values: HashMap<String, String>,
}

impl EnvFile {
    fn load(path: &str) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        let mut values = HashMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((key, value)) = trimmed.split_once('=') else {
                continue;
            };
            values.insert(key.trim().to_string(), unquote_env_value(value.trim()));
        }
        Self { values }
    }
}

fn lookup_env(env_file: &EnvFile, names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        if let Some(value) = env_file.values.get(*name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn unquote_env_value(raw: &str) -> String {
    let value = raw.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn read_jsonl(path: &str) -> Result<Vec<Value>, String> {
    read_jsonl_path(Path::new(path))
}

fn count_jsonl_rows(path: &str) -> Result<usize, String> {
    let file = File::open(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut count = 0;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|err| format!("{path}:{} read error: {err}", index + 1))?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

fn read_jsonl_path(path: &Path) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Value>(trimmed)
            .map_err(|err| format!("{}:{} invalid JSONL row: {err}", path.display(), index + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn append_jsonl(path: &str, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    writeln!(file, "{line}").map_err(|err| format!("failed to write {path}: {err}"))
}

fn append_reject(
    path: &Path,
    reason: &str,
    detail: &str,
    capture: &Value,
    raw_response: Option<Value>,
) -> Result<(), String> {
    let capture_key = capture_key(capture);
    append_jsonl_path(
        path,
        &json!({
            "schema_version": "forge-dataset-review-reject/v1",
            "reason": reason,
            "detail": detail,
            "example_group_id": capture.get("example_group_id").cloned().unwrap_or(Value::Null),
            "capture_key": capture_key,
            "capture": capture,
            "raw_response": raw_response.unwrap_or(Value::Null),
        }),
    )
}

fn append_reject_record(path: &Path, record: &RejectRecord) -> Result<(), String> {
    append_reject(
        path,
        &record.reason,
        &record.detail,
        &record.capture,
        record.raw_response.clone(),
    )
}

fn append_jsonl_path(path: &Path, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    writeln!(file, "{line}").map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn rejects_path(output: &str) -> PathBuf {
    let path = Path::new(output);
    let stem = path.file_stem().and_then(|value| value.to_str());
    let extension = path.extension().and_then(|value| value.to_str());
    let file_name = match (stem, extension) {
        (Some(stem), Some(extension)) => format!("{stem}.rejects.{extension}"),
        (Some(stem), None) => format!("{stem}.rejects"),
        _ => format!("{output}.rejects.jsonl"),
    };
    path.with_file_name(file_name)
}

fn capture_key(capture: &Value) -> String {
    let group = capture
        .get("example_group_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-group");
    let trace = capture.get("proxy_trace").unwrap_or(&Value::Null);
    let turn = trace
        .get("turn")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown-turn".to_string());
    let call_index = trace
        .get("call_index")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown-call".to_string());
    let tool_call_id = trace
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-tool-call");
    format!("{group}:turn-{turn}:call-{call_index}:tool-call-{tool_call_id}")
}

fn row_key(example_group_id: &str, source_bucket: &str, candidate_call: &Value) -> String {
    let candidate = serde_json::to_string(candidate_call).unwrap_or_else(|_| "null".to_string());
    format!("{example_group_id}:{source_bucket}:{candidate}")
}

fn ensure_parent_dir(path: &str) -> Result<(), String> {
    ensure_parent_dir_path(Path::new(path))
}

fn ensure_parent_dir_path(path: &Path) -> Result<(), String> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("failed to create {}: {err}", parent.display()))
}

fn normalize_chat_completions_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture_row() -> Value {
        json!({
            "schema_version": "forge-dataset-capture/v1",
            "example_group_id": "group-1",
            "user_request": "Compare two products.",
            "workflow_state": {
                "required_steps": [],
                "completed_steps": [],
                "pending_steps": [],
                "terminal_tools": ["respond"],
                "recent_errors": []
            },
            "available_tools": [
                {
                    "name": "compare_products",
                    "description": "Compare products.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_ids": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["product_ids"]
                    }
                },
                {
                    "name": "add_to_cart",
                    "description": "Add a product to cart.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "integer"}
                        },
                        "required": ["product_id", "quantity"]
                    }
                }
            ],
            "candidate_call": {
                "name": "compare_products",
                "arguments": {"product_ids": ["SKU-1", "SKU-2"]}
            },
            "tool_result": {"status": "ok", "content": {}},
            "proxy_trace": {"domain": "shopping", "scenario": "compare_headphones"}
        })
    }

    #[test]
    fn reviewer_parser_rejects_unknown_labels() {
        let err = parse_reviewer_decision(json!({
            "label": "synthetic_unrelated_tool",
            "confidence": 0.8,
            "rationale": "bad"
        }))
        .expect_err("unknown label");
        assert!(err.contains("unsupported label"));
    }

    #[test]
    fn verifier_parser_rejects_relabeling() {
        let err = parse_verifier_decision(
            json!({"accepted": true, "label": "valid", "rationale": "changed"}),
            "wrong_tool_semantic",
        )
        .expect_err("invented label");
        assert!(err.contains("invented label"));
    }

    #[test]
    fn training_row_matches_toolcall_training_envelope() {
        let review = ReviewDecision {
            label: "valid".to_string(),
            confidence: 0.9,
            rationale: "ok".to_string(),
            corrected_candidate_call: None,
            raw: json!({}),
        };
        let verifier = VerifierDecision {
            accepted: true,
            rationale: "accepted".to_string(),
            raw: json!({}),
        };
        let capture = capture_row();
        let row = training_row(
            &capture,
            capture["candidate_call"].clone(),
            "valid",
            "real_model_call",
            &review,
            &verifier,
            None,
        );
        assert_eq!(row["schema_version"], TRAINING_SCHEMA_VERSION);
        assert_eq!(
            row["input"]["schema_version"],
            TRAINING_INPUT_SCHEMA_VERSION
        );
        assert_eq!(row["review"]["example_group_id"], "group-1");
        assert!(row["review"]["capture_key"]
            .as_str()
            .expect("capture key")
            .contains("group-1"));
        assert!(row["review"]["row_key"]
            .as_str()
            .expect("row key")
            .contains("real_model_call"));
        assert_eq!(row["label"], "valid");
    }

    #[test]
    fn training_row_metadata_does_not_encode_label() {
        let review = ReviewDecision {
            label: "wrong_arguments_semantic".to_string(),
            confidence: 0.9,
            rationale: "bad argument".to_string(),
            corrected_candidate_call: None,
            raw: json!({}),
        };
        let verifier = VerifierDecision {
            accepted: true,
            rationale: "accepted".to_string(),
            raw: json!({}),
        };
        let capture = capture_row();

        let wrong_args_row = training_row(
            &capture,
            capture["candidate_call"].clone(),
            "wrong_arguments_semantic",
            "targeted_alternative",
            &review,
            &verifier,
            None,
        );
        let clarification_row = training_row(
            &capture,
            capture["candidate_call"].clone(),
            "needs_clarification",
            "targeted_alternative",
            &review,
            &verifier,
            None,
        );

        assert_eq!(
            wrong_args_row["input"]["metadata"]["scenario_family"],
            "shopping"
        );
        assert_eq!(
            wrong_args_row["input"]["metadata"]["requires_transform"],
            false
        );
        assert_eq!(
            clarification_row["input"]["metadata"]["must_acknowledge_missing_data"],
            false
        );
    }

    #[test]
    fn resume_state_tracks_streamed_real_rows() {
        let review = ReviewDecision {
            label: "valid".to_string(),
            confidence: 0.9,
            rationale: "ok".to_string(),
            corrected_candidate_call: None,
            raw: json!({}),
        };
        let verifier = VerifierDecision {
            accepted: true,
            rationale: "accepted".to_string(),
            raw: json!({}),
        };
        let capture = capture_row();
        let row = training_row(
            &capture,
            capture["candidate_call"].clone(),
            "valid",
            "real_model_call",
            &review,
            &verifier,
            None,
        );

        let mut state = ResumeState::default();
        state.record_training_row(&row);

        assert_eq!(state.accepted_real_count, 1);
        assert!(state
            .processed_capture_keys
            .contains(row["review"]["capture_key"].as_str().expect("capture key")));
        assert!(state
            .valid_real_capture_keys
            .contains(row["review"]["capture_key"].as_str().expect("capture key")));
    }

    #[test]
    fn quality_filter_rejects_completed_tool_valid_positive() {
        let mut capture = capture_row();
        capture["workflow_state"]["completed_steps"] = json!(["compare_products"]);
        capture["workflow_state"]["pending_steps"] = json!([]);

        let reason = post_review_quality_reject(
            &capture,
            &capture["candidate_call"],
            "valid",
            "real_model_call",
        )
        .expect("quality rejection");

        assert!(reason.contains("already-completed"));
    }

    #[test]
    fn quality_filter_rejects_wrong_tool_for_semantically_matching_compare_tool() {
        let capture = capture_row();

        let reason = post_review_quality_reject(
            &capture,
            &capture["candidate_call"],
            "wrong_tool_semantic",
            "real_model_call",
        )
        .expect("quality rejection");

        assert!(reason.contains("wrong_arguments_semantic"));
    }

    #[test]
    fn quality_filter_allows_real_competing_wrong_tool() {
        let capture = capture_row();
        let candidate = json!({
            "name": "add_to_cart",
            "arguments": {"product_id": "SKU-1", "quantity": 1}
        });

        assert!(post_review_quality_reject(
            &capture,
            &candidate,
            "wrong_tool_semantic",
            "targeted_alternative",
        )
        .is_none());
    }

    #[test]
    fn proposes_schema_valid_targeted_alternatives() {
        let capture = capture_row();
        let proposals = propose_targeted_alternatives(&capture);
        assert!(proposals
            .iter()
            .any(|proposal| proposal.label == "wrong_arguments_semantic"));
        assert!(proposals
            .iter()
            .any(|proposal| proposal.label == "wrong_tool_semantic"));
        assert!(proposals.iter().any(|proposal| {
            proposal.label == "wrong_tool_semantic"
                && proposal.candidate_call["name"] == "add_to_cart"
        }));
        for proposal in proposals {
            validate_candidate_call(&capture["available_tools"], &proposal.candidate_call)
                .expect("schema-valid proposal");
        }
    }

    #[test]
    fn alternative_cap_keeps_real_rows_as_backbone() {
        assert_eq!(alternative_cap(0, 1.0 / 3.0), 0);
        assert_eq!(alternative_cap(2, 1.0 / 3.0), 0);
        assert_eq!(alternative_cap(3, 1.0 / 3.0), 1);
        assert_eq!(alternative_cap(6, 1.0 / 3.0), 2);
    }

    #[test]
    fn rejects_path_is_sibling_jsonl() {
        assert_eq!(
            rejects_path("target/dataset/training.toolcall.jsonl"),
            PathBuf::from("target/dataset/training.toolcall.rejects.jsonl")
        );
    }

    #[test]
    fn openrouter_request_uses_supported_token_parameter() {
        let client = JsonLlmClient::new(ProviderConfig {
            provider: "openrouter".to_string(),
            chat_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            model: "openrouter/free".to_string(),
            api_key: None,
        });
        let body = client.request_body("system", "user");
        assert_eq!(body["max_tokens"], 1200);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn minimax_request_keeps_openai_compatible_token_parameter() {
        let client = JsonLlmClient::new(ProviderConfig {
            provider: "minimax".to_string(),
            chat_url: "https://api.minimax.io/v1/chat/completions".to_string(),
            model: "MiniMax-M2.7".to_string(),
            api_key: None,
        });
        let body = client.request_body("system", "user");
        assert_eq!(body["max_completion_tokens"], 1200);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn openrouter_model_flag_overrides_env_file_model() {
        let cli = ReviewCli {
            input: "capture.jsonl".to_string(),
            output: "training.jsonl".to_string(),
            env_file: ".env".to_string(),
            provider: "openrouter".to_string(),
            verifier_provider: "same".to_string(),
            minimax_model: String::new(),
            openrouter_model: "openrouter/free".to_string(),
            reviewer_base_url: String::new(),
            reviewer_model: String::new(),
            reviewer_api_key: None,
            verifier_base_url: String::new(),
            verifier_model: String::new(),
            verifier_api_key: None,
            max_alternatives_per_group: 2,
            max_alternative_ratio: 1.0 / 3.0,
            concurrency: 1,
            resume: false,
        };
        let env_file = EnvFile {
            values: HashMap::from([
                (
                    "OPENROUTER_API_KEY".to_string(),
                    "test-openrouter-key".to_string(),
                ),
                (
                    "GENERATETD_OPENROUTER_MODEL".to_string(),
                    "openrouter/owl-alpha".to_string(),
                ),
            ]),
        };

        let config =
            resolve_provider_config_for(&cli, &env_file, ReviewRole::Reviewer, "openrouter")
                .expect("config");

        assert_eq!(config.model, "openrouter/free");
    }

    #[test]
    fn openrouter_env_file_model_is_used_when_flag_is_absent() {
        let cli = ReviewCli {
            input: "capture.jsonl".to_string(),
            output: "training.jsonl".to_string(),
            env_file: ".env".to_string(),
            provider: "openrouter".to_string(),
            verifier_provider: "same".to_string(),
            minimax_model: String::new(),
            openrouter_model: String::new(),
            reviewer_base_url: String::new(),
            reviewer_model: String::new(),
            reviewer_api_key: None,
            verifier_base_url: String::new(),
            verifier_model: String::new(),
            verifier_api_key: None,
            max_alternatives_per_group: 2,
            max_alternative_ratio: 1.0 / 3.0,
            concurrency: 1,
            resume: false,
        };
        let env_file = EnvFile {
            values: HashMap::from([
                (
                    "OPENROUTER_API_KEY".to_string(),
                    "test-openrouter-key".to_string(),
                ),
                (
                    "GENERATETD_OPENROUTER_MODEL".to_string(),
                    "openrouter/owl-alpha".to_string(),
                ),
            ]),
        };

        let config =
            resolve_provider_config_for(&cli, &env_file, ReviewRole::Reviewer, "openrouter")
                .expect("config");

        assert_eq!(config.model, "openrouter/owl-alpha");
    }
}
