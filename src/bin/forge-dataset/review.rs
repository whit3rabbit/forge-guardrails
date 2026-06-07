use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use futures_util::future::join_all;
use serde_json::{json, Value};

use crate::cli::ReviewCli;
use crate::schema::{
    is_allowed_label, validate_candidate_call, TRAINING_INPUT_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
};

pub(crate) mod alternatives;
pub(crate) mod client;
pub(crate) mod config;
pub(crate) mod io;
pub(crate) mod progress;
pub(crate) mod quality;
pub(crate) mod resume;
pub(crate) mod types;

use alternatives::{alternative_cap, propose_targeted_alternatives};
use client::{reviewer_response_schema, verifier_response_schema, JsonLlmClient};
use config::{resolve_provider_config, resolve_provider_config_for, EnvFile};
use io::{
    append_jsonl, append_reject, append_reject_record, capture_key, count_jsonl_rows,
    ensure_parent_dir, ensure_parent_dir_path, rejects_path, row_key,
};
use progress::ReviewProgress;
use quality::post_review_quality_reject;
use resume::ResumeState;
use types::{
    AlternativeProposal, AlternativeReviewRun, CaptureBatchRun, CaptureJob, CaptureReviewOutcome,
    CaptureWrite, ReviewDecision, ReviewRole, VerifierDecision,
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
    let chunk_size = cli.chunk_size;
    let chunk_label = chunk_size
        .map(|value| value.to_string())
        .unwrap_or_else(|| "disabled".to_string());
    eprintln!(
        "review start input={} output={} rejects={} reviewer={} verifier={} resume={} concurrency={} chunk_size={} captures={}",
        cli.input,
        cli.output,
        reject_path.display(),
        reviewer_label,
        verifier_label,
        cli.resume,
        cli.concurrency,
        chunk_label,
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
    let chunk_limit = chunk_size.unwrap_or(cli.concurrency);
    let log_chunks = chunk_size.is_some();
    let mut chunk = Vec::with_capacity(chunk_limit);
    let mut chunk_index = 0;
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
        let cap_key = capture_key(&capture);
        if resume_state.processed_capture_keys.contains(&cap_key) {
            progress.skipped += 1;
            progress.log_skip_if_needed();
            if resume_state.valid_real_capture_keys.contains(&cap_key) {
                alternative_proposals.extend(propose_targeted_alternatives(&capture));
            }
            continue;
        }

        progress.log_reviewing(&capture);
        chunk.push(CaptureJob {
            row_index: progress.seen,
            capture,
        });
        if chunk.len() >= chunk_limit {
            chunk_index += 1;
            flush_capture_chunk(CaptureChunkRun {
                chunk: &mut chunk,
                chunk_index,
                log_chunk: log_chunks,
                concurrency: cli.concurrency,
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
    if !chunk.is_empty() {
        chunk_index += 1;
    }
    flush_capture_chunk(CaptureChunkRun {
        chunk: &mut chunk,
        chunk_index,
        log_chunk: log_chunks,
        concurrency: cli.concurrency,
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

struct CaptureChunkRun<'a> {
    chunk: &'a mut Vec<CaptureJob>,
    chunk_index: usize,
    log_chunk: bool,
    concurrency: usize,
    reviewer: &'a JsonLlmClient,
    verifier: &'a JsonLlmClient,
    reject_path: &'a Path,
    output_path: &'a str,
    resume_state: &'a mut ResumeState,
    progress: &'a mut ReviewProgress,
    alternative_proposals: &'a mut Vec<AlternativeProposal>,
}

async fn flush_capture_chunk(run: CaptureChunkRun<'_>) -> Result<(), String> {
    let CaptureChunkRun {
        chunk,
        chunk_index,
        log_chunk,
        concurrency,
        reviewer,
        verifier,
        reject_path,
        output_path,
        resume_state,
        progress,
        alternative_proposals,
    } = run;
    if chunk.is_empty() {
        return Ok(());
    }

    let first_row = chunk.first().map(|job| job.row_index).unwrap_or(0);
    let last_row = chunk.last().map(|job| job.row_index).unwrap_or(first_row);
    if log_chunk {
        eprintln!(
            "review chunk {} start rows={}..{} size={} concurrency={}",
            chunk_index,
            first_row,
            last_row,
            chunk.len(),
            concurrency
        );
    }

    let jobs = std::mem::take(chunk);
    let mut batch = Vec::with_capacity(concurrency.min(jobs.len()));
    for job in jobs {
        batch.push(job);
        if batch.len() >= concurrency {
            flush_capture_batch(CaptureBatchRun {
                batch: &mut batch,
                reviewer,
                verifier,
                reject_path,
                output_path,
                resume_state,
                progress,
                alternative_proposals,
            })
            .await?;
        }
    }
    flush_capture_batch(CaptureBatchRun {
        batch: &mut batch,
        reviewer,
        verifier,
        reject_path,
        output_path,
        resume_state,
        progress,
        alternative_proposals,
    })
    .await?;

    if log_chunk {
        eprintln!(
            "review chunk {} complete rows={}..{}",
            chunk_index, first_row, last_row
        );
    }
    Ok(())
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

fn training_input_for_review(capture: &Value, candidate_call: &Value) -> Value {
    json!({
        "schema_version": TRAINING_INPUT_SCHEMA_VERSION,
        "user_request": capture.get("user_request").cloned().unwrap_or(Value::Null),
        "workflow_state": capture.get("workflow_state").cloned().unwrap_or(Value::Null),
        "available_tools": capture.get("available_tools").cloned().unwrap_or_else(|| json!([])),
        "candidate_call": candidate_call,
    })
}

async fn review_capture(client: &JsonLlmClient, capture: &Value) -> Result<ReviewDecision, String> {
    let system = concat!(
        "You label Forge tool-call verifier examples. Return only JSON. ",
        "Allowed labels are valid, wrong_tool_semantic, wrong_arguments_semantic, ",
        "tool_not_needed, needs_clarification. Do not use classifier telemetry as truth. ",
        "Judge only the serialized training input: user request, workflow state, available tools, ",
        "and candidate call. Do not rely on tool_result, provenance, run index, or captured outcome; ",
        "those fields are not available to the classifier. Do not mark a tool wrong just because ",
        "another read/inspect tool could also have been used. Terminal tools indicate ",
        "how to answer after tool work is done; they do not forbid a still-useful non-terminal tool. ",
        "If the same non-terminal tool has already completed and there are no pending steps, another ",
        "call to that tool is usually tool_not_needed, not valid. For shopping compare workflows, ",
        "search_products followed by compare_products with product IDs from the search result is valid; ",
        "do not invent a required get_product step unless the user specifically asked for product details."
    );
    let candidate_call = capture
        .get("candidate_call")
        .cloned()
        .unwrap_or(Value::Null);
    let review_input = training_input_for_review(capture, &candidate_call);
    let user = format!(
        "Review this serialized training input and return JSON with \
         label, confidence, rationale, and optional corrected_candidate_call.\n\n{}",
        serde_json::to_string_pretty(&review_input).map_err(|err| err.to_string())?
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
        "serialized training input: user request, workflow state, available tools, and candidate call. ",
        "Do not rely on tool_result, provenance, run index, or captured outcome; those fields are not ",
        "available to the classifier. Reject labels that over-penalize normal multi-step tool use. A successful ",
        "search_products then compare_products sequence is valid for a comparison request. ",
        "If a candidate repeats an already completed non-terminal tool with no pending work, ",
        "the valid label should be rejected."
    );
    let user = json!({
        "task": "Accept or reject the proposed label. Do not relabel.",
        "proposed_label": label,
        "reviewer_rationale": rationale,
        "serialized_training_input": training_input_for_review(capture, candidate_call)
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
        "be semantically wrong for the serialized training input. Judge only the user request, ",
        "workflow state, available tools, and generated candidate call. If the proposed label is ",
        "not clearly correct, return the better label or needs_clarification."
    );
    let user = json!({
        "task": "Review this generated non-valid training alternative.",
        "proposed_label": proposal.label,
        "serialized_training_input": training_input_for_review(&proposal.capture, &proposal.candidate_call)
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
        let r_key = row_key(
            &proposal.example_group_id,
            "targeted_alternative",
            &proposal.candidate_call,
        );
        if resume_state.processed_row_keys.contains(&r_key) {
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
    let cap_key = capture_key(capture);
    let r_key = row_key(example_group_id_str, source_bucket, &candidate_call);
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
            "capture_key": cap_key,
            "row_key": r_key,
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
    fn review_training_input_omits_capture_only_fields() {
        let capture = capture_row();
        let input = training_input_for_review(&capture, &capture["candidate_call"]);

        assert_eq!(input["schema_version"], TRAINING_INPUT_SCHEMA_VERSION);
        assert!(input.get("tool_result").is_none());
        assert!(input.get("proxy_trace").is_none());
        assert!(input.get("metadata").is_none());
        assert_eq!(input["candidate_call"], capture["candidate_call"]);
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
}
