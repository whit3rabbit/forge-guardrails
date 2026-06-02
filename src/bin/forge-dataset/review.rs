use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

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
    let captures = read_jsonl(&cli.input)?;
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
    let reviewer = JsonLlmClient::new(reviewer_config);
    let verifier = JsonLlmClient::new(verifier_config);

    let mut accepted_real_rows = Vec::new();
    let mut accepted_corrected_rows = Vec::new();
    let mut alternative_proposals = Vec::new();

    for capture in captures {
        let decision = match review_capture(&reviewer, &capture).await {
            Ok(decision) => decision,
            Err(err) => {
                append_reject(&reject_path, "reviewer_rejected", &err, &capture, None)?;
                continue;
            }
        };

        if let Some(corrected) = decision.corrected_candidate_call.as_ref() {
            if let Err(err) = validate_candidate_call(&capture["available_tools"], corrected) {
                append_reject(
                    &reject_path,
                    "invalid_corrected_candidate_call",
                    &err,
                    &capture,
                    Some(decision.raw.clone()),
                )?;
                continue;
            }
        }

        let candidate_call = capture["candidate_call"].clone();
        let verifier_decision = match verify_label(
            &verifier,
            &capture,
            &candidate_call,
            &decision.label,
            &decision.rationale,
        )
        .await
        {
            Ok(decision) => decision,
            Err(err) => {
                append_reject(
                    &reject_path,
                    "verifier_invalid_response",
                    &err,
                    &capture,
                    None,
                )?;
                continue;
            }
        };
        if !verifier_decision.accepted {
            append_reject(
                &reject_path,
                "verifier_rejected",
                &verifier_decision.rationale,
                &capture,
                Some(verifier_decision.raw),
            )?;
            continue;
        }

        let corrected_positive = decision
            .corrected_candidate_call
            .as_ref()
            .filter(|_| decision.label != "valid")
            .map(|call| json!({"candidate_call": call}));
        accepted_real_rows.push(training_row(
            &capture,
            candidate_call,
            &decision.label,
            "real_model_call",
            &decision,
            &verifier_decision,
            corrected_positive,
        ));
        if decision.label == "valid" {
            alternative_proposals.extend(propose_targeted_alternatives(&capture));
        }

        if decision.label != "valid" {
            if let Some(corrected) = decision.corrected_candidate_call.as_ref() {
                let corrected_verifier = match verify_label(
                    &verifier,
                    &capture,
                    corrected,
                    "valid",
                    "Reviewer supplied this corrected candidate call as the positive target.",
                )
                .await
                {
                    Ok(decision) => decision,
                    Err(err) => {
                        append_reject(
                            &reject_path,
                            "corrected_positive_verifier_invalid_response",
                            &err,
                            &capture,
                            None,
                        )?;
                        continue;
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
                    accepted_corrected_rows.push(training_row(
                        &capture,
                        corrected.clone(),
                        "valid",
                        "reviewer_corrected_positive",
                        &corrected_decision,
                        &corrected_verifier,
                        None,
                    ));
                } else {
                    append_reject(
                        &reject_path,
                        "corrected_positive_verifier_rejected",
                        &corrected_verifier.rationale,
                        &capture,
                        Some(corrected_verifier.raw),
                    )?;
                }
            }
        }
    }

    let alternative_cap = alternative_cap(accepted_real_rows.len(), cli.max_alternative_ratio);
    let accepted_alternatives = verify_alternatives(
        &reviewer,
        &verifier,
        alternative_proposals,
        alternative_cap,
        cli.max_alternatives_per_group,
        &reject_path,
    )
    .await?;

    for row in accepted_real_rows
        .into_iter()
        .chain(accepted_corrected_rows)
        .chain(accepted_alternatives)
    {
        append_jsonl(&cli.output, &row)?;
    }

    Ok(())
}

async fn review_capture(client: &JsonLlmClient, capture: &Value) -> Result<ReviewDecision, String> {
    let system = concat!(
        "You label Forge tool-call verifier examples. Return only JSON. ",
        "Allowed labels are valid, wrong_tool_semantic, wrong_arguments_semantic, ",
        "tool_not_needed, needs_clarification. Do not use classifier telemetry as truth."
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
        "Do not invent a different label."
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

async fn verify_alternatives(
    reviewer: &JsonLlmClient,
    verifier: &JsonLlmClient,
    proposals: Vec<AlternativeProposal>,
    alternative_cap: usize,
    max_per_group: usize,
    reject_path: &Path,
) -> Result<Vec<Value>, String> {
    let mut accepted = Vec::new();
    let mut per_group: HashMap<String, usize> = HashMap::new();
    for proposal in proposals {
        if accepted.len() >= alternative_cap {
            break;
        }
        let group_count = *per_group.get(&proposal.example_group_id).unwrap_or(&0);
        if group_count >= max_per_group {
            continue;
        }
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
                continue;
            }
        };
        if verifier_decision.accepted {
            accepted.push(training_row(
                &proposal.capture,
                proposal.candidate_call,
                &proposal.label,
                "targeted_alternative",
                &reviewer_decision,
                &verifier_decision,
                None,
            ));
            *per_group
                .entry(proposal.example_group_id.clone())
                .or_insert(0) += 1;
        } else {
            append_reject(
                reject_path,
                "targeted_alternative_verifier_rejected",
                &verifier_decision.rationale,
                &proposal.capture,
                Some(verifier_decision.raw),
            )?;
        }
    }
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
    let mut row = json!({
        "schema_version": TRAINING_SCHEMA_VERSION,
        "input": {
            "schema_version": TRAINING_INPUT_SCHEMA_VERSION,
            "user_request": capture.get("user_request").cloned().unwrap_or(Value::Null),
            "workflow_state": capture.get("workflow_state").cloned().unwrap_or(Value::Null),
            "available_tools": capture.get("available_tools").cloned().unwrap_or_else(|| json!([])),
            "candidate_call": candidate_call,
            "metadata": {
                "scenario_family": capture
                    .get("proxy_trace")
                    .and_then(|trace| trace.get("domain"))
                    .and_then(Value::as_str)
                    .unwrap_or("forge_dataset"),
                "requires_transform": label == "wrong_arguments_semantic",
                "requires_synthesis": false,
                "requires_all_tool_facts": false,
                "must_acknowledge_missing_data": label == "needs_clarification"
            }
        },
        "label": label,
        "review": {
            "source": "forge-dataset",
            "source_bucket": source_bucket,
            "example_group_id": capture.get("example_group_id").cloned().unwrap_or(Value::Null),
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
        json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0,
            "max_completion_tokens": 1200
        })
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
                    lookup_env(
                        env_file,
                        &[
                            "FORGE_DATASET_MINIMAX_MODEL",
                            "GENERATETD_MINIMAX_MODEL",
                            "MINIMAX_MODEL",
                        ],
                    )
                    .unwrap_or_else(|| {
                        if cli.minimax_model.trim().is_empty() {
                            default_minimax_model().to_string()
                        } else {
                            cli.minimax_model.clone()
                        }
                    })
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
                    lookup_env(
                        env_file,
                        &[
                            "FORGE_DATASET_OPENROUTER_MODEL",
                            "GENERATETD_OPENROUTER_MODEL",
                            "OPENROUTER_MODEL",
                        ],
                    )
                    .unwrap_or_else(|| {
                        if cli.openrouter_model.trim().is_empty() {
                            default_openrouter_model().to_string()
                        } else {
                            cli.openrouter_model.clone()
                        }
                    })
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
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Value>(trimmed)
            .map_err(|err| format!("{path}:{} invalid JSONL row: {err}", index + 1))?;
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
    append_jsonl_path(
        path,
        &json!({
            "schema_version": "forge-dataset-review-reject/v1",
            "reason": reason,
            "detail": detail,
            "example_group_id": capture.get("example_group_id").cloned().unwrap_or(Value::Null),
            "capture": capture,
            "raw_response": raw_response.unwrap_or(Value::Null),
        }),
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
        assert_eq!(row["label"], "valid");
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
}
