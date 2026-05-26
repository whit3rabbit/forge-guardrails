use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use forge_guardrails::{ForgeError, Message, MessageType};
use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::cli::Cli;
use crate::scenarios::SmokeScenario;
use crate::startup::default_mode;

pub(crate) struct ClassifierReport<'a> {
    pub(crate) mode: &'a str,
    pub(crate) scores: &'a [Value],
}

pub(crate) struct FinalResponseReport<'a> {
    pub(crate) mode: &'a str,
    pub(crate) scores: &'a [Value],
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn row_for_result(
    backend: &str,
    model: &str,
    ablation: &str,
    cli: &Cli,
    scenario: &SmokeScenario,
    run_idx: usize,
    iterations: i32,
    elapsed: f64,
    result: Result<Value, ForgeError>,
    messages: &[Message],
    compaction_events: usize,
    classifier_report: Option<ClassifierReport<'_>>,
    final_response_report: Option<FinalResponseReport<'_>>,
) -> Value {
    let captured_args = scenario
        .capture
        .lock()
        .expect("capture lock")
        .clone()
        .unwrap_or_default();
    let final_text = terminal_text(&captured_args);
    let accuracy = result
        .as_ref()
        .ok()
        .map(|_| validate_scenario(&scenario.name, &final_text));
    let completeness = result.is_ok();
    let success = completeness && accuracy != Some(false);
    let (error_type, error_message, raw_response) = match &result {
        Ok(_) => (Value::Null, Value::Null, Value::Null),
        Err(err) => (
            json!(error_kind(err)),
            json!(err.to_string()),
            match err {
                ForgeError::ToolCall(tool_err) => tool_err
                    .raw_response
                    .as_ref()
                    .map(|raw| json!(raw))
                    .unwrap_or(Value::Null),
                _ => Value::Null,
            },
        ),
    };
    let stats = message_stats(messages);
    let (tool_sequence, tool_args) = tool_trace(messages);
    let missing_required_steps =
        missing_required_steps(&scenario.workflow.required_steps, messages);
    let required_step_mismatch = !missing_required_steps.is_empty();
    let ideal_iterations = scenario.workflow.required_steps.len() as i32 + 1;
    let wasted_calls = if completeness {
        json!((iterations - ideal_iterations).max(0))
    } else {
        Value::Null
    };

    let mut row = json!({
        "impl": "rust",
        "model": model,
        "backend": backend,
        "mode": cli.mode.clone().unwrap_or_else(|| default_mode(backend).to_string()),
        "ablation": ablation,
        "tool_choice": "auto",
        "scenario": scenario.name,
        "scenario_family": scenario.name,
        "run": run_idx,
        "stream": cli.stream,
        "completeness": completeness,
        "success": success,
        "accuracy": accuracy,
        "iterations": iterations,
        "ideal_iterations": ideal_iterations,
        "wasted_calls": wasted_calls,
        "elapsed_s": (elapsed * 100.0).round() / 100.0,
        "error_type": error_type,
        "failure_kind": error_type.clone(),
        "error_message": error_message,
        "budget_tokens": cli.num_ctx,
        "compaction_events": compaction_events,
        "retry_nudges": stats.retry_nudges,
        "step_nudges": stats.step_nudges,
        "tool_errors": stats.tool_errors,
        "reasoning_msgs": stats.reasoning_msgs,
        "tool_sequence": tool_sequence,
        "tool_args": tool_args,
        "missing_required_steps": missing_required_steps,
        "required_step_mismatch": required_step_mismatch,
        "final_text": final_text,
        "raw_response_on_failure": raw_response,
        "reasoning_budget": cli.reasoning_budget,
    });
    if let Some(report) = classifier_report {
        add_classifier_fields(&mut row, report);
    }
    if let Some(report) = final_response_report {
        add_final_response_fields(&mut row, report);
    }
    row
}

pub(crate) fn write_row(output: Option<&str>, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    if let Some(path) = output {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| err.to_string())?;
        writeln!(file, "{line}").map_err(|err| err.to_string())
    } else {
        println!("{line}");
        Ok(())
    }
}

pub(crate) fn write_hard_negatives(
    output: Option<&str>,
    row: &Value,
    scenario: &SmokeScenario,
) -> Result<(), String> {
    let Some(output) = output else {
        return Ok(());
    };
    let Some(corrected_positive) = scenario.corrected_positive.as_ref() else {
        return Ok(());
    };
    if row.get("success").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(());
    }

    let outcome = json!({
        "scenario": scenario.name,
        "scenario_family": scenario.name,
        "run": row.get("run").cloned().unwrap_or(Value::Null),
        "failure_kind": row.get("failure_kind").cloned().unwrap_or(Value::Null),
        "accuracy": row.get("accuracy").cloned().unwrap_or(Value::Null),
        "corrected_positive": corrected_positive,
    });

    let tool_row = json!({
        "kind": "tool_call",
        "candidate": {
            "tool_sequence": row.get("tool_sequence").cloned().unwrap_or(Value::Null),
            "tool_args": row.get("tool_args").cloned().unwrap_or(Value::Null),
        },
        "classifier_scores": row.get("classifier_scores").cloned().unwrap_or_else(|| json!([])),
        "outcome": outcome.clone(),
    });
    append_jsonl(
        &hard_negative_path(output, "tool_call_hard_negatives"),
        &tool_row,
    )?;

    let final_row = json!({
        "kind": "final_response",
        "candidate": {
            "final_text": row.get("final_text").cloned().unwrap_or(Value::Null),
        },
        "final_response_classifier_scores": row
            .get("final_response_classifier_scores")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "outcome": outcome,
    });
    append_jsonl(
        &hard_negative_path(output, "final_response_hard_negatives"),
        &final_row,
    )
}

fn terminal_text(args: &IndexMap<String, Value>) -> String {
    ["message", "content", "findings"]
        .iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

fn validate_scenario(name: &str, text: &str) -> bool {
    let normalized = text.to_lowercase().replace(',', "");
    match name {
        "basic_2step" => normalized.contains("paris") && normalized.contains("capital"),
        "sequential_3step" => {
            normalized.contains("23")
                && normalized.contains("widget pro")
                && normalized.contains("apac")
        }
        "error_recovery" => normalized.contains("10") && normalized.contains("record"),
        "inconsistent_api_recovery_stateful" => {
            normalized.contains("acc-12345") && normalized.contains("pass")
        }
        _ => false,
    }
}

#[derive(Default)]
struct MessageStats {
    retry_nudges: usize,
    step_nudges: usize,
    tool_errors: usize,
    reasoning_msgs: usize,
}

fn message_stats(messages: &[Message]) -> MessageStats {
    let mut stats = MessageStats::default();
    for message in messages {
        match message.metadata.msg_type {
            MessageType::RetryNudge => stats.retry_nudges += 1,
            MessageType::StepNudge => stats.step_nudges += 1,
            MessageType::ToolResult if message.content.contains("[ToolError]") => {
                stats.tool_errors += 1
            }
            MessageType::Reasoning => stats.reasoning_msgs += 1,
            _ => {}
        }
    }
    stats
}

fn tool_trace(messages: &[Message]) -> (Vec<Value>, Vec<Value>) {
    let mut names = Vec::new();
    let mut args = Vec::new();
    for message in messages {
        if message.metadata.msg_type != MessageType::ToolCall {
            continue;
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                names.push(json!(call.name));
                args.push(json!(call.args));
            }
        }
    }
    (names, args)
}

fn missing_required_steps(required_steps: &[String], messages: &[Message]) -> Vec<Value> {
    let mut called = indexmap::IndexSet::new();
    for message in messages {
        if message.metadata.msg_type != MessageType::ToolCall {
            continue;
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                called.insert(call.name.clone());
            }
        }
    }
    required_steps
        .iter()
        .filter(|step| !called.contains(step.as_str()))
        .map(|step| json!(step))
        .collect()
}

fn error_kind(err: &ForgeError) -> &'static str {
    match err {
        ForgeError::UnsupportedModel(_) => "UnsupportedModelError",
        ForgeError::ToolCall(_) => "ToolCallError",
        ForgeError::ToolExecution(_) => "ToolExecutionError",
        ForgeError::WorkflowCancelled(_) => "WorkflowCancelledError",
        ForgeError::MaxIterations(_) => "MaxIterationsError",
        ForgeError::StepEnforcement(_) => "StepEnforcementError",
        ForgeError::Prerequisite(_) => "PrerequisiteError",
        ForgeError::ContextBudgetExceeded(_) => "ContextBudgetExceeded",
        ForgeError::HardwareDetection(_) => "HardwareDetectionError",
        ForgeError::ContextDiscovery(_) => "ContextDiscoveryError",
        ForgeError::BudgetResolution(_) => "BudgetResolutionError",
        ForgeError::Backend(_) => "BackendError",
        ForgeError::Stream(_) => "StreamError",
    }
}

fn add_classifier_fields(row: &mut Value, report: ClassifierReport<'_>) {
    let max_score = report.scores.iter().max_by(|left, right| {
        let left_conf = left
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        let right_conf = right
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        left_conf.total_cmp(&right_conf)
    });
    let model_version = report
        .scores
        .iter()
        .find_map(|score| score.get("model_version").and_then(Value::as_str));

    if let Some(obj) = row.as_object_mut() {
        obj.insert("classifier_enabled".to_string(), json!(true));
        obj.insert("classifier_mode".to_string(), json!(report.mode));
        obj.insert(
            "classifier_model_version".to_string(),
            model_version.map_or(Value::Null, |value| json!(value)),
        );
        obj.insert("classifier_scores".to_string(), json!(report.scores));
        obj.insert(
            "classifier_max_confidence".to_string(),
            max_score
                .and_then(|score| score.get("confidence").cloned())
                .unwrap_or(Value::Null),
        );
        obj.insert(
            "classifier_predicted_label".to_string(),
            max_score
                .and_then(|score| score.get("label").cloned())
                .unwrap_or(Value::Null),
        );
    }
}

fn add_final_response_fields(row: &mut Value, report: FinalResponseReport<'_>) {
    let max_score = report.scores.iter().max_by(|left, right| {
        let left_conf = left
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        let right_conf = right
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NEG_INFINITY);
        left_conf.total_cmp(&right_conf)
    });
    let model_version = report
        .scores
        .iter()
        .find_map(|score| score.get("model_version").and_then(Value::as_str));

    if let Some(obj) = row.as_object_mut() {
        obj.insert("final_response_classifier_enabled".to_string(), json!(true));
        obj.insert(
            "final_response_classifier_mode".to_string(),
            json!(report.mode),
        );
        obj.insert(
            "final_response_classifier_model_version".to_string(),
            model_version.map_or(Value::Null, |value| json!(value)),
        );
        obj.insert(
            "final_response_classifier_scores".to_string(),
            json!(report.scores),
        );
        obj.insert(
            "final_response_classifier_max_confidence".to_string(),
            max_score
                .and_then(|score| score.get("confidence").cloned())
                .unwrap_or(Value::Null),
        );
        obj.insert(
            "final_response_classifier_predicted_label".to_string(),
            max_score
                .and_then(|score| score.get("label").cloned())
                .unwrap_or(Value::Null),
        );
    }
}

fn append_jsonl(path: &Path, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    writeln!(file, "{line}").map_err(|err| err.to_string())
}

fn hard_negative_path(output: &str, suffix: &str) -> PathBuf {
    let path = Path::new(output);
    let stem = path.file_stem().and_then(|value| value.to_str());
    let extension = path.extension().and_then(|value| value.to_str());
    let file_name = match (stem, extension) {
        (Some(stem), Some(extension)) => format!("{stem}.{suffix}.{extension}"),
        (Some(stem), None) => format!("{stem}.{suffix}"),
        _ => format!("{output}.{suffix}.jsonl"),
    };
    path.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::parse_args;
    use crate::scenarios::build_scenario;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn parse(items: &[&str]) -> Cli {
        parse_args(items.iter().map(|item| item.to_string())).expect("parse")
    }

    #[test]
    fn row_includes_budget_and_batch_compat_fields() {
        let cli = parse(&["--num-ctx", "16384"]);
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        let row = row_for_result(
            "openai-proxy",
            "test-model",
            "reforged",
            &cli,
            &scenario,
            1,
            4,
            1.234,
            Ok(json!(null)),
            &[],
            2,
            None,
            None,
        );

        assert_eq!(row["budget_tokens"], json!(16384));
        assert_eq!(row["ideal_iterations"], json!(2));
        assert_eq!(row["wasted_calls"], json!(2));
        assert_eq!(row["compaction_events"], json!(2));
        assert_eq!(row["missing_required_steps"], json!(["get_country_info"]));
        assert_eq!(row["required_step_mismatch"], json!(true));
        assert!(row.get("classifier_enabled").is_none());
    }

    #[test]
    fn row_includes_classifier_fields_only_when_reported() {
        let cli = parse(&[]);
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        let scores = vec![json!({
            "tool": "report",
            "label": "wrong_tool_semantic",
            "confidence": 0.997,
            "action": "shadow_only",
            "latency_ms": 3.8,
            "model_version": "test-model"
        })];
        let row = row_for_result(
            "openai-proxy",
            "test-model",
            "reforged",
            &cli,
            &scenario,
            1,
            2,
            0.5,
            Ok(json!(null)),
            &[],
            0,
            Some(ClassifierReport {
                mode: "shadow",
                scores: &scores,
            }),
            None,
        );

        assert_eq!(row["classifier_enabled"], json!(true));
        assert_eq!(row["classifier_mode"], json!("shadow"));
        assert_eq!(row["classifier_model_version"], json!("test-model"));
        assert_eq!(
            row["classifier_predicted_label"],
            json!("wrong_tool_semantic")
        );
        assert_eq!(row["classifier_max_confidence"], json!(0.997));
        assert_eq!(row["classifier_scores"], json!(scores));
    }

    #[test]
    fn row_includes_final_response_classifier_fields_only_when_reported() {
        let cli = parse(&[]);
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        let scores = vec![json!({
            "label": "missing_tool_fact",
            "confidence": 0.91,
            "action": "advisory_nudge",
            "latency_ms": 2.5,
            "model_version": "final-test-model"
        })];
        let row = row_for_result(
            "openai-proxy",
            "test-model",
            "reforged",
            &cli,
            &scenario,
            1,
            2,
            0.5,
            Ok(json!(null)),
            &[],
            0,
            None,
            Some(FinalResponseReport {
                mode: "advisory",
                scores: &scores,
            }),
        );

        assert_eq!(row["final_response_classifier_enabled"], json!(true));
        assert_eq!(row["final_response_classifier_mode"], json!("advisory"));
        assert_eq!(
            row["final_response_classifier_model_version"],
            json!("final-test-model")
        );
        assert_eq!(
            row["final_response_classifier_predicted_label"],
            json!("missing_tool_fact")
        );
        assert_eq!(row["final_response_classifier_max_confidence"], json!(0.91));
        assert_eq!(row["final_response_classifier_scores"], json!(scores));
    }

    #[test]
    fn write_hard_negatives_creates_sibling_files_for_failed_rows() {
        let mut dir = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        dir.push(format!(
            "forge-eval-report-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("tempdir");
        let output = dir.join("rows.jsonl");
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        let row = json!({
            "scenario": "basic_2step",
            "run": 1,
            "success": false,
            "accuracy": false,
            "failure_kind": "accuracy_false",
            "tool_sequence": ["get_country_info", "summarize"],
            "tool_args": [{"country": "France"}, {"content": "bad"}],
            "final_text": "bad",
            "classifier_scores": [],
            "final_response_classifier_scores": [],
        });

        write_hard_negatives(output.to_str(), &row, &scenario).expect("write");

        let tool_path = dir.join("rows.tool_call_hard_negatives.jsonl");
        let final_path = dir.join("rows.final_response_hard_negatives.jsonl");
        assert!(fs::read_to_string(tool_path)
            .expect("tool hard negatives")
            .contains("\"kind\":\"tool_call\""));
        assert!(fs::read_to_string(final_path)
            .expect("final hard negatives")
            .contains("\"kind\":\"final_response\""));
    }
}
