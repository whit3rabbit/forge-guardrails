use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use forge_guardrails::{
    final_response_top_k_from_logits, tool_call_top_k_from_logits, ClassifierModelKind,
    CompactEvent, ContextManager, FinalResponseScore, FinalResponseScoreFn, FinalResponseScorer,
    LLMClient, Message, NoCompact, ScorerMode, StreamChunk, ToolCall, ToolCallScore,
    ToolCallScoreFn, ToolCallScorer, WorkflowRunner,
};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::ablation::parse_ablation;
use crate::cli::Cli;
use crate::counting_client::CountingClient;
use crate::report::{
    row_for_result, write_hard_negatives, write_row, ClassifierReport, FinalResponseReport,
};
use crate::scenarios::build_scenario;

pub(crate) async fn run_with_client<C: LLMClient + 'static>(
    client: C,
    cli: &Cli,
    model: &str,
) -> Result<(), String> {
    let client = Arc::new(CountingClient::new(client));
    let scenario_names = if cli.scenarios.is_empty() {
        vec![
            "basic_2step".to_string(),
            "sequential_3step".to_string(),
            "error_recovery".to_string(),
        ]
    } else {
        cli.scenarios.clone()
    };
    let ablation = parse_ablation(&cli.ablation)?;
    let classifier = build_classifier(cli)?;
    let final_response_classifier = build_final_response_classifier(cli)?;

    for scenario_name in &scenario_names {
        for run_idx in 1..=cli.runs {
            let scenario = build_scenario(scenario_name, ablation.use_required_steps)?;
            let emitted: Arc<StdMutex<Vec<Message>>> = Arc::new(StdMutex::new(Vec::new()));
            let emitted_cb = emitted.clone();
            let compactions: Arc<StdMutex<Vec<CompactEvent>>> = Arc::new(StdMutex::new(Vec::new()));
            let compactions_cb = compactions.clone();
            let context = Arc::new(Mutex::new(ContextManager::new(
                Box::new(NoCompact),
                cli.num_ctx,
                Some(Box::new(move |event: &CompactEvent| {
                    compactions_cb
                        .lock()
                        .expect("compaction capture lock")
                        .push(event.clone());
                })),
                None,
                None,
            )));
            let classifier_scores: Arc<StdMutex<Vec<Value>>> = Arc::new(StdMutex::new(Vec::new()));
            let final_response_scores: Arc<StdMutex<Vec<Value>>> =
                Arc::new(StdMutex::new(Vec::new()));
            let mut runner = WorkflowRunner::new(
                client.clone(),
                context,
                15,
                ablation.max_retries,
                2,
                cli.stream,
                Some(Box::new(|_chunk: &StreamChunk| {})),
                Some(Box::new(move |message: &Message| {
                    emitted_cb
                        .lock()
                        .expect("message capture lock")
                        .push(message.clone());
                })),
                ablation.rescue_enabled,
                None,
            );
            if let Some(scorer) = classifier.clone() {
                let scores_cb = classifier_scores.clone();
                let max_latency_ms = cli.classifier_max_latency_ms;
                let callback: Arc<ToolCallScoreFn> =
                    Arc::new(move |call: &ToolCall, score: &ToolCallScore| {
                        if let Some(max_latency_ms) = max_latency_ms {
                            if score.latency_ms > max_latency_ms as f64 {
                                eprintln!(
                                    "warning: tool-call classifier latency {:.1}ms exceeded {}ms",
                                    score.latency_ms, max_latency_ms
                                );
                            }
                        }
                        scores_cb
                            .lock()
                            .expect("classifier score capture lock")
                            .push(classifier_score_json(call, score));
                    });
                runner = runner.with_tool_call_scorer(scorer, Some(callback));
            }
            if let Some(scorer) = final_response_classifier.clone() {
                let scores_cb = final_response_scores.clone();
                let max_latency_ms = cli.final_response_classifier_max_latency_ms;
                let callback: Arc<FinalResponseScoreFn> = Arc::new(move |score| {
                    if let Some(max_latency_ms) = max_latency_ms {
                        if score.latency_ms > max_latency_ms as f64 {
                            eprintln!(
                                "warning: final-response classifier latency {:.1}ms exceeded {}ms",
                                score.latency_ms, max_latency_ms
                            );
                        }
                    }
                    scores_cb
                        .lock()
                        .expect("final response score capture lock")
                        .push(final_response_score_json(score));
                });
                runner = runner.with_final_response_scorer(scorer, Some(callback));
            }

            let before_calls = client.calls();
            let start = Instant::now();
            let result = runner
                .run(&scenario.workflow, &scenario.user_message, None, None, None)
                .await;
            let elapsed = start.elapsed().as_secs_f64();
            let iterations = client.calls() - before_calls;
            let messages = emitted.lock().expect("message capture lock").clone();
            let compaction_events = compactions.lock().expect("compaction capture lock").len();
            let classifier_scores = classifier_scores
                .lock()
                .expect("classifier score capture lock")
                .clone();
            let final_response_scores = final_response_scores
                .lock()
                .expect("final response score capture lock")
                .clone();
            let classifier_report = classifier.as_ref().map(|_| ClassifierReport {
                mode: cli.classifier_mode.as_str(),
                scores: classifier_scores.as_slice(),
            });
            let final_response_report =
                final_response_classifier
                    .as_ref()
                    .map(|_| FinalResponseReport {
                        mode: cli.final_response_classifier_mode.as_str(),
                        scores: final_response_scores.as_slice(),
                    });
            let row = row_for_result(
                &cli.backend,
                model,
                &cli.ablation,
                cli,
                &scenario,
                run_idx,
                iterations,
                elapsed,
                result,
                &messages,
                compaction_events,
                classifier_report,
                final_response_report,
            );
            write_row(cli.output.as_deref(), &row)?;
            write_hard_negatives(cli.output.as_deref(), &row, &scenario, &messages)?;
        }
    }

    Ok(())
}

fn build_classifier(cli: &Cli) -> Result<Option<Arc<dyn ToolCallScorer>>, String> {
    let mode = cli
        .classifier_mode
        .parse::<ScorerMode>()
        .map_err(|err| err.to_string())?;
    if mode == ScorerMode::Disabled {
        return Ok(None);
    }
    let Some(dir) = cli.classifier_dir.as_deref() else {
        return Ok(None);
    };
    let model_kind = cli
        .classifier_model
        .parse::<ClassifierModelKind>()
        .map_err(|err| err.to_string())?;

    #[cfg(feature = "classifier")]
    {
        let scorer =
            forge_guardrails::OnnxToolCallScorer::from_dir_with_model(dir, Some(mode), model_kind)
                .map_err(|err| format!("failed to load classifier artifact: {err}"))?;
        Ok(Some(Arc::new(scorer) as Arc<dyn ToolCallScorer>))
    }

    #[cfg(not(feature = "classifier"))]
    {
        let _ = (dir, model_kind);
        Err("classifier eval requires building with --features classifier".to_string())
    }
}

fn build_final_response_classifier(
    cli: &Cli,
) -> Result<Option<Arc<dyn FinalResponseScorer>>, String> {
    let mode = cli
        .final_response_classifier_mode
        .parse::<ScorerMode>()
        .map_err(|err| err.to_string())?;
    if mode == ScorerMode::Disabled {
        return Ok(None);
    }
    let Some(dir) = cli.final_response_classifier_dir.as_deref() else {
        return Ok(None);
    };
    let model_kind = cli
        .final_response_classifier_model
        .parse::<ClassifierModelKind>()
        .map_err(|err| err.to_string())?;

    #[cfg(feature = "classifier")]
    {
        let scorer = forge_guardrails::OnnxFinalResponseScorer::from_dir_with_model(
            dir,
            Some(mode),
            model_kind,
        )
        .map_err(|err| format!("failed to load final-response classifier artifact: {err}"))?;
        Ok(Some(Arc::new(scorer) as Arc<dyn FinalResponseScorer>))
    }

    #[cfg(not(feature = "classifier"))]
    {
        let _ = (dir, model_kind);
        Err(
            "final-response classifier eval requires building with --features classifier"
                .to_string(),
        )
    }
}

fn classifier_score_json(call: &ToolCall, score: &ToolCallScore) -> Value {
    json!({
        "tool": call.tool.as_str(),
        "label": score.label.as_label().as_ref(),
        "confidence": score.confidence,
        "top_k": tool_call_top_k_from_logits(&score.logits),
        "action": score.action.as_str(),
        "latency_ms": score.latency_ms,
        "model_version": score.model_version.as_str(),
    })
}

fn final_response_score_json(score: &FinalResponseScore) -> Value {
    json!({
        "label": score.label.as_label().as_ref(),
        "confidence": score.confidence,
        "top_k": final_response_top_k_from_logits(&score.logits),
        "action": score.action.as_str(),
        "latency_ms": score.latency_ms,
        "model_version": score.model_version.as_str(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_guardrails::{ClassifierAction, FinalResponseClass, ToolCallClass};
    use indexmap::IndexMap;

    #[test]
    fn classifier_score_json_includes_top_k_probabilities() {
        let score = ToolCallScore {
            label: ToolCallClass::WrongArgumentsSemantic,
            confidence: 0.9,
            logits: vec![0.0, 1.0, 5.0, 2.0, -1.0, 0.5],
            action: ClassifierAction::ShadowOnly,
            model_version: "test".to_string(),
            latency_ms: 1.0,
        };
        let row = classifier_score_json(&ToolCall::new("fetch", IndexMap::new()), &score);

        assert_eq!(row["top_k"][0]["label"], json!("wrong_arguments_semantic"));
        assert_eq!(row["top_k"][0]["logit"], json!(5.0));
        assert!(row["top_k"][0]["confidence"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn final_response_score_json_includes_top_k_probabilities() {
        let score = FinalResponseScore {
            label: FinalResponseClass::FailedToAcknowledgeDataGap,
            confidence: 0.8,
            logits: vec![0.0, 0.5, -1.0, 1.0, 3.0],
            action: ClassifierAction::ShadowOnly,
            model_version: "test-final".to_string(),
            latency_ms: 1.0,
        };
        let row = final_response_score_json(&score);

        assert_eq!(
            row["top_k"][0]["label"],
            json!("failed_to_acknowledge_data_gap")
        );
        assert_eq!(row["top_k"][0]["logit"], json!(3.0));
        assert!(row["top_k"][0]["confidence"].as_f64().unwrap() > 0.0);
    }
}
