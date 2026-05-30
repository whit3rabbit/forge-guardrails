use indexmap::IndexMap;
use serde_json::Value;

use super::cache::ScoreCache;
use super::final_response::OnnxFinalResponseScorer;
use super::softmax;
use super::tool_call::OnnxToolCallScorer;
use crate::clients::base::ToolCall;
use crate::guardrails::scoring::{
    FinalResponseContext, FinalResponseScorer, FinalResponseToolResult, ScorerMode, ToolCallScorer,
};
use crate::guardrails::scoring_context::{ScoringContext, WorkflowStateForScoring};

#[test]
fn softmax_sums_to_one() {
    let probs = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.00001);
}

#[test]
fn score_cache_returns_cached_values_and_refreshes_recency() {
    let mut cache = ScoreCache::new(2);
    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);

    assert_eq!(cache.get("a"), Some(1));
    cache.insert("c".to_string(), 3);

    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get("a"), Some(1));
    assert_eq!(cache.get("b"), None);
    assert_eq!(cache.get("c"), Some(3));
}

#[test]
fn score_cache_does_not_store_when_capacity_is_zero() {
    let mut cache = ScoreCache::new(0);

    cache.insert("a".to_string(), 1);

    assert_eq!(cache.len(), 0);
    assert_eq!(cache.get("a"), None);
}

#[test]
fn onnx_fixture_scores_without_panic_when_test_dir_is_set() {
    let Ok(dir) = std::env::var("FORGE_CLASSIFIER_TEST_DIR") else {
        return;
    };
    let scorer =
        OnnxToolCallScorer::from_dir(dir.as_str(), Some(ScorerMode::Shadow)).expect("scorer");
    let fixture_path = std::path::Path::new(&dir).join("serializer_fixture.json");
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).expect("serializer fixture"))
            .expect("serializer fixture json");
    let ctx = scoring_context_from_fixture(&fixture);
    let candidate = candidate_from_fixture(&fixture);
    let expected_logits = scorer.labels.len();
    let score = scorer.score(&ctx, &candidate).expect("score");

    assert_eq!(score.logits.len(), expected_logits);
}

#[test]
fn final_response_onnx_fixture_scores_without_panic_when_test_dir_is_set() {
    let Ok(dir) = std::env::var("FORGE_FINAL_RESPONSE_CLASSIFIER_TEST_DIR") else {
        return;
    };
    let scorer = OnnxFinalResponseScorer::from_dir(dir.as_str(), Some(ScorerMode::Shadow))
        .expect("final-response scorer");
    let ctx = FinalResponseContext {
        schema_version: "final-response-verifier-input/v1".to_string(),
        user_request: "Summarize the lookup result.".to_string(),
        workflow_state: WorkflowStateForScoring {
            required_steps: vec!["lookup".to_string()],
            completed_steps: vec!["lookup".to_string()],
            pending_steps: Vec::new(),
            terminal_tools: vec!["respond".to_string()],
            recent_errors: Vec::new(),
        },
        required_facts: vec!["Paris is the capital of France.".to_string()],
        tool_trace: vec!["lookup".to_string()],
        tool_results: vec![FinalResponseToolResult {
            tool_name: "lookup".to_string(),
            content: "Paris is the capital of France.".to_string(),
        }],
        candidate_final_response: "Paris is the capital of France.".to_string(),
        metadata: None,
    };
    let expected_logits = scorer.labels.len();
    let score = scorer.score(&ctx).expect("final-response score");

    assert_eq!(score.logits.len(), expected_logits);
}

fn scoring_context_from_fixture(value: &Value) -> ScoringContext {
    ScoringContext {
        schema_version: value["input"]["schema_version"]
            .as_str()
            .expect("schema_version")
            .to_string(),
        user_request: value["input"]["user_request"]
            .as_str()
            .expect("user_request")
            .to_string(),
        workflow_state: serde_json::from_value(value["input"]["workflow_state"].clone())
            .expect("workflow_state"),
        available_tools: serde_json::from_value(value["input"]["available_tools"].clone())
            .expect("available_tools"),
        metadata: None,
    }
}

fn candidate_from_fixture(value: &Value) -> ToolCall {
    let candidate = &value["input"]["candidate_call"];
    let args = candidate["arguments"]
        .as_object()
        .expect("arguments object")
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<IndexMap<_, _>>();
    ToolCall::new(candidate["name"].as_str().expect("name"), args)
}
