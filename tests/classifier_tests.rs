use std::sync::{Arc, Mutex};

use forge_guardrails::guardrails::{
    ArtifactManifest, ClassifierAction, LabelsFile, ScorerMode, ScoringContext, TerminalTool,
    Thresholds, ToolCallClass, ToolCallScore, ToolCallScorer, WorkflowStateForScoring,
};
use forge_guardrails::streaming::{LLMResponse, ToolCall};
use forge_guardrails::{serialize_state_v1, ToolSpecForScoring};
use indexmap::IndexMap;
use serde_json::{json, Value};

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
    }
}

#[test]
fn serialize_state_matches_hf_fixture() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/classifier/serializer_fixture.json"))
            .expect("fixture");
    let ctx = scoring_context_from_fixture(&fixture);
    let candidate = candidate_from_fixture(&fixture);

    let actual = serialize_state_v1(&ctx, &candidate);

    assert_eq!(actual, fixture["serialized"].as_str().expect("serialized"));
}

#[test]
fn labels_match_expected_order() {
    let labels: LabelsFile =
        serde_json::from_str(include_str!("fixtures/classifier/labels.json")).expect("labels");

    labels.validate().expect("valid labels");
    assert_eq!(
        labels.labels,
        vec![
            "valid",
            "wrong_tool_semantic",
            "tool_not_needed",
            "needs_clarification",
            "deterministic_invalid",
        ]
    );
}

#[test]
fn thresholds_parse_and_default_to_shadow() {
    let thresholds: Thresholds =
        serde_json::from_str(include_str!("fixtures/classifier/thresholds.json"))
            .expect("thresholds");

    thresholds.validate().expect("valid thresholds");
    let tool_not_needed = thresholds.for_label(&ToolCallClass::ToolNotNeeded);
    assert_eq!(tool_not_needed.advisory_min_confidence, 0.8);
    assert_eq!(tool_not_needed.enforce_min_confidence, 0.95);

    let unknown = thresholds.for_label(&ToolCallClass::Unknown("new_label".to_string()));
    assert_eq!(unknown.action, "shadow_only");
    assert_eq!(unknown.advisory_min_confidence, 1.01);
    assert_eq!(unknown.enforce_min_confidence, 1.01);
}

#[test]
fn manifest_rejects_wrong_serializer() {
    let mut raw: Value =
        serde_json::from_str(include_str!("fixtures/classifier/artifact_manifest.json"))
            .expect("manifest");
    raw["serializer"] = json!("serialize_state_v2");
    let manifest: ArtifactManifest = serde_json::from_value(raw).expect("manifest");

    let err = manifest.validate().expect_err("wrong serializer rejected");

    assert!(err
        .to_string()
        .contains("unsupported classifier serializer"));
}

#[test]
fn manifest_rejects_wrong_schema_versions() {
    let raw: Value =
        serde_json::from_str(include_str!("fixtures/classifier/artifact_manifest.json"))
            .expect("manifest");

    let mut artifact_raw = raw.clone();
    artifact_raw["artifact_schema_version"] = json!("toolcall-verifier-artifact/v0");
    let artifact_manifest: ArtifactManifest =
        serde_json::from_value(artifact_raw).expect("artifact manifest");
    let artifact_err = artifact_manifest
        .validate()
        .expect_err("wrong artifact schema rejected");
    assert!(artifact_err
        .to_string()
        .contains("unsupported classifier artifact schema"));

    let mut input_raw = raw;
    input_raw["input_schema_version"] = json!("toolcall-verifier-input/v0");
    let input_manifest: ArtifactManifest =
        serde_json::from_value(input_raw).expect("input manifest");
    let input_err = input_manifest
        .validate()
        .expect_err("wrong input schema rejected");
    assert!(input_err
        .to_string()
        .contains("unsupported classifier input schema"));
}

struct FakeScorer {
    calls: Arc<Mutex<usize>>,
    fail: bool,
}

impl ToolCallScorer for FakeScorer {
    fn score(&self, _ctx: &ScoringContext, _candidate: &ToolCall) -> anyhow::Result<ToolCallScore> {
        *self.calls.lock().expect("calls lock") += 1;
        if self.fail {
            anyhow::bail!("fake scorer failure");
        }
        Ok(ToolCallScore {
            label: ToolCallClass::ToolNotNeeded,
            confidence: 0.99,
            logits: vec![0.0, 0.0, 9.0, 0.0, 0.0],
            action: ClassifierAction::ShadowOnly,
            model_version: "fake".to_string(),
            latency_ms: 1.0,
        })
    }
}

fn guardrails_with_scorer(scorer: Arc<dyn ToolCallScorer>) -> forge_guardrails::Guardrails {
    forge_guardrails::Guardrails::new(
        vec!["search".into(), "respond".into()],
        TerminalTool::Single("respond".into()),
        None,
        None,
        3,
        2,
        true,
        3,
        None,
    )
    .with_scorer(scorer)
}

fn minimal_scoring_context() -> ScoringContext {
    ScoringContext {
        schema_version: "toolcall-verifier-input/v1".to_string(),
        user_request: "Find the answer.".to_string(),
        workflow_state: WorkflowStateForScoring {
            required_steps: Vec::new(),
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            terminal_tools: vec!["respond".to_string()],
            recent_errors: Vec::new(),
        },
        available_tools: vec![ToolSpecForScoring {
            name: "search".to_string(),
            description: "Search.".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }],
    }
}

#[test]
fn fake_scorer_runs_only_after_deterministic_checks_pass() {
    let calls = Arc::new(Mutex::new(0));
    let scorer = Arc::new(FakeScorer {
        calls: calls.clone(),
        fail: false,
    });
    let mut guardrails = guardrails_with_scorer(scorer);
    let ctx = minimal_scoring_context();

    let blocked = guardrails.check_with_scoring_context(
        &LLMResponse::ToolCalls(vec![ToolCall::new("missing", IndexMap::new())]),
        &ctx,
    );
    assert_eq!(blocked.action, forge_guardrails::GuardAction::Retry);
    assert_eq!(*calls.lock().expect("calls lock"), 0);

    let allowed = guardrails.check_with_scoring_context(
        &LLMResponse::ToolCalls(vec![ToolCall::new("search", IndexMap::new())]),
        &ctx,
    );
    assert_eq!(allowed.action, forge_guardrails::GuardAction::Execute);
    assert_eq!(*calls.lock().expect("calls lock"), 1);
    assert_eq!(guardrails.last_scores().len(), 1);
}

#[test]
fn fake_scorer_failure_allows_deterministic_path() {
    let calls = Arc::new(Mutex::new(0));
    let scorer = Arc::new(FakeScorer {
        calls: calls.clone(),
        fail: true,
    });
    let mut guardrails = guardrails_with_scorer(scorer);
    let ctx = minimal_scoring_context();

    let result = guardrails.check_with_scoring_context(
        &LLMResponse::ToolCalls(vec![ToolCall::new("search", IndexMap::new())]),
        &ctx,
    );

    assert_eq!(result.action, forge_guardrails::GuardAction::Execute);
    assert_eq!(*calls.lock().expect("calls lock"), 1);
    assert!(guardrails.last_scores().is_empty());
}

#[test]
fn scorer_mode_parses_stable_names() {
    assert_eq!("shadow".parse::<ScorerMode>().unwrap(), ScorerMode::Shadow);
    assert!("block".parse::<ScorerMode>().is_err());
}
