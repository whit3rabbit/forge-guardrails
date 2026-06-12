use std::sync::{Arc, Mutex};

use forge_guardrails::guardrails::{
    final_response_top_k_from_logits, score_final_response_async, score_tool_call_async,
    serialize_final_response_state_v1, tool_call_top_k_from_logits, ArtifactManifest,
    ClassifierAction, ClassifierArtifact, FinalResponseClass, FinalResponseClassifierArtifact,
    FinalResponseContext, FinalResponseScore, FinalResponseScorer, FinalResponseToolResult,
    LabelsFile, NoopFinalResponseScorer, ScorerMode, ScoringContext, ScoringMetadata,
    ScoringPipeline, TerminalTool, Thresholds, ToolCallClass, ToolCallScore, ToolCallScorer,
    WorkflowStateForScoring,
};
use forge_guardrails::streaming::{LLMResponse, ToolCall};
use forge_guardrails::{
    serialize_state_v1, serialize_state_v2, serialize_state_v3, ToolSpecForScoring,
};
#[cfg(feature = "classifier")]
use forge_guardrails::{OnnxScorerOptions, MAX_ONNX_SESSION_POOL_SIZE};
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    let metadata: Option<ScoringMetadata> = value["input"]
        .get("metadata")
        .and_then(|m| serde_json::from_value(m.clone()).ok());
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
        metadata,
    }
}

fn labels_file(labels: &[&str]) -> Value {
    let label_values = labels.iter().map(|label| json!(label)).collect::<Vec<_>>();
    let label2id = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.to_string(), json!(index)))
        .collect::<serde_json::Map<_, _>>();
    let id2label = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (index.to_string(), json!(label)))
        .collect::<serde_json::Map<_, _>>();

    json!({
        "label_mode": "production",
        "labels": label_values,
        "label2id": label2id,
        "id2label": id2label,
    })
}

fn legacy_thresholds() -> Value {
    json!({
        "schema_version": "toolcall-verifier-thresholds/v1",
        "mode": "shadow",
        "default_action": "allow",
        "labels": {
            "valid": {
                "action": "allow",
                "advisory_min_confidence": 0.0,
                "enforce_min_confidence": 1.01
            },
            "wrong_tool_semantic": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 1.01,
                "enforce_min_confidence": 1.01
            },
            "tool_not_needed": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.8,
                "enforce_min_confidence": 0.95
            },
            "needs_clarification": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 1.01,
                "enforce_min_confidence": 1.01
            },
            "deterministic_invalid": {
                "action": "deterministic_only",
                "advisory_min_confidence": 1.01,
                "enforce_min_confidence": 1.01
            }
        }
    })
}

fn six_label_thresholds() -> Value {
    let mut thresholds = legacy_thresholds();
    thresholds["labels"]["wrong_arguments_semantic"] = json!({
        "action": "advisory_then_enforce_after_eval",
        "advisory_min_confidence": 0.90,
        "enforce_min_confidence": 0.995
    });
    thresholds
}

fn write_artifact_dir(labels: Value, thresholds: Value) -> anyhow::Result<PathBuf> {
    let mut dir = std::env::temp_dir();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    dir.push(format!(
        "forge-classifier-test-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)?;
    let manifest = json!({
        "artifact_schema_version": "toolcall-verifier-artifact/v1",
        "model_kind": "classifier",
        "base_model": "test",
        "label_mode": "production",
        "input_schema_version": "toolcall-verifier-input/v1",
        "serializer": "serialize_state_v1",
        "max_length": 128,
        "onnx_file": "model.onnx",
        "quantized_onnx_file": "model.quant.onnx",
        "labels": labels["labels"].clone(),
    });
    std::fs::write(
        dir.join("artifact_manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    std::fs::write(dir.join("labels.json"), serde_json::to_vec(&labels)?)?;
    std::fs::write(
        dir.join("thresholds.json"),
        serde_json::to_vec(&thresholds)?,
    )?;
    Ok(dir)
}

fn final_response_thresholds() -> Value {
    json!({
        "schema_version": "final-response-verifier-thresholds/v1",
        "mode": "shadow",
        "default_action": "allow",
        "labels": {
            "valid_final_response": {
                "action": "allow",
                "advisory_min_confidence": 0.0,
                "enforce_min_confidence": 1.01
            },
            "missing_tool_fact": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.90,
                "enforce_min_confidence": 0.995
            },
            "contradicts_tool_result": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.90,
                "enforce_min_confidence": 0.995
            },
            "unsupported_claim": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.90,
                "enforce_min_confidence": 0.995
            },
            "failed_to_acknowledge_data_gap": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.90,
                "enforce_min_confidence": 0.995
            }
        }
    })
}

fn write_final_response_artifact_dir() -> anyhow::Result<PathBuf> {
    let labels = labels_file(&[
        "valid_final_response",
        "missing_tool_fact",
        "contradicts_tool_result",
        "unsupported_claim",
        "failed_to_acknowledge_data_gap",
    ]);
    let mut dir = std::env::temp_dir();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    dir.push(format!(
        "forge-final-classifier-test-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)?;
    let manifest = json!({
        "artifact_schema_version": "final-response-verifier-artifact/v1",
        "model_kind": "classifier",
        "base_model": "test",
        "label_mode": "production",
        "input_schema_version": "final-response-verifier-input/v1",
        "serializer": "serialize_final_response_state_v1",
        "max_length": 128,
        "onnx_file": "model.onnx",
        "quantized_onnx_file": "model.quant.onnx",
        "labels": labels["labels"].clone(),
    });
    std::fs::write(
        dir.join("artifact_manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    std::fs::write(dir.join("labels.json"), serde_json::to_vec(&labels)?)?;
    std::fs::write(
        dir.join("thresholds.json"),
        serde_json::to_vec(&final_response_thresholds())?,
    )?;
    Ok(dir)
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
fn serialize_state_v3_matches_hf_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/classifier/serializer_fixture_v3.json"
    ))
    .expect("fixture");
    let ctx = scoring_context_from_fixture(&fixture);
    let candidate = candidate_from_fixture(&fixture);

    let actual = serialize_state_v3(&ctx, &candidate);

    assert_eq!(actual, fixture["serialized"].as_str().expect("serialized"));
}

#[test]
fn serialize_final_response_state_v1_matches_fixture_text() {
    let ctx = FinalResponseContext {
        schema_version: "final-response-verifier-input/v1".to_string(),
        user_request: "Summarize the Q4 findings.".to_string(),
        workflow_state: WorkflowStateForScoring {
            required_steps: vec!["fetch_sales_data".to_string(), "analyze_sales".to_string()],
            completed_steps: vec!["fetch_sales_data".to_string(), "analyze_sales".to_string()],
            pending_steps: Vec::new(),
            terminal_tools: vec!["report".to_string()],
            recent_errors: Vec::new(),
        },
        required_facts: vec![
            "23% YoY growth".to_string(),
            "Widget Pro".to_string(),
            "APAC".to_string(),
        ],
        tool_trace: vec![
            "fetch_sales_data".to_string(),
            "analyze_sales".to_string(),
            "report".to_string(),
        ],
        tool_results: vec![FinalResponseToolResult {
            tool_name: "analyze_sales".to_string(),
            content: "Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC."
                .to_string(),
        }],
        candidate_final_response: "Sales improved and the report is complete.".to_string(),
        metadata: Some(ScoringMetadata {
            scenario_family: Some("grounded_synthesis".to_string()),
            requires_transform: Some(false),
            requires_synthesis: Some(true),
            requires_all_tool_facts: Some(true),
            must_acknowledge_missing_data: Some(false),
        }),
    };

    let actual = serialize_final_response_state_v1(&ctx);

    assert!(actual.contains("SCHEMA_VERSION:\nfinal-response-verifier-input/v1"));
    assert!(actual.contains("REQUIRED_FACTS:\n['23% YoY growth', 'Widget Pro', 'APAC']"));
    assert!(actual.contains(
        "analyze_sales: \"Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC.\""
    ));
    assert!(actual.contains("scenario_family=\"grounded_synthesis\""));
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
            "wrong_arguments_semantic",
            "tool_not_needed",
            "needs_clarification",
            "deterministic_invalid",
        ]
    );
}

#[test]
fn six_label_artifacts_are_accepted() {
    let labels: LabelsFile = serde_json::from_value(json!({
        "label_mode": "production",
        "labels": [
            "valid",
            "wrong_tool_semantic",
            "wrong_arguments_semantic",
            "tool_not_needed",
            "needs_clarification",
            "deterministic_invalid"
        ],
        "label2id": {
            "valid": 0,
            "wrong_tool_semantic": 1,
            "wrong_arguments_semantic": 2,
            "tool_not_needed": 3,
            "needs_clarification": 4,
            "deterministic_invalid": 5
        },
        "id2label": {
            "0": "valid",
            "1": "wrong_tool_semantic",
            "2": "wrong_arguments_semantic",
            "3": "tool_not_needed",
            "4": "needs_clarification",
            "5": "deterministic_invalid"
        }
    }))
    .expect("labels");

    labels.validate().expect("six-label artifact accepted");
}

#[test]
fn legacy_five_label_artifact_metadata_loads_from_dir() {
    let labels = labels_file(&[
        "valid",
        "wrong_tool_semantic",
        "tool_not_needed",
        "needs_clarification",
        "deterministic_invalid",
    ]);
    let dir = write_artifact_dir(labels, legacy_thresholds()).expect("artifact dir");

    let artifact = ClassifierArtifact::from_dir(&dir).expect("artifact");

    assert_eq!(artifact.labels.labels.len(), 5);
    assert_eq!(
        artifact.model_path(forge_guardrails::ClassifierModelKind::Quantized),
        dir.join("model.quant.onnx")
    );
}

#[test]
fn six_label_v2_artifact_metadata_loads_from_dir() {
    let labels = labels_file(&[
        "valid",
        "wrong_tool_semantic",
        "wrong_arguments_semantic",
        "tool_not_needed",
        "needs_clarification",
        "deterministic_invalid",
    ]);
    let dir = write_artifact_dir(labels, six_label_thresholds()).expect("artifact dir");

    let artifact = ClassifierArtifact::from_dir(&dir).expect("artifact");

    assert_eq!(artifact.labels.labels.len(), 6);
    let threshold = artifact
        .thresholds
        .for_label(&ToolCallClass::WrongArgumentsSemantic);
    assert_eq!(threshold.advisory_min_confidence, 0.90);
    assert_eq!(threshold.enforce_min_confidence, 0.995);
}

#[test]
fn quantized_failed_shadow_only_blocks_action_modes() {
    let labels = labels_file(&[
        "valid",
        "wrong_tool_semantic",
        "wrong_arguments_semantic",
        "tool_not_needed",
        "needs_clarification",
        "deterministic_invalid",
    ]);
    let dir = write_artifact_dir(labels, six_label_thresholds()).expect("artifact dir");
    let manifest_path = dir.join("artifact_manifest.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
            .expect("manifest json");
    manifest["quantized_active_status"] = json!("failed_shadow_only");
    manifest["quantized_active_allowed"] = json!(false);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest).expect("manifest bytes"),
    )
    .expect("write manifest");

    let artifact = ClassifierArtifact::from_dir(&dir).expect("artifact");

    artifact
        .validate_runtime_mode(
            forge_guardrails::ClassifierModelKind::Quantized,
            ScorerMode::Shadow,
        )
        .expect("quantized shadow stays available");
    artifact
        .validate_runtime_mode(
            forge_guardrails::ClassifierModelKind::Full,
            ScorerMode::Advisory,
        )
        .expect("full model can be used for advisory replay");
    let err = artifact
        .validate_runtime_mode(
            forge_guardrails::ClassifierModelKind::Quantized,
            ScorerMode::Advisory,
        )
        .expect_err("quantized advisory blocked");

    assert!(err
        .to_string()
        .contains("quantized classifier artifact is marked failed_shadow_only"));
}

#[test]
fn six_label_artifact_rejects_missing_wrong_arguments_threshold() {
    let labels = labels_file(&[
        "valid",
        "wrong_tool_semantic",
        "wrong_arguments_semantic",
        "tool_not_needed",
        "needs_clarification",
        "deterministic_invalid",
    ]);
    let dir = write_artifact_dir(labels, legacy_thresholds()).expect("artifact dir");

    let err = ClassifierArtifact::from_dir(&dir).expect_err("threshold missing");

    assert!(err
        .to_string()
        .contains("missing classifier threshold for 'wrong_arguments_semantic'"));
}

#[test]
fn v2_serializer_manifest_is_accepted_when_schema_matches() {
    let labels = labels_file(&[
        "valid",
        "wrong_tool_semantic",
        "wrong_arguments_semantic",
        "tool_not_needed",
        "needs_clarification",
        "deterministic_invalid",
    ]);
    let dir = write_artifact_dir(labels, six_label_thresholds()).expect("artifact dir");
    let manifest_path = dir.join("artifact_manifest.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read manifest"))
            .expect("manifest json");
    manifest["input_schema_version"] = json!("toolcall-verifier-input/v2");
    manifest["serializer"] = json!("serialize_state_v2");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest).expect("manifest bytes"),
    )
    .expect("write manifest");

    let artifact = ClassifierArtifact::from_dir(&dir).expect("v2 artifact");

    assert_eq!(artifact.manifest.serializer, "serialize_state_v2");
}

#[test]
fn final_response_artifact_metadata_loads_from_dir() {
    let dir = write_final_response_artifact_dir().expect("artifact dir");

    let artifact = FinalResponseClassifierArtifact::from_dir(&dir).expect("artifact");

    assert_eq!(artifact.labels.labels.len(), 5);
    let threshold = artifact
        .thresholds
        .for_final_response_label(&FinalResponseClass::MissingToolFact);
    assert_eq!(threshold.advisory_min_confidence, 0.90);
    assert_eq!(threshold.enforce_min_confidence, 0.995);
}

#[test]
fn top_k_tool_call_telemetry_uses_expected_label_order() {
    let top_k = tool_call_top_k_from_logits(&[0.0, 1.0, 5.0, 2.0, -1.0, 0.5]);

    assert_eq!(top_k.len(), 6);
    assert_eq!(top_k[0].label, "wrong_arguments_semantic");
    assert!(top_k[0].confidence > top_k[1].confidence);
    assert_eq!(top_k[0].logit, 5.0);
}

#[test]
fn top_k_final_response_telemetry_uses_expected_label_order() {
    let top_k = final_response_top_k_from_logits(&[0.0, 0.5, -1.0, 1.0, 3.0]);

    assert_eq!(top_k.len(), 5);
    assert_eq!(top_k[0].label, "failed_to_acknowledge_data_gap");
    assert!(top_k[0].confidence > top_k[1].confidence);
    assert_eq!(top_k[0].logit, 3.0);
}

#[test]
fn top_k_telemetry_omits_unknown_label_orders() {
    assert!(tool_call_top_k_from_logits(&[1.0, 0.0]).is_empty());
    assert!(final_response_top_k_from_logits(&[1.0, 0.0]).is_empty());
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
fn wrong_arguments_thresholds_parse_when_present() {
    let thresholds: Thresholds = serde_json::from_value(json!({
        "schema_version": "toolcall-verifier-thresholds/v1",
        "mode": "shadow",
        "default_action": "allow",
        "labels": {
            "valid": {
                "action": "allow",
                "advisory_min_confidence": 0.0,
                "enforce_min_confidence": 1.01
            },
            "wrong_tool_semantic": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 1.01,
                "enforce_min_confidence": 1.01
            },
            "wrong_arguments_semantic": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.90,
                "enforce_min_confidence": 0.995
            },
            "tool_not_needed": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 0.8,
                "enforce_min_confidence": 0.95
            },
            "needs_clarification": {
                "action": "advisory_then_enforce_after_eval",
                "advisory_min_confidence": 1.01,
                "enforce_min_confidence": 1.01
            },
            "deterministic_invalid": {
                "action": "deterministic_only",
                "advisory_min_confidence": 1.01,
                "enforce_min_confidence": 1.01
            }
        }
    }))
    .expect("thresholds");

    thresholds.validate().expect("valid thresholds");
    let wrong_args = thresholds.for_label(&ToolCallClass::WrongArgumentsSemantic);
    assert_eq!(wrong_args.advisory_min_confidence, 0.90);
    assert_eq!(wrong_args.enforce_min_confidence, 0.995);
}

#[test]
fn manifest_rejects_wrong_serializer() {
    let mut raw: Value =
        serde_json::from_str(include_str!("fixtures/classifier/artifact_manifest.json"))
            .expect("manifest");
    raw["serializer"] = json!("serialize_state_unknown");
    let manifest: ArtifactManifest = serde_json::from_value(raw).expect("manifest");

    let err = manifest.validate().expect_err("wrong serializer rejected");

    assert!(err
        .to_string()
        .contains("unsupported classifier input schema"));
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
    action: ClassifierAction,
    label: ToolCallClass,
}

struct BlockingToolCallScorer {
    sleep: Duration,
}

struct BlockingFinalResponseScorer {
    sleep: Duration,
}

struct FakeFinalResponseScorer {
    action: ClassifierAction,
    label: FinalResponseClass,
}

impl ToolCallScorer for FakeScorer {
    fn score(&self, _ctx: &ScoringContext, _candidate: &ToolCall) -> anyhow::Result<ToolCallScore> {
        *self.calls.lock().expect("calls lock") += 1;
        if self.fail {
            anyhow::bail!("fake scorer failure");
        }
        Ok(ToolCallScore {
            label: self.label.clone(),
            confidence: 0.99,
            logits: vec![0.0, 0.0, 9.0, 0.0, 0.0],
            action: self.action,
            model_version: "fake".to_string(),
            latency_ms: 1.0,
        })
    }
}

impl ToolCallScorer for BlockingToolCallScorer {
    fn score(&self, _ctx: &ScoringContext, _candidate: &ToolCall) -> anyhow::Result<ToolCallScore> {
        std::thread::sleep(self.sleep);
        Ok(ToolCallScore {
            label: ToolCallClass::Valid,
            confidence: 1.0,
            logits: vec![9.0, 0.0],
            action: ClassifierAction::Allow,
            model_version: "blocking-fake".to_string(),
            latency_ms: self.sleep.as_secs_f64() * 1000.0,
        })
    }
}

impl FinalResponseScorer for BlockingFinalResponseScorer {
    fn score(&self, _ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore> {
        std::thread::sleep(self.sleep);
        Ok(FinalResponseScore {
            label: FinalResponseClass::ValidFinalResponse,
            confidence: 1.0,
            logits: vec![9.0, 0.0],
            action: ClassifierAction::Allow,
            model_version: "blocking-final-fake".to_string(),
            latency_ms: self.sleep.as_secs_f64() * 1000.0,
        })
    }
}

impl FinalResponseScorer for FakeFinalResponseScorer {
    fn score(&self, _ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore> {
        Ok(FinalResponseScore {
            label: self.label.clone(),
            confidence: 0.99,
            logits: vec![0.0, 9.0],
            action: self.action,
            model_version: "fake-final".to_string(),
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
        metadata: None,
    }
}

fn minimal_final_response_context() -> FinalResponseContext {
    FinalResponseContext {
        schema_version: "final-response-verifier-input/v1".to_string(),
        user_request: "Summarize the facts.".to_string(),
        workflow_state: WorkflowStateForScoring {
            required_steps: Vec::new(),
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            terminal_tools: vec!["respond".to_string()],
            recent_errors: Vec::new(),
        },
        required_facts: vec!["Paris".to_string()],
        tool_trace: vec!["get_country_info".to_string()],
        tool_results: Vec::new(),
        candidate_final_response: "The capital is Paris.".to_string(),
        metadata: None,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn score_tool_call_async_does_not_block_current_thread_runtime() {
    let scorer = Arc::new(BlockingToolCallScorer {
        sleep: Duration::from_millis(80),
    });
    let ctx = Arc::new(minimal_scoring_context());
    let candidate = ToolCall::new("search", IndexMap::new());
    let started = Instant::now();
    let scoring = score_tool_call_async(scorer, ctx, candidate);
    tokio::pin!(scoring);

    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(10)) => {
            assert!(
                started.elapsed() < Duration::from_millis(60),
                "tool-call scoring blocked the current-thread runtime"
            );
        }
        score = &mut scoring => {
            panic!("blocking scorer completed before its delay: {score:?}");
        }
    }

    let score = scoring.await.expect("score");
    assert_eq!(score.action, ClassifierAction::Allow);
}

#[tokio::test]
async fn scoring_pipeline_scores_tool_calls_and_returns_nudge() {
    let calls = Arc::new(Mutex::new(0));
    let scorer = Arc::new(FakeScorer {
        calls: calls.clone(),
        fail: false,
        action: ClassifierAction::AdvisoryNudge,
        label: ToolCallClass::WrongArgumentsSemantic,
    });
    let pipeline = ScoringPipeline::new(Some(scorer), None);
    let candidates = vec![ToolCall::new("search", IndexMap::new())];
    let mut observed_scores = 0;

    let nudge = pipeline
        .score_tool_calls(
            Arc::new(minimal_scoring_context()),
            &candidates,
            |_call, score| {
                observed_scores += 1;
                assert_eq!(score.label, ToolCallClass::WrongArgumentsSemantic);
            },
            |_call, err| panic!("unexpected scoring error: {err}"),
        )
        .await;

    assert!(nudge.is_some());
    assert_eq!(*calls.lock().expect("calls lock"), 1);
    assert_eq!(observed_scores, 1);
}

#[tokio::test]
async fn scoring_pipeline_scores_final_response_and_returns_nudge() {
    let scorer = Arc::new(FakeFinalResponseScorer {
        action: ClassifierAction::Block,
        label: FinalResponseClass::MissingToolFact,
    });
    let pipeline = ScoringPipeline::new(None, Some(scorer));
    let mut observed_scores = 0;

    let nudge = pipeline
        .score_final_response(
            Arc::new(minimal_final_response_context()),
            |score| {
                observed_scores += 1;
                assert_eq!(score.label, FinalResponseClass::MissingToolFact);
            },
            |err| panic!("unexpected final-response scoring error: {err}"),
        )
        .await;

    assert!(nudge.is_some());
    assert_eq!(observed_scores, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn score_final_response_async_does_not_block_current_thread_runtime() {
    let scorer = Arc::new(BlockingFinalResponseScorer {
        sleep: Duration::from_millis(80),
    });
    let ctx = Arc::new(minimal_final_response_context());
    let started = Instant::now();
    let scoring = score_final_response_async(scorer, ctx);
    tokio::pin!(scoring);

    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(10)) => {
            assert!(
                started.elapsed() < Duration::from_millis(60),
                "final-response scoring blocked the current-thread runtime"
            );
        }
        score = &mut scoring => {
            panic!("blocking scorer completed before its delay: {score:?}");
        }
    }

    let score = scoring.await.expect("score");
    assert_eq!(score.action, ClassifierAction::Allow);
}

#[test]
fn serialize_state_v2_includes_scoring_metadata() {
    let mut ctx = minimal_scoring_context();
    ctx.schema_version = "toolcall-verifier-input/v2".to_string();
    ctx.metadata = Some(ScoringMetadata {
        scenario_family: Some("argument_transformation".to_string()),
        requires_transform: Some(true),
        requires_synthesis: Some(false),
        requires_all_tool_facts: Some(true),
        must_acknowledge_missing_data: Some(false),
    });
    let candidate = ToolCall::new("search", IndexMap::new());

    let serialized = serialize_state_v2(&ctx, &candidate);

    assert!(serialized.contains("SCORING_METADATA:"));
    assert!(serialized.contains("scenario_family=\"argument_transformation\""));
    assert!(serialized.contains("requires_transform=true"));
    assert!(serialized.contains("requires_synthesis=false"));
    assert!(serialized.contains("requires_all_tool_facts=true"));
    assert!(serialized.contains("must_acknowledge_missing_data=false"));
}

#[test]
fn fake_scorer_runs_only_after_deterministic_checks_pass() {
    let calls = Arc::new(Mutex::new(0));
    let scorer = Arc::new(FakeScorer {
        calls: calls.clone(),
        fail: false,
        action: ClassifierAction::ShadowOnly,
        label: ToolCallClass::ToolNotNeeded,
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
        action: ClassifierAction::ShadowOnly,
        label: ToolCallClass::ToolNotNeeded,
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
fn fake_scorer_advisory_nudge_blocks_tool_execution() {
    let calls = Arc::new(Mutex::new(0));
    let scorer = Arc::new(FakeScorer {
        calls: calls.clone(),
        fail: false,
        action: ClassifierAction::AdvisoryNudge,
        label: ToolCallClass::WrongArgumentsSemantic,
    });
    let mut guardrails = guardrails_with_scorer(scorer);
    let ctx = minimal_scoring_context();

    let result = guardrails.check_with_scoring_context(
        &LLMResponse::ToolCalls(vec![ToolCall::new("search", IndexMap::new())]),
        &ctx,
    );

    assert_eq!(result.action, forge_guardrails::GuardAction::Retry);
    assert!(result.tool_calls.is_none());
    assert_eq!(*calls.lock().expect("calls lock"), 1);
    let nudge = result.nudge.expect("classifier nudge");
    assert!(nudge
        .content
        .contains("argument values do not match the user request"));
}

#[test]
fn noop_final_response_scorer_allows_valid_final_response() {
    let scorer = NoopFinalResponseScorer;
    let ctx = FinalResponseContext {
        schema_version: "final-response-verifier-input/v1".to_string(),
        user_request: "Summarize the facts.".to_string(),
        workflow_state: WorkflowStateForScoring {
            required_steps: Vec::new(),
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            terminal_tools: vec!["respond".to_string()],
            recent_errors: Vec::new(),
        },
        required_facts: vec!["Paris".to_string()],
        tool_trace: vec!["get_country_info".to_string()],
        tool_results: Vec::new(),
        candidate_final_response: "The capital is Paris.".to_string(),
        metadata: None,
    };

    let score = scorer.score(&ctx).expect("score");

    assert_eq!(score.label, FinalResponseClass::ValidFinalResponse);
    assert_eq!(score.action, ClassifierAction::Allow);
}

#[test]
fn scorer_mode_parses_stable_names() {
    assert_eq!("shadow".parse::<ScorerMode>().unwrap(), ScorerMode::Shadow);
    assert!("block".parse::<ScorerMode>().is_err());
}

#[cfg(feature = "classifier")]
#[test]
fn onnx_scorer_options_validate_bounded_session_pool() {
    assert!(OnnxScorerOptions {
        session_pool_size: 1,
        intra_threads: 1,
    }
    .validate()
    .is_ok());
    assert!(OnnxScorerOptions {
        session_pool_size: MAX_ONNX_SESSION_POOL_SIZE,
        intra_threads: 1,
    }
    .validate()
    .is_ok());

    let err = OnnxScorerOptions {
        session_pool_size: 0,
        intra_threads: 1,
    }
    .validate()
    .expect_err("zero pool rejected");
    assert!(err.to_string().contains("session pool size"));

    let err = OnnxScorerOptions {
        session_pool_size: MAX_ONNX_SESSION_POOL_SIZE + 1,
        intra_threads: 1,
    }
    .validate()
    .expect_err("oversized pool rejected");
    assert!(err.to_string().contains("session pool size"));
}

#[cfg(feature = "classifier")]
#[test]
fn onnx_scorer_options_require_positive_intra_threads() {
    let err = OnnxScorerOptions {
        session_pool_size: 1,
        intra_threads: 0,
    }
    .validate()
    .expect_err("zero threads rejected");

    assert!(err.to_string().contains("intra-op thread count"));
}
