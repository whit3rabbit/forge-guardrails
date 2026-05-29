use super::files::required_files_for_model;
use super::mod_impl::classifier_artifact_needs_download;
use super::paths::{cache_root_from_values, sanitize_repo};
use super::types::ClassifierArtifactKind;
use crate::guardrails::ClassifierModelKind;
use serde_json::json;
use std::path::{Path, PathBuf};

#[test]
fn cache_root_prefers_explicit_then_xdg_then_home() {
    assert_eq!(
        cache_root_from_values(Some("/tmp/forge-cache"), Some("/tmp/xdg"), Some("/home/me"))
            .expect("explicit"),
        PathBuf::from("/tmp/forge-cache")
    );
    assert_eq!(
        cache_root_from_values(None, Some("/tmp/xdg"), Some("/home/me")).expect("xdg"),
        PathBuf::from("/tmp/xdg/forge-guardrails/classifiers")
    );
    assert_eq!(
        cache_root_from_values(None, None, Some("/home/me")).expect("home"),
        PathBuf::from("/home/me/.cache/forge-guardrails/classifiers")
    );
    assert!(cache_root_from_values(None, None, None).is_err());
}

#[test]
fn cached_artifact_dir_uses_stable_layout() {
    let root = cache_root_from_values(Some("/tmp/forge-cache"), None, None).expect("root");
    let dir = root
        .join(ClassifierArtifactKind::ToolCall.as_str())
        .join("owner__name")
        .join("revision")
        .join("onnx");
    let actual = root
        .join(ClassifierArtifactKind::ToolCall.as_str())
        .join(sanitize_repo("owner/name"))
        .join("revision")
        .join("onnx");
    assert_eq!(actual, dir);
}

#[test]
fn required_quantized_tool_call_files_include_runtime_assets() {
    let files = required_files_for_model(
        ClassifierArtifactKind::ToolCall,
        ClassifierModelKind::Quantized,
    );
    assert!(files.contains(&"onnx/tokenizer.json"));
    assert!(files.contains(&"onnx/model_quantized.onnx"));
    assert!(!files.contains(&"onnx/model.onnx"));
}

#[test]
fn missing_required_file_needs_download() {
    let temp = make_temp_dir("missing-required");
    assert!(classifier_artifact_needs_download(
        ClassifierArtifactKind::ToolCall,
        &temp,
        ClassifierModelKind::Quantized
    ));
}

#[test]
fn valid_minimal_tool_call_artifact_skips_download() {
    let temp = make_temp_dir("valid-minimal");
    write_minimal_tool_call_artifact(&temp);
    assert!(!classifier_artifact_needs_download(
        ClassifierArtifactKind::ToolCall,
        &temp,
        ClassifierModelKind::Quantized
    ));
}

fn make_temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "forge-classifier-download-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("tempdir");
    dir
}

fn write_minimal_tool_call_artifact(dir: &Path) {
    std::fs::create_dir_all(dir).expect("dir");
    let labels = vec![
        "valid",
        "wrong_tool_semantic",
        "wrong_arguments_semantic",
        "tool_not_needed",
        "needs_clarification",
        "deterministic_invalid",
    ];
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
    let thresholds = labels
        .iter()
        .map(|label| {
            (
                label.to_string(),
                json!({
                    "action": "shadow_only",
                    "advisory_min_confidence": 1.01,
                    "enforce_min_confidence": 1.01
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    write_json(
        dir.join("artifact_manifest.json"),
        json!({
            "artifact_schema_version": "toolcall-verifier-artifact/v1",
            "input_schema_version": "toolcall-verifier-input/v1",
            "serializer": "serialize_state_v1",
            "max_length": 1280,
            "onnx_file": "model.onnx",
            "quantized_onnx_file": "model_quantized.onnx",
            "labels": labels,
        }),
    );
    write_json(
        dir.join("labels.json"),
        json!({
            "label_mode": "six_label",
            "labels": labels,
            "label2id": label2id,
            "id2label": id2label,
        }),
    );
    write_json(
        dir.join("thresholds.json"),
        json!({
            "schema_version": "toolcall-verifier-thresholds/v1",
            "mode": "shadow",
            "default_action": "shadow_only",
            "labels": thresholds,
        }),
    );
    for file in [
        "input_schema.json",
        "serializer_fixture.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "spm.model",
        "config.json",
        "test_metrics.json",
        "training_metrics.json",
        "training_run_summary.json",
    ] {
        std::fs::write(dir.join(file), "{}").expect("sidecar");
    }
    std::fs::write(dir.join("model_quantized.onnx"), b"onnx").expect("model");
}

fn write_json(path: PathBuf, value: serde_json::Value) {
    std::fs::write(path, serde_json::to_vec(&value).expect("json")).expect("write");
}
