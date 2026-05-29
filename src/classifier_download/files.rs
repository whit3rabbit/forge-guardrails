use super::types::ClassifierArtifactKind;
use crate::guardrails::ClassifierModelKind;

/// Return the required repository files for an artifact and model kind.
pub fn required_files_for_model(
    kind: ClassifierArtifactKind,
    model_kind: ClassifierModelKind,
) -> Vec<&'static str> {
    let mut files = match kind {
        ClassifierArtifactKind::ToolCall => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/input_schema.json",
            "onnx/serializer_fixture.json",
            "onnx/tokenizer.json",
            "onnx/tokenizer_config.json",
            "onnx/special_tokens_map.json",
            "onnx/added_tokens.json",
            "onnx/spm.model",
            "onnx/config.json",
            "onnx/test_metrics.json",
            "onnx/training_metrics.json",
            "onnx/training_run_summary.json",
            "onnx/model_quantized.onnx",
        ],
        ClassifierArtifactKind::FinalResponse => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/input_schema.json",
            "onnx/tokenizer.json",
            "onnx/tokenizer_config.json",
            "onnx/special_tokens_map.json",
            "onnx/added_tokens.json",
            "onnx/spm.model",
            "onnx/config.json",
            "onnx/onnx_parity_report.json",
            "onnx/training_provenance.json",
            "onnx/model_quantized.onnx",
        ],
    };
    if model_kind == ClassifierModelKind::Full {
        files.push("onnx/model.onnx");
    }
    files
}

/// Return files required by runtime loading for an artifact.
pub fn runtime_required_files(kind: ClassifierArtifactKind) -> Vec<&'static str> {
    match kind {
        ClassifierArtifactKind::ToolCall => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/tokenizer.json",
            "onnx/model_quantized.onnx",
        ],
        ClassifierArtifactKind::FinalResponse => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/tokenizer.json",
            "onnx/model_quantized.onnx",
        ],
    }
}

pub(crate) fn optional_sidecar_files(kind: ClassifierArtifactKind) -> Vec<&'static str> {
    match kind {
        ClassifierArtifactKind::ToolCall => vec![
            "onnx/input_schema_v1.json",
            "onnx/input_schema_v2.json",
            "onnx/final_response_input_schema.json",
            "onnx/serializer_fixture_v2.json",
            "onnx/calibration_report.json",
            "onnx/reliability_curves.jsonl",
            "onnx/onnx_parity_report.json",
            "hf_model/input_schema_v1.json",
            "hf_model/input_schema_v2.json",
            "hf_model/final_response_input_schema.json",
            "hf_model/serializer_fixture_v2.json",
            "hf_model/calibration_report.json",
            "hf_model/reliability_curves.jsonl",
            "hf_model/onnx_parity_report.json",
        ],
        ClassifierArtifactKind::FinalResponse => vec![
            "onnx/serializer_fixture.json",
            "onnx/test_metrics.json",
            "onnx/training_metrics.json",
            "onnx/training_run_summary.json",
            "hf_model/added_tokens.json",
            "hf_model/artifact_manifest.json",
            "hf_model/config.json",
            "hf_model/input_schema.json",
            "hf_model/labels.json",
            "hf_model/onnx_parity_report.json",
            "hf_model/special_tokens_map.json",
            "hf_model/spm.model",
            "hf_model/thresholds.json",
            "hf_model/tokenizer_config.json",
            "hf_model/training_provenance.json",
        ],
    }
}
