//! Hugging Face download helpers for local ONNX classifier artifacts.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result as AnyResult};
use hf_hub::repository::{RepoTreeEntry, RepoTypeModel};
use hf_hub::{HFClientSync, HFRepositorySync};

use crate::guardrails::{
    ClassifierArtifact, ClassifierModelKind, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION,
    DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
};

const CLASSIFIER_CACHE_ENV: &str = "FORGE_CLASSIFIER_CACHE_DIR";

/// Classifier artifact family available for download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierArtifactKind {
    /// Tool-call verifier artifact.
    ToolCall,
    /// Final-response verifier artifact.
    FinalResponse,
}

impl ClassifierArtifactKind {
    /// Return the stable artifact name used in CLI output and cache paths.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolCall => "tool-call",
            Self::FinalResponse => "final-response",
        }
    }

    /// Return the default Hugging Face repository for this artifact family.
    pub fn default_repo(self) -> &'static str {
        match self {
            Self::ToolCall => DEFAULT_CLASSIFIER_REPO,
            Self::FinalResponse => DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO,
        }
    }

    /// Return the pinned Hugging Face revision for this artifact family.
    pub fn default_revision(self) -> &'static str {
        match self {
            Self::ToolCall => DEFAULT_CLASSIFIER_REVISION,
            Self::FinalResponse => DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
        }
    }

    /// Return the legacy eval/training output directory for this artifact family.
    pub fn default_eval_output_dir(self) -> PathBuf {
        match self {
            Self::ToolCall => PathBuf::from("target/classifier-artifacts"),
            Self::FinalResponse => PathBuf::from("target/final-response-classifier-artifacts"),
        }
    }

    /// Return the CLI output key for this artifact family.
    pub fn output_key(self) -> &'static str {
        match self {
            Self::ToolCall => "classifier_dir",
            Self::FinalResponse => "final_response_classifier_dir",
        }
    }
}

/// Plan for downloading an artifact in repository-tree layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifierDownloadPlan {
    /// Artifact family.
    pub kind: ClassifierArtifactKind,
    /// Hugging Face repository in owner/name form.
    pub repo: String,
    /// Hugging Face revision.
    pub revision: String,
    /// Local output directory. Repository paths like `onnx/` are created below it.
    pub output_dir: PathBuf,
}

impl ClassifierDownloadPlan {
    /// Return the ONNX artifact directory produced by this plan.
    pub fn artifact_dir(&self) -> PathBuf {
        self.output_dir.join("onnx")
    }
}

/// Result of a classifier artifact preparation operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifierDownloadReport {
    /// Artifact family.
    pub kind: ClassifierArtifactKind,
    /// Hugging Face repository in owner/name form.
    pub repo: String,
    /// Hugging Face revision.
    pub revision: String,
    /// ONNX model kind prepared.
    pub model_kind: ClassifierModelKind,
    /// Runnable local ONNX artifact directory.
    pub artifact_dir: PathBuf,
    /// Whether any file was downloaded during this operation.
    pub downloaded: bool,
    /// Number of optional sidecar files downloaded.
    pub optional_sidecars: usize,
}

/// Return the default user cache root for classifier artifacts.
pub fn default_classifier_cache_root() -> AnyResult<PathBuf> {
    cache_root_from_values(
        std::env::var(CLASSIFIER_CACHE_ENV).ok().as_deref(),
        std::env::var("XDG_CACHE_HOME").ok().as_deref(),
        std::env::var("HOME").ok().as_deref(),
    )
}

/// Return the default cached artifact directory for a pinned artifact.
pub fn default_cached_classifier_artifact_dir(
    kind: ClassifierArtifactKind,
    repo: &str,
    revision: &str,
) -> AnyResult<PathBuf> {
    Ok(default_classifier_cache_root()?
        .join(kind.as_str())
        .join(sanitize_repo(repo))
        .join(revision)
        .join("onnx"))
}

/// Return the default cached tool-call artifact directory.
pub fn default_tool_call_classifier_artifact_dir() -> AnyResult<PathBuf> {
    default_cached_classifier_artifact_dir(
        ClassifierArtifactKind::ToolCall,
        DEFAULT_CLASSIFIER_REPO,
        DEFAULT_CLASSIFIER_REVISION,
    )
}

/// Return true when the local artifact is missing files or fails metadata validation.
pub fn classifier_artifact_needs_download(
    kind: ClassifierArtifactKind,
    artifact_dir: impl AsRef<Path>,
    model_kind: ClassifierModelKind,
) -> bool {
    validate_classifier_artifact_dir(kind, artifact_dir, model_kind).is_err()
}

/// Validate that a local artifact directory has all files required by the Rust scorer.
pub fn validate_classifier_artifact_dir(
    kind: ClassifierArtifactKind,
    artifact_dir: impl AsRef<Path>,
    model_kind: ClassifierModelKind,
) -> AnyResult<()> {
    let artifact_dir = artifact_dir.as_ref();
    for file in required_files_for_model(kind, model_kind) {
        let Some(local_name) = file.strip_prefix("onnx/") else {
            continue;
        };
        anyhow::ensure!(
            artifact_dir.join(local_name).is_file(),
            "{} artifact missing required file '{}'",
            kind.as_str(),
            artifact_dir.join(local_name).display()
        );
    }

    match kind {
        ClassifierArtifactKind::ToolCall => {
            let artifact = ClassifierArtifact::from_dir(artifact_dir)?;
            anyhow::ensure!(
                artifact.model_path(model_kind).is_file(),
                "{} classifier model file missing at {}",
                kind.as_str(),
                artifact.model_path(model_kind).display()
            );
        }
        ClassifierArtifactKind::FinalResponse => {
            let artifact =
                crate::guardrails::FinalResponseClassifierArtifact::from_dir(artifact_dir)?;
            anyhow::ensure!(
                artifact.model_path(model_kind).is_file(),
                "{} classifier model file missing at {}",
                kind.as_str(),
                artifact.model_path(model_kind).display()
            );
        }
    }
    Ok(())
}

/// Download an artifact using the repository-tree layout used by eval tooling.
pub fn download_classifier_artifact_tree(
    plan: &ClassifierDownloadPlan,
    model_kind: ClassifierModelKind,
    mut emit: impl FnMut(String),
) -> AnyResult<ClassifierDownloadReport> {
    let (owner, name) = parse_repo(&plan.repo)?;
    let client = HFClientSync::new()?;
    let repo = client.model(owner, name);

    emit(format!("artifact={}", plan.kind.as_str()));
    emit(format!("repo={}", plan.repo));
    emit(format!("revision={}", plan.revision));

    let available_files = repository_files(&repo, &plan.revision)?;
    let mut downloaded_files = HashSet::new();
    for file in required_files_for_model(plan.kind, model_kind) {
        anyhow::ensure!(
            available_files.contains(file),
            "{} artifact missing required file '{}' at revision {}",
            plan.kind.as_str(),
            file,
            plan.revision
        );
        let path = download_tree_file(&repo, &plan.output_dir, &plan.revision, file)?;
        emit(format!("{file} -> {}", path.display()));
        downloaded_files.insert(file.to_string());
    }

    let mut optional_sidecars = 0usize;
    for file in optional_sidecar_files(plan.kind) {
        if downloaded_files.contains(file) {
            continue;
        }
        if available_files.contains(file) {
            let path = download_tree_file(&repo, &plan.output_dir, &plan.revision, file)?;
            emit(format!("{file} -> {}", path.display()));
            optional_sidecars += 1;
        }
    }

    let missing_runtime_files = runtime_required_files(plan.kind)
        .into_iter()
        .filter(|file| !available_files.contains(*file))
        .collect::<Vec<_>>();
    if !missing_runtime_files.is_empty() {
        emit(format!(
            "runtime_missing={}",
            missing_runtime_files.join(",")
        ));
        emit(format!(
            "warning={} artifact is not runnable by the current Rust ONNX scorer until those files are published",
            plan.kind.as_str()
        ));
    }

    emit(format!("optional_sidecars={optional_sidecars}"));
    emit(format!(
        "{}={}",
        plan.kind.output_key(),
        plan.artifact_dir().display()
    ));

    Ok(ClassifierDownloadReport {
        kind: plan.kind,
        repo: plan.repo.clone(),
        revision: plan.revision.clone(),
        model_kind,
        artifact_dir: plan.artifact_dir(),
        downloaded: true,
        optional_sidecars,
    })
}

/// Ensure that a runnable artifact exists directly in `artifact_dir`.
pub fn ensure_classifier_artifact_dir(
    kind: ClassifierArtifactKind,
    artifact_dir: impl AsRef<Path>,
    repo: &str,
    revision: &str,
    model_kind: ClassifierModelKind,
    mut emit: impl FnMut(String),
) -> AnyResult<ClassifierDownloadReport> {
    let artifact_dir = artifact_dir.as_ref().to_path_buf();
    emit(format!("artifact={}", kind.as_str()));
    emit(format!("repo={repo}"));
    emit(format!("revision={revision}"));
    emit(format!("classifier_model={}", model_kind.as_str()));
    emit(format!("{}={}", kind.output_key(), artifact_dir.display()));

    if validate_classifier_artifact_dir(kind, &artifact_dir, model_kind).is_ok() {
        emit("status=present".to_string());
        return Ok(ClassifierDownloadReport {
            kind,
            repo: repo.to_string(),
            revision: revision.to_string(),
            model_kind,
            artifact_dir,
            downloaded: false,
            optional_sidecars: 0,
        });
    }

    let (owner, name) = parse_repo(repo)?;
    let client = HFClientSync::new()?;
    let hf_repo = client.model(owner, name);
    let available_files = repository_files(&hf_repo, revision)?;
    let mut downloaded_files = HashSet::new();
    for file in required_files_for_model(kind, model_kind) {
        anyhow::ensure!(
            available_files.contains(file),
            "{} artifact missing required file '{}' at revision {}",
            kind.as_str(),
            file,
            revision
        );
        let path = download_onnx_file_to_artifact_dir(&hf_repo, revision, file, &artifact_dir)?;
        emit(format!("{file} -> {}", path.display()));
        downloaded_files.insert(file.to_string());
    }

    let mut optional_sidecars = 0usize;
    for file in optional_sidecar_files(kind) {
        if downloaded_files.contains(file) || !file.starts_with("onnx/") {
            continue;
        }
        if available_files.contains(file) {
            let path = download_onnx_file_to_artifact_dir(&hf_repo, revision, file, &artifact_dir)?;
            emit(format!("{file} -> {}", path.display()));
            optional_sidecars += 1;
        }
    }

    validate_classifier_artifact_dir(kind, &artifact_dir, model_kind)?;
    emit("status=downloaded".to_string());

    Ok(ClassifierDownloadReport {
        kind,
        repo: repo.to_string(),
        revision: revision.to_string(),
        model_kind,
        artifact_dir,
        downloaded: true,
        optional_sidecars,
    })
}

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

fn cache_root_from_values(
    explicit: Option<&str>,
    xdg_cache_home: Option<&str>,
    home: Option<&str>,
) -> AnyResult<PathBuf> {
    if let Some(path) = nonempty_path(explicit) {
        return Ok(path);
    }
    if let Some(path) = nonempty_path(xdg_cache_home) {
        return Ok(path.join("forge-guardrails").join("classifiers"));
    }
    if let Some(path) = nonempty_path(home) {
        return Ok(path
            .join(".cache")
            .join("forge-guardrails")
            .join("classifiers"));
    }
    anyhow::bail!("could not resolve classifier cache directory; set {CLASSIFIER_CACHE_ENV}")
}

fn nonempty_path(value: Option<&str>) -> Option<PathBuf> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn sanitize_repo(repo: &str) -> String {
    repo.replace('/', "__")
}

fn optional_sidecar_files(kind: ClassifierArtifactKind) -> Vec<&'static str> {
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

fn parse_repo(repo: &str) -> AnyResult<(&str, &str)> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("--repo must be in owner/name form"))?;
    anyhow::ensure!(!owner.is_empty(), "--repo owner cannot be empty");
    anyhow::ensure!(!name.is_empty(), "--repo name cannot be empty");
    anyhow::ensure!(!name.contains('/'), "--repo must be in owner/name form");
    Ok((owner, name))
}

fn repository_files(
    repo: &HFRepositorySync<RepoTypeModel>,
    revision: &str,
) -> AnyResult<HashSet<String>> {
    let entries = repo
        .list_tree()
        .revision(revision.to_string())
        .recursive(true)
        .send()
        .with_context(|| format!("failed to list repository files at revision {revision}"))?;
    Ok(entries
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } => Some(path),
            RepoTreeEntry::Directory { .. } => None,
        })
        .collect())
}

fn download_tree_file(
    repo: &HFRepositorySync<RepoTypeModel>,
    output_dir: &Path,
    revision: &str,
    file: &str,
) -> AnyResult<PathBuf> {
    repo.download_file()
        .filename(file.to_string())
        .revision(revision.to_string())
        .local_dir(output_dir.to_path_buf())
        .send()
        .with_context(|| format!("failed to download {file}"))
}

fn download_onnx_file_to_artifact_dir(
    repo: &HFRepositorySync<RepoTypeModel>,
    revision: &str,
    file: &str,
    artifact_dir: &Path,
) -> AnyResult<PathBuf> {
    let local_name = file
        .strip_prefix("onnx/")
        .ok_or_else(|| anyhow::anyhow!("artifact-dir downloads only support onnx files"))?;
    let cached = repo
        .download_file()
        .filename(file.to_string())
        .revision(revision.to_string())
        .send()
        .with_context(|| format!("failed to download {file}"))?;
    let destination = artifact_dir.join(local_name);
    atomic_copy(&cached, &destination)?;
    Ok(destination)
}

fn atomic_copy(source: &Path, destination: &Path) -> AnyResult<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("destination has no parent: {}", destination.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("destination has invalid file name"))?;
    let tmp = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    std::fs::copy(source, &tmp).with_context(|| {
        format!(
            "failed to copy classifier artifact {} to {}",
            source.display(),
            tmp.display()
        )
    })?;
    std::fs::rename(&tmp, destination).with_context(|| {
        format!(
            "failed to move classifier artifact {} to {}",
            tmp.display(),
            destination.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
