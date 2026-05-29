use anyhow::Result as AnyResult;
use hf_hub::HFClientSync;
use std::collections::HashSet;
use std::path::Path;

use super::download::{download_onnx_file_to_artifact_dir, parse_repo, repository_files};
use super::files::{optional_sidecar_files, required_files_for_model};
use super::types::{ClassifierArtifactKind, ClassifierDownloadReport};
use crate::guardrails::{ClassifierArtifact, ClassifierModelKind};

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
