use anyhow::{Context, Result as AnyResult};
use hf_hub::repository::{RepoTreeEntry, RepoTypeModel};
use hf_hub::{HFClientSync, HFRepositorySync};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::files::{optional_sidecar_files, required_files_for_model, runtime_required_files};
use super::types::{ClassifierDownloadPlan, ClassifierDownloadReport};
use crate::guardrails::ClassifierModelKind;

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

pub(crate) fn parse_repo(repo: &str) -> AnyResult<(&str, &str)> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("--repo must be in owner/name form"))?;
    anyhow::ensure!(!owner.is_empty(), "--repo owner cannot be empty");
    anyhow::ensure!(!name.is_empty(), "--repo name cannot be empty");
    anyhow::ensure!(!name.contains('/'), "--repo must be in owner/name form");
    Ok((owner, name))
}

pub(crate) fn repository_files(
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

pub(crate) fn download_tree_file(
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

pub(crate) fn download_onnx_file_to_artifact_dir(
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

pub(crate) fn atomic_copy(source: &Path, destination: &Path) -> AnyResult<()> {
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
