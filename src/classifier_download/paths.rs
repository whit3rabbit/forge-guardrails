use super::types::ClassifierArtifactKind;
use crate::guardrails::{DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION};
use anyhow::Result as AnyResult;
use std::path::PathBuf;

pub(crate) const CLASSIFIER_CACHE_ENV: &str = "FORGE_CLASSIFIER_CACHE_DIR";

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

pub(crate) fn cache_root_from_values(
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

pub(crate) fn nonempty_path(value: Option<&str>) -> Option<PathBuf> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub(crate) fn sanitize_repo(repo: &str) -> String {
    repo.replace('/', "__")
}
