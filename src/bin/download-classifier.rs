//! Download local classifier artifacts for feature-gated ONNX tests.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use forge_guardrails::{
    ClassifierModelKind, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION,
    DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
};
use hf_hub::repository::{RepoTreeEntry, RepoTypeModel};
use hf_hub::{HFClientSync, HFRepositorySync};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ArtifactSelection {
    ToolCall,
    FinalResponse,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactKind {
    ToolCall,
    FinalResponse,
}

impl ArtifactKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ToolCall => "tool-call",
            Self::FinalResponse => "final-response",
        }
    }

    fn default_repo(self) -> &'static str {
        match self {
            Self::ToolCall => DEFAULT_CLASSIFIER_REPO,
            Self::FinalResponse => DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO,
        }
    }

    fn default_revision(self) -> &'static str {
        match self {
            Self::ToolCall => DEFAULT_CLASSIFIER_REVISION,
            Self::FinalResponse => DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
        }
    }

    fn default_output_dir(self) -> PathBuf {
        match self {
            Self::ToolCall => PathBuf::from("target/classifier-artifacts"),
            Self::FinalResponse => PathBuf::from("target/final-response-classifier-artifacts"),
        }
    }

    fn output_key(self) -> &'static str {
        match self {
            Self::ToolCall => "classifier_dir",
            Self::FinalResponse => "final_response_classifier_dir",
        }
    }
}

#[derive(Debug)]
struct DownloadPlan {
    kind: ArtifactKind,
    repo: String,
    revision: String,
    output_dir: PathBuf,
}

#[derive(Debug, Parser)]
#[command(
    name = "download-classifier",
    about = "download ONNX classifier artifacts"
)]
struct Cli {
    /// Artifact family to download.
    #[arg(long, value_enum, default_value = "tool-call")]
    artifact: ArtifactSelection,

    /// Hugging Face model repo, in owner/name form.
    #[arg(long, value_name = "OWNER/NAME")]
    repo: Option<String>,

    /// Hugging Face revision.
    #[arg(long, value_name = "REV")]
    revision: Option<String>,

    /// Output directory for the tool-call artifact, or for final-response when --artifact final-response.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Output directory for the final-response artifact when --artifact both.
    #[arg(long, value_name = "DIR")]
    final_response_output_dir: Option<PathBuf>,

    /// ONNX model to download.
    #[arg(long, default_value = "quantized", value_name = "quantized|full")]
    classifier_model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model_kind =
        ClassifierModelKind::from_str(&cli.classifier_model).map_err(|err| anyhow::anyhow!(err))?;
    let client = HFClientSync::new()?;

    for plan in download_plans(&cli)? {
        download_artifact(&client, &plan, model_kind)?;
    }

    Ok(())
}

fn download_plans(cli: &Cli) -> Result<Vec<DownloadPlan>> {
    if cli.artifact == ArtifactSelection::Both && (cli.repo.is_some() || cli.revision.is_some()) {
        anyhow::bail!("--repo and --revision cannot be combined with --artifact both");
    }

    let kinds = match cli.artifact {
        ArtifactSelection::ToolCall => vec![ArtifactKind::ToolCall],
        ArtifactSelection::FinalResponse => vec![ArtifactKind::FinalResponse],
        ArtifactSelection::Both => vec![ArtifactKind::ToolCall, ArtifactKind::FinalResponse],
    };

    kinds
        .into_iter()
        .map(|kind| {
            let repo = cli
                .repo
                .clone()
                .unwrap_or_else(|| kind.default_repo().to_string());
            let revision = cli
                .revision
                .clone()
                .unwrap_or_else(|| kind.default_revision().to_string());
            let output_dir = output_dir_for(cli, kind);
            Ok(DownloadPlan {
                kind,
                repo,
                revision,
                output_dir,
            })
        })
        .collect()
}

fn output_dir_for(cli: &Cli, kind: ArtifactKind) -> PathBuf {
    match kind {
        ArtifactKind::ToolCall => cli
            .output_dir
            .clone()
            .unwrap_or_else(|| kind.default_output_dir()),
        ArtifactKind::FinalResponse if cli.artifact == ArtifactSelection::FinalResponse => cli
            .output_dir
            .clone()
            .unwrap_or_else(|| kind.default_output_dir()),
        ArtifactKind::FinalResponse => cli
            .final_response_output_dir
            .clone()
            .unwrap_or_else(|| kind.default_output_dir()),
    }
}

fn download_artifact(
    client: &HFClientSync,
    plan: &DownloadPlan,
    model_kind: ClassifierModelKind,
) -> Result<()> {
    let (owner, name) = parse_repo(&plan.repo)?;
    let repo = client.model(owner, name);
    let onnx_dir = plan.output_dir.join("onnx");

    println!("artifact={}", plan.kind.as_str());
    println!("repo={}", plan.repo);
    println!("revision={}", plan.revision);

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
        download_file(&repo, &plan.output_dir, &plan.revision, file)?;
        downloaded_files.insert(file.to_string());
    }

    let mut optional_downloaded = 0usize;
    for file in optional_sidecar_files(plan.kind) {
        if downloaded_files.contains(file) {
            continue;
        }
        if available_files.contains(file) {
            download_file(&repo, &plan.output_dir, &plan.revision, file)?;
            optional_downloaded += 1;
        }
    }

    let missing_runtime_files = runtime_required_files(plan.kind)
        .into_iter()
        .filter(|file| !available_files.contains(*file))
        .collect::<Vec<_>>();
    if !missing_runtime_files.is_empty() {
        println!("runtime_missing={}", missing_runtime_files.join(","));
        println!(
            "warning={} artifact is not runnable by the current Rust ONNX scorer until those files are published",
            plan.kind.as_str()
        );
    }

    println!("optional_sidecars={optional_downloaded}");
    println!("{}={}", plan.kind.output_key(), onnx_dir.display());
    Ok(())
}

fn parse_repo(repo: &str) -> Result<(&str, &str)> {
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
) -> Result<HashSet<String>> {
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

fn download_file(
    repo: &HFRepositorySync<RepoTypeModel>,
    output_dir: &Path,
    revision: &str,
    file: &str,
) -> Result<PathBuf> {
    let path = repo
        .download_file()
        .filename(file.to_string())
        .revision(revision.to_string())
        .local_dir(output_dir.to_path_buf())
        .send()
        .with_context(|| format!("failed to download {file}"))?;
    println!("{file} -> {}", path.display());
    Ok(path)
}

fn required_files_for_model(
    artifact: ArtifactKind,
    model_kind: ClassifierModelKind,
) -> Vec<&'static str> {
    let mut files = match artifact {
        ArtifactKind::ToolCall => vec![
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
        ArtifactKind::FinalResponse => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/input_schema.json",
            "onnx/tokenizer_config.json",
            "onnx/special_tokens_map.json",
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

fn runtime_required_files(artifact: ArtifactKind) -> Vec<&'static str> {
    match artifact {
        ArtifactKind::ToolCall => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/tokenizer.json",
            "onnx/model_quantized.onnx",
        ],
        ArtifactKind::FinalResponse => vec![
            "onnx/artifact_manifest.json",
            "onnx/labels.json",
            "onnx/thresholds.json",
            "onnx/tokenizer.json",
            "onnx/model_quantized.onnx",
        ],
    }
}

fn optional_sidecar_files(artifact: ArtifactKind) -> Vec<&'static str> {
    match artifact {
        ArtifactKind::ToolCall => vec![
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
        ArtifactKind::FinalResponse => vec![
            "onnx/tokenizer.json",
            "onnx/added_tokens.json",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_keep_existing_tool_call_output() {
        let cli = Cli {
            artifact: ArtifactSelection::ToolCall,
            repo: None,
            revision: None,
            output_dir: None,
            final_response_output_dir: None,
            classifier_model: "quantized".to_string(),
        };
        let plans = download_plans(&cli).expect("plans");
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].kind, ArtifactKind::ToolCall);
        assert_eq!(plans[0].repo, DEFAULT_CLASSIFIER_REPO);
        assert_eq!(plans[0].revision, DEFAULT_CLASSIFIER_REVISION);
        assert_eq!(
            plans[0].output_dir,
            PathBuf::from("target/classifier-artifacts")
        );
    }

    #[test]
    fn final_response_defaults_to_separate_output() {
        let cli = Cli {
            artifact: ArtifactSelection::FinalResponse,
            repo: None,
            revision: None,
            output_dir: None,
            final_response_output_dir: None,
            classifier_model: "quantized".to_string(),
        };
        let plans = download_plans(&cli).expect("plans");
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].kind, ArtifactKind::FinalResponse);
        assert_eq!(plans[0].repo, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO);
        assert_eq!(
            plans[0].revision,
            DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION
        );
        assert_eq!(
            plans[0].output_dir,
            PathBuf::from("target/final-response-classifier-artifacts")
        );
    }

    #[test]
    fn final_response_download_does_not_require_unpublished_tokenizer_json() {
        let required =
            required_files_for_model(ArtifactKind::FinalResponse, ClassifierModelKind::Quantized);
        assert!(!required.contains(&"onnx/tokenizer.json"));
        assert!(
            runtime_required_files(ArtifactKind::FinalResponse).contains(&"onnx/tokenizer.json")
        );
    }
}
