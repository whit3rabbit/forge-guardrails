//! Download local classifier artifacts for feature-gated ONNX tests.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::Parser;
use forge_guardrails::{ClassifierModelKind, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION};
use hf_hub::repository::{RepoTreeEntry, RepoTypeModel};
use hf_hub::{HFClientSync, HFRepositorySync};

#[derive(Debug, Parser)]
#[command(
    name = "download-classifier",
    about = "download ONNX tool-call classifier artifacts"
)]
struct Cli {
    /// Hugging Face model repo, in owner/name form.
    #[arg(long, default_value = DEFAULT_CLASSIFIER_REPO, value_name = "OWNER/NAME")]
    repo: String,

    /// Hugging Face revision.
    #[arg(long, default_value = DEFAULT_CLASSIFIER_REVISION, value_name = "REV")]
    revision: String,

    /// Output directory. Files are written under OUTPUT/onnx.
    #[arg(
        long,
        default_value = "target/classifier-artifacts",
        value_name = "DIR"
    )]
    output_dir: PathBuf,

    /// ONNX model to download.
    #[arg(long, default_value = "quantized", value_name = "quantized|full")]
    classifier_model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model_kind =
        ClassifierModelKind::from_str(&cli.classifier_model).map_err(|err| anyhow::anyhow!(err))?;
    let (owner, name) = parse_repo(&cli.repo)?;
    let output_dir = cli.output_dir;
    let onnx_dir = output_dir.join("onnx");

    let client = HFClientSync::new()?;
    let repo = client.model(owner, name);
    println!("repo={}", cli.repo);
    println!("revision={}", cli.revision);

    let available_files = repository_files(&repo, &cli.revision)?;
    for file in required_files_for_model(model_kind) {
        download_file(&repo, &output_dir, &cli.revision, file)?;
    }

    let mut optional_downloaded = 0usize;
    for file in optional_sidecar_files() {
        if available_files.contains(file) {
            download_file(&repo, &output_dir, &cli.revision, file)?;
            optional_downloaded += 1;
        }
    }
    println!("optional_sidecars={optional_downloaded}");
    println!("classifier_dir={}", onnx_dir.display());
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

fn required_files_for_model(model_kind: ClassifierModelKind) -> Vec<&'static str> {
    let mut files = vec![
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
    ];
    if model_kind == ClassifierModelKind::Full {
        files.push("onnx/model.onnx");
    }
    files
}

fn optional_sidecar_files() -> Vec<&'static str> {
    vec![
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
    ]
}
