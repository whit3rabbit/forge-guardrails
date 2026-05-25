//! Download local classifier artifacts for feature-gated ONNX tests.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::Parser;
use forge_guardrails::{ClassifierModelKind, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION};
use hf_hub::HFClientSync;

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
    for file in files_for_model(model_kind) {
        let path = repo
            .download_file()
            .filename(file.to_string())
            .revision(cli.revision.clone())
            .local_dir(output_dir.clone())
            .send()
            .with_context(|| format!("failed to download {file}"))?;
        println!("{file} -> {}", path.display());
    }

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

fn files_for_model(model_kind: ClassifierModelKind) -> Vec<&'static str> {
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
