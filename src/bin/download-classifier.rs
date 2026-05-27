//! Download local classifier artifacts for feature-gated ONNX tests.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use forge_guardrails::{
    download_classifier_artifact_tree, ClassifierArtifactKind, ClassifierDownloadPlan,
    ClassifierModelKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ArtifactSelection {
    ToolCall,
    FinalResponse,
    Both,
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

    for plan in download_plans(&cli)? {
        download_classifier_artifact_tree(&plan, model_kind, |line| println!("{line}"))?;
    }

    Ok(())
}

fn download_plans(cli: &Cli) -> Result<Vec<ClassifierDownloadPlan>> {
    if cli.artifact == ArtifactSelection::Both && (cli.repo.is_some() || cli.revision.is_some()) {
        anyhow::bail!("--repo and --revision cannot be combined with --artifact both");
    }

    artifact_kinds(cli.artifact)
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
            Ok(ClassifierDownloadPlan {
                kind,
                repo,
                revision,
                output_dir,
            })
        })
        .collect()
}

fn artifact_kinds(selection: ArtifactSelection) -> Vec<ClassifierArtifactKind> {
    match selection {
        ArtifactSelection::ToolCall => vec![ClassifierArtifactKind::ToolCall],
        ArtifactSelection::FinalResponse => vec![ClassifierArtifactKind::FinalResponse],
        ArtifactSelection::Both => vec![
            ClassifierArtifactKind::ToolCall,
            ClassifierArtifactKind::FinalResponse,
        ],
    }
}

fn output_dir_for(cli: &Cli, kind: ClassifierArtifactKind) -> PathBuf {
    match kind {
        ClassifierArtifactKind::ToolCall => cli
            .output_dir
            .clone()
            .unwrap_or_else(|| kind.default_eval_output_dir()),
        ClassifierArtifactKind::FinalResponse
            if cli.artifact == ArtifactSelection::FinalResponse =>
        {
            cli.output_dir
                .clone()
                .unwrap_or_else(|| kind.default_eval_output_dir())
        }
        ClassifierArtifactKind::FinalResponse => cli
            .final_response_output_dir
            .clone()
            .unwrap_or_else(|| kind.default_eval_output_dir()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_guardrails::{
        required_files_for_model, runtime_required_files, DEFAULT_CLASSIFIER_REPO,
        DEFAULT_CLASSIFIER_REVISION, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO,
        DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
    };

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
        assert_eq!(plans[0].kind, ClassifierArtifactKind::ToolCall);
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
        assert_eq!(plans[0].kind, ClassifierArtifactKind::FinalResponse);
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
    fn final_response_download_requires_runtime_tokenizer_json() {
        let required = required_files_for_model(
            ClassifierArtifactKind::FinalResponse,
            ClassifierModelKind::Quantized,
        );
        assert!(required.contains(&"onnx/tokenizer.json"));
        assert!(
            runtime_required_files(ClassifierArtifactKind::FinalResponse)
                .contains(&"onnx/tokenizer.json")
        );
    }
}
