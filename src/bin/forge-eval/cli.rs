use crate::ablation::parse_ablation;
use forge_guardrails::{ClassifierModelKind, ScorerMode};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cli {
    pub(crate) backend: String,
    pub(crate) model: Option<String>,
    pub(crate) gguf: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) runs: usize,
    pub(crate) num_ctx: i64,
    pub(crate) scenarios: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) output: Option<String>,
    pub(crate) ablation: String,
    pub(crate) mode: Option<String>,
    pub(crate) reasoning_budget: Option<String>,
    pub(crate) anthropic_api_key: Option<String>,
    pub(crate) classifier_dir: Option<String>,
    pub(crate) classifier_mode: String,
    pub(crate) classifier_model: String,
    pub(crate) classifier_max_latency_ms: Option<u64>,
    pub(crate) final_response_classifier_dir: Option<String>,
    pub(crate) final_response_classifier_mode: String,
    pub(crate) final_response_classifier_model: String,
    pub(crate) final_response_classifier_max_latency_ms: Option<u64>,
}

pub(crate) fn parse_args<I>(args: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut cli = Cli {
        backend: "openai-proxy".to_string(),
        model: None,
        gguf: None,
        base_url: None,
        runs: 1,
        num_ctx: 8192,
        scenarios: Vec::new(),
        stream: false,
        output: None,
        ablation: "reforged".to_string(),
        mode: None,
        reasoning_budget: None,
        anthropic_api_key: None,
        classifier_dir: env_optional("FORGE_CLASSIFIER_DIR"),
        classifier_mode: env_or("FORGE_CLASSIFIER_MODE", "shadow"),
        classifier_model: env_classifier_model(),
        classifier_max_latency_ms: env_optional_u64("FORGE_CLASSIFIER_MAX_LATENCY_MS")?,
        final_response_classifier_dir: env_optional("FORGE_FINAL_RESPONSE_CLASSIFIER_DIR"),
        final_response_classifier_mode: env_or("FORGE_FINAL_RESPONSE_CLASSIFIER_MODE", "shadow"),
        final_response_classifier_model: env_or(
            "FORGE_FINAL_RESPONSE_CLASSIFIER_MODEL",
            "quantized",
        ),
        final_response_classifier_max_latency_ms: env_optional_u64(
            "FORGE_FINAL_RESPONSE_CLASSIFIER_MAX_LATENCY_MS",
        )?,
    };

    let values: Vec<String> = args.into_iter().collect();
    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--backend" => cli.backend = take_one(&values, &mut index, "--backend")?,
            "--model" => cli.model = Some(take_one(&values, &mut index, "--model")?),
            "--gguf" => cli.gguf = Some(take_one(&values, &mut index, "--gguf")?),
            "--base-url" => cli.base_url = Some(take_one(&values, &mut index, "--base-url")?),
            "--runs" => {
                let raw = take_one(&values, &mut index, "--runs")?;
                cli.runs = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --runs value: {raw}"))?;
            }
            "--num-ctx" => {
                let raw = take_one(&values, &mut index, "--num-ctx")?;
                cli.num_ctx = raw
                    .parse::<i64>()
                    .map_err(|_| format!("invalid --num-ctx value: {raw}"))?;
            }
            "--scenario" => {
                cli.scenarios = take_many(&values, &mut index, "--scenario")?;
            }
            "--stream" => cli.stream = true,
            "--output" => cli.output = Some(take_one(&values, &mut index, "--output")?),
            "--ablation" => cli.ablation = take_one(&values, &mut index, "--ablation")?,
            "--mode" | "--llamafile-mode" => {
                let flag = values[index].clone();
                cli.mode = Some(take_one(&values, &mut index, &flag)?)
            }
            "--reasoning-budget" => {
                cli.reasoning_budget = Some(take_one(&values, &mut index, "--reasoning-budget")?)
            }
            "--anthropic-api-key" => {
                cli.anthropic_api_key = Some(take_one(&values, &mut index, "--anthropic-api-key")?)
            }
            "--classifier-dir" => {
                cli.classifier_dir = Some(take_one(&values, &mut index, "--classifier-dir")?)
            }
            "--classifier-mode" => {
                cli.classifier_mode = take_one(&values, &mut index, "--classifier-mode")?
            }
            "--classifier-model" => {
                cli.classifier_model = take_one(&values, &mut index, "--classifier-model")?
            }
            "--classifier-max-latency-ms" => {
                cli.classifier_max_latency_ms = Some(take_u64(
                    &values,
                    &mut index,
                    "--classifier-max-latency-ms",
                )?)
            }
            "--final-response-classifier-dir" => {
                cli.final_response_classifier_dir = Some(take_one(
                    &values,
                    &mut index,
                    "--final-response-classifier-dir",
                )?)
            }
            "--final-response-classifier-mode" => {
                cli.final_response_classifier_mode =
                    take_one(&values, &mut index, "--final-response-classifier-mode")?
            }
            "--final-response-classifier-model" => {
                cli.final_response_classifier_model =
                    take_one(&values, &mut index, "--final-response-classifier-model")?
            }
            "--final-response-classifier-max-latency-ms" => {
                cli.final_response_classifier_max_latency_ms = Some(take_u64(
                    &values,
                    &mut index,
                    "--final-response-classifier-max-latency-ms",
                )?)
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag: {flag}")),
            value => return Err(format!("unexpected argument: {value}")),
        }
        index += 1;
    }

    if cli.runs == 0 {
        return Err("--runs must be at least 1".to_string());
    }
    if cli.num_ctx <= 0 {
        return Err("--num-ctx must be at least 1".to_string());
    }
    parse_ablation(&cli.ablation)?;
    ScorerMode::from_str(&cli.classifier_mode)?;
    ClassifierModelKind::from_str(&cli.classifier_model)?;
    ScorerMode::from_str(&cli.final_response_classifier_mode)?;
    ClassifierModelKind::from_str(&cli.final_response_classifier_model)?;
    Ok(cli)
}

fn take_one(values: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    values
        .get(*index)
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn take_u64(values: &[String], index: &mut usize, flag: &str) -> Result<u64, String> {
    let raw = take_one(values, index, flag)?;
    raw.parse::<u64>()
        .map_err(|_| format!("{flag} must be a non-negative integer, got '{raw}'"))
}

fn take_many(values: &[String], index: &mut usize, flag: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    *index += 1;
    while *index < values.len() {
        let value = &values[*index];
        if value.starts_with("--") {
            *index -= 1;
            break;
        }
        out.push(value.clone());
        *index += 1;
    }
    if out.is_empty() {
        Err(format!("{flag} requires at least one value"))
    } else {
        Ok(out)
    }
}

fn env_optional(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_or(key: &str, default: &str) -> String {
    env_optional(key).unwrap_or_else(|| default.to_string())
}

fn env_classifier_model() -> String {
    if let Some(value) = env_optional("FORGE_CLASSIFIER_MODEL") {
        return value;
    }
    match std::env::var("FORGE_CLASSIFIER_USE_QUANTIZED") {
        Ok(raw)
            if matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            ) =>
        {
            "full".to_string()
        }
        _ => "quantized".to_string(),
    }
}

fn env_optional_u64(key: &str) -> Result<Option<u64>, String> {
    let Some(raw) = env_optional(key) else {
        return Ok(None);
    };
    raw.parse::<u64>()
        .map(Some)
        .map_err(|_| format!("{key} must be a non-negative integer, got '{raw}'"))
}

pub(crate) fn print_help() {
    println!(
        "forge-eval\n\n\
         Usage: forge-eval --backend openai-proxy --base-url URL --model MODEL [options]\n\n\
         Options:\n\
           --backend openai-proxy|ollama|llamaserver|llamafile|anthropic\n\
           --model MODEL\n\
           --gguf PATH\n\
           --base-url URL\n\
           --runs N\n\
           --num-ctx TOKENS (default: 8192; also sent as Ollama num_ctx)\n\
           --scenario NAME [NAME ...]\n\
           --stream\n\
           --ablation reforged|no_rescue|no_nudge|no_steps|no_recovery|no_compact|bare\n\
           --output PATH\n\
           --mode native|prompt|auto\n\
           --reasoning-budget TOKENS (metadata only; start local server with the same flag)\n\
           --anthropic-api-key KEY\n\
           --classifier-dir PATH\n\
           --classifier-mode disabled|shadow|advisory|enforce (default: shadow)\n\
           --classifier-model quantized|full (default: quantized)\n\
           --classifier-max-latency-ms MS\n\
           --final-response-classifier-dir PATH\n\
           --final-response-classifier-mode disabled|shadow|advisory|enforce (default: shadow)\n\
           --final-response-classifier-model quantized|full (default: quantized)\n\
           --final-response-classifier-max-latency-ms MS"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(items: &[&str]) -> Cli {
        parse_args(items.iter().map(|item| item.to_string())).expect("parse")
    }

    #[test]
    fn parses_multiple_scenarios() {
        let cli = parse(&[
            "--backend",
            "openai-proxy",
            "--model",
            "test-model",
            "--scenario",
            "basic_2step",
            "sequential_3step",
            "--stream",
        ]);
        assert_eq!(
            cli.scenarios,
            vec!["basic_2step".to_string(), "sequential_3step".to_string()]
        );
        assert!(cli.stream);
        assert_eq!(cli.num_ctx, 8192);
        assert_eq!(cli.classifier_dir, None);
        assert_eq!(cli.classifier_mode, "shadow");
        assert_eq!(cli.classifier_model, "quantized");
        assert_eq!(cli.classifier_max_latency_ms, None);
        assert_eq!(cli.final_response_classifier_dir, None);
        assert_eq!(cli.final_response_classifier_mode, "shadow");
        assert_eq!(cli.final_response_classifier_model, "quantized");
        assert_eq!(cli.final_response_classifier_max_latency_ms, None);
    }

    #[test]
    fn parses_num_ctx() {
        let cli = parse(&["--num-ctx", "16384"]);
        assert_eq!(cli.num_ctx, 16384);
    }

    #[test]
    fn rejects_zero_runs() {
        let err = parse_args(["--runs".to_string(), "0".to_string()]).unwrap_err();
        assert!(err.contains("at least 1"));
    }

    #[test]
    fn rejects_zero_num_ctx() {
        let err = parse_args(["--num-ctx".to_string(), "0".to_string()]).unwrap_err();
        assert!(err.contains("at least 1"));
    }

    #[test]
    fn parses_classifier_flags() {
        let cli = parse(&[
            "--classifier-dir",
            "target/classifier-artifacts/onnx",
            "--classifier-mode",
            "shadow",
            "--classifier-model",
            "quantized",
            "--classifier-max-latency-ms",
            "25",
        ]);
        assert_eq!(
            cli.classifier_dir.as_deref(),
            Some("target/classifier-artifacts/onnx")
        );
        assert_eq!(cli.classifier_mode, "shadow");
        assert_eq!(cli.classifier_model, "quantized");
        assert_eq!(cli.classifier_max_latency_ms, Some(25));
    }

    #[test]
    fn parses_final_response_classifier_flags() {
        let cli = parse(&[
            "--final-response-classifier-dir",
            "target/final-response-artifacts/onnx",
            "--final-response-classifier-mode",
            "advisory",
            "--final-response-classifier-model",
            "full",
            "--final-response-classifier-max-latency-ms",
            "40",
        ]);
        assert_eq!(
            cli.final_response_classifier_dir.as_deref(),
            Some("target/final-response-artifacts/onnx")
        );
        assert_eq!(cli.final_response_classifier_mode, "advisory");
        assert_eq!(cli.final_response_classifier_model, "full");
        assert_eq!(cli.final_response_classifier_max_latency_ms, Some(40));
    }
}
