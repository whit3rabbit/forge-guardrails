#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Command {
    Prompts(PromptsCli),
    Capture(CaptureCli),
    Review(Box<ReviewCli>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptsCli {
    pub(crate) model: String,
    pub(crate) output: String,
    pub(crate) domains: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CaptureCli {
    pub(crate) proxy_base_url: String,
    pub(crate) model: String,
    pub(crate) output: String,
    pub(crate) max_turns: usize,
    pub(crate) domains: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReviewCli {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) env_file: String,
    pub(crate) provider: String,
    pub(crate) verifier_provider: String,
    pub(crate) minimax_model: String,
    pub(crate) openrouter_model: String,
    pub(crate) reviewer_base_url: String,
    pub(crate) reviewer_model: String,
    pub(crate) reviewer_api_key: Option<String>,
    pub(crate) verifier_base_url: String,
    pub(crate) verifier_model: String,
    pub(crate) verifier_api_key: Option<String>,
    pub(crate) max_alternatives_per_group: usize,
    pub(crate) max_alternative_ratio: f64,
}

const DEFAULT_PROXY_BASE_URL: &str = "http://127.0.0.1:8081/v1";
const DEFAULT_PROMPTS_OUTPUT: &str = "target/dataset/tool_prompts.jsonl";
const DEFAULT_CAPTURE_OUTPUT: &str = "target/dataset/capture.jsonl";
const DEFAULT_REVIEW_OUTPUT: &str = "target/dataset/training.toolcall.jsonl";
const DEFAULT_DOMAINS: &str = "repo_docs,shopping,calendar,support";
const DEFAULT_ENV_FILE: &str = "notebook/generatetd/.env";
const DEFAULT_MINIMAX_MODEL: &str = "MiniMax-M2.7";
const DEFAULT_OPENROUTER_MODEL: &str = "deepseek/deepseek-v4-flash:free";

pub(crate) fn parse_args<I>(args: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let values: Vec<String> = args.into_iter().collect();
    let Some(command) = values.first().map(String::as_str) else {
        return Err("__help__".to_string());
    };
    match command {
        "--help" | "-h" => Err("__help__".to_string()),
        "prompts" => parse_prompts(&values[1..]),
        "capture" => parse_capture(&values[1..]),
        "review" => parse_review(&values[1..]),
        other => Err(format!("unknown command: {other}")),
    }
}

fn parse_prompts(values: &[String]) -> Result<Command, String> {
    let mut cli = PromptsCli {
        model: "test-model".to_string(),
        output: DEFAULT_PROMPTS_OUTPUT.to_string(),
        domains: parse_domains(DEFAULT_DOMAINS)?,
    };

    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--model" => cli.model = take_one(values, &mut index, "--model")?,
            "--output" => cli.output = take_one(values, &mut index, "--output")?,
            "--domains" => {
                cli.domains = parse_domains(&take_one(values, &mut index, "--domains")?)?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown prompts flag: {flag}")),
            value => return Err(format!("unexpected prompts argument: {value}")),
        }
        index += 1;
    }

    if cli.model.trim().is_empty() {
        return Err("--model must not be empty".to_string());
    }
    Ok(Command::Prompts(cli))
}

fn parse_capture(values: &[String]) -> Result<Command, String> {
    let mut cli = CaptureCli {
        proxy_base_url: DEFAULT_PROXY_BASE_URL.to_string(),
        model: String::new(),
        output: DEFAULT_CAPTURE_OUTPUT.to_string(),
        max_turns: 4,
        domains: parse_domains(DEFAULT_DOMAINS)?,
    };

    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--proxy-base-url" => {
                cli.proxy_base_url = take_one(values, &mut index, "--proxy-base-url")?
            }
            "--model" => cli.model = take_one(values, &mut index, "--model")?,
            "--output" => cli.output = take_one(values, &mut index, "--output")?,
            "--max-turns" => {
                cli.max_turns = take_usize(values, &mut index, "--max-turns")?;
            }
            "--domains" => {
                cli.domains = parse_domains(&take_one(values, &mut index, "--domains")?)?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown capture flag: {flag}")),
            value => return Err(format!("unexpected capture argument: {value}")),
        }
        index += 1;
    }

    if cli.model.trim().is_empty() {
        return Err("capture requires --model".to_string());
    }
    if cli.max_turns == 0 {
        return Err("--max-turns must be at least 1".to_string());
    }
    Ok(Command::Capture(cli))
}

fn parse_review(values: &[String]) -> Result<Command, String> {
    let mut cli = ReviewCli {
        input: DEFAULT_CAPTURE_OUTPUT.to_string(),
        output: DEFAULT_REVIEW_OUTPUT.to_string(),
        env_file: DEFAULT_ENV_FILE.to_string(),
        provider: "auto".to_string(),
        verifier_provider: "same".to_string(),
        minimax_model: DEFAULT_MINIMAX_MODEL.to_string(),
        openrouter_model: DEFAULT_OPENROUTER_MODEL.to_string(),
        reviewer_base_url: String::new(),
        reviewer_model: String::new(),
        reviewer_api_key: None,
        verifier_base_url: String::new(),
        verifier_model: String::new(),
        verifier_api_key: None,
        max_alternatives_per_group: 2,
        max_alternative_ratio: 1.0 / 3.0,
    };

    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--input" => cli.input = take_one(values, &mut index, "--input")?,
            "--output" => cli.output = take_one(values, &mut index, "--output")?,
            "--env-file" => cli.env_file = take_one(values, &mut index, "--env-file")?,
            "--provider" => cli.provider = take_one(values, &mut index, "--provider")?,
            "--verifier-provider" => {
                cli.verifier_provider = take_one(values, &mut index, "--verifier-provider")?
            }
            "--reviewer-base-url" => {
                cli.reviewer_base_url = take_one(values, &mut index, "--reviewer-base-url")?
            }
            "--reviewer-model" => {
                cli.reviewer_model = take_one(values, &mut index, "--reviewer-model")?
            }
            "--reviewer-api-key" => {
                cli.reviewer_api_key = Some(take_one(values, &mut index, "--reviewer-api-key")?)
            }
            "--minimax-model" => {
                cli.minimax_model = take_one(values, &mut index, "--minimax-model")?;
            }
            "--openrouter-model" => {
                cli.openrouter_model = take_one(values, &mut index, "--openrouter-model")?;
            }
            "--verifier-base-url" => {
                cli.verifier_base_url = take_one(values, &mut index, "--verifier-base-url")?
            }
            "--verifier-model" => {
                cli.verifier_model = take_one(values, &mut index, "--verifier-model")?
            }
            "--verifier-api-key" => {
                cli.verifier_api_key = Some(take_one(values, &mut index, "--verifier-api-key")?)
            }
            "--max-alternatives-per-group" => {
                cli.max_alternatives_per_group =
                    take_usize(values, &mut index, "--max-alternatives-per-group")?;
            }
            "--max-alternative-ratio" => {
                cli.max_alternative_ratio =
                    take_f64(values, &mut index, "--max-alternative-ratio")?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown review flag: {flag}")),
            value => return Err(format!("unexpected review argument: {value}")),
        }
        index += 1;
    }

    validate_provider_flag(
        "--provider",
        &cli.provider,
        &["auto", "minimax", "openrouter"],
    )?;
    validate_provider_flag(
        "--verifier-provider",
        &cli.verifier_provider,
        &["same", "auto", "minimax", "openrouter"],
    )?;
    if !(0.0..=1.0).contains(&cli.max_alternative_ratio) {
        return Err("--max-alternative-ratio must be between 0.0 and 1.0".to_string());
    }

    Ok(Command::Review(Box::new(cli)))
}

fn validate_provider_flag(flag: &str, value: &str, allowed: &[&str]) -> Result<(), String> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(format!("{flag} must be one of {}", allowed.join("|")))
    }
}

fn take_one(values: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    values
        .get(*index)
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn take_usize(values: &[String], index: &mut usize, flag: &str) -> Result<usize, String> {
    let raw = take_one(values, index, flag)?;
    raw.parse::<usize>()
        .map_err(|_| format!("{flag} must be a non-negative integer, got '{raw}'"))
}

fn take_f64(values: &[String], index: &mut usize, flag: &str) -> Result<f64, String> {
    let raw = take_one(values, index, flag)?;
    raw.parse::<f64>()
        .map_err(|_| format!("{flag} must be a number, got '{raw}'"))
}

fn parse_domains(raw: &str) -> Result<Vec<String>, String> {
    let domains = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if domains.is_empty() {
        Err("--domains must include at least one domain".to_string())
    } else {
        Ok(domains)
    }
}

pub(crate) fn default_minimax_model() -> &'static str {
    DEFAULT_MINIMAX_MODEL
}

pub(crate) fn default_openrouter_model() -> &'static str {
    DEFAULT_OPENROUTER_MODEL
}

pub(crate) fn print_help() {
    println!(
        "forge-dataset\n\n\
         Usage:\n\
           forge-dataset prompts [options]\n\
           forge-dataset capture --model MODEL [options]\n\
           forge-dataset review --provider auto|minimax|openrouter [options]\n\n\
         Prompt options:\n\
           --model MODEL (default: test-model)\n\
           --output PATH (default: target/dataset/tool_prompts.jsonl)\n\
          --domains CSV (default: repo_docs,shopping,calendar,support; also supports forge_eval)\n\n\
         Capture options:\n\
           --proxy-base-url URL (default: http://127.0.0.1:8081/v1)\n\
           --model MODEL\n\
           --output PATH (default: target/dataset/capture.jsonl)\n\
           --max-turns N (default: 4)\n\
          --domains CSV (default: repo_docs,shopping,calendar,support; also supports forge_eval)\n\n\
         Review options:\n\
           --input PATH (default: target/dataset/capture.jsonl)\n\
           --output PATH (default: target/dataset/training.toolcall.jsonl)\n\
           --env-file PATH (default: notebook/generatetd/.env)\n\
           --provider auto|minimax|openrouter (default: auto)\n\
           --verifier-provider same|auto|minimax|openrouter (default: same)\n\
           --minimax-model MODEL (default: MiniMax-M2.7)\n\
           --openrouter-model MODEL (default: deepseek/deepseek-v4-flash:free)\n\
           --reviewer-base-url URL (manual override)\n\
           --reviewer-model MODEL (manual override)\n\
           --reviewer-api-key KEY (or FORGE_DATASET_REVIEWER_API_KEY / OPENAI_API_KEY)\n\
           --verifier-base-url URL (manual override)\n\
           --verifier-model MODEL (manual override)\n\
           --verifier-api-key KEY (or FORGE_DATASET_VERIFIER_API_KEY / OPENAI_API_KEY)\n\
           --max-alternatives-per-group N (default: 2)\n\
           --max-alternative-ratio R (default: 0.333333)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(items: &[&str]) -> Result<Command, String> {
        parse_args(items.iter().map(|item| item.to_string()))
    }

    #[test]
    fn capture_defaults_match_contract() {
        let command = parse(&["capture", "--model", "test-model"]).expect("parse");
        let Command::Capture(cli) = command else {
            panic!("expected capture");
        };
        assert_eq!(cli.proxy_base_url, DEFAULT_PROXY_BASE_URL);
        assert_eq!(cli.output, DEFAULT_CAPTURE_OUTPUT);
        assert_eq!(cli.max_turns, 4);
        assert_eq!(
            cli.domains,
            vec!["repo_docs", "shopping", "calendar", "support"]
        );
    }

    #[test]
    fn prompts_defaults_match_contract() {
        let command = parse(&["prompts"]).expect("parse");
        let Command::Prompts(cli) = command else {
            panic!("expected prompts");
        };
        assert_eq!(cli.model, "test-model");
        assert_eq!(cli.output, DEFAULT_PROMPTS_OUTPUT);
        assert_eq!(
            cli.domains,
            vec!["repo_docs", "shopping", "calendar", "support"]
        );
    }

    #[test]
    fn review_defaults_to_auto_provider() {
        let command = parse(&["review"]).expect("parse");
        let Command::Review(cli) = command else {
            panic!("expected review");
        };
        assert_eq!(cli.provider, "auto");
        assert_eq!(cli.verifier_provider, "same");
        assert_eq!(cli.env_file, DEFAULT_ENV_FILE);
    }

    #[test]
    fn review_parses_caps() {
        let command = parse(&[
            "review",
            "--provider",
            "openrouter",
            "--verifier-provider",
            "minimax",
            "--max-alternatives-per-group",
            "3",
            "--max-alternative-ratio",
            "0.25",
        ])
        .expect("parse");
        let Command::Review(cli) = command else {
            panic!("expected review");
        };
        assert_eq!(cli.provider, "openrouter");
        assert_eq!(cli.verifier_provider, "minimax");
        assert_eq!(cli.max_alternatives_per_group, 3);
        assert_eq!(cli.max_alternative_ratio, 0.25);
    }
}
