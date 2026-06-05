#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Command {
    Prompts(PromptsCli),
    Capture(CaptureCli),
    Review(Box<ReviewCli>),
    AgentLogs(Box<AgentLogsCli>),
    Assemble(AssembleCli),
    Validate(ValidateCli),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptsCli {
    pub(crate) model: String,
    pub(crate) output: String,
    pub(crate) domains: Vec<String>,
    pub(crate) runs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CaptureCli {
    pub(crate) proxy_base_url: String,
    pub(crate) model: String,
    pub(crate) output: String,
    pub(crate) max_turns: usize,
    pub(crate) domains: Vec<String>,
    pub(crate) runs: usize,
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
    pub(crate) concurrency: usize,
    pub(crate) resume: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentLogsCli {
    pub(crate) out: String,
    pub(crate) env_file: String,
    pub(crate) provider: String,
    pub(crate) verifier_provider: String,
    pub(crate) minimax_model: String,
    pub(crate) openrouter_model: String,
    pub(crate) minimax_api_key: Option<String>,
    pub(crate) openrouter_api_key: Option<String>,
    pub(crate) no_api: bool,
    pub(crate) limit: Option<usize>,
    pub(crate) since: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) include_codex: bool,
    pub(crate) include_claude: bool,
    pub(crate) synthetic_balanced: usize,
    pub(crate) synthetic_missing_argument: usize,
    pub(crate) synthetic_tool_not_needed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AssembleCli {
    pub(crate) inputs: Vec<String>,
    pub(crate) out_dir: String,
    pub(crate) combined_output: String,
    pub(crate) drop_conflicts: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidateCli {
    pub(crate) inputs: Vec<String>,
}

const DEFAULT_PROXY_BASE_URL: &str = "http://127.0.0.1:8081/v1";
const DEFAULT_PROMPTS_OUTPUT: &str = "target/dataset/tool_prompts.jsonl";
const DEFAULT_CAPTURE_OUTPUT: &str = "target/dataset/capture.jsonl";
const DEFAULT_REVIEW_OUTPUT: &str = "target/dataset/training.toolcall.jsonl";
const DEFAULT_AGENT_LOGS_OUT: &str = "target/dataset/agent_logs";
const DEFAULT_ASSEMBLE_OUT_DIR: &str = "target/dataset/assembled";
const DEFAULT_COMBINED_OUTPUT: &str = "training.toolcall.combined.jsonl";
const DEFAULT_DOMAINS: &str = "repo_docs,shopping,calendar,support";
const DEFAULT_ENV_FILE: &str = "notebook/generatetd/.env";
const DEFAULT_MINIMAX_MODEL: &str = "MiniMax-M2.7";
const DEFAULT_OPENROUTER_MODEL: &str = "openrouter/free";

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
        "agent-logs" => parse_agent_logs(&values[1..]),
        "assemble" => parse_assemble(&values[1..]),
        "validate" => parse_validate(&values[1..]),
        other => Err(format!("unknown command: {other}")),
    }
}

fn parse_prompts(values: &[String]) -> Result<Command, String> {
    let mut cli = PromptsCli {
        model: "test-model".to_string(),
        output: DEFAULT_PROMPTS_OUTPUT.to_string(),
        domains: parse_domains(DEFAULT_DOMAINS)?,
        runs: 1,
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
            "--runs" => cli.runs = take_usize(values, &mut index, "--runs")?,
            flag if flag.starts_with("--") => return Err(format!("unknown prompts flag: {flag}")),
            value => return Err(format!("unexpected prompts argument: {value}")),
        }
        index += 1;
    }

    if cli.model.trim().is_empty() {
        return Err("--model must not be empty".to_string());
    }
    if cli.runs == 0 {
        return Err("--runs must be at least 1".to_string());
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
        runs: 1,
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
            "--runs" => cli.runs = take_usize(values, &mut index, "--runs")?,
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
    if cli.runs == 0 {
        return Err("--runs must be at least 1".to_string());
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
        minimax_model: String::new(),
        openrouter_model: String::new(),
        reviewer_base_url: String::new(),
        reviewer_model: String::new(),
        reviewer_api_key: None,
        verifier_base_url: String::new(),
        verifier_model: String::new(),
        verifier_api_key: None,
        max_alternatives_per_group: 2,
        max_alternative_ratio: 1.0 / 3.0,
        concurrency: 1,
        resume: false,
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
            "--concurrency" => {
                cli.concurrency = take_usize(values, &mut index, "--concurrency")?;
            }
            "--resume" => cli.resume = true,
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
    if cli.concurrency == 0 || cli.concurrency > 32 {
        return Err("--concurrency must be between 1 and 32".to_string());
    }

    Ok(Command::Review(Box::new(cli)))
}

fn parse_agent_logs(values: &[String]) -> Result<Command, String> {
    let mut cli = AgentLogsCli {
        out: DEFAULT_AGENT_LOGS_OUT.to_string(),
        env_file: DEFAULT_ENV_FILE.to_string(),
        provider: "auto".to_string(),
        verifier_provider: "same".to_string(),
        minimax_model: String::new(),
        openrouter_model: String::new(),
        minimax_api_key: None,
        openrouter_api_key: None,
        no_api: false,
        limit: None,
        since: None,
        project: None,
        include_codex: true,
        include_claude: true,
        synthetic_balanced: 0,
        synthetic_missing_argument: 0,
        synthetic_tool_not_needed: 0,
    };

    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--out" | "--out-dir" => cli.out = take_one(values, &mut index, "--out")?,
            "--env-file" => cli.env_file = take_one(values, &mut index, "--env-file")?,
            "--provider" => cli.provider = take_one(values, &mut index, "--provider")?,
            "--verifier-provider" => {
                cli.verifier_provider = take_one(values, &mut index, "--verifier-provider")?
            }
            "--minimax-model" => {
                cli.minimax_model = take_one(values, &mut index, "--minimax-model")?;
            }
            "--openrouter-model" => {
                cli.openrouter_model = take_one(values, &mut index, "--openrouter-model")?;
            }
            "--minimax-api-key" => {
                cli.minimax_api_key = Some(take_one(values, &mut index, "--minimax-api-key")?);
            }
            "--openrouter-api-key" => {
                cli.openrouter_api_key =
                    Some(take_one(values, &mut index, "--openrouter-api-key")?);
            }
            "--no-api" => cli.no_api = true,
            "--limit" => cli.limit = Some(take_usize(values, &mut index, "--limit")?),
            "--since" => cli.since = Some(take_one(values, &mut index, "--since")?),
            "--project" => cli.project = Some(take_one(values, &mut index, "--project")?),
            "--include-codex" => cli.include_codex = true,
            "--no-codex" => cli.include_codex = false,
            "--include-claude" => cli.include_claude = true,
            "--no-claude" => cli.include_claude = false,
            "--synthetic-balanced" => {
                cli.synthetic_balanced = take_usize(values, &mut index, "--synthetic-balanced")?;
            }
            "--synthetic-missing-argument" => {
                cli.synthetic_missing_argument =
                    take_usize(values, &mut index, "--synthetic-missing-argument")?;
            }
            "--synthetic-tool-not-needed" => {
                cli.synthetic_tool_not_needed =
                    take_usize(values, &mut index, "--synthetic-tool-not-needed")?;
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown agent-logs flag: {flag}"));
            }
            value => return Err(format!("unexpected agent-logs argument: {value}")),
        }
        index += 1;
    }

    validate_provider_flag(
        "--provider",
        &cli.provider,
        &["auto", "minimax", "openrouter", "none"],
    )?;
    validate_provider_flag(
        "--verifier-provider",
        &cli.verifier_provider,
        &["same", "auto", "minimax", "openrouter", "none"],
    )?;
    if cli.synthetic_balanced > 0
        && (cli.synthetic_missing_argument > 0 || cli.synthetic_tool_not_needed > 0)
    {
        return Err(
            "--synthetic-balanced cannot be combined with per-type synthetic count flags"
                .to_string(),
        );
    }
    if !cli.include_codex && !cli.include_claude {
        return Err("agent-logs requires at least one of Codex or Claude logs".to_string());
    }

    Ok(Command::AgentLogs(Box::new(cli)))
}

fn parse_assemble(values: &[String]) -> Result<Command, String> {
    let mut cli = AssembleCli {
        inputs: Vec::new(),
        out_dir: DEFAULT_ASSEMBLE_OUT_DIR.to_string(),
        combined_output: DEFAULT_COMBINED_OUTPUT.to_string(),
        drop_conflicts: false,
    };

    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--input" => cli.inputs.push(take_one(values, &mut index, "--input")?),
            "--out-dir" => cli.out_dir = take_one(values, &mut index, "--out-dir")?,
            "--combined-output" => {
                cli.combined_output = take_one(values, &mut index, "--combined-output")?
            }
            "--drop-conflicts" => cli.drop_conflicts = true,
            flag if flag.starts_with("--") => return Err(format!("unknown assemble flag: {flag}")),
            value => cli.inputs.push(value.to_string()),
        }
        index += 1;
    }

    if cli.inputs.is_empty() {
        return Err("assemble requires at least one --input path".to_string());
    }
    if cli.combined_output.trim().is_empty() {
        return Err("--combined-output must not be empty".to_string());
    }
    Ok(Command::Assemble(cli))
}

fn parse_validate(values: &[String]) -> Result<Command, String> {
    let mut cli = ValidateCli { inputs: Vec::new() };

    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--input" => cli.inputs.push(take_one(values, &mut index, "--input")?),
            flag if flag.starts_with("--") => return Err(format!("unknown validate flag: {flag}")),
            value => cli.inputs.push(value.to_string()),
        }
        index += 1;
    }

    if cli.inputs.is_empty() {
        return Err("validate requires at least one input path".to_string());
    }
    Ok(Command::Validate(cli))
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
           forge-dataset review --provider auto|minimax|openrouter [options]\n\
           forge-dataset agent-logs [options]\n\
           forge-dataset assemble --input PATH [--input PATH ...] [options]\n\
           forge-dataset validate --input PATH [--input PATH ...]\n\n\
         Prompt options:\n\
           --model MODEL (default: test-model)\n\
           --output PATH (default: target/dataset/tool_prompts.jsonl)\n\
           --runs N (default: 1)\n\
           --domains CSV (default: repo_docs,shopping,calendar,support; also supports forge_eval)\n\n\
         Capture options:\n\
           --proxy-base-url URL (default: http://127.0.0.1:8081/v1)\n\
           --model MODEL\n\
           --output PATH (default: target/dataset/capture.jsonl)\n\
           --max-turns N (default: 4)\n\
           --runs N (default: 1)\n\
           --domains CSV (default: repo_docs,shopping,calendar,support; also supports forge_eval)\n\n\
         Review options:\n\
           --input PATH (default: target/dataset/capture.jsonl)\n\
           --output PATH (default: target/dataset/training.toolcall.jsonl)\n\
           --env-file PATH (default: notebook/generatetd/.env)\n\
           --provider auto|minimax|openrouter (default: auto)\n\
           --verifier-provider same|auto|minimax|openrouter (default: same)\n\
           --minimax-model MODEL (default: MiniMax-M2.7)\n\
           --openrouter-model MODEL (default: openrouter/free)\n\
           --reviewer-base-url URL (manual override)\n\
           --reviewer-model MODEL (manual override)\n\
           --reviewer-api-key KEY (or FORGE_DATASET_REVIEWER_API_KEY / OPENAI_API_KEY)\n\
           --verifier-base-url URL (manual override)\n\
           --verifier-model MODEL (manual override)\n\
           --verifier-api-key KEY (or FORGE_DATASET_VERIFIER_API_KEY / OPENAI_API_KEY)\n\
           --max-alternatives-per-group N (default: 2)\n\
           --max-alternative-ratio R (default: 0.333333)\n\
           --concurrency N (default: 1; max: 32; parallelizes capture review)\n\
           --resume (skip rows already present in output/rejects)\n\n\
         Agent log options:\n\
           --out DIR (default: target/dataset/agent_logs)\n\
           --env-file PATH (default: notebook/generatetd/.env)\n\
           --provider auto|minimax|openrouter|none (default: auto)\n\
           --verifier-provider same|auto|minimax|openrouter|none (default: same)\n\
           --minimax-model MODEL (default: MiniMax-M2.7)\n\
           --openrouter-model MODEL (default: openrouter/free)\n\
           --minimax-api-key KEY / --openrouter-api-key KEY\n\
           --no-api\n\
           --limit N\n\
           --since YYYY-MM-DD\n\
           --project TEXT\n\
           --include-codex / --no-codex\n\
           --include-claude / --no-claude\n\
           --synthetic-balanced N\n\
           --synthetic-missing-argument N\n\
           --synthetic-tool-not-needed N\n\n\
         Assemble options:\n\
           --input PATH (may be repeated; positional paths are also accepted)\n\
           --out-dir DIR (default: target/dataset/assembled)\n\
           --combined-output NAME (default: training.toolcall.combined.jsonl)\n\
           --drop-conflicts (exclude all inputs with conflicting labels)\n\n\
         Validate options:\n\
           --input PATH (may be repeated; positional paths are also accepted)"
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
        assert_eq!(cli.runs, 1);
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
        assert_eq!(cli.runs, 1);
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
        assert_eq!(cli.concurrency, 1);
        assert!(!cli.resume);
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

    #[test]
    fn review_parses_resume() {
        let command = parse(&["review", "--resume"]).expect("parse");
        let Command::Review(cli) = command else {
            panic!("expected review");
        };
        assert!(cli.resume);
    }

    #[test]
    fn review_parses_concurrency() {
        let command = parse(&["review", "--concurrency", "4"]).expect("parse");
        let Command::Review(cli) = command else {
            panic!("expected review");
        };
        assert_eq!(cli.concurrency, 4);
    }

    #[test]
    fn review_rejects_zero_concurrency() {
        let err = parse(&["review", "--concurrency", "0"]).expect_err("invalid");
        assert!(err.contains("--concurrency must be between 1 and 32"));
    }

    #[test]
    fn validate_accepts_repeated_inputs_and_positionals() {
        let command =
            parse(&["validate", "--input", "target/a.jsonl", "target/b.jsonl"]).expect("parse");
        let Command::Validate(cli) = command else {
            panic!("expected validate");
        };
        assert_eq!(cli.inputs, vec!["target/a.jsonl", "target/b.jsonl"]);
    }

    #[test]
    fn agent_logs_parses_backend_options() {
        let command = parse(&[
            "agent-logs",
            "--out-dir",
            "target/dataset/run/agent_logs",
            "--provider",
            "openrouter",
            "--verifier-provider",
            "minimax",
            "--openrouter-model",
            "openrouter/owl-alpha",
            "--limit",
            "25",
            "--since",
            "2026-05-01",
            "--project",
            "forge-rs",
            "--no-claude",
            "--synthetic-balanced",
            "10",
        ])
        .expect("parse");
        let Command::AgentLogs(cli) = command else {
            panic!("expected agent-logs");
        };
        assert_eq!(cli.out, "target/dataset/run/agent_logs");
        assert_eq!(cli.provider, "openrouter");
        assert_eq!(cli.verifier_provider, "minimax");
        assert_eq!(cli.openrouter_model, "openrouter/owl-alpha");
        assert_eq!(cli.limit, Some(25));
        assert_eq!(cli.since.as_deref(), Some("2026-05-01"));
        assert_eq!(cli.project.as_deref(), Some("forge-rs"));
        assert!(!cli.include_claude);
        assert_eq!(cli.synthetic_balanced, 10);
    }

    #[test]
    fn agent_logs_rejects_balanced_and_per_type_synthetic_mix() {
        let err = parse(&[
            "agent-logs",
            "--synthetic-balanced",
            "10",
            "--synthetic-missing-argument",
            "1",
        ])
        .expect_err("invalid");
        assert!(err.contains("--synthetic-balanced cannot be combined"));
    }

    #[test]
    fn assemble_accepts_repeated_inputs_and_output_name() {
        let command = parse(&[
            "assemble",
            "--input",
            "target/proxy.jsonl",
            "target/agent.jsonl",
            "--out-dir",
            "target/dataset/combined",
            "--combined-output",
            "training.toolcall.private.jsonl",
        ])
        .expect("parse");
        let Command::Assemble(cli) = command else {
            panic!("expected assemble");
        };
        assert_eq!(cli.inputs, vec!["target/proxy.jsonl", "target/agent.jsonl"]);
        assert_eq!(cli.out_dir, "target/dataset/combined");
        assert_eq!(cli.combined_output, "training.toolcall.private.jsonl");
        assert!(!cli.drop_conflicts);
    }

    #[test]
    fn assemble_parses_drop_conflicts() {
        let command = parse(&[
            "assemble",
            "--input",
            "target/proxy.jsonl",
            "--drop-conflicts",
        ])
        .expect("parse");
        let Command::Assemble(cli) = command else {
            panic!("expected assemble");
        };
        assert!(cli.drop_conflicts);
    }

    #[test]
    fn parses_runs_for_prompt_and_capture() {
        let command = parse(&["prompts", "--runs", "3"]).expect("parse");
        let Command::Prompts(cli) = command else {
            panic!("expected prompts");
        };
        assert_eq!(cli.runs, 3);

        let command = parse(&["capture", "--model", "test-model", "--runs", "4"]).expect("parse");
        let Command::Capture(cli) = command else {
            panic!("expected capture");
        };
        assert_eq!(cli.runs, 4);
    }
}
