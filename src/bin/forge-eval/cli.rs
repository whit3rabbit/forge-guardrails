use crate::ablation::parse_ablation;

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
           --anthropic-api-key KEY"
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
}
