use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use crate::cli::{default_minimax_model, default_openrouter_model, AgentLogsCli};

pub(crate) fn run(cli: AgentLogsCli) -> Result<(), String> {
    let env_file = EnvFile::load(&cli.env_file);
    let command = build_generatetd_command(&cli, &env_file)?;
    let out_dir = absolute_output_path(&cli.out);
    fs::create_dir_all(&out_dir)
        .map_err(|err| format!("failed to create {}: {err}", out_dir.display()))?;
    eprintln!(
        "agent-logs running backend={} cwd={} out={}",
        command.program,
        command.current_dir.display(),
        out_dir.display()
    );
    let mut process = ProcessCommand::new(&command.program);
    process
        .args(&command.args)
        .current_dir(&command.current_dir)
        .envs(command.env.iter().map(|(key, value)| (key, value)));
    let status = process
        .status()
        .map_err(|err| format!("failed to run generatetd backend: {err}"))?;
    if !status.success() {
        return Err(format!("generatetd backend exited with {status}"));
    }
    let tool_output = out_dir.join("tool_call_training.jsonl");
    if !tool_output.exists() {
        return Err(format!(
            "generatetd backend did not write {}",
            tool_output.display()
        ));
    }
    println!("Agent log tool rows: {}", tool_output.display());
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GeneratedCommand {
    program: String,
    args: Vec<String>,
    current_dir: PathBuf,
    env: Vec<(String, String)>,
}

fn build_generatetd_command(
    cli: &AgentLogsCli,
    env_file: &EnvFile,
) -> Result<GeneratedCommand, String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let backend_dir = repo_root.join("notebook/generatetd");
    if !backend_dir.is_dir() {
        return Err(format!(
            "generatetd backend directory not found: {}",
            backend_dir.display()
        ));
    }

    let minimax_model = resolve_value(
        &cli.minimax_model,
        "GENERATETD_MINIMAX_MODEL",
        env_file,
        default_minimax_model(),
    );
    let openrouter_model = resolve_value(
        &cli.openrouter_model,
        "GENERATETD_OPENROUTER_MODEL",
        env_file,
        default_openrouter_model(),
    );
    let minimax_key = resolve_optional(cli.minimax_api_key.as_deref(), "MINIMAX_API_KEY", env_file);
    let openrouter_key = resolve_optional(
        cli.openrouter_api_key.as_deref(),
        "OPENROUTER_API_KEY",
        env_file,
    );
    let output_dir = absolute_output_path(&cli.out);

    let provider = if cli.no_api { "none" } else { &cli.provider };
    let mut args = vec![
        "-m".to_string(),
        "generatetd".to_string(),
        "generate".to_string(),
        "--out".to_string(),
        output_dir.display().to_string(),
        "--provider".to_string(),
        provider.to_string(),
        "--serializer".to_string(),
        "v2".to_string(),
        "--tool-calls-only".to_string(),
        "--no-notebook-adapter".to_string(),
        "--minimax-model".to_string(),
        minimax_model,
        "--openrouter-model".to_string(),
        openrouter_model,
    ];

    if cli.no_api || cli.provider == "none" {
        args.push("--no-api".to_string());
    } else {
        args.push("--llm-review".to_string());
        args.push("--verify-review".to_string());
        args.push("--verifier-provider".to_string());
        args.push(cli.verifier_provider.clone());
    }
    if let Some(limit) = cli.limit {
        args.push("--limit".to_string());
        args.push(limit.to_string());
    }
    if let Some(since) = cli.since.as_ref() {
        args.push("--since".to_string());
        args.push(since.clone());
    }
    if let Some(project) = cli.project.as_ref() {
        args.push("--project".to_string());
        args.push(project.clone());
    }
    if !cli.include_codex {
        args.push("--no-codex".to_string());
    }
    if !cli.include_claude {
        args.push("--no-claude".to_string());
    }
    if cli.synthetic_balanced > 0 {
        args.push("--synthetic-balanced".to_string());
        args.push(cli.synthetic_balanced.to_string());
    }
    if cli.synthetic_missing_argument > 0 {
        args.push("--synthetic-missing-argument".to_string());
        args.push(cli.synthetic_missing_argument.to_string());
    }
    if cli.synthetic_tool_not_needed > 0 {
        args.push("--synthetic-tool-not-needed".to_string());
        args.push(cli.synthetic_tool_not_needed.to_string());
    }

    let mut env = Vec::new();
    if let Some(key) = minimax_key {
        env.push(("MINIMAX_API_KEY".to_string(), key));
    }
    if let Some(key) = openrouter_key {
        env.push(("OPENROUTER_API_KEY".to_string(), key));
    }

    Ok(GeneratedCommand {
        program: env::var("PYTHON").unwrap_or_else(|_| "python3".to_string()),
        args,
        current_dir: backend_dir,
        env,
    })
}

fn resolve_value(flag: &str, env_key: &str, env_file: &EnvFile, default: &str) -> String {
    if !flag.trim().is_empty() {
        return flag.to_string();
    }
    if let Ok(value) = env::var(env_key) {
        if !value.trim().is_empty() {
            return value;
        }
    }
    env_file
        .get(env_key)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| default.to_string())
}

fn absolute_output_path(path: &str) -> PathBuf {
    let raw = PathBuf::from(path);
    if raw.is_absolute() {
        raw
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(raw)
    }
}

fn resolve_optional(flag: Option<&str>, env_key: &str, env_file: &EnvFile) -> Option<String> {
    if let Some(value) = flag.filter(|value| !value.trim().is_empty()) {
        return Some(value.to_string());
    }
    if let Ok(value) = env::var(env_key) {
        if !value.trim().is_empty() {
            return Some(value);
        }
    }
    env_file
        .get(env_key)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
}

#[derive(Debug, Default)]
struct EnvFile {
    values: BTreeMap<String, String>,
}

impl EnvFile {
    fn load(path: &str) -> Self {
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };
        let mut values = BTreeMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((key, value)) = trimmed.split_once('=') else {
                continue;
            };
            values.insert(key.trim().to_string(), unquote_env_value(value.trim()));
        }
        Self { values }
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

fn unquote_env_value(value: &str) -> String {
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli() -> AgentLogsCli {
        AgentLogsCli {
            out: "target/dataset/run/agent_logs".to_string(),
            env_file: "missing.env".to_string(),
            provider: "openrouter".to_string(),
            verifier_provider: "minimax".to_string(),
            minimax_model: "MiniMax-M2.7".to_string(),
            openrouter_model: "openrouter/owl-alpha".to_string(),
            minimax_api_key: Some("mm-key".to_string()),
            openrouter_api_key: Some("or-key".to_string()),
            no_api: false,
            limit: Some(10),
            since: Some("2026-05-01".to_string()),
            project: Some("forge-rs".to_string()),
            include_codex: true,
            include_claude: false,
            synthetic_balanced: 20,
            synthetic_missing_argument: 0,
            synthetic_tool_not_needed: 0,
        }
    }

    #[test]
    fn builds_generatetd_command_for_reviewed_tool_call_backend() {
        let command = build_generatetd_command(&cli(), &EnvFile::default()).expect("command");
        assert!(!command.program.is_empty());
        assert_eq!(
            command.args[0..3],
            [
                "-m".to_string(),
                "generatetd".to_string(),
                "generate".to_string()
            ]
        );
        assert!(command.args.contains(&"--tool-calls-only".to_string()));
        assert!(command.args.contains(&"--serializer".to_string()));
        assert!(command.args.contains(&"v2".to_string()));
        assert!(command.args.contains(&"--llm-review".to_string()));
        assert!(command.args.contains(&"--verify-review".to_string()));
        assert!(command.args.contains(&"--no-claude".to_string()));
        assert!(command
            .env
            .contains(&("OPENROUTER_API_KEY".to_string(), "or-key".to_string())));
    }

    #[test]
    fn builds_no_api_command_without_review_flags() {
        let mut cli = cli();
        cli.no_api = true;
        let command = build_generatetd_command(&cli, &EnvFile::default()).expect("command");
        assert!(command.args.contains(&"--no-api".to_string()));
        assert!(!command.args.contains(&"--llm-review".to_string()));
        assert!(!command.args.contains(&"--verify-review".to_string()));
    }
}
