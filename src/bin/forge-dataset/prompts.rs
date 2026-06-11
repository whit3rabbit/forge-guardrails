use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde_json::{json, Value};

use crate::cli::PromptsCli;
use crate::stub_tools::{available_tools_for_training, registries_for_domains, request_tools};

const PROMPT_SCHEMA_VERSION: &str = "forge-dataset-tool-prompt/v1";

pub(crate) fn run(cli: PromptsCli) -> Result<(), String> {
    ensure_parent_dir(&cli.output)?;
    touch_jsonl(&cli.output)?;
    let registries = registries_for_domains(&cli.domains)?;
    for run_index in 0..cli.runs {
        for registry in &registries {
            for scenario in &registry.scenarios {
                let row = prompt_row(
                    &cli.model,
                    run_index,
                    registry.domain,
                    scenario.id,
                    scenario.user_request,
                    Value::Array(request_tools(registry)),
                    Value::Array(available_tools_for_training(registry)),
                );
                append_jsonl(&cli.output, &row)?;
            }
        }
    }
    Ok(())
}

fn prompt_row(
    model: &str,
    run_index: usize,
    domain: &str,
    scenario: &str,
    user_request: &str,
    tools: Value,
    available_tools: Value,
) -> Value {
    json!({
        "schema_version": PROMPT_SCHEMA_VERSION,
        "domain": domain,
        "scenario": scenario,
        "run_index": run_index,
        "user_request": user_request,
        "request": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are using a deterministic, harmless stub tool registry for dataset capture. Use tools only when they are needed for the user's request."
                },
                {"role": "user", "content": user_request}
            ],
            "tools": tools,
            "stream": false
        },
        "available_tools": available_tools,
        "metadata": {
            "private_agent_log": true,
            "public_export_allowed": false,
            "source": "forge-dataset",
            "run_index": run_index
        }
    })
}

fn append_jsonl(path: &str, row: &Value) -> Result<(), String> {
    let mut line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    line.push('\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    file.write_all(line.as_bytes())
        .map_err(|err| format!("failed to write {path}: {err}"))?;
    file.flush()
        .map_err(|err| format!("failed to flush {path}: {err}"))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {path}: {err}"))
}

fn touch_jsonl(path: &str) -> Result<(), String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    file.sync_data()
        .map_err(|err| format!("failed to sync {path}: {err}"))
}

fn ensure_parent_dir(path: &str) -> Result<(), String> {
    let Some(parent) = Path::new(path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("failed to create {}: {err}", parent.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stub_tools::{available_tools_for_training, registries_for_domains, request_tools};

    #[test]
    fn prompt_row_contains_openai_tool_request_and_private_metadata() {
        let registry = registries_for_domains(&["repo_docs".to_string()])
            .expect("registry")
            .remove(0);
        let scenario = registry.scenarios.first().expect("scenario");
        let row = prompt_row(
            "test-model",
            0,
            registry.domain,
            scenario.id,
            scenario.user_request,
            Value::Array(request_tools(&registry)),
            Value::Array(available_tools_for_training(&registry)),
        );

        assert_eq!(row["schema_version"], PROMPT_SCHEMA_VERSION);
        assert_eq!(row["request"]["model"], "test-model");
        assert_eq!(row["run_index"], 0);
        assert_eq!(row["request"]["tools"][0]["type"], "function");
        assert_eq!(row["metadata"]["private_agent_log"], true);
        assert_eq!(row["metadata"]["public_export_allowed"], false);
    }

    #[test]
    fn prompts_runs_expand_scenario_payloads() {
        let output = std::env::temp_dir().join(format!(
            "forge-dataset-prompts-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        run(PromptsCli {
            model: "test-model".to_string(),
            output: output.display().to_string(),
            domains: vec!["forge_eval".to_string()],
            runs: 2,
        })
        .expect("run");

        let lines = std::fs::read_to_string(&output).expect("read");
        assert_eq!(lines.lines().count(), 10);
        std::fs::remove_file(output).expect("remove");
    }
}
