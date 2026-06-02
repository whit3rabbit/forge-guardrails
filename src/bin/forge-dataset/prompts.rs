use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde_json::{json, Value};

use crate::cli::PromptsCli;
use crate::stub_tools::{available_tools_for_training, registries_for_domains, request_tools};

const PROMPT_SCHEMA_VERSION: &str = "forge-dataset-tool-prompt/v1";

pub(crate) fn run(cli: PromptsCli) -> Result<(), String> {
    ensure_parent_dir(&cli.output)?;
    let registries = registries_for_domains(&cli.domains)?;
    for registry in &registries {
        for scenario in &registry.scenarios {
            let row = prompt_row(
                &cli.model,
                registry.domain,
                scenario.id,
                scenario.user_request,
                Value::Array(request_tools(registry)),
                Value::Array(available_tools_for_training(registry)),
            );
            append_jsonl(&cli.output, &row)?;
        }
    }
    Ok(())
}

fn prompt_row(
    model: &str,
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
            "source": "forge-dataset"
        }
    })
}

fn append_jsonl(path: &str, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {path}: {err}"))?;
    writeln!(file, "{line}").map_err(|err| format!("failed to write {path}: {err}"))
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
            registry.domain,
            scenario.id,
            scenario.user_request,
            Value::Array(request_tools(&registry)),
            Value::Array(available_tools_for_training(&registry)),
        );

        assert_eq!(row["schema_version"], PROMPT_SCHEMA_VERSION);
        assert_eq!(row["request"]["model"], "test-model");
        assert_eq!(row["request"]["tools"][0]["type"], "function");
        assert_eq!(row["metadata"]["private_agent_log"], true);
        assert_eq!(row["metadata"]["public_export_allowed"], false);
    }
}
