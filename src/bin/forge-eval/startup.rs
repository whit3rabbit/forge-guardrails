use std::env;
use std::path::Path;

use forge_guardrails::{AnthropicClient, AnyLlmProxyClient, LlamafileClient, OllamaClient};

use crate::cli::Cli;
use crate::runner::run_with_client;

const DEFAULT_PROXY_URL: &str = "http://127.0.0.1:8081/v1";
const DEFAULT_LOCAL_URL: &str = "http://127.0.0.1:8080/v1";

pub(crate) async fn run_cli(cli: Cli) -> Result<(), String> {
    match cli.backend.as_str() {
        "openai-proxy" => {
            let model = require_model(&cli)?;
            let base_url = cli.base_url.as_deref().unwrap_or(DEFAULT_PROXY_URL);
            let client = AnyLlmProxyClient::new(model.clone()).with_base_url(base_url);
            run_with_client(client, &cli, &model).await
        }
        "ollama" => {
            let model = require_model(&cli)?;
            let client = configured_ollama_client(&cli, &model);
            run_with_client(client, &cli, &model).await
        }
        "llamaserver" | "llamafile" => {
            let gguf = cli
                .gguf
                .as_deref()
                .ok_or_else(|| format!("--backend {} requires --gguf", cli.backend))?;
            let model = Path::new(gguf)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or(gguf)
                .to_string();
            let default_mode = if cli.backend == "llamaserver" {
                "native"
            } else {
                "prompt"
            };
            let mode = cli.mode.as_deref().unwrap_or(default_mode);
            let base_url = cli.base_url.as_deref().unwrap_or(DEFAULT_LOCAL_URL);
            let client = LlamafileClient::new(gguf)
                .with_base_url(base_url)
                .with_mode(mode);
            run_with_client(client, &cli, &model).await
        }
        "anthropic" => {
            let model = require_model(&cli)?;
            let api_key = cli
                .anthropic_api_key
                .clone()
                .or_else(|| env::var("ANTHROPIC_API_KEY").ok());
            let client = AnthropicClient::new(model.clone(), api_key);
            run_with_client(client, &cli, &model).await
        }
        other => Err(format!("unknown backend: {other}")),
    }
}

pub(crate) fn configured_ollama_client(cli: &Cli, model: &str) -> OllamaClient {
    let mut client = OllamaClient::new(model.to_string());
    if let Some(base_url) = cli.base_url.as_deref() {
        client = client.with_base_url(base_url);
    }
    client.set_num_ctx(Some(cli.num_ctx));
    client
}

pub(crate) fn default_mode(backend: &str) -> &'static str {
    match backend {
        "llamafile" => "prompt",
        "llamaserver" | "ollama" | "anthropic" => "native",
        "openai-proxy" => "proxy",
        _ => "unknown",
    }
}

fn require_model(cli: &Cli) -> Result<String, String> {
    cli.model
        .clone()
        .ok_or_else(|| format!("--backend {} requires --model", cli.backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::parse_args;
    use forge_guardrails::LLMClient;

    fn parse(items: &[&str]) -> Cli {
        parse_args(items.iter().map(|item| item.to_string())).expect("parse")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn configured_ollama_client_sets_num_ctx() {
        let cli = parse(&[
            "--backend",
            "ollama",
            "--model",
            "llama3",
            "--base-url",
            "http://127.0.0.1:11434",
            "--num-ctx",
            "4096",
        ]);
        let client = configured_ollama_client(&cli, "llama3");
        assert_eq!(client.get_context_length().await.unwrap(), Some(4096));
    }

    #[test]
    fn backend_default_modes_match_contract() {
        assert_eq!(default_mode("llamaserver"), "native");
        assert_eq!(default_mode("llamafile"), "prompt");
        assert_eq!(default_mode("openai-proxy"), "proxy");
    }
}
