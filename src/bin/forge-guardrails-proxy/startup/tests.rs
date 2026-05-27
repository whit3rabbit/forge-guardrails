use clap::Parser;

use super::managed::{build_managed_startup, managed_extra_flags};
use super::modes::{build_env_startup, build_external_startup, normalize_openai_base_url};
use crate::cli::{Cli, CliBackend};
use crate::client::ClientFactory;
use crate::config::{
    DEFAULT_BACKEND_PORT, DEFAULT_CLI_HOST, DEFAULT_ENV_CONTEXT_TOKENS, DEFAULT_ENV_HOST,
    DEFAULT_ENV_MODEL, DEFAULT_MAX_RETRIES,
};

fn parse(args: &[&str]) -> Cli {
    Cli::try_parse_from(std::iter::once("forge-guardrails-proxy").chain(args.iter().copied()))
        .expect("parse")
}

#[test]
fn external_startup_uses_cli_flags_without_launching_backend() {
    let cli = parse(&[
        "--backend-url",
        "http://localhost:8080",
        "--budget-tokens",
        "4096",
        "--host",
        "127.0.0.1",
        "--port",
        "18081",
        "--model",
        "default-model",
        "--max-retries",
        "7",
        "--no-rescue",
        "--serialize",
        "-v",
    ]);
    let startup = build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url"))
        .expect("startup");

    assert_eq!(startup.config.host, "127.0.0.1");
    assert_eq!(startup.config.port, 18081);
    assert_eq!(startup.config.default_model, "default-model");
    assert_eq!(startup.config.context_tokens, 4096);
    assert_eq!(startup.config.max_retries, 7);
    assert!(!startup.config.rescue_enabled);
    assert!(startup.config.serialize_requests);
    assert!(startup.config.verbose);
    assert!(startup.managed_server.is_none());
    match startup.client_factory {
        ClientFactory::DirectOpenAi {
            base_url,
            context_tokens,
            ..
        } => {
            assert_eq!(base_url, "http://localhost:8080/v1");
            assert_eq!(context_tokens, 4096);
        }
        _ => panic!("expected direct OpenAI client factory"),
    }
}

#[test]
fn external_startup_rejects_llamafile_runtime() {
    let cli = parse(&[
        "--backend-url",
        "http://localhost:8080",
        "--llamafile-runtime",
        "/opt/forge/bin/llamafile",
    ]);
    let err = match build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url")) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--llamafile-runtime requires --backend llamafile"));
}

#[test]
fn external_startup_rejects_extra_flags() {
    let cli = parse(&[
        "--backend-url",
        "http://localhost:8080",
        "--extra-flags",
        "--",
        "--reasoning-budget",
        "0",
    ]);
    let err = match build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url")) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--extra-flags requires managed"));
}

#[test]
fn external_anthropic_startup_uses_direct_anthropic_factory() {
    let cli = parse(&[
        "--backend-url",
        "http://localhost:8080",
        "--backend-protocol",
        "anthropic",
        "--budget-tokens",
        "4096",
        "--model",
        "claude-3",
    ]);
    let startup = build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url"))
        .expect("startup");

    assert_eq!(startup.config.default_model, "claude-3");
    assert_eq!(startup.config.context_tokens, 4096);
    match startup.client_factory {
        ClientFactory::DirectAnthropic {
            base_url,
            context_tokens,
            ..
        } => {
            assert_eq!(base_url, "http://localhost:8080/v1");
            assert_eq!(context_tokens, 4096);
        }
        _ => panic!("expected direct Anthropic client factory"),
    }
}

#[test]
fn external_prompt_mode_uses_llamafile_client_factory() {
    let cli = parse(&[
        "--backend-url",
        "http://localhost:8080",
        "--mode",
        "prompt",
        "--budget-tokens",
        "4096",
    ]);
    let startup = build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url"))
        .expect("startup");

    match startup.client_factory {
        ClientFactory::DirectLlamafile {
            base_url,
            mode,
            context_tokens,
            ..
        } => {
            assert_eq!(base_url, "http://localhost:8080/v1");
            assert_eq!(mode, "prompt");
            assert_eq!(context_tokens, 4096);
        }
        _ => panic!("expected direct llamafile client factory"),
    }
}

#[test]
fn external_anthropic_rejects_prompt_mode() {
    let cli = parse(&[
        "--backend-url",
        "http://localhost:8080",
        "--backend-protocol",
        "anthropic",
        "--mode",
        "prompt",
    ]);
    let err = match build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url")) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--mode prompt is not supported"));
}

#[test]
fn env_fallback_rejects_managed_only_flags() {
    let cli = parse(&["--gguf", "model.gguf"]);
    let err = match build_env_startup(&cli) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--gguf requires --backend"));
}

#[test]
fn env_fallback_rejects_llamafile_runtime() {
    let cli = parse(&["--llamafile-runtime", "/opt/forge/bin/llamafile"]);
    let err = match build_env_startup(&cli) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--llamafile-runtime requires --backend llamafile"));
}

#[test]
fn env_fallback_rejects_first_class_backend_flags() {
    let cli = parse(&["--reasoning-budget", "0"]);
    let err = match build_env_startup(&cli) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("require managed --backend"));
}

#[test]
fn env_fallback_rejects_prompt_mode() {
    let cli = parse(&["--mode", "prompt"]);
    let err = match build_env_startup(&cli) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--mode prompt requires"));
}

#[test]
fn managed_startup_rejects_anthropic_protocol_before_launch() {
    let cli = parse(&["--backend", "ollama", "--backend-protocol", "anthropic"]);
    let err = match build_managed_startup(&cli, CliBackend::Ollama) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--backend-protocol anthropic requires external mode"));
}

#[test]
fn managed_ollama_rejects_prompt_mode_before_launch() {
    let cli = parse(&["--backend", "ollama", "--mode", "prompt"]);
    let err = match build_managed_startup(&cli, CliBackend::Ollama) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("--mode prompt is not supported with --backend ollama"));
}

#[test]
fn managed_ollama_requires_model_before_launch() {
    let cli = parse(&["--backend", "ollama"]);
    let err = match build_managed_startup(&cli, CliBackend::Ollama) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("requires --model"));
}

#[test]
fn managed_ollama_rejects_llamafile_runtime_before_launch() {
    let cli = parse(&[
        "--backend",
        "ollama",
        "--model",
        "llama3",
        "--llamafile-runtime",
        "/opt/forge/bin/llamafile",
    ]);
    let err = match build_managed_startup(&cli, CliBackend::Ollama) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("does not accept --llamafile-runtime"));
}

#[test]
fn managed_ollama_rejects_backend_tuning_flags_before_launch() {
    let cli = parse(&[
        "--backend",
        "ollama",
        "--model",
        "llama3",
        "--reasoning-budget",
        "0",
    ]);
    let err = match build_managed_startup(&cli, CliBackend::Ollama) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("require --backend llamaserver or llamafile"));
}

#[test]
fn managed_extra_flags_include_first_class_reasoning_flags() {
    let cli = parse(&[
        "--backend",
        "llamaserver",
        "--gguf",
        "model.gguf",
        "--reasoning-budget",
        "0",
        "--extra-flags",
        "--",
        "--reasoning-format",
        "auto",
    ]);
    assert_eq!(
        managed_extra_flags(&cli).expect("flags"),
        vec![
            "--reasoning-format".to_string(),
            "auto".to_string(),
            "--reasoning-budget".to_string(),
            "0".to_string()
        ]
    );
}

#[test]
fn managed_llamaserver_requires_gguf_before_launch() {
    let cli = parse(&["--backend", "llamaserver"]);
    let err = match build_managed_startup(&cli, CliBackend::Llamaserver) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("requires --gguf"));
}

#[test]
fn managed_llamaserver_rejects_model_before_launch() {
    let cli = parse(&["--backend", "llamaserver", "--model", "llama3"]);
    let err = match build_managed_startup(&cli, CliBackend::Llamaserver) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("does not accept --model"));
}

#[test]
fn managed_llamaserver_rejects_llamafile_runtime_before_launch() {
    let cli = parse(&[
        "--backend",
        "llamaserver",
        "--gguf",
        "model.gguf",
        "--llamafile-runtime",
        "/opt/forge/bin/llamafile",
    ]);
    let err = match build_managed_startup(&cli, CliBackend::Llamaserver) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("does not accept --llamafile-runtime"));
}

#[test]
fn managed_llamafile_requires_runtime_before_launch() {
    let cli = parse(&["--backend", "llamafile", "--gguf", "model.gguf"]);
    let err = match build_managed_startup(&cli, CliBackend::Llamafile) {
        Err(err) => err,
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("requires --llamafile-runtime"));
}

#[test]
fn normalizes_external_backend_url() {
    assert_eq!(
        normalize_openai_base_url("http://localhost:8080").expect("url"),
        "http://localhost:8080/v1"
    );
    assert_eq!(
        normalize_openai_base_url("http://localhost:8080/v1/").expect("url"),
        "http://localhost:8080/v1"
    );
}

#[test]
fn keeps_known_defaults_visible_to_startup() {
    assert_eq!(DEFAULT_BACKEND_PORT, 8080);
    assert_eq!(DEFAULT_CLI_HOST, "127.0.0.1");
    assert_eq!(DEFAULT_ENV_HOST, "0.0.0.0");
    assert_eq!(DEFAULT_ENV_MODEL, "gpt-4o-mini");
    assert_eq!(DEFAULT_ENV_CONTEXT_TOKENS, 128_000);
    assert_eq!(DEFAULT_MAX_RETRIES, 3);
}
