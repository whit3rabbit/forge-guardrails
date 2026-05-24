use std::path::Path;

use forge_guardrails::{AnyLlmRuntimeClient, LLMClient, LlamafileClient, ServerManager};
use reqwest::Url;

use crate::cli::{Cli, CliBackend, CliBackendProtocol, CliMode};
use crate::client::ClientFactory;
use crate::config::ProxyConfig;
use crate::config::{
    apply_env_cli_overrides, cli_host, cli_max_retries, cli_model, cli_port,
    normalized_extra_flags, require_cli_gguf, require_cli_llamafile_runtime, require_cli_model,
    resolve_serialize, validate_nonzero_u16, validate_optional_positive_i64, validate_positive_i64,
    DEFAULT_ENV_CONTEXT_TOKENS, DEFAULT_EXTERNAL_CONTEXT_TOKENS, DEFAULT_EXTERNAL_MODEL,
};
use crate::upstream::{
    direct_anthropic_api_key, direct_local_openai_upstream_from_env, direct_openai_api_key,
};

pub(crate) struct Startup {
    pub(crate) config: ProxyConfig,
    pub(crate) client_factory: ClientFactory,
    pub(crate) managed_server: Option<ServerManager>,
}

pub(crate) fn build_startup(cli: Cli) -> Result<Startup, String> {
    if cli.backend_url.is_none() && cli.backend.is_none() {
        return build_env_startup(&cli);
    }
    if let Some(backend_url) = cli.backend_url.as_deref() {
        return build_external_startup(&cli, backend_url);
    }
    let backend = cli.backend.expect("backend checked above");
    build_managed_startup(&cli, backend)
}

fn build_env_startup(cli: &Cli) -> Result<Startup, String> {
    if cli.backend_protocol == CliBackendProtocol::Anthropic {
        return Err("--backend-protocol anthropic requires --backend-url".to_string());
    }
    if cli.mode == CliMode::Prompt {
        return Err("--mode prompt requires --backend-url or a llama managed backend".to_string());
    }
    if cli.gguf.is_some() {
        return Err("--gguf requires --backend".to_string());
    }
    if cli.llamafile_runtime.is_some() {
        return Err("--llamafile-runtime requires --backend llamafile".to_string());
    }
    if !normalized_extra_flags(&cli.extra_flags).is_empty() {
        return Err("--extra-flags requires --backend".to_string());
    }

    let mut config = ProxyConfig::from_env()?;
    apply_env_cli_overrides(&mut config, cli)?;
    let client_factory = build_env_client_factory(&config);
    Ok(Startup {
        config,
        client_factory,
        managed_server: None,
    })
}

fn build_external_startup(cli: &Cli, backend_url: &str) -> Result<Startup, String> {
    if cli.llamafile_runtime.is_some() {
        return Err("--llamafile-runtime requires --backend llamafile".to_string());
    }
    if cli.backend_protocol == CliBackendProtocol::Anthropic && cli.mode == CliMode::Prompt {
        return Err("--mode prompt is not supported with --backend-protocol anthropic".to_string());
    }

    let base_url = if cli.backend_protocol == CliBackendProtocol::Anthropic {
        normalize_anthropic_base_url(backend_url)?
    } else {
        normalize_openai_base_url(backend_url)?
    };
    let context_tokens = match cli.budget_tokens {
        Some(tokens) => validate_positive_i64(tokens, "--budget-tokens")?,
        None if cli.backend_protocol == CliBackendProtocol::Anthropic => DEFAULT_ENV_CONTEXT_TOKENS,
        None => discover_external_context_tokens(&base_url),
    };
    let config = ProxyConfig {
        host: cli_host(cli)?,
        port: cli_port(cli)?,
        default_model: cli_model(cli, DEFAULT_EXTERNAL_MODEL)?,
        context_tokens,
        max_retries: cli_max_retries(cli)?,
        rescue_enabled: !cli.no_rescue,
        serialize_requests: resolve_serialize(cli, false),
        verbose: cli.verbose,
    };
    let client_factory = match (cli.backend_protocol, cli.mode) {
        (CliBackendProtocol::Anthropic, _) => ClientFactory::DirectAnthropic {
            base_url,
            api_key: direct_anthropic_api_key(),
            context_tokens,
        },
        (CliBackendProtocol::Openai, CliMode::Prompt) => ClientFactory::DirectLlamafile {
            base_url,
            mode: cli.mode.as_str().to_string(),
            context_tokens,
        },
        (CliBackendProtocol::Openai, CliMode::Native) => ClientFactory::DirectOpenAi {
            base_url,
            api_key: direct_openai_api_key(&[]),
            context_tokens,
        },
    };
    Ok(Startup {
        config,
        client_factory,
        managed_server: None,
    })
}

fn build_managed_startup(cli: &Cli, backend: CliBackend) -> Result<Startup, String> {
    if cli.backend_protocol == CliBackendProtocol::Anthropic {
        return Err(
            "--backend-protocol anthropic requires external mode (--backend-url)".to_string(),
        );
    }
    let backend_name = backend.as_str();
    let backend_port = validate_nonzero_u16(cli.backend_port, "--backend-port")?;
    let budget_tokens = validate_optional_positive_i64(cli.budget_tokens, "--budget-tokens")?;
    let budget_mode = forge_guardrails::BudgetMode::from(cli.budget_mode);
    let extra_flags = normalized_extra_flags(&cli.extra_flags);
    let proxy_host = cli_host(cli)?;
    let proxy_port = cli_port(cli)?;
    let max_retries = cli_max_retries(cli)?;
    let serialize_requests = resolve_serialize(cli, true);

    let (default_model, client_factory, managed_server, context_tokens) = match backend {
        CliBackend::Ollama => {
            if cli.mode == CliMode::Prompt {
                return Err("--mode prompt is not supported with --backend ollama".to_string());
            }
            let model = require_cli_model(cli)?;
            if cli.gguf.is_some() {
                return Err("--backend ollama does not accept --gguf".to_string());
            }
            if cli.llamafile_runtime.is_some() {
                return Err("--backend ollama does not accept --llamafile-runtime".to_string());
            }
            let (server, context) = forge_guardrails::setup_backend(
                backend_name,
                Some(&model),
                None,
                None,
                budget_mode,
                budget_tokens,
                backend_port as i64,
                cli.mode.as_str(),
                &extra_flags,
                None,
                None,
                None,
                false,
            )?;
            let context_tokens = context.budget();
            (
                model.clone(),
                ClientFactory::ManagedOllama {
                    model,
                    context_tokens,
                },
                Some(server),
                context_tokens,
            )
        }
        CliBackend::Llamaserver => {
            if cli.model.is_some() {
                return Err(format!(
                    "--backend {backend_name} does not accept --model; use --gguf"
                ));
            }
            if cli.llamafile_runtime.is_some() {
                return Err("--backend llamaserver does not accept --llamafile-runtime".to_string());
            }
            let gguf = require_cli_gguf(cli, backend_name)?;
            let (server, context) = forge_guardrails::setup_backend(
                backend_name,
                None,
                Some(Path::new(&gguf)),
                None,
                budget_mode,
                budget_tokens,
                backend_port as i64,
                cli.mode.as_str(),
                &extra_flags,
                None,
                None,
                None,
                false,
            )?;
            let context_tokens = context.budget();
            let model = gguf_model_identity(&gguf);
            (
                model,
                ClientFactory::ManagedLlamafile {
                    gguf_path: gguf,
                    base_url: format!("http://127.0.0.1:{backend_port}/v1"),
                    mode: cli.mode.as_str().to_string(),
                },
                Some(server),
                context_tokens,
            )
        }
        CliBackend::Llamafile => {
            if cli.model.is_some() {
                return Err(format!(
                    "--backend {backend_name} does not accept --model; use --gguf"
                ));
            }
            let gguf = require_cli_gguf(cli, backend_name)?;
            let runtime = require_cli_llamafile_runtime(cli)?;
            let (server, context) = forge_guardrails::setup_backend(
                backend_name,
                None,
                Some(Path::new(&gguf)),
                Some(Path::new(&runtime)),
                budget_mode,
                budget_tokens,
                backend_port as i64,
                cli.mode.as_str(),
                &extra_flags,
                None,
                None,
                None,
                false,
            )?;
            let context_tokens = context.budget();
            let model = gguf_model_identity(&gguf);
            (
                model,
                ClientFactory::ManagedLlamafile {
                    gguf_path: gguf,
                    base_url: format!("http://127.0.0.1:{backend_port}/v1"),
                    mode: cli.mode.as_str().to_string(),
                },
                Some(server),
                context_tokens,
            )
        }
    };

    let config = ProxyConfig {
        host: proxy_host,
        port: proxy_port,
        default_model,
        context_tokens,
        max_retries,
        rescue_enabled: !cli.no_rescue,
        serialize_requests,
        verbose: cli.verbose,
    };

    Ok(Startup {
        config,
        client_factory,
        managed_server,
    })
}

fn build_env_client_factory(config: &ProxyConfig) -> ClientFactory {
    if let Some(upstream) = direct_local_openai_upstream_from_env() {
        eprintln!(
            "warning: using direct local OpenAI-compatible upstream at {}; anyllm routing is bypassed for this local operator endpoint",
            upstream.base_url
        );
        return ClientFactory::DirectOpenAi {
            base_url: upstream.base_url,
            api_key: upstream.api_key,
            context_tokens: config.context_tokens,
        };
    }

    let load_result = anyllm_proxy::config::MultiConfig::load();
    ClientFactory::Runtime(
        AnyLlmRuntimeClient::from_multi_config_with_model_router(
            config.default_model.clone(),
            load_result.multi_config,
            load_result.model_router,
        )
        .with_context_length(config.context_tokens),
    )
}

fn normalize_openai_base_url(raw: &str) -> Result<String, String> {
    let trimmed = crate::config::validate_nonempty(raw, "--backend-url")?.trim_end_matches('/');
    Url::parse(trimmed).map_err(|err| format!("invalid --backend-url: {err}"))?;
    if trimmed.ends_with("/v1") {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("{trimmed}/v1"))
    }
}

fn normalize_anthropic_base_url(raw: &str) -> Result<String, String> {
    normalize_openai_base_url(raw)
}

fn discover_external_context_tokens(base_url: &str) -> i64 {
    let client = LlamafileClient::new(DEFAULT_EXTERNAL_MODEL)
        .with_base_url(base_url)
        .with_mode("native");
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return DEFAULT_EXTERNAL_CONTEXT_TOKENS;
    };
    match runtime.block_on(client.get_context_length()) {
        Ok(Some(tokens)) if tokens > 0 => tokens,
        _ => DEFAULT_EXTERNAL_CONTEXT_TOKENS,
    }
}

fn gguf_model_identity(gguf: &str) -> String {
    Path::new(gguf)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or(gguf)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_BACKEND_PORT, DEFAULT_CLI_HOST, DEFAULT_ENV_CONTEXT_TOKENS, DEFAULT_ENV_HOST,
        DEFAULT_ENV_MODEL, DEFAULT_MAX_RETRIES,
    };
    use clap::Parser;

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
        let startup =
            build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url"))
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
        let err =
            match build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url")) {
                Err(err) => err,
                Ok(_) => panic!("expected error"),
            };
        assert!(err.contains("--llamafile-runtime requires --backend llamafile"));
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
        let startup =
            build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url"))
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
        let startup =
            build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url"))
                .expect("startup");

        match startup.client_factory {
            ClientFactory::DirectLlamafile {
                base_url,
                mode,
                context_tokens,
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
        let err =
            match build_external_startup(&cli, cli.backend_url.as_deref().expect("backend url")) {
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
}
