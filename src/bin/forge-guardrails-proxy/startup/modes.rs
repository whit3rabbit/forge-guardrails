use forge_guardrails::{AnyLlmRuntimeClient, LLMClient, LlamafileClient};
use reqwest::Url;

use crate::cli::{Cli, CliBackendProtocol, CliMode};
use crate::client::ClientFactory;
use crate::config::{
    apply_env_cli_overrides, classifier_settings_from_env_cli, cli_host, cli_max_retries,
    cli_model, cli_port, final_response_classifier_settings_from_env_cli, normalized_extra_flags,
    redact_secrets_from_env_cli, resolve_serialize, schema_compression_from_env_cli,
    tool_call_policy_from_env_cli, tool_output_compression_from_env_cli, validate_nonempty,
    validate_positive_i64, DEFAULT_ENV_CONTEXT_TOKENS, DEFAULT_EXTERNAL_CONTEXT_TOKENS,
    DEFAULT_INTERNAL_MODEL,
};
use crate::upstream::{
    direct_anthropic_api_key, direct_local_openai_upstream_from_env, direct_openai_api_key,
};

use super::managed::reject_managed_llama_options;
use super::Startup;

pub(super) fn build_env_startup(cli: &Cli) -> Result<Startup, String> {
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
    reject_managed_llama_options(cli)?;

    let mut config = crate::config::ProxyConfig::from_env()?;
    apply_env_cli_overrides(&mut config, cli)?;
    let client_factory = build_env_client_factory(&config);
    Ok(Startup {
        config,
        client_factory,
        managed_server: None,
        scorer: None,
        final_response_scorer: None,
    })
}

pub(super) fn build_external_startup(cli: &Cli, backend_url: &str) -> Result<Startup, String> {
    if cli.llamafile_runtime.is_some() {
        return Err("--llamafile-runtime requires --backend llamafile".to_string());
    }
    if !normalized_extra_flags(&cli.extra_flags).is_empty() {
        return Err(
            "--extra-flags requires managed --backend llamaserver or llamafile".to_string(),
        );
    }
    reject_managed_llama_options(cli)?;
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
    let (classifier_dir, classifier_mode, classifier_model, classifier_auto_download) =
        classifier_settings_from_env_cli(cli)?;
    let (
        final_response_classifier_dir,
        final_response_classifier_mode,
        final_response_classifier_model,
    ) = final_response_classifier_settings_from_env_cli(cli)?;
    let tool_output_compression = tool_output_compression_from_env_cli(cli)?;
    let tool_call_policy = tool_call_policy_from_env_cli(cli)?;
    let schema_compression = schema_compression_from_env_cli(cli)?;
    let redact_secrets = redact_secrets_from_env_cli(cli)?;
    let config = crate::config::ProxyConfig {
        host: cli_host(cli)?,
        port: cli_port(cli)?,
        default_model: cli_model(cli, DEFAULT_INTERNAL_MODEL)?,
        default_model_explicit: cli.model.is_some(),
        context_tokens,
        max_retries: cli_max_retries(cli)?,
        rescue_enabled: !cli.no_rescue,
        serialize_requests: resolve_serialize(cli, false),
        verbose: cli.verbose,
        classifier_dir,
        classifier_mode,
        classifier_model,
        classifier_auto_download,
        classifier_max_latency_ms: cli.classifier_max_latency_ms,
        final_response_classifier_dir,
        final_response_classifier_mode,
        final_response_classifier_model,
        final_response_classifier_max_latency_ms: cli.final_response_classifier_max_latency_ms,
        tool_output_compression,
        tool_call_policy,
        schema_compression,
        redact_secrets,
    };
    let client_factory = match (cli.backend_protocol, cli.mode) {
        (CliBackendProtocol::Anthropic, _) => ClientFactory::DirectAnthropic {
            base_url,
            api_key: direct_anthropic_api_key(),
            http_client: reqwest::Client::new(),
            context_tokens,
        },
        (CliBackendProtocol::Openai, CliMode::Prompt) => ClientFactory::DirectLlamafile {
            base_url,
            mode: cli.mode.as_str().to_string(),
            http_client: reqwest::Client::new(),
            context_tokens,
        },
        (CliBackendProtocol::Openai, CliMode::Native) => ClientFactory::DirectOpenAi {
            base_url,
            api_key: direct_openai_api_key(&[]),
            http_client: reqwest::Client::new(),
            context_tokens,
        },
    };
    Ok(Startup {
        config,
        client_factory,
        managed_server: None,
        scorer: None,
        final_response_scorer: None,
    })
}

fn build_env_client_factory(config: &crate::config::ProxyConfig) -> ClientFactory {
    if let Some(upstream) = direct_local_openai_upstream_from_env() {
        eprintln!(
            "warning: using direct local OpenAI-compatible upstream at {}; anyllm routing is bypassed for this local operator endpoint",
            upstream.base_url
        );
        return ClientFactory::DirectOpenAi {
            base_url: upstream.base_url,
            api_key: upstream.api_key,
            http_client: reqwest::Client::new(),
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

pub(super) fn normalize_openai_base_url(raw: &str) -> Result<String, String> {
    let trimmed = validate_nonempty(raw, "--backend-url")?.trim_end_matches('/');
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
    let client = LlamafileClient::new(DEFAULT_INTERNAL_MODEL)
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
