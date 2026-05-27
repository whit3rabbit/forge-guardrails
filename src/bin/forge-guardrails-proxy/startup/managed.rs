use std::path::Path;

use crate::cli::{Cli, CliBackend, CliBackendProtocol, CliMode};
use crate::client::ClientFactory;
use crate::config::{
    classifier_settings_from_env_cli, cli_host, cli_max_retries, cli_port,
    final_response_classifier_settings_from_env_cli, normalized_extra_flags, require_cli_gguf,
    require_cli_llamafile_runtime, require_cli_model, resolve_serialize, validate_nonempty,
    validate_nonzero_u16, validate_optional_positive_i64,
};

use super::Startup;

pub(super) fn build_managed_startup(cli: &Cli, backend: CliBackend) -> Result<Startup, String> {
    if cli.backend_protocol == CliBackendProtocol::Anthropic {
        return Err(
            "--backend-protocol anthropic requires external mode (--backend-url)".to_string(),
        );
    }
    let backend_name = backend.as_str();
    let backend_port = validate_nonzero_u16(cli.backend_port, "--backend-port")?;
    let budget_tokens = validate_optional_positive_i64(cli.budget_tokens, "--budget-tokens")?;
    let budget_mode = forge_guardrails::BudgetMode::from(cli.budget_mode);
    let extra_flags = managed_extra_flags(cli)?;
    let cache_type_k = optional_nonempty(cli.cache_type_k.as_deref(), "--cache-type-k")?;
    let cache_type_v = optional_nonempty(cli.cache_type_v.as_deref(), "--cache-type-v")?;
    let n_slots = validate_optional_positive_i64(cli.slots, "--slots")?;
    let kv_unified = cli.kv_unified;
    let proxy_host = cli_host(cli)?;
    let proxy_port = cli_port(cli)?;
    let max_retries = cli_max_retries(cli)?;
    let serialize_requests = resolve_serialize(cli, true);
    let (classifier_dir, classifier_mode, classifier_model) =
        classifier_settings_from_env_cli(cli)?;
    let (
        final_response_classifier_dir,
        final_response_classifier_mode,
        final_response_classifier_model,
    ) = final_response_classifier_settings_from_env_cli(cli)?;

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
            if has_managed_llama_options(cli) || !extra_flags.is_empty() {
                return Err(
                    "cache, slot, reasoning, and extra backend flags require --backend llamaserver or llamafile"
                        .to_string(),
                );
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
                &[],
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
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
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
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
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

    let config = crate::config::ProxyConfig {
        host: proxy_host,
        port: proxy_port,
        default_model,
        context_tokens,
        max_retries,
        rescue_enabled: !cli.no_rescue,
        serialize_requests,
        verbose: cli.verbose,
        classifier_dir,
        classifier_mode,
        classifier_model,
        classifier_max_latency_ms: cli.classifier_max_latency_ms,
        final_response_classifier_dir,
        final_response_classifier_mode,
        final_response_classifier_model,
        final_response_classifier_max_latency_ms: cli.final_response_classifier_max_latency_ms,
    };

    Ok(Startup {
        config,
        client_factory,
        managed_server,
        scorer: None,
        final_response_scorer: None,
    })
}

pub(super) fn managed_extra_flags(cli: &Cli) -> Result<Vec<String>, String> {
    let mut flags = normalized_extra_flags(&cli.extra_flags);
    if let Some(value) = cli.reasoning_budget.as_deref() {
        flags.push("--reasoning-budget".to_string());
        flags.push(validate_nonempty(value, "--reasoning-budget")?.to_string());
    }
    if let Some(value) = cli.reasoning_format.as_deref() {
        flags.push("--reasoning-format".to_string());
        flags.push(validate_nonempty(value, "--reasoning-format")?.to_string());
    }
    Ok(flags)
}

fn optional_nonempty<'a>(value: Option<&'a str>, label: &str) -> Result<Option<&'a str>, String> {
    value.map(|raw| validate_nonempty(raw, label)).transpose()
}

pub(super) fn reject_managed_llama_options(cli: &Cli) -> Result<(), String> {
    if has_managed_llama_options(cli) {
        return Err(
            "cache, slot, and reasoning flags require managed --backend llamaserver or llamafile"
                .to_string(),
        );
    }
    Ok(())
}

fn has_managed_llama_options(cli: &Cli) -> bool {
    cli.cache_type_k.is_some()
        || cli.cache_type_v.is_some()
        || cli.slots.is_some()
        || cli.kv_unified
        || cli.reasoning_budget.is_some()
        || cli.reasoning_format.is_some()
}

fn gguf_model_identity(gguf: &str) -> String {
    Path::new(gguf)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or(gguf)
        .to_string()
}
