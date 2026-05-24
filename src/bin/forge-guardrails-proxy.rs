use std::env;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyllm_providers::ProviderProtocol;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, options, post};
use axum::Router;
use clap::{ArgAction, Parser, ValueEnum};
use forge_guardrails::{
    AnyLlmProxyClient, AnyLlmRuntimeClient, ApiFormat, BackendError, ChunkStream,
    ContextDiscoveryError, ContextManager, HTTPServer, LLMClient, LLMResponse, LlamafileClient,
    NoCompact, OllamaClient, SamplingParams, ServerManager, StreamError, ToolSpec,
};
use reqwest::Url;
use serde_json::{json, Value};
use tokio::sync::Mutex as TokioMutex;

const DEFAULT_PROXY_PORT: u16 = 8081;
const DEFAULT_BACKEND_PORT: u16 = 8080;
const DEFAULT_ENV_CONTEXT_TOKENS: i64 = 128_000;
const DEFAULT_EXTERNAL_CONTEXT_TOKENS: i64 = 8192;
const DEFAULT_ENV_HOST: &str = "0.0.0.0";
const DEFAULT_CLI_HOST: &str = "127.0.0.1";
const DEFAULT_ENV_MODEL: &str = "gpt-4o-mini";
const DEFAULT_EXTERNAL_MODEL: &str = "default";
const DEFAULT_MAX_RETRIES: i32 = 3;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "forge-guardrails-proxy",
    about = "forge proxy - OpenAI-compatible proxy with guardrails"
)]
struct Cli {
    /// URL of externally managed backend (external mode).
    #[arg(long, value_name = "URL", conflicts_with = "backend")]
    backend_url: Option<String>,

    /// Backend type (managed mode).
    #[arg(long, value_enum, value_name = "BACKEND")]
    backend: Option<CliBackend>,

    /// Model name (required for ollama).
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,

    /// Path to GGUF file (llamaserver/llamafile).
    #[arg(long, value_name = "PATH")]
    gguf: Option<String>,

    /// Trusted llamafile runtime binary path (managed llamafile).
    #[arg(long, value_name = "PATH")]
    llamafile_runtime: Option<String>,

    /// Backend port (default: 8080).
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT, value_name = "PORT")]
    backend_port: u16,

    /// Context budget mode (default: backend).
    #[arg(long, value_enum, default_value = "backend", value_name = "MODE")]
    budget_mode: CliBudgetMode,

    /// Manual token budget.
    #[arg(long, value_name = "N")]
    budget_tokens: Option<i64>,

    /// Additional backend CLI flags. Use: --extra-flags -- --flag value
    #[arg(
        long,
        value_name = "FLAG",
        num_args = 0..,
        allow_hyphen_values = true,
        trailing_var_arg = true
    )]
    extra_flags: Vec<String>,

    /// Proxy listen host (default: 127.0.0.1 in CLI mode).
    #[arg(long, value_name = "HOST")]
    host: Option<String>,

    /// Proxy listen port (default: 8081).
    #[arg(long, value_name = "PORT")]
    port: Option<u16>,

    /// Force request serialization.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_serialize")]
    serialize: bool,

    /// Disable request serialization.
    #[arg(long, action = ArgAction::SetTrue)]
    no_serialize: bool,

    /// Max retries per request (default: 3).
    #[arg(long, value_name = "N")]
    max_retries: Option<i32>,

    /// Disable rescue parsing.
    #[arg(long, action = ArgAction::SetTrue)]
    no_rescue: bool,

    /// Verbose logging.
    #[arg(short, long, action = ArgAction::SetTrue)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliBackend {
    Llamaserver,
    Llamafile,
    Ollama,
}

impl CliBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Llamaserver => "llamaserver",
            Self::Llamafile => "llamafile",
            Self::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliBudgetMode {
    Backend,
    Manual,
    ForgeFull,
    ForgeFast,
}

impl From<CliBudgetMode> for forge_guardrails::BudgetMode {
    fn from(mode: CliBudgetMode) -> Self {
        match mode {
            CliBudgetMode::Backend => Self::Backend,
            CliBudgetMode::Manual => Self::Manual,
            CliBudgetMode::ForgeFull => Self::ForgeFull,
            CliBudgetMode::ForgeFast => Self::ForgeFast,
        }
    }
}

#[derive(Clone)]
struct ProxyConfig {
    host: String,
    port: u16,
    default_model: String,
    context_tokens: i64,
    max_retries: i32,
    rescue_enabled: bool,
    serialize_requests: bool,
    verbose: bool,
}

#[derive(Clone)]
struct AppState {
    config: Arc<ProxyConfig>,
    client_factory: Arc<ClientFactory>,
    request_mutex: Arc<TokioMutex<()>>,
}

enum ClientFactory {
    Runtime(AnyLlmRuntimeClient),
    DirectOpenAi {
        base_url: String,
        api_key: Option<String>,
        context_tokens: i64,
    },
    ManagedLlamafile {
        gguf_path: String,
        base_url: String,
        mode: String,
    },
    ManagedOllama {
        model: String,
        context_tokens: i64,
    },
}

enum RoutedClient {
    Runtime(AnyLlmRuntimeClient),
    DirectOpenAi(AnyLlmProxyClient),
    ManagedLlamafile(LlamafileClient),
    ManagedOllama(OllamaClient),
}

struct DirectOpenAiUpstream {
    base_url: String,
    api_key: Option<String>,
}

struct Startup {
    config: ProxyConfig,
    client_factory: ClientFactory,
    managed_server: Option<ServerManager>,
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run_main(cli) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run_main(cli: Cli) -> Result<(), String> {
    apply_litellm_env_aliases();
    let startup = build_startup(cli)?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("failed to build tokio runtime: {err}"))?;

    runtime.block_on(serve(
        startup.config,
        startup.client_factory,
        startup.managed_server,
    ))
}

async fn serve(
    config: ProxyConfig,
    client_factory: ClientFactory,
    managed_server: Option<ServerManager>,
) -> Result<(), String> {
    let result = serve_inner(config, client_factory).await;
    if let Some(server) = managed_server {
        if let Err(err) = server.stop() {
            let stop_err = format!("failed to stop managed backend: {err}");
            if result.is_ok() {
                return Err(stop_err);
            }
            eprintln!("warning: {stop_err}");
        }
    }
    result
}

async fn serve_inner(config: ProxyConfig, client_factory: ClientFactory) -> Result<(), String> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|err| format!("invalid bind address: {err}"))?;
    let state = AppState {
        config: Arc::new(config.clone()),
        client_factory: Arc::new(client_factory),
        request_mutex: Arc::new(TokioMutex::new(())),
    };

    eprintln!(
        "forge-guardrails-proxy listening on http://{}:{}",
        config.host, config.port
    );
    eprintln!(
        "warning: inbound auth is not enforced; do not expose this proxy publicly without an auth layer"
    );
    if config.verbose {
        eprintln!(
            "proxy config: model={}, context_tokens={}, max_retries={}, rescue_enabled={}, serialize_requests={}",
            config.default_model,
            config.context_tokens,
            config.max_retries,
            config.rescue_enabled,
            config.serialize_requests
        );
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", options(cors_preflight))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages", options(cors_preflight))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|err| format!("failed to bind {addr}: {err}"))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| format!("server failed: {err}"))
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let ctrl_c = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(err) => {
                    eprintln!("warning: failed to install SIGTERM handler: {err}");
                    std::future::pending::<()>().await;
                }
            }
        };
        tokio::select! {
            _ = ctrl_c => {}
            _ = terminate => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn health() -> Response {
    build_response(200, "application/json", json!({"status": "ok"}).to_string())
}

async fn models(State(state): State<AppState>) -> Response {
    build_response(
        200,
        "application/json",
        json!({
            "object": "list",
            "data": [{
                "id": state.config.default_model,
                "object": "model",
                "created": 0,
                "owned_by": "forge-guardrails"
            }]
        })
        .to_string(),
    )
}

async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    proxy_post(state, "/v1/chat/completions", body, extract_openai_model).await
}

async fn anthropic_messages(State(state): State<AppState>, body: Bytes) -> Response {
    proxy_post(state, "/v1/messages", body, extract_anthropic_model).await
}

async fn cors_preflight() -> Response {
    build_response(204, "", String::new())
}

async fn proxy_post(
    state: AppState,
    path: &'static str,
    body: Bytes,
    model_from_body: fn(&[u8], &str) -> String,
) -> Response {
    let model = model_from_body(body.as_ref(), &state.config.default_model);
    let client = Arc::new(state.client_factory.client_for_model(model.clone()));
    let context_manager = Arc::new(TokioMutex::new(ContextManager::new(
        Box::new(NoCompact),
        state.config.context_tokens,
        None,
        None,
        None,
    )));
    let server = HTTPServer::new(
        &state.config.host,
        state.config.port,
        false,
        state.config.max_retries,
        state.config.rescue_enabled,
        &model,
    );

    let _guard = if state.config.serialize_requests {
        Some(state.request_mutex.lock().await)
    } else {
        None
    };

    let (status, content_type, _headers, response_body) = server
        .handle_request("POST", path, body.as_ref(), &client, &context_manager)
        .await;
    build_response(status, content_type, response_body)
}

fn build_startup(cli: Cli) -> Result<Startup, String> {
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

    let base_url = normalize_openai_base_url(backend_url)?;
    let context_tokens = match cli.budget_tokens {
        Some(tokens) => validate_positive_i64(tokens, "--budget-tokens")?,
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
    let client_factory = ClientFactory::DirectOpenAi {
        base_url,
        api_key: direct_openai_api_key(&[]),
        context_tokens,
    };
    Ok(Startup {
        config,
        client_factory,
        managed_server: None,
    })
}

fn build_managed_startup(cli: &Cli, backend: CliBackend) -> Result<Startup, String> {
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
                "native",
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
                "native",
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
                    mode: "native".to_string(),
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
                "native",
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
                    mode: "native".to_string(),
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

fn apply_env_cli_overrides(config: &mut ProxyConfig, cli: &Cli) -> Result<(), String> {
    if let Some(host) = cli.host.as_deref() {
        config.host = validate_nonempty(host, "--host")?.to_string();
    }
    if let Some(port) = cli.port {
        config.port = validate_nonzero_u16(port, "--port")?;
    }
    if let Some(model) = cli.model.as_deref() {
        config.default_model = validate_nonempty(model, "--model")?.to_string();
    }
    if let Some(tokens) = cli.budget_tokens {
        config.context_tokens = validate_positive_i64(tokens, "--budget-tokens")?;
    }
    if let Some(max_retries) = cli.max_retries {
        config.max_retries = validate_nonnegative_i32(max_retries, "--max-retries")?;
    }
    if cli.no_rescue {
        config.rescue_enabled = false;
    }
    if cli.serialize {
        config.serialize_requests = true;
    } else if cli.no_serialize {
        config.serialize_requests = false;
    }
    if cli.verbose {
        config.verbose = true;
    }
    Ok(())
}

fn cli_host(cli: &Cli) -> Result<String, String> {
    Ok(cli
        .host
        .as_deref()
        .map(|host| validate_nonempty(host, "--host").map(ToOwned::to_owned))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_CLI_HOST.to_string()))
}

fn cli_port(cli: &Cli) -> Result<u16, String> {
    validate_nonzero_u16(cli.port.unwrap_or(DEFAULT_PROXY_PORT), "--port")
}

fn cli_model(cli: &Cli, default: &str) -> Result<String, String> {
    Ok(cli
        .model
        .as_deref()
        .map(|model| validate_nonempty(model, "--model").map(ToOwned::to_owned))
        .transpose()?
        .unwrap_or_else(|| default.to_string()))
}

fn cli_max_retries(cli: &Cli) -> Result<i32, String> {
    validate_nonnegative_i32(
        cli.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
        "--max-retries",
    )
}

fn require_cli_model(cli: &Cli) -> Result<String, String> {
    cli.model
        .as_deref()
        .map(|model| validate_nonempty(model, "--model").map(ToOwned::to_owned))
        .transpose()?
        .ok_or_else(|| "--backend ollama requires --model".to_string())
}

fn require_cli_gguf(cli: &Cli, backend: &str) -> Result<String, String> {
    cli.gguf
        .as_deref()
        .map(|gguf| validate_nonempty(gguf, "--gguf").map(ToOwned::to_owned))
        .transpose()?
        .ok_or_else(|| format!("--backend {backend} requires --gguf"))
}

fn require_cli_llamafile_runtime(cli: &Cli) -> Result<String, String> {
    cli.llamafile_runtime
        .as_deref()
        .map(|runtime| validate_nonempty(runtime, "--llamafile-runtime").map(ToOwned::to_owned))
        .transpose()?
        .ok_or_else(|| "--backend llamafile requires --llamafile-runtime".to_string())
}

fn validate_nonempty<'a>(value: &'a str, label: &str) -> Result<&'a str, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{label} cannot be empty"))
    } else {
        Ok(trimmed)
    }
}

fn validate_nonzero_u16(value: u16, label: &str) -> Result<u16, String> {
    if value == 0 {
        Err(format!("{label} cannot be 0"))
    } else {
        Ok(value)
    }
}

fn validate_optional_positive_i64(value: Option<i64>, label: &str) -> Result<Option<i64>, String> {
    value
        .map(|tokens| validate_positive_i64(tokens, label))
        .transpose()
}

fn validate_positive_i64(value: i64, label: &str) -> Result<i64, String> {
    if value <= 0 {
        Err(format!("{label} must be positive"))
    } else {
        Ok(value)
    }
}

fn validate_nonnegative_i32(value: i32, label: &str) -> Result<i32, String> {
    if value < 0 {
        Err(format!("{label} must be non-negative"))
    } else {
        Ok(value)
    }
}

fn resolve_serialize(cli: &Cli, default: bool) -> bool {
    if cli.serialize {
        true
    } else if cli.no_serialize {
        false
    } else {
        default
    }
}

fn normalized_extra_flags(flags: &[String]) -> Vec<String> {
    if flags.first().is_some_and(|flag| flag == "--") {
        flags[1..].to_vec()
    } else {
        flags.to_vec()
    }
}

fn normalize_openai_base_url(raw: &str) -> Result<String, String> {
    let trimmed = validate_nonempty(raw, "--backend-url")?.trim_end_matches('/');
    Url::parse(trimmed).map_err(|err| format!("invalid --backend-url: {err}"))?;
    if trimmed.ends_with("/v1") {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("{trimmed}/v1"))
    }
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

impl ClientFactory {
    fn client_for_model(&self, model: String) -> RoutedClient {
        match self {
            Self::Runtime(client) => RoutedClient::Runtime(client.for_model(model)),
            Self::DirectOpenAi {
                base_url,
                api_key,
                context_tokens,
            } => {
                let mut client = AnyLlmProxyClient::new(model)
                    .with_base_url(base_url)
                    .with_context_length(*context_tokens);
                if let Some(api_key) = api_key {
                    client = client.with_api_key(api_key.clone());
                }
                RoutedClient::DirectOpenAi(client)
            }
            Self::ManagedLlamafile {
                gguf_path,
                base_url,
                mode,
            } => RoutedClient::ManagedLlamafile(
                LlamafileClient::new(gguf_path)
                    .with_base_url(base_url)
                    .with_mode(mode),
            ),
            Self::ManagedOllama {
                model,
                context_tokens,
            } => {
                let client = OllamaClient::new(model.clone());
                client.set_num_ctx(Some(*context_tokens));
                RoutedClient::ManagedOllama(client)
            }
        }
    }
}

impl LLMClient for RoutedClient {
    fn api_format(&self) -> ApiFormat {
        match self {
            Self::Runtime(client) => client.api_format(),
            Self::DirectOpenAi(client) => client.api_format(),
            Self::ManagedLlamafile(client) => client.api_format(),
            Self::ManagedOllama(client) => client.api_format(),
        }
    }

    fn last_usage(&self) -> Option<forge_guardrails::TokenUsage> {
        match self {
            Self::Runtime(client) => client.last_usage(),
            Self::DirectOpenAi(client) => client.last_usage(),
            Self::ManagedLlamafile(client) => client.last_usage(),
            Self::ManagedOllama(client) => client.last_usage(),
        }
    }

    fn last_call_info(&self) -> Option<forge_guardrails::LLMCallInfo> {
        match self {
            Self::Runtime(client) => client.last_call_info(),
            Self::DirectOpenAi(client) => client.last_call_info(),
            Self::ManagedLlamafile(client) => client.last_call_info(),
            Self::ManagedOllama(client) => client.last_call_info(),
        }
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        match self {
            Self::Runtime(client) => client.send(messages, tools, sampling).await,
            Self::DirectOpenAi(client) => client.send(messages, tools, sampling).await,
            Self::ManagedLlamafile(client) => client.send(messages, tools, sampling).await,
            Self::ManagedOllama(client) => client.send(messages, tools, sampling).await,
        }
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        match self {
            Self::Runtime(client) => client.send_stream(messages, tools, sampling).await,
            Self::DirectOpenAi(client) => client.send_stream(messages, tools, sampling).await,
            Self::ManagedLlamafile(client) => client.send_stream(messages, tools, sampling).await,
            Self::ManagedOllama(client) => client.send_stream(messages, tools, sampling).await,
        }
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        match self {
            Self::Runtime(client) => client.get_context_length().await,
            Self::DirectOpenAi(client) => client.get_context_length().await,
            Self::ManagedLlamafile(client) => client.get_context_length().await,
            Self::ManagedOllama(client) => client.get_context_length().await,
        }
    }
}

fn build_response(status: u16, content_type: &str, body: String) -> Response {
    let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = (status_code, body).into_response();
    if !content_type.is_empty() {
        if let Ok(value) = HeaderValue::from_str(content_type) {
            response.headers_mut().insert(header::CONTENT_TYPE, value);
        }
    }
    for (name, value) in HTTPServer::cors_headers() {
        if let Some(header_name) = cors_header_name(name) {
            response
                .headers_mut()
                .insert(header_name, HeaderValue::from_static(value));
        }
    }
    response
}

fn cors_header_name(name: &str) -> Option<HeaderName> {
    match name {
        "Access-Control-Allow-Origin" => {
            Some(HeaderName::from_static("access-control-allow-origin"))
        }
        "Access-Control-Allow-Methods" => {
            Some(HeaderName::from_static("access-control-allow-methods"))
        }
        "Access-Control-Allow-Headers" => {
            Some(HeaderName::from_static("access-control-allow-headers"))
        }
        _ => None,
    }
}

fn extract_openai_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

fn extract_anthropic_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

fn extract_json_model(body: &[u8], default_model: &str) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("model")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| default_model.to_string())
}

fn apply_litellm_env_aliases() {
    for (key, value) in anyllm_proxy::config::env_aliases::compute_env_aliases() {
        // SAFETY: this runs before any tokio runtime or application thread is
        // started, matching anyllm_proxy's own startup rule.
        unsafe {
            env::set_var(key, value);
        }
    }
}

fn direct_local_openai_upstream_from_env() -> Option<DirectOpenAiUpstream> {
    direct_local_openai_upstream(
        env::var("OPENAI_BASE_URL").ok().as_deref(),
        env::var("BACKEND").ok().as_deref(),
        env::var("PROXY_CONFIG").is_ok(),
    )
}

fn direct_local_openai_upstream(
    openai_base_url: Option<&str>,
    backend: Option<&str>,
    proxy_config_set: bool,
) -> Option<DirectOpenAiUpstream> {
    if proxy_config_set {
        return None;
    }

    let backend = backend
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("openai");

    if backend.eq_ignore_ascii_case("openai") {
        let raw = openai_base_url?.trim();
        return local_openai_upstream(raw, None);
    }

    let provider = anyllm_providers::get_provider(&backend.to_ascii_lowercase())?;
    if provider.protocol != ProviderProtocol::OpenAICompat {
        return None;
    }

    let base_url = openai_base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(provider.default_base_url);
    local_openai_upstream(base_url, Some(provider.env_vars))
}

fn local_openai_upstream(
    base_url: &str,
    provider_env_vars: Option<&[&str]>,
) -> Option<DirectOpenAiUpstream> {
    if !is_exact_local_url(base_url) {
        return None;
    }
    Some(DirectOpenAiUpstream {
        base_url: base_url.to_string(),
        api_key: direct_openai_api_key(provider_env_vars.unwrap_or(&[])),
    })
}

fn direct_openai_api_key(provider_env_vars: &[&str]) -> Option<String> {
    std::iter::once("OPENAI_API_KEY")
        .chain(provider_env_vars.iter().copied())
        .find_map(|key| {
            env::var(key)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn is_exact_local_url(raw: &str) -> bool {
    let Ok(parsed) = Url::parse(raw) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    matches!(
        host.to_ascii_lowercase().as_str(),
        "host.docker.internal" | "localhost" | "127.0.0.1" | "::1" | "[::1]"
    )
}

impl ProxyConfig {
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            host: env_string(&["FORGE_HOST"], DEFAULT_ENV_HOST),
            port: env_u16(
                &["FORGE_PORT", "PORT", "LISTEN_PORT"],
                DEFAULT_PROXY_PORT,
                "FORGE_PORT",
            )?,
            default_model: env_string(&["FORGE_MODEL", "SMALL_MODEL"], DEFAULT_ENV_MODEL),
            context_tokens: env_i64(
                &["FORGE_CONTEXT_TOKENS"],
                DEFAULT_ENV_CONTEXT_TOKENS,
                "FORGE_CONTEXT_TOKENS",
            )?,
            max_retries: env_i32(
                &["FORGE_MAX_RETRIES"],
                DEFAULT_MAX_RETRIES,
                "FORGE_MAX_RETRIES",
            )?,
            rescue_enabled: env_bool("FORGE_RESCUE_ENABLED", true)?,
            serialize_requests: env_bool("FORGE_SERIALIZE_REQUESTS", false)?,
            verbose: false,
        })
    }
}

fn env_string(keys: &[&str], default: &str) -> String {
    keys.iter()
        .find_map(|key| env::var(key).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_u16(keys: &[&str], default: u16, label: &str) -> Result<u16, String> {
    match keys.iter().find_map(|key| env::var(key).ok()) {
        Some(raw) => {
            let value = raw
                .parse::<u16>()
                .map_err(|_| format!("{label} must be a number in 1-65535, got '{raw}'"))?;
            if value == 0 {
                return Err(format!("{label} cannot be 0"));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

fn env_i64(keys: &[&str], default: i64, label: &str) -> Result<i64, String> {
    match keys.iter().find_map(|key| env::var(key).ok()) {
        Some(raw) => {
            let value = raw
                .parse::<i64>()
                .map_err(|_| format!("{label} must be a positive integer, got '{raw}'"))?;
            if value <= 0 {
                return Err(format!("{label} must be positive"));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

fn env_i32(keys: &[&str], default: i32, label: &str) -> Result<i32, String> {
    match keys.iter().find_map(|key| env::var(key).ok()) {
        Some(raw) => {
            let value = raw
                .parse::<i32>()
                .map_err(|_| format!("{label} must be a non-negative integer, got '{raw}'"))?;
            if value < 0 {
                return Err(format!("{label} must be non-negative"));
            }
            Ok(value)
        }
        None => Ok(default),
    }
}

fn env_bool(key: &str, default: bool) -> Result<bool, String> {
    match env::var(key) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!("{key} must be true or false, got '{raw}'")),
        },
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("forge-guardrails-proxy").chain(args.iter().copied()))
            .expect("parse")
    }

    fn parse_err(args: &[&str]) -> ErrorKind {
        Cli::try_parse_from(std::iter::once("forge-guardrails-proxy").chain(args.iter().copied()))
            .expect_err("expected parse error")
            .kind()
    }

    fn sample_config() -> ProxyConfig {
        ProxyConfig {
            host: DEFAULT_ENV_HOST.to_string(),
            port: DEFAULT_PROXY_PORT,
            default_model: DEFAULT_ENV_MODEL.to_string(),
            context_tokens: DEFAULT_ENV_CONTEXT_TOKENS,
            max_retries: DEFAULT_MAX_RETRIES,
            rescue_enabled: true,
            serialize_requests: false,
            verbose: false,
        }
    }

    #[test]
    fn clap_defaults_keep_env_fallback_mode() {
        let cli = parse(&[]);
        assert!(cli.backend_url.is_none());
        assert!(cli.backend.is_none());
        assert!(cli.llamafile_runtime.is_none());
        assert_eq!(cli.backend_port, DEFAULT_BACKEND_PORT);
        assert_eq!(cli.budget_mode, CliBudgetMode::Backend);
        assert_eq!(cli.budget_tokens, None);
        assert!(!cli.serialize);
        assert!(!cli.no_serialize);
        assert!(!cli.no_rescue);
    }

    #[test]
    fn clap_help_is_available() {
        assert_eq!(parse_err(&["--help"]), ErrorKind::DisplayHelp);
    }

    #[test]
    fn clap_rejects_invalid_backend() {
        assert_eq!(
            parse_err(&["--backend", "not-a-backend"]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn clap_rejects_backend_url_with_backend() {
        assert_eq!(
            parse_err(&[
                "--backend-url",
                "http://localhost:8080",
                "--backend",
                "ollama"
            ]),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn clap_rejects_serialize_and_no_serialize() {
        assert_eq!(
            parse_err(&[
                "--backend-url",
                "http://localhost:8080",
                "--serialize",
                "--no-serialize"
            ]),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn extra_flags_after_separator_are_normalized() {
        let cli = parse(&[
            "--backend",
            "llamaserver",
            "--gguf",
            "model.gguf",
            "--extra-flags",
            "--",
            "--reasoning-format",
            "auto",
        ]);
        assert_eq!(
            normalized_extra_flags(&cli.extra_flags),
            vec!["--reasoning-format".to_string(), "auto".to_string()]
        );
    }

    #[test]
    fn llamafile_runtime_flag_is_parsed() {
        let cli = parse(&[
            "--backend",
            "llamafile",
            "--gguf",
            "model.gguf",
            "--llamafile-runtime",
            "/opt/forge/bin/llamafile",
        ]);
        assert_eq!(
            cli.llamafile_runtime.as_deref(),
            Some("/opt/forge/bin/llamafile")
        );
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
    fn env_fallback_accepts_safe_cli_overrides() {
        let cli = parse(&[
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
            "--model",
            "env-override",
            "--budget-tokens",
            "2048",
            "--max-retries",
            "0",
            "--no-rescue",
            "--serialize",
            "-v",
        ]);
        let mut config = sample_config();
        apply_env_cli_overrides(&mut config, &cli).expect("overrides");
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 9090);
        assert_eq!(config.default_model, "env-override");
        assert_eq!(config.context_tokens, 2048);
        assert_eq!(config.max_retries, 0);
        assert!(!config.rescue_enabled);
        assert!(config.serialize_requests);
        assert!(config.verbose);
    }

    #[test]
    fn managed_mode_serializes_by_default() {
        let cli = parse(&["--backend", "llamaserver", "--gguf", "model.gguf"]);
        assert!(resolve_serialize(&cli, true));
    }

    #[test]
    fn managed_mode_can_disable_serialization() {
        let cli = parse(&[
            "--backend",
            "llamaserver",
            "--gguf",
            "model.gguf",
            "--no-serialize",
        ]);
        assert!(!resolve_serialize(&cli, true));
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
    fn extracts_openai_request_model() {
        let body = br#"{"model":"forge-virtual","messages":[]}"#;
        assert_eq!(extract_openai_model(body, "default"), "forge-virtual");
    }

    #[test]
    fn extracts_anthropic_request_model() {
        let body = br#"{"model":"claude-sonnet","messages":[],"max_tokens":64}"#;
        assert_eq!(extract_anthropic_model(body, "default"), "claude-sonnet");
    }

    #[test]
    fn model_extraction_falls_back_for_invalid_json() {
        assert_eq!(extract_openai_model(b"not json", "default"), "default");
    }

    #[test]
    fn model_extraction_falls_back_for_empty_model() {
        let body = br#"{"model":"   ","messages":[]}"#;
        assert_eq!(extract_anthropic_model(body, "default"), "default");
    }

    #[test]
    fn default_proxy_port_matches_python_reference() {
        assert_eq!(DEFAULT_PROXY_PORT, 8081);
    }

    #[test]
    fn direct_local_upstream_allows_host_docker_internal() {
        assert_eq!(
            direct_local_openai_upstream(Some("http://host.docker.internal:11434/v1"), None, false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://host.docker.internal:11434/v1")
        );
    }

    #[test]
    fn direct_local_upstream_does_not_override_proxy_config() {
        assert!(direct_local_openai_upstream(
            Some("http://host.docker.internal:11434/v1"),
            None,
            true
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_does_not_override_non_openai_backend() {
        assert!(direct_local_openai_upstream(
            Some("http://host.docker.internal:11434/v1"),
            Some("anthropic"),
            false
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_rejects_host_suffix_trick() {
        assert!(direct_local_openai_upstream(
            Some("http://host.docker.internal.example.com/v1"),
            None,
            false
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_uses_ollama_catalog_default() {
        assert_eq!(
            direct_local_openai_upstream(None, Some("ollama"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn direct_local_upstream_uses_ollama_local_override() {
        assert_eq!(
            direct_local_openai_upstream(
                Some("http://host.docker.internal:11434/v1"),
                Some("ollama"),
                false
            )
            .map(|upstream| upstream.base_url)
            .as_deref(),
            Some("http://host.docker.internal:11434/v1")
        );
    }

    #[test]
    fn direct_local_upstream_uses_lm_studio_catalog_default() {
        assert_eq!(
            direct_local_openai_upstream(None, Some("lm_studio"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:1234/v1")
        );
    }

    #[test]
    fn direct_local_upstream_uses_hosted_vllm_catalog_default() {
        assert_eq!(
            direct_local_openai_upstream(None, Some("hosted_vllm"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:8000/v1")
        );
    }

    #[test]
    fn direct_local_upstream_keeps_public_provider_default_on_runtime_path() {
        assert!(direct_local_openai_upstream(None, Some("groq"), false).is_none());
    }

    #[test]
    fn direct_local_upstream_allows_public_provider_when_overridden_to_local() {
        assert_eq!(
            direct_local_openai_upstream(Some("http://localhost:9999/v1"), Some("groq"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:9999/v1")
        );
    }

    #[test]
    fn direct_local_upstream_rejects_non_openai_compat_provider() {
        assert!(direct_local_openai_upstream(
            Some("http://localhost:9999/v1"),
            Some("anthropic"),
            false
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_rejects_malformed_url() {
        assert!(direct_local_openai_upstream(Some("not a url"), None, false).is_none());
    }

    #[test]
    fn exact_local_url_allows_ipv6_loopback() {
        assert!(is_exact_local_url("http://[::1]:11434/v1"));
    }
}
