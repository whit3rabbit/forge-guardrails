use clap::{ArgAction, Parser, ValueEnum};

use crate::config::DEFAULT_BACKEND_PORT;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "forge-guardrails-proxy",
    about = "forge proxy - OpenAI-compatible proxy with guardrails"
)]
pub(crate) struct Cli {
    /// URL of externally managed backend (external mode).
    #[arg(long, value_name = "URL", conflicts_with = "backend")]
    pub(crate) backend_url: Option<String>,

    /// Backend type (managed mode).
    #[arg(long, value_enum, value_name = "BACKEND")]
    pub(crate) backend: Option<CliBackend>,

    /// Model name (required for ollama).
    #[arg(long, value_name = "MODEL")]
    pub(crate) model: Option<String>,

    /// Path to GGUF file (llamaserver/llamafile).
    #[arg(long, value_name = "PATH")]
    pub(crate) gguf: Option<String>,

    /// Trusted llamafile runtime binary path (managed llamafile).
    #[arg(long, value_name = "PATH")]
    pub(crate) llamafile_runtime: Option<String>,

    /// Backend port (default: 8080).
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT, value_name = "PORT")]
    pub(crate) backend_port: u16,

    /// Context budget mode (default: backend).
    #[arg(long, value_enum, default_value = "backend", value_name = "MODE")]
    pub(crate) budget_mode: CliBudgetMode,

    /// Manual token budget.
    #[arg(long, value_name = "N")]
    pub(crate) budget_tokens: Option<i64>,

    /// Additional backend CLI flags. Use: --extra-flags -- --flag value
    #[arg(
        long,
        value_name = "FLAG",
        num_args = 0..,
        allow_hyphen_values = true,
        trailing_var_arg = true
    )]
    pub(crate) extra_flags: Vec<String>,

    /// KV cache type for K cache in managed llama backends.
    #[arg(long, value_name = "TYPE")]
    pub(crate) cache_type_k: Option<String>,

    /// KV cache type for V cache in managed llama backends.
    #[arg(long, value_name = "TYPE")]
    pub(crate) cache_type_v: Option<String>,

    /// Number of managed llama backend slots.
    #[arg(long, value_name = "N")]
    pub(crate) slots: Option<i64>,

    /// Use unified KV cache in managed llama backends.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(crate) kv_unified: bool,

    /// Reasoning budget for managed llama backends.
    #[arg(long, value_name = "TOKENS")]
    pub(crate) reasoning_budget: Option<String>,

    /// Reasoning format for managed llama backends.
    #[arg(long, value_name = "FORMAT")]
    pub(crate) reasoning_format: Option<String>,

    /// Function-calling mode for llama-compatible OpenAI-shape backends.
    #[arg(long, value_enum, default_value = "native", value_name = "MODE")]
    pub(crate) mode: CliMode,

    /// Wire protocol used by an external backend.
    #[arg(long, value_enum, default_value = "openai", value_name = "PROTOCOL")]
    pub(crate) backend_protocol: CliBackendProtocol,

    /// Proxy listen host (default: 127.0.0.1 in CLI mode).
    #[arg(long, value_name = "HOST")]
    pub(crate) host: Option<String>,

    /// Proxy listen port (default: 8081).
    #[arg(long, value_name = "PORT")]
    pub(crate) port: Option<u16>,

    /// Force request serialization.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_serialize")]
    pub(crate) serialize: bool,

    /// Disable request serialization.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(crate) no_serialize: bool,

    /// Max retries per request (default: 3).
    #[arg(long, value_name = "N")]
    pub(crate) max_retries: Option<i32>,

    /// Enable the tool-call ONNX classifier shortcut in advisory mode.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "classify_download")]
    pub(crate) classify: bool,

    /// Download the default tool-call ONNX classifier artifact and exit.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "classify")]
    pub(crate) classify_download: bool,

    /// Local classifier artifact directory.
    #[arg(long, value_name = "PATH")]
    pub(crate) classifier_dir: Option<String>,

    /// Classifier mode.
    #[arg(long, value_name = "disabled|shadow|advisory|enforce")]
    pub(crate) classifier_mode: Option<String>,

    /// Classifier ONNX model file.
    #[arg(long, value_name = "quantized|full")]
    pub(crate) classifier_model: Option<String>,

    /// Warn when tool-call classifier latency exceeds this many milliseconds.
    #[arg(long, value_name = "MS")]
    pub(crate) classifier_max_latency_ms: Option<u64>,

    /// Local final-response classifier artifact directory.
    #[arg(long, value_name = "PATH")]
    pub(crate) final_response_classifier_dir: Option<String>,

    /// Final-response classifier mode.
    #[arg(long, value_name = "disabled|shadow|advisory|enforce")]
    pub(crate) final_response_classifier_mode: Option<String>,

    /// Final-response classifier ONNX model file.
    #[arg(long, value_name = "quantized|full")]
    pub(crate) final_response_classifier_model: Option<String>,

    /// Warn when final-response classifier latency exceeds this many milliseconds.
    #[arg(long, value_name = "MS")]
    pub(crate) final_response_classifier_max_latency_ms: Option<u64>,

    /// Tool-output compression mode.
    #[arg(long, value_name = "disabled|safe|standard|aggressive")]
    pub(crate) tool_output_compression: Option<String>,

    /// Tool-call policy nudge mode.
    #[arg(long, value_name = "disabled|standard")]
    pub(crate) tool_call_policy: Option<String>,

    /// Disable rescue parsing.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(crate) no_rescue: bool,

    /// Verbose logging.
    #[arg(short, long, action = ArgAction::SetTrue)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliBackend {
    Llamaserver,
    Llamafile,
    Ollama,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliMode {
    Native,
    Prompt,
}

impl CliMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Prompt => "prompt",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliBackendProtocol {
    Openai,
    Anthropic,
}

impl CliBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Llamaserver => "llamaserver",
            Self::Llamafile => "llamafile",
            Self::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliBudgetMode {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{normalized_extra_flags, DEFAULT_BACKEND_PORT};
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

    #[test]
    fn clap_defaults_keep_env_fallback_mode() {
        let cli = parse(&[]);
        assert!(cli.backend_url.is_none());
        assert!(cli.backend.is_none());
        assert!(cli.llamafile_runtime.is_none());
        assert_eq!(cli.backend_port, DEFAULT_BACKEND_PORT);
        assert_eq!(cli.budget_mode, CliBudgetMode::Backend);
        assert_eq!(cli.budget_tokens, None);
        assert_eq!(cli.cache_type_k, None);
        assert_eq!(cli.cache_type_v, None);
        assert_eq!(cli.slots, None);
        assert!(!cli.kv_unified);
        assert_eq!(cli.reasoning_budget, None);
        assert_eq!(cli.reasoning_format, None);
        assert_eq!(cli.mode, CliMode::Native);
        assert_eq!(cli.backend_protocol, CliBackendProtocol::Openai);
        assert!(!cli.serialize);
        assert!(!cli.no_serialize);
        assert!(!cli.no_rescue);
        assert!(!cli.classify);
        assert!(!cli.classify_download);
        assert_eq!(cli.classifier_dir, None);
        assert_eq!(cli.classifier_mode, None);
        assert_eq!(cli.classifier_model, None);
        assert_eq!(cli.classifier_max_latency_ms, None);
        assert_eq!(cli.final_response_classifier_dir, None);
        assert_eq!(cli.final_response_classifier_mode, None);
        assert_eq!(cli.final_response_classifier_model, None);
        assert_eq!(cli.final_response_classifier_max_latency_ms, None);
        assert_eq!(cli.tool_output_compression, None);
        assert_eq!(cli.tool_call_policy, None);
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
    fn clap_rejects_classify_and_classify_download() {
        assert_eq!(
            parse_err(&["--classify", "--classify-download"]),
            ErrorKind::ArgumentConflict
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
    fn first_class_managed_backend_flags_are_parsed() {
        let cli = parse(&[
            "--backend",
            "llamaserver",
            "--gguf",
            "model.gguf",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--slots",
            "4",
            "--kv-unified",
            "--reasoning-budget",
            "0",
            "--reasoning-format",
            "auto",
        ]);
        assert_eq!(cli.cache_type_k.as_deref(), Some("q8_0"));
        assert_eq!(cli.cache_type_v.as_deref(), Some("q8_0"));
        assert_eq!(cli.slots, Some(4));
        assert!(cli.kv_unified);
        assert_eq!(cli.reasoning_budget.as_deref(), Some("0"));
        assert_eq!(cli.reasoning_format.as_deref(), Some("auto"));
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
    fn classifier_flags_are_parsed() {
        let cli = parse(&[
            "--classify",
            "--classifier-dir",
            "target/classifier-artifacts/onnx",
            "--classifier-mode",
            "shadow",
            "--classifier-model",
            "quantized",
            "--classifier-max-latency-ms",
            "25",
            "--final-response-classifier-dir",
            "target/final-response-artifacts/onnx",
            "--final-response-classifier-mode",
            "advisory",
            "--final-response-classifier-model",
            "full",
            "--final-response-classifier-max-latency-ms",
            "40",
        ]);
        assert!(cli.classify);
        assert_eq!(
            cli.classifier_dir.as_deref(),
            Some("target/classifier-artifacts/onnx")
        );
        assert_eq!(cli.classifier_mode.as_deref(), Some("shadow"));
        assert_eq!(cli.classifier_model.as_deref(), Some("quantized"));
        assert_eq!(cli.classifier_max_latency_ms, Some(25));
        assert_eq!(
            cli.final_response_classifier_dir.as_deref(),
            Some("target/final-response-artifacts/onnx")
        );
        assert_eq!(
            cli.final_response_classifier_mode.as_deref(),
            Some("advisory")
        );
        assert_eq!(cli.final_response_classifier_model.as_deref(), Some("full"));
        assert_eq!(cli.final_response_classifier_max_latency_ms, Some(40));
    }

    #[test]
    fn classify_download_flag_is_parsed() {
        let cli = parse(&["--classify-download"]);
        assert!(cli.classify_download);
        assert!(!cli.classify);
    }

    #[test]
    fn mode_and_backend_protocol_are_parsed() {
        let cli = parse(&[
            "--backend-url",
            "http://localhost:8080",
            "--mode",
            "prompt",
            "--backend-protocol",
            "anthropic",
        ]);
        assert_eq!(cli.mode, CliMode::Prompt);
        assert_eq!(cli.backend_protocol, CliBackendProtocol::Anthropic);
    }
}
