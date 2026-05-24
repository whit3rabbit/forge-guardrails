use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use forge_guardrails::clients::base::LLMCallInfo;
use forge_guardrails::error::ToolError;
use forge_guardrails::workflow::ToolCallable;
use forge_guardrails::{
    AnthropicClient, AnyLlmProxyClient, ApiFormat, ChunkStream, ContextManager, ForgeError,
    LLMClient, LLMResponse, LlamafileClient, Message, MessageType, NoCompact, OllamaClient,
    SamplingParams, StreamChunk, ToolDef, ToolSpec, Workflow, WorkflowRunner,
};
use indexmap::IndexMap;
use serde_json::{json, Value};
use tokio::sync::Mutex;

const DEFAULT_PROXY_URL: &str = "http://127.0.0.1:8081/v1";
const DEFAULT_LOCAL_URL: &str = "http://127.0.0.1:8080/v1";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Cli {
    backend: String,
    model: Option<String>,
    gguf: Option<String>,
    base_url: Option<String>,
    runs: usize,
    scenarios: Vec<String>,
    stream: bool,
    output: Option<String>,
    ablation: String,
    mode: Option<String>,
    reasoning_budget: Option<String>,
    anthropic_api_key: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct Ablation {
    rescue_enabled: bool,
    max_retries: i32,
    use_required_steps: bool,
}

struct SmokeScenario {
    name: String,
    workflow: Workflow,
    user_message: String,
    capture: Arc<StdMutex<Option<IndexMap<String, Value>>>>,
}

struct CountingClient<C> {
    inner: C,
    calls: AtomicI32,
}

impl<C> CountingClient<C> {
    fn new(inner: C) -> Self {
        Self {
            inner,
            calls: AtomicI32::new(0),
        }
    }

    fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl<C: LLMClient> LLMClient for CountingClient<C> {
    fn api_format(&self) -> ApiFormat {
        self.inner.api_format()
    }

    fn last_usage(&self) -> Option<forge_guardrails::TokenUsage> {
        self.inner.last_usage()
    }

    fn last_call_info(&self) -> Option<LLMCallInfo> {
        self.inner.last_call_info()
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.send(messages, tools, sampling).await
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.send_stream(messages, tools, sampling).await
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        self.inner.get_context_length().await
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let cli = match parse_args(env::args().skip(1)) {
        Ok(cli) => cli,
        Err(message) if message == "__help__" => {
            print_help();
            return;
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("run `forge-eval --help` for usage");
            std::process::exit(2);
        }
    };

    if let Err(err) = run_cli(cli).await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run_cli(cli: Cli) -> Result<(), String> {
    match cli.backend.as_str() {
        "openai-proxy" => {
            let model = require_model(&cli)?;
            let base_url = cli.base_url.as_deref().unwrap_or(DEFAULT_PROXY_URL);
            let client = AnyLlmProxyClient::new(model.clone()).with_base_url(base_url);
            run_with_client(client, &cli, &model).await
        }
        "ollama" => {
            let model = require_model(&cli)?;
            let client = OllamaClient::new(model.clone());
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

async fn run_with_client<C: LLMClient + 'static>(
    client: C,
    cli: &Cli,
    model: &str,
) -> Result<(), String> {
    let client = Arc::new(CountingClient::new(client));
    let scenario_names = if cli.scenarios.is_empty() {
        vec![
            "basic_2step".to_string(),
            "sequential_3step".to_string(),
            "error_recovery".to_string(),
        ]
    } else {
        cli.scenarios.clone()
    };
    let ablation = parse_ablation(&cli.ablation)?;

    for scenario_name in &scenario_names {
        for run_idx in 1..=cli.runs {
            let scenario = build_scenario(scenario_name, ablation.use_required_steps)?;
            let emitted: Arc<StdMutex<Vec<Message>>> = Arc::new(StdMutex::new(Vec::new()));
            let emitted_cb = emitted.clone();
            let context = Arc::new(Mutex::new(ContextManager::new(
                Box::new(NoCompact),
                8192,
                None,
                None,
                None,
            )));
            let runner = WorkflowRunner::new(
                client.clone(),
                context,
                15,
                ablation.max_retries,
                2,
                cli.stream,
                Some(Box::new(|_chunk: &StreamChunk| {})),
                Some(Box::new(move |message: &Message| {
                    emitted_cb
                        .lock()
                        .expect("message capture lock")
                        .push(message.clone());
                })),
                ablation.rescue_enabled,
                None,
            );

            let before_calls = client.calls();
            let start = Instant::now();
            let result = runner
                .run(&scenario.workflow, &scenario.user_message, None, None, None)
                .await;
            let elapsed = start.elapsed().as_secs_f64();
            let iterations = client.calls() - before_calls;
            let messages = emitted.lock().expect("message capture lock").clone();
            let row = row_for_result(
                &cli.backend,
                model,
                &cli.ablation,
                cli,
                &scenario,
                run_idx,
                iterations,
                elapsed,
                result,
                &messages,
            );
            write_row(cli.output.as_deref(), &row)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn row_for_result(
    backend: &str,
    model: &str,
    ablation: &str,
    cli: &Cli,
    scenario: &SmokeScenario,
    run_idx: usize,
    iterations: i32,
    elapsed: f64,
    result: Result<Value, ForgeError>,
    messages: &[Message],
) -> Value {
    let captured_args = scenario
        .capture
        .lock()
        .expect("capture lock")
        .clone()
        .unwrap_or_default();
    let final_text = terminal_text(&captured_args);
    let accuracy = result
        .as_ref()
        .ok()
        .map(|_| validate_scenario(&scenario.name, &final_text));
    let completeness = result.is_ok();
    let success = completeness && accuracy != Some(false);
    let (error_type, error_message, raw_response) = match &result {
        Ok(_) => (Value::Null, Value::Null, Value::Null),
        Err(err) => (
            json!(error_kind(err)),
            json!(err.to_string()),
            match err {
                ForgeError::ToolCall(tool_err) => tool_err
                    .raw_response
                    .as_ref()
                    .map(|raw| json!(raw))
                    .unwrap_or(Value::Null),
                _ => Value::Null,
            },
        ),
    };
    let stats = message_stats(messages);
    let (tool_sequence, tool_args) = tool_trace(messages);

    json!({
        "impl": "rust",
        "model": model,
        "backend": backend,
        "mode": cli.mode.clone().unwrap_or_else(|| default_mode(backend).to_string()),
        "ablation": ablation,
        "tool_choice": "auto",
        "scenario": scenario.name,
        "run": run_idx,
        "stream": cli.stream,
        "completeness": completeness,
        "success": success,
        "accuracy": accuracy,
        "iterations": iterations,
        "elapsed_s": (elapsed * 100.0).round() / 100.0,
        "error_type": error_type,
        "error_message": error_message,
        "retry_nudges": stats.retry_nudges,
        "step_nudges": stats.step_nudges,
        "tool_errors": stats.tool_errors,
        "reasoning_msgs": stats.reasoning_msgs,
        "tool_sequence": tool_sequence,
        "tool_args": tool_args,
        "final_text": final_text,
        "raw_response_on_failure": raw_response,
        "reasoning_budget": cli.reasoning_budget,
    })
}

fn write_row(output: Option<&str>, row: &Value) -> Result<(), String> {
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    if let Some(path) = output {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| err.to_string())?;
        writeln!(file, "{line}").map_err(|err| err.to_string())
    } else {
        println!("{line}");
        Ok(())
    }
}

fn build_scenario(name: &str, use_required_steps: bool) -> Result<SmokeScenario, String> {
    match name {
        "basic_2step" => basic_2step(use_required_steps),
        "sequential_3step" => sequential_3step(use_required_steps),
        "error_recovery" => error_recovery(use_required_steps),
        other => Err(format!("unsupported scenario: {other}")),
    }
}

fn basic_2step(use_required_steps: bool) -> Result<SmokeScenario, String> {
    let capture = Arc::new(StdMutex::new(None));
    let mut tools = IndexMap::new();
    tools.insert(
        "get_country_info".to_string(),
        make_tool(
            "get_country_info",
            "Look up facts about a country.",
            json!({
                "type": "object",
                "properties": {"country": {"type": "string", "description": "Country name"}},
                "required": ["country"]
            }),
            |_args| {
                Ok(json!(
                    "The capital of France is Paris. Population: 2.1 million."
                ))
            },
        )?,
    );
    tools.insert(
        "summarize".to_string(),
        terminal_tool(
            "summarize",
            "Summarize content and provide the final answer.",
            json!({
                "type": "object",
                "properties": {"content": {"type": "string", "description": "The content to summarize"}},
                "required": ["content"]
            }),
            "content",
            capture.clone(),
        )?,
    );
    let required = if use_required_steps {
        vec!["get_country_info".to_string()]
    } else {
        Vec::new()
    };
    let workflow = Workflow::new(
        "basic_2step",
        "Simple 2-step information retrieval and summary",
        tools,
        required,
        "summarize".to_string().into(),
        "You are a helpful assistant. First use get_country_info, then summarize.",
    )?;
    Ok(SmokeScenario {
        name: "basic_2step".to_string(),
        workflow,
        user_message: "What is the capital of France?".to_string(),
        capture,
    })
}

fn sequential_3step(use_required_steps: bool) -> Result<SmokeScenario, String> {
    let capture = Arc::new(StdMutex::new(None));
    let mut tools = IndexMap::new();
    tools.insert(
        "fetch_sales_data".to_string(),
        make_tool(
            "fetch_sales_data",
            "Fetch sales data for a given quarter and year.",
            json!({
                "type": "object",
                "properties": {
                    "quarter": {"type": "integer", "description": "Quarter number"},
                    "year": {"type": "integer", "description": "Four-digit year"}
                },
                "required": ["quarter", "year"]
            }),
            |_args| {
                Ok(json!(
                    "Dataset: 150 records, 12 columns, covering Q1-Q4 2024 sales data."
                ))
            },
        )?,
    );
    tools.insert(
        "analyze_sales".to_string(),
        make_tool(
            "analyze_sales",
            "Analyze the loaded sales data and produce findings.",
            json!({"type": "object", "properties": {}}),
            |_args| Ok(json!("Analysis: Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC.")),
        )?,
    );
    tools.insert(
        "report".to_string(),
        terminal_tool(
            "report",
            "Produce a final report from findings.",
            json!({
                "type": "object",
                "properties": {"findings": {"type": "string", "description": "The findings to include in the report"}},
                "required": ["findings"]
            }),
            "findings",
            capture.clone(),
        )?,
    );
    let required = if use_required_steps {
        vec!["fetch_sales_data".to_string(), "analyze_sales".to_string()]
    } else {
        Vec::new()
    };
    let workflow = Workflow::new(
        "sequential_3step",
        "Fetch data, analyze, then report",
        tools,
        required,
        "report".to_string().into(),
        "You are a data analyst assistant. Fetch data, analyze it, then report.",
    )?;
    Ok(SmokeScenario {
        name: "sequential_3step".to_string(),
        workflow,
        user_message: "Generate a sales report from the Q4 2024 dataset.".to_string(),
        capture,
    })
}

fn error_recovery(use_required_steps: bool) -> Result<SmokeScenario, String> {
    let capture = Arc::new(StdMutex::new(None));
    let mut tools = IndexMap::new();
    tools.insert(
        "fetch".to_string(),
        make_tool(
            "fetch",
            "Fetch records. The count parameter must be a numeric string.",
            json!({
                "type": "object",
                "properties": {"count": {"type": "string", "description": "Zero-padded 4-digit count"}},
                "required": ["count"]
            }),
            |args| {
                let count = args.get("count").and_then(Value::as_str).unwrap_or("");
                if count.len() == 4 && count.chars().all(|c| c.is_ascii_digit()) {
                    Ok(json!(format!("Fetched {} records.", count.parse::<i64>().unwrap_or(0))))
                } else {
                    Err(ToolError::Execution(format!(
                        "count must be a zero-padded 4-digit string, got '{count}'"
                    )))
                }
            },
        )?,
    );
    tools.insert(
        "summarize".to_string(),
        terminal_tool(
            "summarize",
            "Summarize the fetched content.",
            json!({
                "type": "object",
                "properties": {"content": {"type": "string", "description": "The content to summarize"}},
                "required": ["content"]
            }),
            "content",
            capture.clone(),
        )?,
    );
    let required = if use_required_steps {
        vec!["fetch".to_string()]
    } else {
        Vec::new()
    };
    let workflow = Workflow::new(
        "error_recovery",
        "Fetch with validation, then summarize",
        tools,
        required,
        "summarize".to_string().into(),
        "You are a helpful assistant. Fetch the requested records, then summarize them.",
    )?;
    Ok(SmokeScenario {
        name: "error_recovery".to_string(),
        workflow,
        user_message: "Fetch 10 records and summarize them.".to_string(),
        capture,
    })
}

fn make_tool<F>(name: &str, description: &str, schema: Value, func: F) -> Result<ToolDef, String>
where
    F: Fn(IndexMap<String, Value>) -> Result<Value, ToolError> + Send + Sync + 'static,
{
    let spec = ToolSpec::from_json_schema(name, description, &schema)?;
    let func = Arc::new(func);
    let callable: ToolCallable = Arc::new(move |args| {
        let func = func.clone();
        Box::pin(async move { func(args) })
    });
    Ok(ToolDef::new(spec, callable))
}

fn terminal_tool(
    name: &str,
    description: &str,
    schema: Value,
    output_key: &'static str,
    capture: Arc<StdMutex<Option<IndexMap<String, Value>>>>,
) -> Result<ToolDef, String> {
    make_tool(name, description, schema, move |args| {
        *capture.lock().expect("capture lock") = Some(args.clone());
        Ok(args.get(output_key).cloned().unwrap_or_else(|| json!("")))
    })
}

fn terminal_text(args: &IndexMap<String, Value>) -> String {
    ["message", "content", "findings"]
        .iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

fn validate_scenario(name: &str, text: &str) -> bool {
    let normalized = text.to_lowercase().replace(',', "");
    match name {
        "basic_2step" => normalized.contains("paris") && normalized.contains("capital"),
        "sequential_3step" => {
            normalized.contains("23")
                && normalized.contains("widget pro")
                && normalized.contains("apac")
        }
        "error_recovery" => normalized.contains("10") && normalized.contains("record"),
        _ => false,
    }
}

#[derive(Default)]
struct MessageStats {
    retry_nudges: usize,
    step_nudges: usize,
    tool_errors: usize,
    reasoning_msgs: usize,
}

fn message_stats(messages: &[Message]) -> MessageStats {
    let mut stats = MessageStats::default();
    for message in messages {
        match message.metadata.msg_type {
            MessageType::RetryNudge => stats.retry_nudges += 1,
            MessageType::StepNudge => stats.step_nudges += 1,
            MessageType::ToolResult if message.content.contains("[ToolError]") => {
                stats.tool_errors += 1
            }
            MessageType::Reasoning => stats.reasoning_msgs += 1,
            _ => {}
        }
    }
    stats
}

fn tool_trace(messages: &[Message]) -> (Vec<Value>, Vec<Value>) {
    let mut names = Vec::new();
    let mut args = Vec::new();
    for message in messages {
        if message.metadata.msg_type != MessageType::ToolCall {
            continue;
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                names.push(json!(call.name));
                args.push(json!(call.args));
            }
        }
    }
    (names, args)
}

fn error_kind(err: &ForgeError) -> &'static str {
    match err {
        ForgeError::UnsupportedModel(_) => "UnsupportedModelError",
        ForgeError::ToolCall(_) => "ToolCallError",
        ForgeError::ToolExecution(_) => "ToolExecutionError",
        ForgeError::WorkflowCancelled(_) => "WorkflowCancelledError",
        ForgeError::MaxIterations(_) => "MaxIterationsError",
        ForgeError::StepEnforcement(_) => "StepEnforcementError",
        ForgeError::Prerequisite(_) => "PrerequisiteError",
        ForgeError::ContextBudgetExceeded(_) => "ContextBudgetExceeded",
        ForgeError::HardwareDetection(_) => "HardwareDetectionError",
        ForgeError::ContextDiscovery(_) => "ContextDiscoveryError",
        ForgeError::BudgetResolution(_) => "BudgetResolutionError",
        ForgeError::Backend(_) => "BackendError",
        ForgeError::Stream(_) => "StreamError",
    }
}

fn parse_ablation(name: &str) -> Result<Ablation, String> {
    let ablation = match name {
        "reforged" => Ablation {
            rescue_enabled: true,
            max_retries: 5,
            use_required_steps: true,
        },
        "no_rescue" => Ablation {
            rescue_enabled: false,
            max_retries: 5,
            use_required_steps: true,
        },
        "no_steps" => Ablation {
            rescue_enabled: true,
            max_retries: 5,
            use_required_steps: false,
        },
        "no_recovery" | "no_nudge" => Ablation {
            rescue_enabled: false,
            max_retries: 0,
            use_required_steps: true,
        },
        "bare" => Ablation {
            rescue_enabled: false,
            max_retries: 0,
            use_required_steps: false,
        },
        "no_compact" => Ablation {
            rescue_enabled: true,
            max_retries: 5,
            use_required_steps: true,
        },
        other => return Err(format!("unsupported ablation: {other}")),
    };
    Ok(ablation)
}

fn default_mode(backend: &str) -> &'static str {
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

fn parse_args<I>(args: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut cli = Cli {
        backend: "openai-proxy".to_string(),
        model: None,
        gguf: None,
        base_url: None,
        runs: 1,
        scenarios: Vec::new(),
        stream: false,
        output: None,
        ablation: "reforged".to_string(),
        mode: None,
        reasoning_budget: None,
        anthropic_api_key: None,
    };

    let values: Vec<String> = args.into_iter().collect();
    let mut index = 0;
    while index < values.len() {
        match values[index].as_str() {
            "--help" | "-h" => return Err("__help__".to_string()),
            "--backend" => cli.backend = take_one(&values, &mut index, "--backend")?,
            "--model" => cli.model = Some(take_one(&values, &mut index, "--model")?),
            "--gguf" => cli.gguf = Some(take_one(&values, &mut index, "--gguf")?),
            "--base-url" => cli.base_url = Some(take_one(&values, &mut index, "--base-url")?),
            "--runs" => {
                let raw = take_one(&values, &mut index, "--runs")?;
                cli.runs = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --runs value: {raw}"))?;
            }
            "--scenario" => {
                cli.scenarios = take_many(&values, &mut index, "--scenario")?;
            }
            "--stream" => cli.stream = true,
            "--output" => cli.output = Some(take_one(&values, &mut index, "--output")?),
            "--ablation" => cli.ablation = take_one(&values, &mut index, "--ablation")?,
            "--mode" | "--llamafile-mode" => {
                let flag = values[index].clone();
                cli.mode = Some(take_one(&values, &mut index, &flag)?)
            }
            "--reasoning-budget" => {
                cli.reasoning_budget = Some(take_one(&values, &mut index, "--reasoning-budget")?)
            }
            "--anthropic-api-key" => {
                cli.anthropic_api_key = Some(take_one(&values, &mut index, "--anthropic-api-key")?)
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag: {flag}")),
            value => return Err(format!("unexpected argument: {value}")),
        }
        index += 1;
    }

    if cli.runs == 0 {
        return Err("--runs must be at least 1".to_string());
    }
    parse_ablation(&cli.ablation)?;
    Ok(cli)
}

fn take_one(values: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    values
        .get(*index)
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn take_many(values: &[String], index: &mut usize, flag: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    *index += 1;
    while *index < values.len() {
        let value = &values[*index];
        if value.starts_with("--") {
            *index -= 1;
            break;
        }
        out.push(value.clone());
        *index += 1;
    }
    if out.is_empty() {
        Err(format!("{flag} requires at least one value"))
    } else {
        Ok(out)
    }
}

fn print_help() {
    println!(
        "forge-eval\n\n\
         Usage: forge-eval --backend openai-proxy --base-url URL --model MODEL [options]\n\n\
         Options:\n\
           --backend openai-proxy|ollama|llamaserver|llamafile|anthropic\n\
           --model MODEL\n\
           --gguf PATH\n\
           --base-url URL\n\
           --runs N\n\
           --scenario NAME [NAME ...]\n\
           --stream\n\
           --ablation reforged|no_rescue|no_nudge|no_steps|no_recovery|no_compact|bare\n\
           --output PATH\n\
           --mode native|prompt|auto\n\
           --reasoning-budget TOKENS\n\
           --anthropic-api-key KEY"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(items: &[&str]) -> Cli {
        parse_args(items.iter().map(|item| item.to_string())).expect("parse")
    }

    #[test]
    fn parses_multiple_scenarios() {
        let cli = parse(&[
            "--backend",
            "openai-proxy",
            "--model",
            "test-model",
            "--scenario",
            "basic_2step",
            "sequential_3step",
            "--stream",
        ]);
        assert_eq!(
            cli.scenarios,
            vec!["basic_2step".to_string(), "sequential_3step".to_string()]
        );
        assert!(cli.stream);
    }

    #[test]
    fn rejects_zero_runs() {
        let err = parse_args(["--runs".to_string(), "0".to_string()]).unwrap_err();
        assert!(err.contains("at least 1"));
    }

    #[test]
    fn backend_default_modes_match_contract() {
        assert_eq!(default_mode("llamaserver"), "native");
        assert_eq!(default_mode("llamafile"), "prompt");
        assert_eq!(default_mode("openai-proxy"), "proxy");
    }

    #[test]
    fn builds_plumbing_scenario() {
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        assert_eq!(scenario.workflow.required_steps, vec!["get_country_info"]);
        assert!(scenario.workflow.terminal_tools.contains("summarize"));
    }
}
