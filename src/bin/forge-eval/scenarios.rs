use std::sync::{Arc, Mutex as StdMutex};

use forge_guardrails::error::ToolError;
use forge_guardrails::workflow::ToolCallable;
use forge_guardrails::{ToolDef, ToolSpec, Workflow};
use indexmap::IndexMap;
use serde_json::{json, Value};

pub(crate) struct SmokeScenario {
    pub(crate) name: String,
    pub(crate) workflow: Workflow,
    pub(crate) user_message: String,
    pub(crate) capture: Arc<StdMutex<Option<IndexMap<String, Value>>>>,
    pub(crate) corrected_positive: Option<Value>,
}

pub(crate) fn build_scenario(
    name: &str,
    use_required_steps: bool,
) -> Result<SmokeScenario, String> {
    match name {
        "basic_2step" => basic_2step(use_required_steps),
        "sequential_3step" => sequential_3step(use_required_steps),
        "error_recovery" => error_recovery(use_required_steps),
        "inconsistent_api_recovery_stateful" => {
            inconsistent_api_recovery_stateful(use_required_steps)
        }
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
        corrected_positive: Some(json!({"final_text": "The capital of France is Paris."})),
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
        corrected_positive: Some(
            json!({"final_text": "Q4 sales were 1200 units with 15% growth."}),
        ),
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
                    Ok(json!(format!(
                        "Fetched {} records.",
                        count.parse::<i64>().unwrap_or(0)
                    )))
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
        corrected_positive: Some(
            json!({"final_text": "Fetched 10 records and summarized the record count."}),
        ),
    })
}

fn inconsistent_api_recovery_stateful(use_required_steps: bool) -> Result<SmokeScenario, String> {
    let capture = Arc::new(StdMutex::new(None));
    let mut tools = IndexMap::new();
    tools.insert(
        "legacy_list_accounts".to_string(),
        make_tool(
            "legacy_list_accounts",
            "List available accounts in the legacy audit system.",
            json!({"type": "object", "properties": {}}),
            |_args| Ok(json!("Accounts: ACC-12345 Acme Corp")),
        )?,
    );
    tools.insert(
        "legacy_submit_audit".to_string(),
        terminal_tool(
            "legacy_submit_audit",
            "Submit the final compliance audit report.",
            json!({
                "type": "object",
                "properties": {
                    "report": {"type": "string", "description": "Final audit report"}
                },
                "required": ["report"]
            }),
            "report",
            capture.clone(),
        )?,
    );
    let required = if use_required_steps {
        vec!["legacy_list_accounts".to_string()]
    } else {
        Vec::new()
    };
    let workflow = Workflow::new(
        "inconsistent_api_recovery_stateful",
        "Stateful legacy audit smoke scenario",
        tools,
        required,
        "legacy_submit_audit".to_string().into(),
        "You are a compliance audit assistant. List accounts before submitting the final audit.",
    )?;
    Ok(SmokeScenario {
        name: "inconsistent_api_recovery_stateful".to_string(),
        workflow,
        user_message: concat!(
            "Run a compliance audit for Acme Corp. Include account ACC-12345 and ",
            "compliance_status PASS in the submitted report."
        )
        .to_string(),
        capture,
        corrected_positive: Some(json!({
            "account_id": "ACC-12345",
            "compliance_status": "PASS"
        })),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_plumbing_scenario() {
        let scenario = build_scenario("basic_2step", true).expect("scenario");
        assert_eq!(scenario.workflow.required_steps, vec!["get_country_info"]);
        assert!(scenario.workflow.terminal_tools.contains("summarize"));
    }

    #[test]
    fn builds_inconsistent_api_recovery_stateful_scenario() {
        let scenario =
            build_scenario("inconsistent_api_recovery_stateful", true).expect("scenario");
        assert_eq!(
            scenario.workflow.required_steps,
            vec!["legacy_list_accounts"]
        );
        assert!(scenario
            .workflow
            .terminal_tools
            .contains("legacy_submit_audit"));
    }
}
