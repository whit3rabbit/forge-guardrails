use super::HandlerError;
use crate::clients::base::LLMRequestOptions;
use crate::core::tool_spec::ToolSpec;
use crate::tools::respond::{respond_spec, RESPOND_TOOL_NAME};
use indexmap::IndexSet;
use serde_json::{Map, Value};

pub(super) const FORGE_EXTENSION_FIELD: &str = "_forge";
pub(super) const FORGE_REQUIRED_STEPS_FIELD: &str = "required_steps";
pub(super) const FORGE_TERMINAL_TOOLS_FIELD: &str = "terminal_tools";
pub(super) const FORGE_RETURN_RAW_ON_GUARDRAIL_FAILURE_FIELD: &str =
    "return_raw_on_guardrail_failure";

#[derive(Debug, Clone)]
pub(super) struct ProxyStepContract {
    pub(super) required_steps: Vec<String>,
    pub(super) terminal_tools: Vec<String>,
}

pub(super) fn extract_proxy_step_contract(
    body: &Value,
) -> Result<Option<ProxyStepContract>, HandlerError> {
    let Some(forge_obj) = forge_object(body)? else {
        return Ok(None);
    };

    let required_steps =
        parse_forge_string_array_field(forge_obj, FORGE_REQUIRED_STEPS_FIELD)?.unwrap_or_default();
    let terminal_tools = parse_forge_string_array_field(forge_obj, FORGE_TERMINAL_TOOLS_FIELD)?
        .unwrap_or_else(|| vec![RESPOND_TOOL_NAME.to_string()]);

    Ok(Some(ProxyStepContract {
        required_steps,
        terminal_tools,
    }))
}

fn parse_forge_string_array_field(
    forge_obj: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<String>>, HandlerError> {
    let Some(value) = forge_obj.get(field) else {
        return Ok(None);
    };
    let Some(items) = value.as_array() else {
        return Err(HandlerError::BadRequest(format!(
            "{FORGE_EXTENSION_FIELD}.{field} must be an array of strings"
        )));
    };
    let mut strings = Vec::with_capacity(items.len());
    for item in items {
        let Some(s) = item.as_str() else {
            return Err(HandlerError::BadRequest(format!(
                "{FORGE_EXTENSION_FIELD}.{field} must be an array of strings"
            )));
        };
        strings.push(s.to_string());
    }
    Ok(Some(strings))
}

pub(super) fn add_proxy_respond_tool_if_needed(
    tool_specs: &mut Vec<ToolSpec>,
    contract: Option<&ProxyStepContract>,
) -> bool {
    let has_real_terminal = contract.is_some_and(|contract| {
        contract
            .terminal_tools
            .iter()
            .any(|tool| tool != RESPOND_TOOL_NAME)
    });
    if has_real_terminal {
        return false;
    }
    tool_specs.push(respond_spec());
    true
}

fn normalize_proxy_terminal_tools(
    terminal_tools: Vec<String>,
    respond_injected: bool,
) -> Result<Vec<String>, HandlerError> {
    let mut terminal_tools = if terminal_tools.is_empty() {
        vec![RESPOND_TOOL_NAME.to_string()]
    } else {
        terminal_tools
    };
    if !respond_injected {
        terminal_tools.retain(|tool| tool != RESPOND_TOOL_NAME);
    }
    if terminal_tools.is_empty() {
        return Err(HandlerError::BadRequest(format!(
            "{FORGE_EXTENSION_FIELD}.{FORGE_TERMINAL_TOOLS_FIELD} has no available terminal tools"
        )));
    }
    Ok(terminal_tools)
}

pub(super) fn validate_proxy_step_contract(
    contract: Option<ProxyStepContract>,
    tool_names: &IndexSet<String>,
    respond_injected: bool,
) -> Result<Option<ProxyStepContract>, HandlerError> {
    let Some(contract) = contract else {
        return Ok(None);
    };

    validate_proxy_name_list(FORGE_REQUIRED_STEPS_FIELD, &contract.required_steps)?;
    for step in &contract.required_steps {
        if !tool_names.contains(step.as_str()) {
            return Err(HandlerError::BadRequest(format!(
                "{FORGE_EXTENSION_FIELD}.{FORGE_REQUIRED_STEPS_FIELD} contains unknown tool '{step}'"
            )));
        }
    }

    let terminal_tools = normalize_proxy_terminal_tools(contract.terminal_tools, respond_injected)?;
    validate_proxy_name_list(FORGE_TERMINAL_TOOLS_FIELD, &terminal_tools)?;

    let required_set: IndexSet<&str> = contract.required_steps.iter().map(String::as_str).collect();
    for terminal in &terminal_tools {
        if !tool_names.contains(terminal.as_str()) {
            return Err(HandlerError::BadRequest(format!(
                "{FORGE_EXTENSION_FIELD}.{FORGE_TERMINAL_TOOLS_FIELD} contains unknown tool '{terminal}'"
            )));
        }
        if required_set.contains(terminal.as_str()) {
            return Err(HandlerError::BadRequest(format!(
                "{FORGE_EXTENSION_FIELD}.{FORGE_TERMINAL_TOOLS_FIELD} contains required step '{terminal}'"
            )));
        }
    }

    Ok(Some(ProxyStepContract {
        required_steps: contract.required_steps,
        terminal_tools,
    }))
}

fn validate_proxy_name_list(field: &str, values: &[String]) -> Result<(), HandlerError> {
    let mut seen = IndexSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(HandlerError::BadRequest(format!(
                "{FORGE_EXTENSION_FIELD}.{field} contains an empty tool name"
            )));
        }
        if !seen.insert(value.as_str()) {
            return Err(HandlerError::BadRequest(format!(
                "{FORGE_EXTENSION_FIELD}.{field} contains duplicate tool '{value}'"
            )));
        }
    }
    Ok(())
}

fn forge_object(body: &Value) -> Result<Option<&serde_json::Map<String, Value>>, HandlerError> {
    let Some(forge) = body.get(FORGE_EXTENSION_FIELD) else {
        return Ok(None);
    };
    forge.as_object().map(Some).ok_or_else(|| {
        HandlerError::BadRequest(format!("{FORGE_EXTENSION_FIELD} must be an object"))
    })
}

pub(super) fn extract_forge_bool_field(body: &Value, field: &str) -> Result<bool, HandlerError> {
    let Some(forge_obj) = forge_object(body)? else {
        return Ok(false);
    };
    let Some(value) = forge_obj.get(field) else {
        return Ok(false);
    };
    value.as_bool().ok_or_else(|| {
        HandlerError::BadRequest(format!("{FORGE_EXTENSION_FIELD}.{field} must be a boolean"))
    })
}

pub(super) fn extract_stream_include_usage(body: &Value) -> Result<bool, HandlerError> {
    let Some(stream_options) = body.get("stream_options") else {
        return Ok(false);
    };
    let Some(stream_options_obj) = stream_options.as_object() else {
        return Err(HandlerError::BadRequest(
            "stream_options must be an object".to_string(),
        ));
    };
    let Some(include_usage) = stream_options_obj.get("include_usage") else {
        return Ok(false);
    };
    include_usage.as_bool().ok_or_else(|| {
        HandlerError::BadRequest("stream_options.include_usage must be a boolean".to_string())
    })
}

pub(super) fn sanitize_guarded_request_options(
    mut options: LLMRequestOptions,
    step_contract: Option<&ProxyStepContract>,
) -> Result<LLMRequestOptions, HandlerError> {
    let has_required_steps =
        step_contract.is_some_and(|contract| !contract.required_steps.is_empty());
    options.passthrough = sanitize_guarded_passthrough(options.passthrough, has_required_steps)?;
    options.inbound_anthropic_body =
        sanitize_guarded_anthropic_body(options.inbound_anthropic_body, has_required_steps)?;
    Ok(options)
}

fn sanitize_guarded_passthrough(
    passthrough: Option<Map<String, Value>>,
    has_required_steps: bool,
) -> Result<Option<Map<String, Value>>, HandlerError> {
    let Some(mut passthrough) = passthrough else {
        return Ok(None);
    };
    passthrough.remove("response_format");
    if let Some(tool_choice) = passthrough.get("tool_choice") {
        validate_guarded_openai_tool_choice(tool_choice, has_required_steps)?;
    }
    Ok(if passthrough.is_empty() {
        None
    } else {
        Some(passthrough)
    })
}

pub(super) fn sanitize_guarded_anthropic_body(
    body: Option<Value>,
    has_required_steps: bool,
) -> Result<Option<Value>, HandlerError> {
    let Some(mut body) = body else {
        return Ok(None);
    };
    if let Some(obj) = body.as_object_mut() {
        obj.remove("response_format");
        if let Some(tool_choice) = obj.get("tool_choice") {
            validate_guarded_anthropic_tool_choice(tool_choice, has_required_steps)?;
        }
    }
    Ok(Some(body))
}

fn validate_guarded_openai_tool_choice(
    value: &Value,
    has_required_steps: bool,
) -> Result<(), HandlerError> {
    match value {
        Value::String(choice) if choice == "none" => Err(HandlerError::BadRequest(
            "tool_choice=none is incompatible with guarded tool execution".to_string(),
        )),
        Value::Object(_) if has_required_steps => Err(HandlerError::BadRequest(
            "forced tool_choice is incompatible with _forge.required_steps".to_string(),
        )),
        _ => Ok(()),
    }
}

fn validate_guarded_anthropic_tool_choice(
    value: &Value,
    has_required_steps: bool,
) -> Result<(), HandlerError> {
    let choice_type = match value {
        Value::Object(obj) => obj.get("type").and_then(Value::as_str),
        Value::String(choice) => Some(choice.as_str()),
        _ => None,
    };
    match choice_type {
        Some("none") => Err(HandlerError::BadRequest(
            "tool_choice=none is incompatible with guarded tool execution".to_string(),
        )),
        Some("tool") if has_required_steps => Err(HandlerError::BadRequest(
            "forced tool_choice is incompatible with _forge.required_steps".to_string(),
        )),
        _ => Ok(()),
    }
}
