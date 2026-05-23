//! Synthetic respond tool for signalling completion.
//!
//! Provides a structured alternative to bare text responses,
//! particularly useful for small local models.

use crate::error::ToolResolutionError;
use crate::tool_spec::{ParamModel, ToolSpec};
use crate::workflow::ToolDef;
use indexmap::IndexMap;

/// Protocol-level constant for the respond tool name.
/// Must match exactly for tool resolution, proxy injection, and respond-call stripping.
pub const RESPOND_TOOL_NAME: &str = "respond";

/// Return the ToolSpec for the respond tool.
///
/// Description text is injected verbatim into LLM prompts and must
/// be preserved for prompt compatibility.
pub fn respond_spec() -> ToolSpec {
    ToolSpec {
        name: RESPOND_TOOL_NAME.to_string(),
        description: "Send a message to the user. Use this tool to chat with the user, \
             ask questions, clarify their request, or report the final result."
            .to_string(),
        parameters: ParamModel::Object {
            description: None,
            required: true,
            properties: {
                let mut props = IndexMap::new();
                props.insert(
                    "message".to_string(),
                    ParamModel::String {
                        description: Some("The message to send to the user".to_string()),
                        required: true,
                        default: None,
                        enum_values: None,
                    },
                );
                props
            },
        },
    }
}

/// Identity callable: returns the 'message' argument unchanged.
fn respond_callable(args: Vec<String>) -> Result<String, ToolResolutionError> {
    // The caller serializes named args as key=value strings.
    // For the respond tool, the convention is a single "message=<value>" arg.
    // If args is empty, return empty string.
    if args.is_empty() {
        return Ok(String::new());
    }
    // Extract the value from "message=<value>" format
    for arg in &args {
        if let Some(value) = arg.strip_prefix("message=") {
            return Ok(value.to_string());
        }
    }
    // Fallback: return the first arg as-is (handles cases where
    // the calling convention passes just the value)
    Ok(args[0].clone())
}

/// Return a complete ToolDef for the respond tool.
///
/// The spec matches respond_spec(). The callable is an identity function:
/// given a 'message' argument, it returns that argument unchanged.
pub fn respond_tool() -> ToolDef {
    ToolDef::new(respond_spec(), respond_callable as _)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respond_tool_name_is_respond() {
        assert_eq!(RESPOND_TOOL_NAME, "respond");
    }

    #[test]
    fn respond_spec_has_correct_name() {
        let spec = respond_spec();
        assert_eq!(spec.name, "respond");
    }

    #[test]
    fn respond_spec_has_description() {
        let spec = respond_spec();
        assert!(!spec.description.is_empty());
    }

    #[test]
    fn respond_spec_has_message_param() {
        let spec = respond_spec();
        match &spec.parameters {
            ParamModel::Object { properties, .. } => {
                assert!(properties.contains_key("message"));
                let msg_param = &properties["message"];
                match msg_param {
                    ParamModel::String {
                        required,
                        description,
                        ..
                    } => {
                        assert!(*required);
                        assert!(description.is_some());
                    }
                    _ => panic!("message param should be String type"),
                }
            }
            _ => panic!("parameters should be Object type"),
        }
    }

    #[test]
    fn respond_tool_spec_matches_respond_spec() {
        let tool = respond_tool();
        let spec = respond_spec();
        assert_eq!(tool.spec.name, spec.name);
        assert_eq!(tool.spec.description, spec.description);
    }

    #[test]
    fn respond_callable_returns_message() {
        let result = respond_callable(vec!["message=hello world".to_string()]);
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn respond_callable_empty_returns_empty() {
        let result = respond_callable(vec![]);
        assert_eq!(result.unwrap(), "");
    }
}
