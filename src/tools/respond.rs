//! Synthetic respond tool for signalling completion.
//!
//! Provides a structured alternative to bare text responses,
//! particularly useful for small local models.

use crate::core::tool_spec::{ParamModel, ToolSpec};
use crate::core::workflow::ToolDef;
use crate::error::ToolResolutionError;
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
        description: "Respond to the user with a message. Use this when the user is chatting, \
             asking a question, when you need to ask a clarifying question before \
             proceeding, or when no other tool action is needed. Also use this \
             after completing the user's request to report the result."
            .to_string(),
        parameters: ParamModel::Object {
            description: None,
            required: true,
            properties: {
                let mut props = IndexMap::new();
                props.insert(
                    "message".to_string(),
                    ParamModel::String {
                        description: Some("The message to send to the user.".to_string()),
                        required: true,
                        default: None,
                        enum_values: None,
                    },
                );
                props
            },
        },
        json_schema: None,
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
    Err(
        ToolResolutionError::new("respond tool requires a message argument")
            .with_tool_name(RESPOND_TOOL_NAME),
    )
}

/// Return a complete ToolDef for the respond tool.
///
/// The spec matches respond_spec(). The callable is an identity function:
/// given a 'message' argument, it returns that argument unchanged.
pub fn respond_tool() -> ToolDef {
    ToolDef::new(
        respond_spec(),
        respond_callable as fn(Vec<String>) -> Result<String, ToolResolutionError>,
    )
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
                        assert_eq!(
                            description.as_deref(),
                            Some("The message to send to the user.")
                        );
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

    #[test]
    fn respond_callable_rejects_malformed_nonempty_args() {
        let err = respond_callable(vec!["hello world".to_string()]).unwrap_err();
        assert_eq!(err.tool_name.as_deref(), Some(RESPOND_TOOL_NAME));
        assert!(err.message.contains("message argument"));
    }
}
