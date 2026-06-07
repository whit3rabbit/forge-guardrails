use serde_json::Value;

pub(crate) fn post_review_quality_reject(
    capture: &Value,
    candidate_call: &Value,
    label: &str,
    source_bucket: &str,
) -> Option<String> {
    if label == "valid" && repeats_completed_non_terminal_tool(capture, candidate_call) {
        return Some(format!(
            "{source_bucket} proposed valid for an already-completed non-terminal tool with no pending workflow steps"
        ));
    }

    if label == "wrong_tool_semantic" && looks_like_wrong_tool_overreach(capture, candidate_call) {
        return Some(
            "candidate uses a semantically matching tool for the request; if arguments are bad, this should be wrong_arguments_semantic, not wrong_tool_semantic"
                .to_string(),
        );
    }

    None
}

pub(crate) fn repeats_completed_non_terminal_tool(capture: &Value, candidate_call: &Value) -> bool {
    let Some(name) = candidate_call.get("name").and_then(Value::as_str) else {
        return false;
    };
    if name == "respond" {
        return false;
    }
    let Some(workflow_state) = capture.get("workflow_state") else {
        return false;
    };
    let pending_empty = workflow_state
        .get("pending_steps")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty);
    if !pending_empty {
        return false;
    }
    workflow_state
        .get("completed_steps")
        .and_then(Value::as_array)
        .is_some_and(|completed| completed.iter().any(|step| step.as_str() == Some(name)))
}

pub(crate) fn looks_like_wrong_tool_overreach(capture: &Value, candidate_call: &Value) -> bool {
    let Some(name) = candidate_call.get("name").and_then(Value::as_str) else {
        return false;
    };
    let request = capture
        .get("user_request")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    if request.contains("compare") && name == "compare_products" {
        return true;
    }
    if request.contains("list") && name == "list_files" {
        return true;
    }
    if request.contains("look up") && name == "lookup_ticket" {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn capture_row() -> Value {
        json!({
            "schema_version": "forge-dataset-capture/v1",
            "example_group_id": "group-1",
            "user_request": "Compare two products.",
            "workflow_state": {
                "required_steps": [],
                "completed_steps": [],
                "pending_steps": [],
                "terminal_tools": ["respond"],
                "recent_errors": []
            },
            "available_tools": [
                {
                    "name": "compare_products",
                    "description": "Compare products.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_ids": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["product_ids"]
                    }
                },
                {
                    "name": "add_to_cart",
                    "description": "Add a product to cart.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "integer"}
                        },
                        "required": ["product_id", "quantity"]
                    }
                }
            ],
            "candidate_call": {
                "name": "compare_products",
                "arguments": {"product_ids": ["SKU-1", "SKU-2"]}
            },
            "tool_result": {"status": "ok", "content": {}},
            "proxy_trace": {"domain": "shopping", "scenario": "compare_headphones"}
        })
    }

    #[test]
    fn quality_filter_rejects_completed_tool_valid_positive() {
        let mut capture = capture_row();
        capture["workflow_state"]["completed_steps"] = json!(["compare_products"]);
        capture["workflow_state"]["pending_steps"] = json!([]);

        let reason = post_review_quality_reject(
            &capture,
            &capture["candidate_call"],
            "valid",
            "real_model_call",
        )
        .expect("quality rejection");

        assert!(reason.contains("already-completed"));
    }

    #[test]
    fn quality_filter_rejects_wrong_tool_for_semantically_matching_compare_tool() {
        let capture = capture_row();

        let reason = post_review_quality_reject(
            &capture,
            &capture["candidate_call"],
            "wrong_tool_semantic",
            "real_model_call",
        )
        .expect("quality rejection");

        assert!(reason.contains("wrong_arguments_semantic"));
    }

    #[test]
    fn quality_filter_allows_real_competing_wrong_tool() {
        let capture = capture_row();
        let candidate = json!({
            "name": "add_to_cart",
            "arguments": {"product_id": "SKU-1", "quantity": 1}
        });

        assert!(post_review_quality_reject(
            &capture,
            &candidate,
            "wrong_tool_semantic",
            "targeted_alternative",
        )
        .is_none());
    }
}
