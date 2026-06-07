use serde_json::{json, Value};

use super::types::AlternativeProposal;
use crate::schema::{mutated_arguments_for_tool, tool_by_name, validate_candidate_call};

pub(crate) fn propose_targeted_alternatives(capture: &Value) -> Vec<AlternativeProposal> {
    let mut proposals = Vec::new();
    let available_tools = &capture["available_tools"];
    let original = &capture["candidate_call"];
    let Some(current_name) = original.get("name").and_then(Value::as_str) else {
        return proposals;
    };
    let example_group_id = capture
        .get("example_group_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-group")
        .to_string();

    if let Some(tool) = tool_by_name(available_tools, current_name) {
        if let Some(mutated_args) =
            mutated_arguments_for_tool(tool, original.get("arguments").unwrap_or(&Value::Null))
        {
            let candidate_call = json!({"name": current_name, "arguments": mutated_args});
            if validate_candidate_call(available_tools, &candidate_call).is_ok()
                && candidate_call != *original
            {
                proposals.push(AlternativeProposal {
                    capture: capture.clone(),
                    example_group_id: example_group_id.clone(),
                    candidate_call,
                    label: "wrong_arguments_semantic".to_string(),
                });
            }
        }
    }

    if let Some(candidate_call) = curated_wrong_tool_candidate(capture, current_name) {
        if validate_candidate_call(available_tools, &candidate_call).is_ok() {
            proposals.push(AlternativeProposal {
                capture: capture.clone(),
                example_group_id,
                candidate_call,
                label: "wrong_tool_semantic".to_string(),
            });
        }
    }
    proposals
}

pub(crate) fn curated_wrong_tool_candidate(capture: &Value, current_name: &str) -> Option<Value> {
    let domain = capture
        .get("proxy_trace")
        .and_then(|trace| trace.get("domain"))
        .and_then(Value::as_str)?;
    let scenario = capture
        .get("proxy_trace")
        .and_then(|trace| trace.get("scenario"))
        .and_then(Value::as_str)?;

    let candidate = match (domain, scenario) {
        ("shopping", "compare_headphones") => {
            json!({"name": "add_to_cart", "arguments": {"product_id": "SKU-HEADPHONES-1", "quantity": 1}})
        }
        ("shopping", "add_dock") => {
            json!({"name": "compare_products", "arguments": {"product_ids": ["SKU-DOCK-1", "SKU-HEADPHONES-1"]}})
        }
        ("shopping", "product_details") => {
            json!({"name": "add_to_cart", "arguments": {"product_id": "SKU-HEADPHONES-1", "quantity": 1}})
        }
        ("calendar", "find_and_hold") => {
            json!({"name": "list_events", "arguments": {"date": "2026-06-05"}})
        }
        ("calendar", "list_events") => {
            json!({"name": "create_calendar_hold", "arguments": {"slot_id": "slot-001"}})
        }
        ("calendar", "cancel_hold") => {
            json!({"name": "list_events", "arguments": {"date": "2026-06-05"}})
        }
        ("support", "lookup_ticket") => {
            json!({"name": "update_ticket", "arguments": {"ticket_id": "TCK-1001", "note": "Reviewed by dataset stub."}})
        }
        ("support", "search_kb") => {
            json!({"name": "lookup_ticket", "arguments": {"ticket_id": "TCK-1001"}})
        }
        ("support", "escalate_ticket") => {
            json!({"name": "update_ticket", "arguments": {"ticket_id": "TCK-2002", "note": "Customer is blocked from billing."}})
        }
        ("forge_eval", "smoke_eval") => {
            json!({"name": "run_release_eval", "arguments": {"scenario": "basic_2step", "runs": 1}})
        }
        ("forge_eval", "release_eval") => {
            json!({"name": "run_smoke_eval", "arguments": {"scenario": "basic_2step", "runs": 1}})
        }
        ("forge_eval", "diagnose_failure") => {
            json!({"name": "report_result", "arguments": {"summary": "Failure diagnosed."}})
        }
        ("forge_eval", "inspect_workflow_state") => {
            json!({"name": "report_result", "arguments": {"summary": "Workflow state checked."}})
        }
        ("forge_eval", "fetch_zero_padded") => {
            json!({"name": "summarize_records", "arguments": {"content": "Fetched 0010 records."}})
        }
        _ => return None,
    };

    if candidate.get("name").and_then(Value::as_str) == Some(current_name) {
        None
    } else {
        Some(candidate)
    }
}

pub(crate) fn alternative_cap(real_row_count: usize, ratio: f64) -> usize {
    ((real_row_count as f64 * ratio) + 1e-9).floor() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn proposes_schema_valid_targeted_alternatives() {
        let capture = capture_row();
        let proposals = propose_targeted_alternatives(&capture);
        assert!(proposals
            .iter()
            .any(|proposal| proposal.label == "wrong_arguments_semantic"));
        assert!(proposals
            .iter()
            .any(|proposal| proposal.label == "wrong_tool_semantic"));
        assert!(proposals.iter().any(|proposal| {
            proposal.label == "wrong_tool_semantic"
                && proposal.candidate_call["name"] == "add_to_cart"
        }));
        for proposal in proposals {
            validate_candidate_call(&capture["available_tools"], &proposal.candidate_call)
                .expect("schema-valid proposal");
        }
    }

    #[test]
    fn alternative_cap_keeps_real_rows_as_backbone() {
        assert_eq!(alternative_cap(0, 1.0 / 3.0), 0);
        assert_eq!(alternative_cap(2, 1.0 / 3.0), 0);
        assert_eq!(alternative_cap(3, 1.0 / 3.0), 1);
        assert_eq!(alternative_cap(6, 1.0 / 3.0), 2);
    }
}
