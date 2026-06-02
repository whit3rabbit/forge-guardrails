use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
pub(crate) struct StubRegistry {
    pub(crate) domain: &'static str,
    pub(crate) tools: Vec<StubTool>,
    pub(crate) scenarios: Vec<StubScenario>,
}

#[derive(Debug, Clone)]
pub(crate) struct StubTool {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) parameters: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct StubScenario {
    pub(crate) id: &'static str,
    pub(crate) user_request: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolExecution {
    pub(crate) status: ToolExecutionStatus,
    pub(crate) content: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolExecutionStatus {
    Ok,
    Error,
}

impl ToolExecutionStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
        }
    }
}

pub(crate) fn registries_for_domains(domains: &[String]) -> Result<Vec<StubRegistry>, String> {
    let mut registries = Vec::new();
    for domain in domains {
        let registry = match domain.as_str() {
            "repo_docs" => repo_docs_registry(),
            "shopping" => shopping_registry(),
            "calendar" => calendar_registry(),
            "support" => support_registry(),
            "forge_eval" => forge_eval_registry(),
            other => return Err(format!("unsupported dataset domain: {other}")),
        };
        registries.push(registry);
    }
    Ok(registries)
}

pub(crate) fn request_tools(registry: &StubRegistry) -> Vec<Value> {
    registry
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect()
}

pub(crate) fn available_tools_for_training(registry: &StubRegistry) -> Vec<Value> {
    let mut tools = registry
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect::<Vec<_>>();
    let respond = forge_guardrails::respond::respond_spec();
    tools.push(json!({
        "name": respond.name,
        "description": respond.description,
        "parameters": respond.get_json_schema(),
    }));
    tools
}

pub(crate) fn execute_tool(
    registry: &StubRegistry,
    name: &str,
    arguments: &Value,
) -> ToolExecution {
    let args = arguments.as_object().cloned().unwrap_or_default();
    match registry.domain {
        "repo_docs" => execute_repo_docs(name, &args),
        "shopping" => execute_shopping(name, &args),
        "calendar" => execute_calendar(name, &args),
        "support" => execute_support(name, &args),
        "forge_eval" => execute_forge_eval(name, &args),
        _ => error(format!("unknown registry '{}'", registry.domain)),
    }
}

pub(crate) fn content_as_message(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn repo_docs_registry() -> StubRegistry {
    StubRegistry {
        domain: "repo_docs",
        tools: vec![
            StubTool {
                name: "search_docs",
                description: "Search repository documentation by semantic query.",
                parameters: object_schema(json!({
                    "query": {"type": "string", "description": "Search query for docs"}
                }), vec!["query"]),
            },
            StubTool {
                name: "read_doc",
                description: "Read a repository documentation file by path.",
                parameters: object_schema(json!({
                    "path": {"type": "string", "description": "Repository-relative documentation path"}
                }), vec!["path"]),
            },
            StubTool {
                name: "list_files",
                description: "List repository files matching a glob.",
                parameters: object_schema(json!({
                    "glob": {"type": "string", "description": "Repository-relative glob pattern"}
                }), vec!["glob"]),
            },
            StubTool {
                name: "summarize_file",
                description: "Summarize a repository file by path.",
                parameters: object_schema(json!({
                    "path": {"type": "string", "description": "Repository-relative file path"}
                }), vec!["path"]),
            },
        ],
        scenarios: vec![
            StubScenario {
                id: "compression_docs",
                user_request: "Find the docs about proxy tool-output compression and summarize the relevant file.",
            },
            StubScenario {
                id: "classifier_logging",
                user_request: "Which repository docs mention classifier telemetry logging?",
            },
            StubScenario {
                id: "list_backend_docs",
                user_request: "List markdown files related to backend setup.",
            },
        ],
    }
}

fn shopping_registry() -> StubRegistry {
    StubRegistry {
        domain: "shopping",
        tools: vec![
            StubTool {
                name: "search_products",
                description: "Search the product catalog by natural-language query.",
                parameters: object_schema(
                    json!({
                        "query": {"type": "string", "description": "Product search query"}
                    }),
                    vec!["query"],
                ),
            },
            StubTool {
                name: "get_product",
                description: "Get details for one product by product_id.",
                parameters: object_schema(
                    json!({
                        "product_id": {"type": "string", "description": "Stable product identifier"}
                    }),
                    vec!["product_id"],
                ),
            },
            StubTool {
                name: "compare_products",
                description: "Compare two or more products by product_ids.",
                parameters: object_schema(
                    json!({
                        "product_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Product identifiers to compare"
                        }
                    }),
                    vec!["product_ids"],
                ),
            },
            StubTool {
                name: "add_to_cart",
                description: "Add a product to the shopping cart.",
                parameters: object_schema(
                    json!({
                        "product_id": {"type": "string", "description": "Product identifier"},
                        "quantity": {"type": "integer", "description": "Positive quantity to add"}
                    }),
                    vec!["product_id", "quantity"],
                ),
            },
        ],
        scenarios: vec![
            StubScenario {
                id: "compare_headphones",
                user_request: "Compare two good noise cancelling headphones under $200.",
            },
            StubScenario {
                id: "add_dock",
                user_request: "Add two units of SKU-DOCK-1 to my cart.",
            },
            StubScenario {
                id: "product_details",
                user_request: "Show me details for product SKU-HEADPHONES-1.",
            },
        ],
    }
}

fn calendar_registry() -> StubRegistry {
    StubRegistry {
        domain: "calendar",
        tools: vec![
            StubTool {
                name: "find_free_slots",
                description: "Find free calendar slots for a date and duration.",
                parameters: object_schema(
                    json!({
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                        "duration_minutes": {"type": "integer", "description": "Requested duration in minutes"}
                    }),
                    vec!["date", "duration_minutes"],
                ),
            },
            StubTool {
                name: "create_calendar_hold",
                description: "Create a tentative hold for a previously returned slot_id.",
                parameters: object_schema(
                    json!({
                        "slot_id": {"type": "string", "description": "Slot identifier returned by find_free_slots"}
                    }),
                    vec!["slot_id"],
                ),
            },
            StubTool {
                name: "list_events",
                description: "List existing events on a date.",
                parameters: object_schema(
                    json!({
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                    }),
                    vec!["date"],
                ),
            },
            StubTool {
                name: "cancel_hold",
                description: "Cancel a tentative calendar hold.",
                parameters: object_schema(
                    json!({
                        "hold_id": {"type": "string", "description": "Hold identifier to cancel"}
                    }),
                    vec!["hold_id"],
                ),
            },
        ],
        scenarios: vec![
            StubScenario {
                id: "find_and_hold",
                user_request:
                    "Find a free 30 minute slot on 2026-06-05 and create a calendar hold.",
            },
            StubScenario {
                id: "list_events",
                user_request: "List my calendar events on 2026-06-03.",
            },
            StubScenario {
                id: "cancel_hold",
                user_request: "Cancel hold hold-002.",
            },
        ],
    }
}

fn support_registry() -> StubRegistry {
    StubRegistry {
        domain: "support",
        tools: vec![
            StubTool {
                name: "lookup_ticket",
                description: "Look up a support ticket by ticket_id.",
                parameters: object_schema(
                    json!({
                        "ticket_id": {"type": "string", "description": "Support ticket identifier"}
                    }),
                    vec!["ticket_id"],
                ),
            },
            StubTool {
                name: "search_kb",
                description: "Search support knowledge base articles.",
                parameters: object_schema(
                    json!({
                        "query": {"type": "string", "description": "Knowledge-base search query"}
                    }),
                    vec!["query"],
                ),
            },
            StubTool {
                name: "update_ticket",
                description: "Append an internal note to a support ticket.",
                parameters: object_schema(
                    json!({
                        "ticket_id": {"type": "string", "description": "Support ticket identifier"},
                        "note": {"type": "string", "description": "Note to append"}
                    }),
                    vec!["ticket_id", "note"],
                ),
            },
            StubTool {
                name: "escalate_ticket",
                description: "Escalate a ticket with a concrete reason.",
                parameters: object_schema(
                    json!({
                        "ticket_id": {"type": "string", "description": "Support ticket identifier"},
                        "reason": {"type": "string", "description": "Escalation reason"}
                    }),
                    vec!["ticket_id", "reason"],
                ),
            },
        ],
        scenarios: vec![
            StubScenario {
                id: "lookup_ticket",
                user_request: "Look up support ticket TCK-1001 and tell me its status.",
            },
            StubScenario {
                id: "search_kb",
                user_request: "Find knowledge-base guidance for password reset failures.",
            },
            StubScenario {
                id: "escalate_ticket",
                user_request:
                    "Escalate ticket TCK-2002 because the customer is blocked from billing.",
            },
        ],
    }
}

fn forge_eval_registry() -> StubRegistry {
    StubRegistry {
        domain: "forge_eval",
        tools: vec![
            StubTool {
                name: "run_smoke_eval",
                description: "Run a small Forge smoke eval scenario.",
                parameters: object_schema(
                    json!({
                        "scenario": {"type": "string", "description": "Smoke scenario name"},
                        "runs": {"type": "integer", "description": "Number of runs"}
                    }),
                    vec!["scenario", "runs"],
                ),
            },
            StubTool {
                name: "run_release_eval",
                description: "Run a Forge release eval scenario.",
                parameters: object_schema(
                    json!({
                        "scenario": {"type": "string", "description": "Release scenario name"},
                        "runs": {"type": "integer", "description": "Number of runs"}
                    }),
                    vec!["scenario", "runs"],
                ),
            },
            StubTool {
                name: "fetch_records",
                description: "Fetch records. The count argument must be a zero-padded 4-digit numeric string.",
                parameters: object_schema(
                    json!({
                        "count": {"type": "string", "description": "Zero-padded 4-digit count, such as 0010"}
                    }),
                    vec!["count"],
                ),
            },
            StubTool {
                name: "summarize_records",
                description: "Summarize fetched record content.",
                parameters: object_schema(
                    json!({
                        "content": {"type": "string", "description": "Record content to summarize"}
                    }),
                    vec!["content"],
                ),
            },
            StubTool {
                name: "report_result",
                description: "Submit a concise terminal report for a completed Forge workflow.",
                parameters: object_schema(
                    json!({
                        "summary": {"type": "string", "description": "Final workflow summary"}
                    }),
                    vec!["summary"],
                ),
            },
            StubTool {
                name: "inspect_workflow_state",
                description: "Inspect the current Forge workflow state. Takes no arguments.",
                parameters: object_schema(json!({}), vec![]),
            },
            StubTool {
                name: "diagnose_failure",
                description: "Diagnose a Forge eval or proxy failure code.",
                parameters: object_schema(
                    json!({
                        "error_code": {"type": "string", "description": "Failure code to diagnose"}
                    }),
                    vec!["error_code"],
                ),
            },
        ],
        scenarios: vec![
            StubScenario {
                id: "fetch_zero_padded",
                user_request: "Fetch exactly 0010 records and summarize the count.",
            },
            StubScenario {
                id: "inspect_workflow_state",
                user_request: "Check the current Forge workflow state before continuing.",
            },
            StubScenario {
                id: "smoke_eval",
                user_request: "Run the basic_2step smoke eval once.",
            },
            StubScenario {
                id: "release_eval",
                user_request: "Run the basic_2step release eval once.",
            },
            StubScenario {
                id: "diagnose_failure",
                user_request: "Diagnose failure code TOOL_CALL_REJECTED from the last Forge eval.",
            },
        ],
    }
}

fn execute_repo_docs(name: &str, args: &Map<String, Value>) -> ToolExecution {
    match name {
        "search_docs" => {
            let query = string_arg(args, "query");
            ok(json!({
                "matches": [
                    {"path": "docs/COMPRESSION.md", "title": "Tool output compression", "score": 0.94},
                    {"path": "docs/EVAL_GUIDE.md", "title": "Eval classifier telemetry", "score": 0.81}
                ],
                "query": query
            }))
        }
        "read_doc" => {
            let path = string_arg(args, "path");
            ok(json!({
                "path": path,
                "content": "Forge proxy docs describe opt-in guarded traffic, classifier telemetry, and private local eval artifacts."
            }))
        }
        "list_files" => {
            let glob = string_arg(args, "glob");
            ok(json!({
                "glob": glob,
                "files": ["docs/BACKEND_SETUP.md", "docs/EVAL_GUIDE.md", "docs/PROXY_FLOW.md"]
            }))
        }
        "summarize_file" => {
            let path = string_arg(args, "path");
            ok(json!({
                "path": path,
                "summary": "The file documents proxy setup, safe opt-in behavior, and local verification commands."
            }))
        }
        other => error(format!("unknown repo_docs tool '{other}'")),
    }
}

fn execute_shopping(name: &str, args: &Map<String, Value>) -> ToolExecution {
    match name {
        "search_products" => {
            let query = string_arg(args, "query");
            ok(json!({
                "query": query,
                "products": [
                    {"product_id": "SKU-HEADPHONES-1", "name": "QuietBand 45", "price": 179},
                    {"product_id": "SKU-HEADPHONES-2", "name": "FocusPods Lite", "price": 149}
                ]
            }))
        }
        "get_product" => {
            let product_id = string_arg(args, "product_id");
            ok(json!({
                "product_id": product_id,
                "name": product_name(&product_id),
                "price": if product_id == "SKU-DOCK-1" { 89 } else { 179 },
                "in_stock": true
            }))
        }
        "compare_products" => {
            let product_ids = args
                .get("product_ids")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            ok(json!({
                "product_ids": product_ids,
                "winner": "SKU-HEADPHONES-1",
                "rationale": "Best noise cancellation while staying under $200."
            }))
        }
        "add_to_cart" => {
            let product_id = string_arg(args, "product_id");
            let quantity = args.get("quantity").and_then(Value::as_i64).unwrap_or(1);
            if quantity <= 0 {
                return error("quantity must be positive");
            }
            ok(json!({
                "cart_item": {"product_id": product_id, "quantity": quantity},
                "cart_status": "held"
            }))
        }
        other => error(format!("unknown shopping tool '{other}'")),
    }
}

fn execute_calendar(name: &str, args: &Map<String, Value>) -> ToolExecution {
    match name {
        "find_free_slots" => {
            let date = string_arg(args, "date");
            let duration = args
                .get("duration_minutes")
                .and_then(Value::as_i64)
                .unwrap_or(30);
            ok(json!({
                "date": date,
                "duration_minutes": duration,
                "slots": [
                    {"slot_id": "slot-001", "start": "2026-06-05T10:00:00-05:00"},
                    {"slot_id": "slot-002", "start": "2026-06-05T14:30:00-05:00"}
                ]
            }))
        }
        "create_calendar_hold" => {
            let slot_id = string_arg(args, "slot_id");
            ok(json!({
                "hold_id": "hold-001",
                "slot_id": slot_id,
                "status": "tentative"
            }))
        }
        "list_events" => {
            let date = string_arg(args, "date");
            ok(json!({
                "date": date,
                "events": [
                    {"title": "Design review", "start": "2026-06-03T09:00:00-05:00"},
                    {"title": "Support handoff", "start": "2026-06-03T13:00:00-05:00"}
                ]
            }))
        }
        "cancel_hold" => {
            let hold_id = string_arg(args, "hold_id");
            ok(json!({"hold_id": hold_id, "status": "cancelled"}))
        }
        other => error(format!("unknown calendar tool '{other}'")),
    }
}

fn execute_support(name: &str, args: &Map<String, Value>) -> ToolExecution {
    match name {
        "lookup_ticket" => {
            let ticket_id = string_arg(args, "ticket_id");
            ok(json!({
                "ticket_id": ticket_id,
                "status": if ticket_id == "TCK-2002" { "blocked" } else { "open" },
                "category": if ticket_id == "TCK-2002" { "billing" } else { "login" }
            }))
        }
        "search_kb" => {
            let query = string_arg(args, "query");
            ok(json!({
                "query": query,
                "articles": [
                    {"article_id": "KB-101", "title": "Password reset troubleshooting"},
                    {"article_id": "KB-207", "title": "Account lockout recovery"}
                ]
            }))
        }
        "update_ticket" => {
            let ticket_id = string_arg(args, "ticket_id");
            let note = string_arg(args, "note");
            ok(json!({"ticket_id": ticket_id, "note": note, "status": "updated"}))
        }
        "escalate_ticket" => {
            let ticket_id = string_arg(args, "ticket_id");
            let reason = string_arg(args, "reason");
            ok(json!({"ticket_id": ticket_id, "reason": reason, "status": "escalated"}))
        }
        other => error(format!("unknown support tool '{other}'")),
    }
}

fn execute_forge_eval(name: &str, args: &Map<String, Value>) -> ToolExecution {
    match name {
        "run_smoke_eval" => {
            let scenario = string_arg(args, "scenario");
            let runs = args.get("runs").and_then(Value::as_i64).unwrap_or(1);
            ok(json!({"suite": "smoke", "scenario": scenario, "runs": runs, "status": "queued"}))
        }
        "run_release_eval" => {
            let scenario = string_arg(args, "scenario");
            let runs = args.get("runs").and_then(Value::as_i64).unwrap_or(1);
            ok(json!({"suite": "release", "scenario": scenario, "runs": runs, "status": "queued"}))
        }
        "fetch_records" => {
            let count = string_arg(args, "count");
            if count.len() != 4 || !count.chars().all(|ch| ch.is_ascii_digit()) {
                return error("count must be a zero-padded 4-digit numeric string like 0010");
            }
            ok(
                json!({"count": count, "records": ["record-0001", "record-0002"], "status": "fetched"}),
            )
        }
        "summarize_records" => {
            let content = string_arg(args, "content");
            ok(json!({"summary": format!("Summarized {}", content)}))
        }
        "report_result" => {
            let summary = string_arg(args, "summary");
            ok(json!({"summary": summary, "status": "reported"}))
        }
        "inspect_workflow_state" => ok(json!({
            "required_steps": ["fetch_records", "summarize_records"],
            "completed_steps": [],
            "pending_steps": ["fetch_records", "summarize_records"]
        })),
        "diagnose_failure" => {
            let error_code = string_arg(args, "error_code");
            ok(
                json!({"error_code": error_code, "diagnosis": "The candidate tool call was rejected by semantic verification."}),
            )
        }
        other => error(format!("unknown forge_eval tool '{other}'")),
    }
}

fn object_schema(properties: Value, required: Vec<&str>) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

fn string_arg(args: &Map<String, Value>, name: &str) -> String {
    args.get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn product_name(product_id: &str) -> &'static str {
    match product_id {
        "SKU-DOCK-1" => "USB-C Dock Pro",
        "SKU-HEADPHONES-2" => "FocusPods Lite",
        _ => "QuietBand 45",
    }
}

fn ok(content: Value) -> ToolExecution {
    ToolExecution {
        status: ToolExecutionStatus::Ok,
        content,
    }
}

fn error(message: impl Into<String>) -> ToolExecution {
    ToolExecution {
        status: ToolExecutionStatus::Error,
        content: json!({"error": message.into()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registries_include_requested_tool_sets() {
        let registries = registries_for_domains(&[
            "repo_docs".to_string(),
            "shopping".to_string(),
            "calendar".to_string(),
            "support".to_string(),
            "forge_eval".to_string(),
        ])
        .expect("registries");
        let names = registries
            .iter()
            .flat_map(|registry| registry.tools.iter().map(|tool| tool.name))
            .collect::<Vec<_>>();
        assert!(names.contains(&"search_docs"));
        assert!(names.contains(&"add_to_cart"));
        assert!(names.contains(&"create_calendar_hold"));
        assert!(names.contains(&"escalate_ticket"));
        assert!(names.contains(&"fetch_records"));
        assert!(names.contains(&"inspect_workflow_state"));
    }

    #[test]
    fn request_tools_are_openai_function_tools() {
        let registry = shopping_registry();
        let tools = request_tools(&registry);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "search_products");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn available_tools_include_proxy_respond_tool() {
        let registry = repo_docs_registry();
        let tools = available_tools_for_training(&registry);
        assert!(tools
            .iter()
            .any(|tool| tool["name"].as_str() == Some("respond")));
    }

    #[test]
    fn deterministic_tool_execution_is_bounded_and_harmless() {
        let registry = shopping_registry();
        let result = execute_tool(
            &registry,
            "add_to_cart",
            &json!({"product_id": "SKU-DOCK-1", "quantity": 2}),
        );
        assert_eq!(result.status, ToolExecutionStatus::Ok);
        assert_eq!(result.content["cart_item"]["quantity"], 2);
    }

    #[test]
    fn forge_eval_tools_cover_protected_valid_slices() {
        let registry = forge_eval_registry();
        let no_args = execute_tool(&registry, "inspect_workflow_state", &json!({}));
        assert_eq!(no_args.status, ToolExecutionStatus::Ok);

        let valid_count = execute_tool(&registry, "fetch_records", &json!({"count": "0010"}));
        assert_eq!(valid_count.status, ToolExecutionStatus::Ok);

        let bad_count = execute_tool(&registry, "fetch_records", &json!({"count": "10"}));
        assert_eq!(bad_count.status, ToolExecutionStatus::Error);
    }
}
