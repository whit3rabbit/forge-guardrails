#![cfg(feature = "secrets-scanner")]

use forge_guardrails::redact_proxy_request_inputs;
use serde_json::json;

const SECRET: &str = "ghp_n0tArEaLsEcReTgHuBpAt1234567890AbCde";

#[test]
fn redacts_openai_message_content_tool_results_and_prior_tool_arguments() {
    let mut body = json!({
        "model": SECRET,
        "messages": [
            {"role": "system", "content": format!("system sees {SECRET}")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": format!("please inspect {SECRET}")},
                    {"type": "image_url", "image_url": {"url": format!("https://example.invalid/{SECRET}")}}
                ]
            },
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_keep_id",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": serde_json::to_string(&json!({"query": SECRET})).unwrap()
                        }
                    },
                    {
                        "id": "call_plain",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{ \"query\" : \"plain\" }"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_keep_id",
                "name": "lookup",
                "content": format!("tool result leaked {SECRET}")
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": format!("schema keeps {SECRET} untouched"),
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }]
    });

    let summary = redact_proxy_request_inputs(&mut body).expect("redaction succeeds");

    assert!(summary.fields_redacted >= 4);
    assert_eq!(body["model"], SECRET);
    assert_eq!(body["messages"][2]["tool_calls"][0]["id"], "call_keep_id");
    assert_eq!(
        body["messages"][2]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(
        body["tools"][0]["function"]["description"],
        format!("schema keeps {SECRET} untouched")
    );

    assert!(!body["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains(SECRET));
    assert!(!body["messages"][1]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains(SECRET));
    assert!(body["messages"][1]["content"][1]["image_url"]["url"]
        .as_str()
        .unwrap()
        .contains(SECRET));
    assert!(!body["messages"][3]["content"]
        .as_str()
        .unwrap()
        .contains(SECRET));

    let args: serde_json::Value = serde_json::from_str(
        body["messages"][2]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .expect("arguments stay valid JSON");
    assert_eq!(args["query"], "[REDACTED_SECRET]");
    assert_eq!(
        body["messages"][2]["tool_calls"][1]["function"]["arguments"],
        "{ \"query\" : \"plain\" }"
    );
}

#[test]
fn redacts_anthropic_system_message_tool_result_and_tool_use_input() {
    let mut body = json!({
        "model": SECRET,
        "max_tokens": 128,
        "system": [{"type": "text", "text": format!("system {SECRET}")}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": format!("hello {SECRET}")}]},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_keep_id",
                    "name": "lookup",
                    "input": {"token": SECRET}
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_keep_id",
                    "content": format!("tool result {SECRET}")
                }]
            }
        ],
        "tools": [{
            "name": "lookup",
            "description": format!("tool schema keeps {SECRET}"),
            "input_schema": {"type": "object", "properties": {"token": {"type": "string"}}}
        }]
    });

    let summary = redact_proxy_request_inputs(&mut body).expect("redaction succeeds");

    assert!(summary.fields_redacted >= 4);
    assert_eq!(body["model"], SECRET);
    assert_eq!(body["messages"][1]["content"][0]["id"], "toolu_keep_id");
    assert_eq!(body["messages"][1]["content"][0]["name"], "lookup");
    assert_eq!(
        body["tools"][0]["description"],
        format!("tool schema keeps {SECRET}")
    );

    assert!(!body["system"][0]["text"].as_str().unwrap().contains(SECRET));
    assert!(!body["messages"][0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains(SECRET));
    assert_eq!(
        body["messages"][1]["content"][0]["input"]["token"],
        "[REDACTED_SECRET]"
    );
    assert!(!body["messages"][2]["content"][0]["content"]
        .as_str()
        .unwrap()
        .contains(SECRET));
}
