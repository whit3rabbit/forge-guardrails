use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::Value;

use super::io::{capture_key, read_jsonl, read_jsonl_path, row_key};

#[derive(Debug, Default)]
pub(crate) struct ResumeState {
    pub(crate) processed_capture_keys: HashSet<String>,
    pub(crate) processed_row_keys: HashSet<String>,
    pub(crate) valid_real_capture_keys: HashSet<String>,
    pub(crate) alternatives_per_group: HashMap<String, usize>,
    pub(crate) accepted_real_count: usize,
    pub(crate) accepted_alternative_count: usize,
}

impl ResumeState {
    pub(crate) fn load(output_path: &str, reject_path: &Path) -> Result<Self, String> {
        let mut state = Self::default();
        state.load_training_output(output_path)?;
        state.load_rejects(reject_path)?;
        Ok(state)
    }

    pub(crate) fn load_training_output(&mut self, path: &str) -> Result<(), String> {
        if !Path::new(path).exists() {
            return Ok(());
        }
        for row in read_jsonl(path)? {
            self.record_training_row(&row);
        }
        Ok(())
    }

    pub(crate) fn load_rejects(&mut self, path: &Path) -> Result<(), String> {
        if !path.exists() {
            return Ok(());
        }
        for row in read_jsonl_path(path)? {
            if let Some(cap_key) = row.get("capture_key").and_then(Value::as_str) {
                self.processed_capture_keys.insert(cap_key.to_string());
            } else if let Some(capture) = row.get("capture") {
                self.processed_capture_keys.insert(capture_key(capture));
            }
        }
        Ok(())
    }

    pub(crate) fn record_training_row(&mut self, row: &Value) {
        let review = row.get("review").unwrap_or(&Value::Null);
        let source_bucket = review
            .get("source_bucket")
            .and_then(Value::as_str)
            .unwrap_or("");
        if let Some(r_key) = review.get("row_key").and_then(Value::as_str) {
            self.processed_row_keys.insert(r_key.to_string());
        } else if let (Some(example_group_id), Some(candidate_call)) = (
            review.get("example_group_id").and_then(Value::as_str),
            row.get("input")
                .and_then(|input| input.get("candidate_call")),
        ) {
            self.processed_row_keys.insert(row_key(
                example_group_id,
                source_bucket,
                candidate_call,
            ));
        }
        if let Some(cap_key) = review.get("capture_key").and_then(Value::as_str) {
            if source_bucket == "real_model_call" {
                self.processed_capture_keys.insert(cap_key.to_string());
                self.accepted_real_count += 1;
                if row.get("label").and_then(Value::as_str) == Some("valid") {
                    self.valid_real_capture_keys.insert(cap_key.to_string());
                }
            }
        }
        if source_bucket == "targeted_alternative" {
            self.accepted_alternative_count += 1;
            if let Some(group_id) = review.get("example_group_id").and_then(Value::as_str) {
                *self
                    .alternatives_per_group
                    .entry(group_id.to_string())
                    .or_insert(0) += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{ReviewDecision, VerifierDecision};
    use super::*;
    use crate::schema::{TRAINING_INPUT_SCHEMA_VERSION, TRAINING_SCHEMA_VERSION};
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

    fn training_row(
        capture: &Value,
        candidate_call: Value,
        label: &str,
        source_bucket: &str,
        review: &ReviewDecision,
        verifier: &VerifierDecision,
    ) -> Value {
        let example_group_id = capture
            .get("example_group_id")
            .cloned()
            .unwrap_or(Value::Null);
        let example_group_id_str = example_group_id.as_str().unwrap_or("unknown-group");
        let cap_key = capture_key(capture);
        let r_key = row_key(example_group_id_str, source_bucket, &candidate_call);
        json!({
            "schema_version": TRAINING_SCHEMA_VERSION,
            "input": {
                "schema_version": TRAINING_INPUT_SCHEMA_VERSION,
                "user_request": capture.get("user_request").cloned().unwrap_or(Value::Null),
                "workflow_state": capture.get("workflow_state").cloned().unwrap_or(Value::Null),
                "available_tools": capture.get("available_tools").cloned().unwrap_or_else(|| json!([])),
                "candidate_call": candidate_call,
            },
            "label": label,
            "review": {
                "source": "forge-dataset",
                "source_bucket": source_bucket,
                "example_group_id": example_group_id,
                "capture_key": cap_key,
                "row_key": r_key,
                "reviewer": {
                    "label": review.label,
                    "confidence": review.confidence,
                    "rationale": review.rationale
                },
                "verifier": {
                    "accepted": verifier.accepted,
                    "rationale": verifier.rationale
                }
            }
        })
    }

    #[test]
    fn resume_state_tracks_streamed_real_rows() {
        let review = ReviewDecision {
            label: "valid".to_string(),
            confidence: 0.9,
            rationale: "ok".to_string(),
            corrected_candidate_call: None,
            raw: json!({}),
        };
        let verifier = VerifierDecision {
            accepted: true,
            rationale: "accepted".to_string(),
            raw: json!({}),
        };
        let capture = capture_row();
        let row = training_row(
            &capture,
            capture["candidate_call"].clone(),
            "valid",
            "real_model_call",
            &review,
            &verifier,
        );

        let mut state = ResumeState::default();
        state.record_training_row(&row);

        assert_eq!(state.accepted_real_count, 1);
        assert!(state
            .processed_capture_keys
            .contains(row["review"]["capture_key"].as_str().expect("capture key")));
        assert!(state
            .valid_real_capture_keys
            .contains(row["review"]["capture_key"].as_str().expect("capture key")));
    }
}
