//! Tool schema description compression for proxy-intercepted requests.
//!
//! Minification is conservative: it only touches description strings in tool
//! schemas (top-level `ToolSpec.description` and every `"description"` value
//! inside the JSON Schema). Names, parameter names, types, `required` arrays,
//! and all other schema structure are never modified.

use std::str::FromStr;

use serde_json::Value;

use crate::core::tool_spec::ToolSpec;

/// Controls how tool schema descriptions are compressed before being forwarded
/// upstream. Default is `Disabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchemaCompressionMode {
    /// No mutation. Default.
    #[default]
    Disabled,
    /// Trim trailing whitespace per line, collapse consecutive blank lines to
    /// at most one, collapse internal space/tab runs to one space (outside
    /// fenced code blocks), and drop `"description": ""` keys.
    Minify,
}

impl SchemaCompressionMode {
    /// Returns the canonical string representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Minify => "minify",
        }
    }
}

impl FromStr for SchemaCompressionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "disabled" => Ok(Self::Disabled),
            "minify" => Ok(Self::Minify),
            _ => Err(format!(
                "unknown schema compression mode '{s}'; expected disabled or minify"
            )),
        }
    }
}

/// Summary of changes made by one schema compression pass.
#[derive(Debug, Clone, Default)]
pub struct SchemaCompressionStats {
    /// Number of description fields that were shortened.
    pub descriptions_changed: usize,
    /// Number of empty (or whitespace-only) description fields that were dropped.
    pub descriptions_dropped: usize,
}

/// Minify tool schema descriptions in place. Returns stats on what changed.
///
/// Transforms apply to `spec.description` and every `"description"` string
/// inside `spec.json_schema` up to recursion depth 32. Tool names, parameter
/// names, types, and `required` arrays are untouched.
pub fn compress_tool_schemas(
    specs: &mut [ToolSpec],
    mode: SchemaCompressionMode,
) -> SchemaCompressionStats {
    let mut stats = SchemaCompressionStats::default();
    if mode == SchemaCompressionMode::Disabled {
        return stats;
    }
    for spec in specs.iter_mut() {
        if let Some(minified) = minify_description(&spec.description) {
            spec.description = minified;
            stats.descriptions_changed += 1;
        }
        if let Some(json_schema) = spec.json_schema.as_mut() {
            let (c, d) = minify_schema_descriptions(json_schema, 0);
            stats.descriptions_changed += c;
            stats.descriptions_dropped += d;
        }
    }
    stats
}

/// Minify tool schema descriptions in a raw Anthropic request body.
///
/// Walks `body["tools"][*]["description"]` and
/// `body["tools"][*]["input_schema"]` and applies the same transforms as
/// `compress_tool_schemas`. Returns true if any description was changed or
/// dropped. Both paths share `minify_description`, so the resulting strings
/// are byte-identical.
pub fn patch_anthropic_tool_schemas(body: &mut Value, mode: SchemaCompressionMode) -> bool {
    if mode == SchemaCompressionMode::Disabled {
        return false;
    }
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut any_changed = false;
    for tool in tools.iter_mut() {
        let Some(obj) = tool.as_object_mut() else {
            continue;
        };
        if let Some(Value::String(desc)) = obj.get_mut("description") {
            if let Some(minified) = minify_description(desc) {
                *desc = minified;
                any_changed = true;
            }
        }
        if let Some(schema) = obj.get_mut("input_schema") {
            let (c, d) = minify_schema_descriptions(schema, 0);
            if c + d > 0 {
                any_changed = true;
            }
        }
    }
    any_changed
}

const MAX_SCHEMA_RECURSION_DEPTH: usize = 32;

/// Returns `Some(minified)` if the text was changed, `None` if already clean.
pub(crate) fn minify_description(desc: &str) -> Option<String> {
    let result = apply_minify(desc);
    if result == desc {
        None
    } else {
        Some(result)
    }
}

fn apply_minify(desc: &str) -> String {
    let mut in_fence = false;
    let mut consecutive_blanks = 0usize;
    let mut lines_out: Vec<String> = Vec::new();

    for line in desc.lines() {
        let trimmed_end = line.trim_end();

        // Fenced code block boundary — toggle and pass through.
        if trimmed_end.starts_with("```") {
            in_fence = !in_fence;
            consecutive_blanks = 0;
            lines_out.push(trimmed_end.to_string());
            continue;
        }

        if in_fence {
            lines_out.push(trimmed_end.to_string());
            consecutive_blanks = 0;
            continue;
        }

        let processed = collapse_internal_whitespace(trimmed_end);
        if processed.is_empty() {
            consecutive_blanks += 1;
            if consecutive_blanks <= 1 {
                lines_out.push(String::new());
            }
        } else {
            consecutive_blanks = 0;
            lines_out.push(processed);
        }
    }

    while lines_out.last().is_some_and(|l| l.is_empty()) {
        lines_out.pop();
    }
    lines_out.join("\n")
}

fn collapse_internal_whitespace(line: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let mut last_was_space = false;
    for ch in line.chars() {
        if ch == ' ' || ch == '\t' {
            if !last_was_space && !result.is_empty() {
                result.push(' ');
            }
            last_was_space = true;
        } else {
            result.push(ch);
            last_was_space = false;
        }
    }
    result
}

/// Walk a JSON value and minify every `"description"` string key.
/// Returns (changed_count, dropped_count).
fn minify_schema_descriptions(value: &mut Value, depth: usize) -> (usize, usize) {
    if depth > MAX_SCHEMA_RECURSION_DEPTH {
        return (0, 0);
    }
    let mut changed = 0usize;
    let mut dropped = 0usize;
    match value {
        Value::Object(obj) => {
            // Compute the new value (or drop) for "description" before mutating.
            let desc_action = obj.get("description").and_then(Value::as_str).map(|d| {
                if d.is_empty() {
                    None // drop
                } else {
                    let m = apply_minify(d);
                    if m.is_empty() {
                        None // drop
                    } else if m != d {
                        Some(m) // replace
                    } else {
                        Some(d.to_string()) // unchanged — will not count
                    }
                }
            });
            match desc_action {
                Some(None) => {
                    obj.remove("description");
                    dropped += 1;
                }
                Some(Some(ref new_val)) => {
                    // Only count as changed if the value actually differs.
                    let was_changed = obj
                        .get("description")
                        .and_then(Value::as_str)
                        .is_some_and(|old| old != new_val.as_str());
                    if was_changed {
                        obj.insert("description".to_string(), Value::String(new_val.clone()));
                        changed += 1;
                    }
                }
                None => {} // key absent — nothing to do
            }
            // Recurse into all other values (collect keys first to avoid borrow issues).
            let keys: Vec<String> = obj.keys().cloned().collect();
            for key in keys {
                if let Some(v) = obj.get_mut(&key) {
                    let (c, d) = minify_schema_descriptions(v, depth + 1);
                    changed += c;
                    dropped += d;
                }
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                let (c, d) = minify_schema_descriptions(v, depth + 1);
                changed += c;
                dropped += d;
            }
        }
        _ => {}
    }
    (changed, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_mode_from_str_roundtrips() {
        assert_eq!(
            SchemaCompressionMode::from_str("disabled").unwrap(),
            SchemaCompressionMode::Disabled
        );
        assert_eq!(
            SchemaCompressionMode::from_str("minify").unwrap(),
            SchemaCompressionMode::Minify
        );
        assert_eq!(
            SchemaCompressionMode::from_str("MINIFY").unwrap(),
            SchemaCompressionMode::Minify
        );
    }

    #[test]
    fn schema_mode_invalid_returns_err() {
        let err = SchemaCompressionMode::from_str("gzip").unwrap_err();
        assert!(err.contains("gzip"), "error should mention the bad input");
        assert!(
            err.contains("disabled") || err.contains("minify"),
            "error should list valid options"
        );
    }

    #[test]
    fn minify_collapses_internal_whitespace() {
        assert_eq!(
            minify_description("  hello   world  ").unwrap(),
            "hello world"
        );
    }

    #[test]
    fn minify_trims_trailing_whitespace() {
        assert_eq!(
            minify_description("hello   \nworld   ").unwrap(),
            "hello\nworld"
        );
    }

    #[test]
    fn minify_collapses_excess_blank_lines() {
        let desc = "a\n\n\n\nb";
        let result = minify_description(desc).unwrap();
        assert!(result.contains("a") && result.contains("b"));
        assert!(
            !result.contains("\n\n\n"),
            "should collapse to max 2 blanks"
        );
    }

    #[test]
    fn minify_preserves_fenced_code_block_content() {
        let desc = "Before:\n```\n  preserved   whitespace  \n```\nAfter";
        // Content inside fence should be left alone (only trailing whitespace trimmed).
        if let Some(result) = minify_description(desc) {
            assert!(
                result.contains("  preserved   whitespace"),
                "fenced content must not be collapsed"
            );
        }
        // "After" must still be present.
        let result = apply_minify(desc);
        assert!(result.contains("After"));
    }

    #[test]
    fn minify_idempotent() {
        let desc = "  hello   world  \n\n\n\nline2   ";
        let once = apply_minify(desc);
        let twice = apply_minify(&once);
        assert_eq!(once, twice, "minification must be idempotent");
    }

    #[test]
    fn minify_unchanged_returns_none() {
        assert_eq!(minify_description("already clean"), None);
    }

    #[test]
    fn compress_tool_schemas_disabled_noop() {
        use crate::core::tool_spec::param_model::ParamModel;
        let original_desc = "  A   tool  ";
        let mut specs = vec![ToolSpec {
            name: "tool".to_string(),
            description: original_desc.to_string(),
            parameters: ParamModel::Object {
                description: None,
                required: true,
                properties: Default::default(),
            },
            json_schema: None,
        }];
        let stats = compress_tool_schemas(&mut specs, SchemaCompressionMode::Disabled);
        assert_eq!(specs[0].description, original_desc);
        assert_eq!(stats.descriptions_changed, 0);
        assert_eq!(stats.descriptions_dropped, 0);
    }

    #[test]
    fn compress_tool_schemas_minifies_descriptions() {
        use crate::core::tool_spec::param_model::ParamModel;
        let mut specs = vec![ToolSpec {
            name: "tool".to_string(),
            description: "  A   tool  ".to_string(),
            parameters: ParamModel::Object {
                description: None,
                required: true,
                properties: Default::default(),
            },
            json_schema: Some(json!({
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "  A   param  "
                    }
                }
            })),
        }];
        let stats = compress_tool_schemas(&mut specs, SchemaCompressionMode::Minify);
        assert_eq!(specs[0].description, "A tool");
        assert!(stats.descriptions_changed >= 1);
        let pdesc = specs[0].json_schema.as_ref().unwrap()["properties"]["param"]["description"]
            .as_str()
            .unwrap();
        assert_eq!(pdesc, "A param");
    }

    #[test]
    fn patch_anthropic_disabled_noop() {
        let mut body = json!({
            "tools": [{"name": "bash", "description": "  Run   a   command  "}]
        });
        let changed = patch_anthropic_tool_schemas(&mut body, SchemaCompressionMode::Disabled);
        assert!(!changed);
        assert_eq!(
            body["tools"][0]["description"].as_str().unwrap(),
            "  Run   a   command  "
        );
    }

    #[test]
    fn patch_anthropic_minifies_descriptions() {
        let mut body = json!({
            "tools": [{
                "name": "bash",
                "description": "  Run   a   command  ",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "  The   command  "
                        }
                    }
                }
            }]
        });
        let changed = patch_anthropic_tool_schemas(&mut body, SchemaCompressionMode::Minify);
        assert!(changed);
        assert_eq!(
            body["tools"][0]["description"].as_str().unwrap(),
            "Run a command"
        );
        assert_eq!(
            body["tools"][0]["input_schema"]["properties"]["cmd"]["description"]
                .as_str()
                .unwrap(),
            "The command"
        );
    }

    #[test]
    fn toolspec_and_anthropic_paths_byte_identical() {
        let raw = "  Runs   a   shell   command  ";
        let via_toolspec = minify_description(raw).unwrap_or_else(|| raw.to_string());
        let mut body = json!({"tools": [{"name": "bash", "description": raw}]});
        patch_anthropic_tool_schemas(&mut body, SchemaCompressionMode::Minify);
        let via_anthropic = body["tools"][0]["description"].as_str().unwrap();
        assert_eq!(
            via_toolspec, via_anthropic,
            "ToolSpec path and Anthropic body path must be byte-identical"
        );
    }
}
