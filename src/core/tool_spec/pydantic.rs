//! Pydantic JSON Schema translation logic for ToolSpec.

use serde_json::{json, Map, Value};
use std::collections::HashSet;

/// Build a Pydantic-compatible JSON Schema from a standard JSON Schema.
pub fn build_pydantic_json_schema(name: &str, schema: &Value) -> Result<Value, String> {
    let empty_properties = Map::new();
    let properties = match schema.get("properties") {
        Some(Value::Object(obj)) => obj,
        Some(_) => return Err("properties must be an object".to_string()),
        None => &empty_properties,
    };
    let required_order: Vec<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let model_name = to_pascal_params(name);
    let mut defs = Map::new();
    let mut root =
        build_pydantic_model_schema(&model_name, properties, &required_order, &mut defs)?;
    if !defs.is_empty() {
        if let Value::Object(root_obj) = root {
            let mut ordered_root = Map::new();
            ordered_root.insert("$defs".to_string(), Value::Object(defs));
            for (key, value) in root_obj {
                ordered_root.insert(key, value);
            }
            root = Value::Object(ordered_root);
        }
    }
    Ok(root)
}

fn build_pydantic_model_schema(
    model_name: &str,
    properties: &Map<String, Value>,
    required_order: &[String],
    defs: &mut Map<String, Value>,
) -> Result<Value, String> {
    let mut prop_schemas = Map::new();
    let required_fields: HashSet<&str> = required_order.iter().map(String::as_str).collect();

    for (field_name, prop) in properties {
        let is_required = required_fields.contains(field_name.as_str());
        let schema = pydantic_property_schema(prop, field_name, model_name, is_required, defs)?;
        prop_schemas.insert(field_name.clone(), schema);
    }

    let mut model = Map::new();
    model.insert("properties".to_string(), Value::Object(prop_schemas));
    let required = required_order
        .iter()
        .filter(|name| properties.contains_key(*name))
        .cloned()
        .map(Value::String)
        .collect::<Vec<_>>();
    if !required.is_empty() {
        model.insert("required".to_string(), Value::Array(required));
    }
    model.insert("title".to_string(), Value::String(model_name.to_string()));
    model.insert("type".to_string(), Value::String("object".to_string()));
    Ok(Value::Object(model))
}

fn pydantic_property_schema(
    prop: &Value,
    field_name: &str,
    model_name_prefix: &str,
    required: bool,
    defs: &mut Map<String, Value>,
) -> Result<Value, String> {
    let base = pydantic_type_schema(prop, field_name, model_name_prefix, defs)?;
    let description = prop
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_string);
    let title = field_title(field_name);

    if required {
        if let Value::Object(base_obj) = base {
            if base_obj.contains_key("$ref") {
                let mut result = base_obj;
                if let Some(desc) = description {
                    result.insert("description".to_string(), Value::String(desc));
                }
                return Ok(Value::Object(result));
            }

            let mut result = Map::new();
            let prop_type = prop.get("type").and_then(Value::as_str).unwrap_or("string");
            if prop.get("enum").is_some() {
                if let Some(desc) = description {
                    result.insert("description".to_string(), Value::String(desc));
                }
                insert_schema_key(&mut result, &base_obj, "const");
                insert_schema_key(&mut result, &base_obj, "enum");
                result.insert("title".to_string(), Value::String(title));
                insert_schema_key(&mut result, &base_obj, "type");
            } else if prop_type == "array" {
                if let Some(desc) = description {
                    result.insert("description".to_string(), Value::String(desc));
                }
                insert_schema_key(&mut result, &base_obj, "items");
                result.insert("title".to_string(), Value::String(title));
                insert_schema_key(&mut result, &base_obj, "type");
            } else if prop_type == "object" {
                insert_schema_key(&mut result, &base_obj, "additionalProperties");
                if let Some(desc) = description {
                    result.insert("description".to_string(), Value::String(desc));
                }
                result.insert("title".to_string(), Value::String(title));
                insert_schema_key(&mut result, &base_obj, "type");
            } else {
                if let Some(desc) = description {
                    result.insert("description".to_string(), Value::String(desc));
                }
                result.insert("title".to_string(), Value::String(title));
                for (key, value) in &base_obj {
                    result.insert(key.clone(), value.clone());
                }
            }
            append_remaining_schema_keys(&mut result, &base_obj);
            return Ok(Value::Object(result));
        }
        return Ok(base);
    }

    let is_ref = matches!(&base, Value::Object(obj) if obj.contains_key("$ref"));
    let mut result = Map::new();
    result.insert(
        "anyOf".to_string(),
        Value::Array(vec![base, json!({"type": "null"})]),
    );
    result.insert(
        "default".to_string(),
        prop.get("default").cloned().unwrap_or(Value::Null),
    );
    if let Some(desc) = description {
        result.insert("description".to_string(), Value::String(desc));
    }
    if !is_ref {
        result.insert("title".to_string(), Value::String(title));
    }
    Ok(Value::Object(result))
}

fn pydantic_type_schema(
    prop: &Value,
    field_name: &str,
    model_name_prefix: &str,
    defs: &mut Map<String, Value>,
) -> Result<Value, String> {
    if let Some(enum_values) = prop.get("enum").and_then(Value::as_array) {
        let mut obj = Map::new();
        if enum_values.len() == 1 {
            obj.insert("const".to_string(), enum_values[0].clone());
        } else {
            obj.insert("enum".to_string(), Value::Array(enum_values.clone()));
        }
        obj.insert(
            "type".to_string(),
            Value::String(
                prop.get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("string")
                    .to_string(),
            ),
        );
        return Ok(Value::Object(obj));
    }

    match prop.get("type").and_then(Value::as_str).unwrap_or("string") {
        "string" | "integer" | "number" | "boolean" => Ok(json!({
            "type": prop.get("type").and_then(Value::as_str).unwrap_or("string")
        })),
        "object" => {
            let sub_props = prop.get("properties").and_then(Value::as_object);
            if let Some(sub_props) = sub_props.filter(|props| !props.is_empty()) {
                let sub_required: Vec<String> = prop
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let model_name =
                    format!("{}_{}", model_name_prefix, capitalize_for_model(field_name));
                let model =
                    build_pydantic_model_schema(&model_name, sub_props, &sub_required, defs)?;
                defs.insert(model_name.clone(), model);
                Ok(json!({"$ref": format!("#/$defs/{}", model_name)}))
            } else {
                Ok(json!({"additionalProperties": true, "type": "object"}))
            }
        }
        "array" => {
            let item_schema = match prop.get("items") {
                Some(items) => pydantic_type_schema(
                    items,
                    &format!("{}Item", field_name),
                    model_name_prefix,
                    defs,
                )?,
                None => Value::Object(Map::new()),
            };
            Ok(json!({"items": item_schema, "type": "array"}))
        }
        _ => Ok(Value::Object(Map::new())),
    }
}

fn insert_schema_key(target: &mut Map<String, Value>, source: &Map<String, Value>, key: &str) {
    if let Some(value) = source.get(key) {
        target.insert(key.to_string(), value.clone());
    }
}

fn append_remaining_schema_keys(target: &mut Map<String, Value>, source: &Map<String, Value>) {
    for (key, value) in source {
        if !target.contains_key(key) {
            target.insert(key.clone(), value.clone());
        }
    }
}

fn to_pascal_params(name: &str) -> String {
    format!(
        "{}Params",
        name.split('_')
            .map(capitalize_for_model)
            .collect::<String>()
    )
}

fn capitalize_for_model(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => {
            let head = first.to_uppercase().collect::<String>();
            let tail = chars.as_str().to_lowercase();
            format!("{}{}", head, tail)
        }
        None => String::new(),
    }
}

fn field_title(name: &str) -> String {
    name.split('_')
        .map(capitalize_for_model)
        .collect::<Vec<_>>()
        .join(" ")
}
