use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

/// Describes the schema for a single tool parameter field.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamModel {
    /// A string parameter type.
    String {
        /// Optional parameter description.
        description: Option<String>,
        /// Whether the parameter is required.
        required: bool,
        /// Optional default value.
        default: Option<Value>,
        /// Optional list of valid enum values.
        enum_values: Option<Vec<String>>,
    },
    /// A numeric parameter type (floating point).
    Number {
        /// Optional parameter description.
        description: Option<String>,
        /// Whether the parameter is required.
        required: bool,
        /// Optional default value.
        default: Option<Value>,
    },
    /// A boolean parameter type (true/false).
    Boolean {
        /// Optional parameter description.
        description: Option<String>,
        /// Whether the parameter is required.
        required: bool,
        /// Optional default value.
        default: Option<Value>,
    },
    /// An integer parameter type.
    Integer {
        /// Optional parameter description.
        description: Option<String>,
        /// Whether the parameter is required.
        required: bool,
        /// Optional default value.
        default: Option<Value>,
    },
    /// A nested object parameter type.
    Object {
        /// Optional parameter description.
        description: Option<String>,
        /// Whether the parameter is required.
        required: bool,
        /// Map of property names to their parameter models.
        properties: IndexMap<String, ParamModel>,
    },
    /// An array of items parameter type.
    Array {
        /// Optional parameter description.
        description: Option<String>,
        /// Whether the parameter is required.
        required: bool,
        /// The parameter model of individual items in the array.
        items: Box<ParamModel>,
    },
    /// An unsupported or custom parameter type.
    Unsupported {
        /// The name of the unsupported type.
        type_name: String,
    },
}

impl ParamModel {
    /// Returns true if this parameter is required.
    pub fn is_required(&self) -> bool {
        match self {
            Self::String { required, .. }
            | Self::Number { required, .. }
            | Self::Boolean { required, .. }
            | Self::Integer { required, .. }
            | Self::Object { required, .. }
            | Self::Array { required, .. } => *required,
            Self::Unsupported { .. } => false,
        }
    }
}

/// Tool parameter schema validated from a JSON Schema definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// The unique identifier/name of the tool.
    pub name: String,
    /// Human-readable explanation of what the tool does and when to call it.
    pub description: String,
    /// Schema definition of arguments accepted by this tool.
    pub parameters: ParamModel,
    /// Optional pre-computed/cached JSON Schema representation.
    pub json_schema: Option<Value>,
}

impl ToolSpec {
    /// Construct a ToolSpec from a JSON Schema dictionary.
    ///
    /// Handles: string, number, boolean, integer, object (nested), array,
    /// enum constraints, required fields, and defaults.
    /// Returns an error string for unsupported schema constructs.
    pub fn from_json_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: &Value,
    ) -> Result<Self, String> {
        let name = name.into();
        let description = description.into();
        let empty_properties = Value::Object(Map::new());
        let properties = schema.get("properties").unwrap_or(&empty_properties);
        let required_order: Vec<String> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let required_fields: HashSet<String> = required_order.iter().cloned().collect();

        let param = Self::parse_object_param(properties, &required_fields)?;
        let json_schema = Self::build_pydantic_json_schema(&name, schema)?;
        Ok(Self {
            name,
            description,
            parameters: param,
            json_schema: Some(json_schema),
        })
    }

    fn parse_object_param(
        props: &Value,
        required_fields: &HashSet<String>,
    ) -> Result<ParamModel, String> {
        let props_obj = props
            .as_object()
            .ok_or_else(|| "properties must be an object".to_string())?;
        let mut fields = IndexMap::new();
        for (key, schema_val) in props_obj {
            let is_required = required_fields.contains(key);
            fields.insert(
                key.clone(),
                Self::parse_single_param(schema_val, is_required)?,
            );
        }
        Ok(ParamModel::Object {
            description: None,
            required: true,
            properties: fields,
        })
    }

    fn parse_single_param(schema: &Value, required: bool) -> Result<ParamModel, String> {
        let type_name = schema
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("string");
        let description = schema
            .get("description")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());
        let default = schema.get("default").cloned();
        let enum_values = schema.get("enum").and_then(|e| e.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>()
        });

        match type_name {
            "string" => Ok(ParamModel::String {
                description,
                required,
                default,
                enum_values,
            }),
            "number" => Ok(ParamModel::Number {
                description,
                required,
                default,
            }),
            "boolean" => Ok(ParamModel::Boolean {
                description,
                required,
                default,
            }),
            "integer" => Ok(ParamModel::Integer {
                description,
                required,
                default,
            }),
            "object" => {
                let nested_props = schema.get("properties");
                let nested_required: HashSet<String> = schema
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let properties = match nested_props {
                    Some(p) => {
                        let p_obj = p
                            .as_object()
                            .ok_or_else(|| "nested properties must be an object".to_string())?;
                        let mut fields = IndexMap::new();
                        for (key, val) in p_obj {
                            let is_req = nested_required.contains(key);
                            fields.insert(key.clone(), Self::parse_single_param(val, is_req)?);
                        }
                        fields
                    }
                    None => IndexMap::new(),
                };
                Ok(ParamModel::Object {
                    description,
                    required,
                    properties,
                })
            }
            "array" => {
                let items = match schema.get("items") {
                    Some(items_schema) => Self::parse_single_param(items_schema, true)?,
                    None => ParamModel::Unsupported {
                        type_name: "any".to_string(),
                    },
                };
                Ok(ParamModel::Array {
                    description,
                    required,
                    items: Box::new(items),
                })
            }
            other => Ok(ParamModel::Unsupported {
                type_name: other.to_string(),
            }),
        }
    }

    /// Emit the JSON Schema for this tool spec's parameters.
    pub fn get_json_schema(&self) -> Value {
        if let Some(schema) = &self.json_schema {
            return schema.clone();
        }
        self.reconstruct_json_schema()
    }

    fn reconstruct_json_schema(&self) -> Value {
        let mut schema = Map::new();
        schema.insert("type".into(), Value::String("object".into()));
        let (properties, required) = self.param_to_schema(&self.parameters);
        schema.insert("properties".into(), Value::Object(properties));
        if !required.is_empty() {
            schema.insert(
                "required".into(),
                Value::Array(required.into_iter().map(Value::String).collect()),
            );
        }
        Value::Object(schema)
    }

    fn build_pydantic_json_schema(name: &str, schema: &Value) -> Result<Value, String> {
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

        let model_name = Self::to_pascal_params(name);
        let mut defs = Map::new();
        let mut root =
            Self::build_pydantic_model_schema(&model_name, properties, &required_order, &mut defs)?;
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
            let schema =
                Self::pydantic_property_schema(prop, field_name, model_name, is_required, defs)?;
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
        let base = Self::pydantic_type_schema(prop, field_name, model_name_prefix, defs)?;
        let description = prop
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string);
        let title = Self::field_title(field_name);

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
                    Self::insert_schema_key(&mut result, &base_obj, "const");
                    Self::insert_schema_key(&mut result, &base_obj, "enum");
                    result.insert("title".to_string(), Value::String(title));
                    Self::insert_schema_key(&mut result, &base_obj, "type");
                } else if prop_type == "array" {
                    if let Some(desc) = description {
                        result.insert("description".to_string(), Value::String(desc));
                    }
                    Self::insert_schema_key(&mut result, &base_obj, "items");
                    result.insert("title".to_string(), Value::String(title));
                    Self::insert_schema_key(&mut result, &base_obj, "type");
                } else if prop_type == "object" {
                    Self::insert_schema_key(&mut result, &base_obj, "additionalProperties");
                    if let Some(desc) = description {
                        result.insert("description".to_string(), Value::String(desc));
                    }
                    result.insert("title".to_string(), Value::String(title));
                    Self::insert_schema_key(&mut result, &base_obj, "type");
                } else {
                    if let Some(desc) = description {
                        result.insert("description".to_string(), Value::String(desc));
                    }
                    result.insert("title".to_string(), Value::String(title));
                    for (key, value) in &base_obj {
                        result.insert(key.clone(), value.clone());
                    }
                }
                Self::append_remaining_schema_keys(&mut result, &base_obj);
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
                    let model_name = format!(
                        "{}_{}",
                        model_name_prefix,
                        Self::capitalize_for_model(field_name)
                    );
                    let model = Self::build_pydantic_model_schema(
                        &model_name,
                        sub_props,
                        &sub_required,
                        defs,
                    )?;
                    defs.insert(model_name.clone(), model);
                    Ok(json!({"$ref": format!("#/$defs/{}", model_name)}))
                } else {
                    Ok(json!({"additionalProperties": true, "type": "object"}))
                }
            }
            "array" => {
                let item_schema = match prop.get("items") {
                    Some(items) => Self::pydantic_type_schema(
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
                .map(Self::capitalize_for_model)
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
            .map(Self::capitalize_for_model)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn param_to_schema(&self, param: &ParamModel) -> (Map<String, Value>, Vec<String>) {
        match param {
            ParamModel::Object {
                properties,
                required,
                ..
            } => {
                let mut props = Map::new();
                let mut req = Vec::new();
                for (name, model) in properties {
                    let (nested_props, _nested_req) = self.single_param_to_schema(model);
                    props.insert(name.clone(), Value::Object(nested_props));
                    if *required && model.is_required() {
                        req.push(name.clone());
                    }
                }
                (props, req)
            }
            _ => (Map::new(), Vec::new()),
        }
    }

    fn single_param_to_schema(&self, param: &ParamModel) -> (Map<String, Value>, Vec<String>) {
        let mut map = Map::new();
        match param {
            ParamModel::String {
                description,
                enum_values,
                default,
                ..
            } => {
                map.insert("type".into(), Value::String("string".into()));
                if let Some(desc) = description {
                    map.insert("description".into(), Value::String(desc.clone()));
                }
                if let Some(enums) = enum_values {
                    map.insert(
                        "enum".into(),
                        Value::Array(enums.iter().map(|s| Value::String(s.clone())).collect()),
                    );
                }
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Number {
                description,
                default,
                ..
            } => {
                map.insert("type".into(), Value::String("number".into()));
                if let Some(desc) = description {
                    map.insert("description".into(), Value::String(desc.clone()));
                }
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Boolean {
                description,
                default,
                ..
            } => {
                map.insert("type".into(), Value::String("boolean".into()));
                if let Some(desc) = description {
                    map.insert("description".into(), Value::String(desc.clone()));
                }
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Integer {
                description,
                default,
                ..
            } => {
                map.insert("type".into(), Value::String("integer".into()));
                if let Some(desc) = description {
                    map.insert("description".into(), Value::String(desc.clone()));
                }
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Object {
                description,
                properties,
                ..
            } => {
                map.insert("type".into(), Value::String("object".into()));
                if let Some(desc) = description {
                    map.insert("description".into(), Value::String(desc.clone()));
                }
                let (nested_props, nested_req) = self.collect_object_schema(properties);
                map.insert("properties".into(), Value::Object(nested_props));
                if !nested_req.is_empty() {
                    map.insert(
                        "required".into(),
                        Value::Array(nested_req.into_iter().map(Value::String).collect()),
                    );
                }
            }
            ParamModel::Array {
                description, items, ..
            } => {
                map.insert("type".into(), Value::String("array".into()));
                if let Some(desc) = description {
                    map.insert("description".into(), Value::String(desc.clone()));
                }
                let (item_schema, _) = self.single_param_to_schema(items);
                map.insert("items".into(), Value::Object(item_schema));
            }
            ParamModel::Unsupported { type_name } => {
                map.insert("type".into(), Value::String(type_name.clone()));
            }
        }
        (map, Vec::new())
    }

    fn collect_object_schema(
        &self,
        properties: &IndexMap<String, ParamModel>,
    ) -> (Map<String, Value>, Vec<String>) {
        let mut props = Map::new();
        let mut required = Vec::new();
        for (name, model) in properties {
            let (schema_map, _) = self.single_param_to_schema(model);
            props.insert(name.clone(), Value::Object(schema_map));
            if model.is_required() {
                required.push(name.clone());
            }
        }
        (props, required)
    }
}
