use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::collections::HashSet;

/// Describes the schema for a single tool parameter field.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamModel {
    String {
        description: Option<String>,
        required: bool,
        default: Option<Value>,
        enum_values: Option<Vec<String>>,
    },
    Number {
        description: Option<String>,
        required: bool,
        default: Option<Value>,
    },
    Boolean {
        description: Option<String>,
        required: bool,
        default: Option<Value>,
    },
    Integer {
        description: Option<String>,
        required: bool,
        default: Option<Value>,
    },
    Object {
        description: Option<String>,
        required: bool,
        properties: IndexMap<String, ParamModel>,
    },
    Array {
        description: Option<String>,
        required: bool,
        items: Box<ParamModel>,
    },
    Unsupported {
        type_name: String,
    },
}

impl ParamModel {
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
    pub name: String,
    pub description: String,
    pub parameters: ParamModel,
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
        let properties = schema
            .get("properties")
            .ok_or_else(|| "schema must contain 'properties'".to_string())?;
        let required_fields: HashSet<String> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let param = Self::parse_object_param(properties, &required_fields)?;
        Ok(Self {
            name,
            description,
            parameters: param,
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
                let items_schema = schema
                    .get("items")
                    .ok_or_else(|| "array type must have 'items'".to_string())?;
                let items = Self::parse_single_param(items_schema, true)?;
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
                enum_values,
                default,
                ..
            } => {
                map.insert("type".into(), Value::String("string".into()));
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
            ParamModel::Number { default, .. } => {
                map.insert("type".into(), Value::String("number".into()));
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Boolean { default, .. } => {
                map.insert("type".into(), Value::String("boolean".into()));
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Integer { default, .. } => {
                map.insert("type".into(), Value::String("integer".into()));
                if let Some(d) = default {
                    map.insert("default".into(), d.clone());
                }
            }
            ParamModel::Object { properties, .. } => {
                map.insert("type".into(), Value::String("object".into()));
                let (nested_props, nested_req) = self.collect_object_schema(properties);
                map.insert("properties".into(), Value::Object(nested_props));
                if !nested_req.is_empty() {
                    map.insert(
                        "required".into(),
                        Value::Array(nested_req.into_iter().map(Value::String).collect()),
                    );
                }
            }
            ParamModel::Array { items, .. } => {
                map.insert("type".into(), Value::String("array".into()));
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
