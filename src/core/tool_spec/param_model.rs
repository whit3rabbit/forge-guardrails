//! Parameter model schema representation.

use indexmap::IndexMap;
use serde_json::Value;

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
