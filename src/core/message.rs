use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;
use std::fmt;

/// Message role: system, user, assistant, or tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Message type classification for metadata tagging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    SystemPrompt,
    UserInput,
    ToolCall,
    ToolResult,
    Reasoning,
    TextResponse,
    StepNudge,
    PrerequisiteNudge,
    RetryNudge,
    ContextWarning,
    Summary,
}

impl MessageType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SystemPrompt => "system_prompt",
            Self::UserInput => "user_input",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Reasoning => "reasoning",
            Self::TextResponse => "text_response",
            Self::StepNudge => "step_nudge",
            Self::PrerequisiteNudge => "prerequisite_nudge",
            Self::RetryNudge => "retry_nudge",
            Self::ContextWarning => "context_warning",
            Self::Summary => "summary",
        }
    }
}

impl fmt::Display for MessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Immutable metadata attached to a message.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageMeta {
    pub msg_type: MessageType,
    pub step_index: Option<i64>,
    pub original_type: Option<MessageType>,
    pub token_estimate: Option<i64>,
}

impl MessageMeta {
    pub fn new(msg_type: MessageType) -> Self {
        Self {
            msg_type,
            step_index: None,
            original_type: None,
            token_estimate: None,
        }
    }

    pub fn with_step_index(mut self, idx: i64) -> Self {
        self.step_index = Some(idx);
        self
    }

    pub fn with_original_type(mut self, t: MessageType) -> Self {
        self.original_type = Some(t);
        self
    }

    pub fn with_token_estimate(mut self, est: i64) -> Self {
        self.token_estimate = Some(est);
        self
    }
}

/// Immutable representation of a single tool call within a message.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallInfo {
    pub name: String,
    pub args: Option<IndexMap<String, Value>>,
    pub call_id: String,
}

impl ToolCallInfo {
    pub fn new(
        name: impl Into<String>,
        args: Option<IndexMap<String, Value>>,
        call_id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            args,
            call_id: call_id.into(),
        }
    }

    fn effective_args(&self) -> &IndexMap<String, Value> {
        static EMPTY: std::sync::OnceLock<IndexMap<String, Value>> = std::sync::OnceLock::new();
        self.args
            .as_ref()
            .unwrap_or_else(|| EMPTY.get_or_init(IndexMap::new))
    }
}

fn json_dumps_default_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => serde_json::to_string(v).unwrap_or_else(|_| "\"\"".to_string()),
        Value::Array(values) => {
            let inner = values
                .iter()
                .map(json_dumps_default_value)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{}]", inner)
        }
        Value::Object(values) => {
            let inner = values
                .iter()
                .map(|(key, val)| {
                    let key = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
                    format!("{}: {}", key, json_dumps_default_value(val))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{}}}", inner)
        }
    }
}

fn json_dumps_default_object(values: &IndexMap<String, Value>) -> String {
    let inner = values
        .iter()
        .map(|(key, val)| {
            let key = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
            format!("{}: {}", key, json_dumps_default_value(val))
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{}}}", inner)
}

/// A conversation message with dual serialization format support.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub metadata: MessageMeta,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<ToolCallInfo>>,
}

impl Message {
    pub fn new(role: MessageRole, content: impl Into<String>, metadata: MessageMeta) -> Self {
        Self {
            role,
            content: content.into(),
            metadata,
            tool_name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    pub fn with_tool_calls(mut self, calls: Vec<ToolCallInfo>) -> Self {
        self.tool_calls = Some(calls);
        self
    }

    /// Serialize this message for an LLM API.
    ///
    /// Format "ollama" (default): tool calls have no id/type, args as dict.
    /// Format "openai": tool calls have id, type="function", args as JSON string.
    pub fn serialize(&self, format: &str) -> Value {
        match format {
            "ollama" => self.serialize_ollama(),
            "openai" => self.serialize_openai(),
            _ => self.serialize_ollama(),
        }
    }

    fn serialize_ollama(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "role".to_string(),
            Value::String(self.role.as_str().to_string()),
        );
        map.insert("content".to_string(), Value::String(self.content.clone()));

        if let Some(calls) = &self.tool_calls {
            let tool_calls_json: Vec<Value> = calls
                .iter()
                .map(|tc| {
                    let mut entry = serde_json::Map::new();
                    let mut func = serde_json::Map::new();
                    func.insert("name".to_string(), Value::String(tc.name.clone()));
                    func.insert(
                        "arguments".to_string(),
                        serde_json::to_value(tc.effective_args())
                            .unwrap_or(Value::Object(serde_json::Map::new())),
                    );
                    entry.insert("function".to_string(), Value::Object(func));
                    Value::Object(entry)
                })
                .collect();
            map.insert("tool_calls".to_string(), Value::Array(tool_calls_json));
        }

        if self.role == MessageRole::Tool {
            if let Some(name) = &self.tool_name {
                map.insert("tool_name".to_string(), Value::String(name.clone()));
            }
        }

        Value::Object(map)
    }

    fn serialize_openai(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "role".to_string(),
            Value::String(self.role.as_str().to_string()),
        );
        map.insert("content".to_string(), Value::String(self.content.clone()));

        if let Some(calls) = &self.tool_calls {
            let tool_calls_json: Vec<Value> = calls
                .iter()
                .map(|tc| {
                    let mut entry = serde_json::Map::new();
                    entry.insert("id".to_string(), Value::String(tc.call_id.clone()));
                    entry.insert("type".to_string(), Value::String("function".to_string()));
                    let mut func = serde_json::Map::new();
                    func.insert("name".to_string(), Value::String(tc.name.clone()));
                    let args_str = json_dumps_default_object(tc.effective_args());
                    func.insert("arguments".to_string(), Value::String(args_str));
                    entry.insert("function".to_string(), Value::Object(func));
                    Value::Object(entry)
                })
                .collect();
            map.insert("tool_calls".to_string(), Value::Array(tool_calls_json));
        }

        if self.role == MessageRole::Tool {
            if let Some(name) = &self.tool_name {
                map.insert("name".to_string(), Value::String(name.clone()));
            }
            if let Some(id) = &self.tool_call_id {
                map.insert("tool_call_id".to_string(), Value::String(id.clone()));
            }
        }

        Value::Object(map)
    }
}
