use super::preserve_trailing_newline;
use regex_lite::Regex;
use serde_json::Value;
use std::sync::LazyLock;

const JSON_TABLE_MIN_BYTES: usize = 128;

static TIMESTAMP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\b\d{4}-\d{2}-\d{2}[T ][0-9:.]+Z?\b"#).expect("valid timestamp regex")
});

static HASH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\b[0-9a-f]{32,64}\b"#).expect("valid hash regex"));

pub(super) fn minify_json(output: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(output.trim()).ok()?;
    serde_json::to_string(&parsed).ok()
}

pub(super) fn minimize_table_whitespace(output: &str) -> Option<String> {
    if looks_like_jsonl(output) {
        return None;
    }

    let mut changed = false;
    let lines = output
        .lines()
        .map(|line| {
            if line.matches('|').count() < 2 {
                return line.to_string();
            }
            let minimized = line.split('|').map(str::trim).collect::<Vec<_>>().join("|");
            if minimized != line {
                changed = true;
            }
            minimized
        })
        .collect::<Vec<_>>();
    changed.then(|| preserve_trailing_newline(output, lines.join("\n")))
}

pub(super) fn looks_like_jsonl(output: &str) -> bool {
    let output = output.trim();
    if output.is_empty() {
        return false;
    }
    if serde_json::from_str::<Value>(output).is_ok() {
        return true;
    }
    let lines = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let mut count = 0;
    for line in lines {
        if !(line.starts_with('{') || line.starts_with('[')) {
            return false;
        }
        if serde_json::from_str::<Value>(line).is_err() {
            return false;
        }
        count += 1;
    }
    count > 0
}

pub(super) fn fold_repeated_lines(output: &str) -> Option<String> {
    let lines = output.lines().collect::<Vec<_>>();
    if lines.len() < 3 {
        return None;
    }

    let mut result = Vec::new();
    let mut changed = false;
    let mut idx = 0usize;
    while idx < lines.len() {
        if idx + 3 < lines.len()
            && lines[idx] == lines[idx + 2]
            && lines[idx + 1] == lines[idx + 3]
            && lines[idx] != lines[idx + 1]
        {
            let first = lines[idx];
            let second = lines[idx + 1];
            let mut count = 2usize;
            let mut next = idx + 4;
            while next + 1 < lines.len() && lines[next] == first && lines[next + 1] == second {
                count += 1;
                next += 2;
            }
            result.push(format!("{count}x [{first}, {second}]"));
            changed = true;
            idx = next;
            continue;
        }

        if idx + 1 < lines.len() && lines[idx] == lines[idx + 1] {
            let line = lines[idx];
            let mut count = 2usize;
            let mut next = idx + 2;
            while next < lines.len() && lines[next] == line {
                count += 1;
                next += 1;
            }
            result.push(format!("{count}x {line}"));
            changed = true;
            idx = next;
            continue;
        }

        result.push(lines[idx].to_string());
        idx += 1;
    }

    changed.then(|| preserve_trailing_newline(output, result.join("\n")))
}

pub(super) fn normalize_whitespace(output: &str) -> Option<String> {
    let mut result = Vec::new();
    let mut blank_count = 0usize;
    let mut changed = false;
    for line in output.split('\n') {
        let trimmed = line.trim_end();
        if trimmed.len() != line.len() {
            changed = true;
        }
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                result.push(String::new());
            } else {
                changed = true;
            }
        } else {
            blank_count = 0;
            result.push(trimmed.to_string());
        }
    }
    changed.then(|| preserve_trailing_newline(output, result.join("\n")))
}

pub(super) fn normalize_dynamic_log_noise(output: &str) -> Option<String> {
    let timestamps = TIMESTAMP_RE.replace_all(output, "[timestamp]").to_string();
    let hashes = HASH_RE.replace_all(&timestamps, "[hash]").to_string();
    (hashes != output).then_some(hashes)
}

pub(super) fn json_array_to_table(output: &str) -> Option<String> {
    if output.len() < JSON_TABLE_MIN_BYTES {
        return None;
    }

    let parsed: Value = serde_json::from_str(output.trim()).ok()?;
    let Value::Array(items) = parsed else {
        return None;
    };
    if items.len() < 2 {
        return None;
    }

    let mut columns = Vec::<ColumnSpec>::new();
    for item in &items {
        let obj = item.as_object()?;
        for key in obj.keys() {
            if !columns.iter().any(|column| column.name == *key) {
                columns.push(ColumnSpec::new(key.clone()));
            }
        }
    }
    if columns.is_empty() {
        return None;
    }

    for item in &items {
        let obj = item.as_object()?;
        for column in &mut columns {
            match obj.get(&column.name) {
                Some(Value::Array(_) | Value::Object(_)) => return None,
                Some(Value::Null) | None => {
                    column.nullable = true;
                }
                Some(value) => column.observe(value)?,
            }
        }
    }

    let schema = columns
        .iter()
        .map(ColumnSpec::schema_entry)
        .collect::<Vec<_>>()
        .join(",");
    let mut lines = vec![format!("[{}]{{{schema}}}", items.len())];
    for item in &items {
        let obj = item.as_object()?;
        let row = columns
            .iter()
            .map(|column| match obj.get(&column.name) {
                Some(Value::Null) | None => String::new(),
                Some(value) => scalar_to_csv(value),
            })
            .collect::<Vec<_>>()
            .join(",");
        lines.push(row);
    }
    Some(lines.join("\n"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColumnSpec {
    name: String,
    kind: Option<ScalarKind>,
    nullable: bool,
}

impl ColumnSpec {
    fn new(name: String) -> Self {
        Self {
            name,
            kind: None,
            nullable: false,
        }
    }

    fn observe(&mut self, value: &Value) -> Option<()> {
        let next = ScalarKind::from_value(value)?;
        // Mixed-type columns abort the transform: an int 5 and a string "5"
        // would render identically, breaking the lossless guarantee.
        self.kind = Some(match self.kind {
            Some(existing) => existing.merge(next)?,
            None => next,
        });
        Some(())
    }

    fn schema_entry(&self) -> String {
        let suffix = if self.nullable { "?" } else { "" };
        format!(
            "{}:{}{}",
            self.name,
            self.kind.unwrap_or(ScalarKind::Null).as_str(),
            suffix
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    Null,
    Bool,
    Int,
    Float,
    String,
}

impl ScalarKind {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Null => Some(Self::Null),
            Value::Bool(_) => Some(Self::Bool),
            Value::Number(number) => {
                if number.is_i64() || number.is_u64() {
                    Some(Self::Int)
                } else {
                    Some(Self::Float)
                }
            }
            Value::String(_) => Some(Self::String),
            Value::Array(_) | Value::Object(_) => None,
        }
    }

    fn merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Null, kind) | (kind, Self::Null) => Some(kind),
            (Self::Int, Self::Float) | (Self::Float, Self::Int) => Some(Self::Float),
            (left, right) if left == right => Some(left),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool => "bool",
            Self::Int => "int",
            Self::Float => "float",
            Self::String => "string",
        }
    }
}

fn scalar_to_csv(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) if needs_csv_quote(value) => csv_quote(value),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => String::new(),
    }
}

fn needs_csv_quote(value: &str) -> bool {
    // Empty strings are quoted so a bare empty cell always means null/missing.
    value.is_empty()
        || value.contains(',')
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
}

fn csv_quote(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('"');
    for ch in value.chars() {
        if ch == '"' {
            quoted.push('"');
        }
        quoted.push(ch);
    }
    quoted.push('"');
    quoted
}
