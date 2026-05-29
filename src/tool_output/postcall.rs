use super::preserve_trailing_newline;
use regex_lite::Regex;
use serde_json::Value;
use std::sync::LazyLock;

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

fn looks_like_jsonl(output: &str) -> bool {
    let Some(first) = output.lines().map(str::trim).find(|line| !line.is_empty()) else {
        return false;
    };
    if !(first.starts_with('{') || first.starts_with('[')) {
        return false;
    }

    for line in output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if !(line.starts_with('{') || line.starts_with('[')) {
            return true;
        }
        if serde_json::from_str::<Value>(line).is_err() {
            return true;
        }
    }
    true
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
    let parsed: Value = serde_json::from_str(output.trim()).ok()?;
    let Value::Array(items) = parsed else {
        return None;
    };
    if items.len() < 2 {
        return None;
    }
    let first = items.first()?.as_object()?;
    let keys: Vec<&String> = first.keys().collect();
    if keys.is_empty() {
        return None;
    }
    let mut rows = Vec::new();
    for item in &items {
        let obj = item.as_object()?;
        if obj.len() != keys.len() || !keys.iter().all(|key| obj.contains_key(*key)) {
            return None;
        }
        let mut row = Vec::new();
        for key in &keys {
            let value = obj.get(*key)?;
            if matches!(value, Value::Array(_) | Value::Object(_)) {
                return None;
            }
            row.push(match value {
                Value::String(value) => value.clone(),
                _ => value.to_string(),
            });
        }
        rows.push(row.join("\t"));
    }
    let header = keys
        .iter()
        .map(|key| key.as_str())
        .collect::<Vec<_>>()
        .join("\t");
    Some(format!(
        "[{} rows]\n{}\n{}",
        items.len(),
        header,
        rows.join("\n")
    ))
}
