use super::preserve_unknown_or_empty_summary;
use serde_json::Value;

const ERROR_PATTERNS: &[&str] = &["error[", "error:", "warning[", "failed", "panicked"];

pub(in crate::tool_output) fn filter_cargo_output(command: &str, output: &str) -> String {
    if command.contains("test") {
        return filter_cargo_test(output);
    }
    if command.contains("clippy") || command.contains("build") || command.contains("check") {
        return filter_cargo_build(output);
    }
    if output.len() > 10_000 {
        return format!("{}\n... (truncated)", truncate_chars(output, 5000));
    }
    output.to_string()
}

fn filter_cargo_build(output: &str) -> String {
    if let Some(filtered) = filter_cargo_json_messages(output) {
        return filtered;
    }

    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut in_block = false;
    let mut block = Vec::new();

    for line in output.lines() {
        if line.starts_with("error[") || line.starts_with("warning[") {
            push_block(&mut errors, &mut warnings, &mut block);
            in_block = true;
            block = vec![line.to_string()];
        } else if in_block {
            block.push(line.to_string());
            if line.trim().is_empty() {
                push_block(&mut errors, &mut warnings, &mut block);
                in_block = false;
            }
        }

        if line.starts_with("error: could not compile") {
            errors.push(line.to_string());
        }
    }
    push_block(&mut errors, &mut warnings, &mut block);

    let result = format_cargo_diagnostics(&errors, &warnings);
    if result.is_empty() {
        preserve_unknown_or_empty_summary(output, "(compiled successfully)")
    } else {
        result
    }
}

fn filter_cargo_json_messages(output: &str) -> Option<String> {
    let mut saw_json = false;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    for line in output.lines() {
        let Ok(parsed) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        saw_json = true;
        if parsed.get("reason").and_then(Value::as_str) != Some("compiler-message") {
            continue;
        }
        let Some(message) = parsed.get("message") else {
            continue;
        };
        let level = message
            .get("level")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let rendered = message
            .get("rendered")
            .and_then(Value::as_str)
            .or_else(|| message.get("message").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        if rendered.trim().is_empty() {
            continue;
        }
        if level == "error" {
            errors.push(rendered);
        } else {
            warnings.push(rendered);
        }
    }

    let result = format_cargo_diagnostics(&errors, &warnings);
    if saw_json && !result.is_empty() {
        Some(result)
    } else {
        None
    }
}

fn format_cargo_diagnostics(errors: &[String], warnings: &[String]) -> String {
    let mut result = String::new();
    if !errors.is_empty() {
        result.push_str(&format!(
            "Errors ({}):\n{}\n",
            errors.len(),
            errors
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n")
        ));
        if errors.len() > 5 {
            result.push_str(&format!("... and {} more\n", errors.len() - 5));
        }
    }
    if !warnings.is_empty() && warnings.len() <= 10 {
        result.push_str(&format!(
            "Warnings ({}):\n{}\n",
            warnings.len(),
            warnings.join("\n")
        ));
    } else if warnings.len() > 10 {
        result.push_str(&format!("Warnings: {} (truncated)\n", warnings.len()));
    }

    if !result.is_empty() && !result.ends_with("\n\n") {
        result.push('\n');
    }
    result
}

fn push_block(errors: &mut Vec<String>, warnings: &mut Vec<String>, block: &mut Vec<String>) {
    if block.is_empty() {
        return;
    }
    let joined = block.join("\n");
    if joined.contains("error[") {
        errors.push(joined);
    } else {
        warnings.push(joined);
    }
    block.clear();
}

fn filter_cargo_test(output: &str) -> String {
    let mut failures = Vec::new();
    let mut summary = Vec::new();

    for line in output.lines() {
        if line.starts_with("test ") && line.ends_with("... FAILED") {
            failures.push(line.to_string());
        } else if line.starts_with("test result:") || starts_running_test_count(line) {
            summary.push(line.to_string());
        } else if contains_error_signal(line) {
            failures.push(line.to_string());
        }
    }

    if failures.is_empty() {
        return if summary.is_empty() {
            preserve_unknown_or_empty_summary(output, "(all tests passed)")
        } else {
            summary.join("\n")
        };
    }

    let mut result = format!("FAILURES ({}):\n", failures.len());
    result.push_str(
        &failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n"),
    );
    if failures.len() > 10 {
        result.push_str(&format!("\n... and {} more", failures.len() - 10));
    }
    if !summary.is_empty() {
        result.push('\n');
        result.push_str(&summary.join("\n"));
    }
    result
}

fn contains_error_signal(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    ERROR_PATTERNS.iter().any(|pattern| lower.contains(pattern))
        || lower.contains("thread") && lower.contains("panicked")
}

fn starts_running_test_count(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("running ") else {
        return false;
    };
    let Some((count, suffix)) = rest.split_once(' ') else {
        return false;
    };
    count.parse::<usize>().is_ok() && suffix.starts_with("test")
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
}
