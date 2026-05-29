use super::preserve_unknown_or_empty_summary;

pub(super) fn filter_make_output(output: &str) -> String {
    let lines = output.lines().collect::<Vec<_>>();
    let mut result = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        if !is_progress_line(line) {
            result.push((*line).to_string());
            continue;
        }
        if contains_issue(line) {
            result.push((*line).to_string());
            continue;
        }
        let prev = idx
            .checked_sub(1)
            .and_then(|prev| lines.get(prev))
            .copied()
            .unwrap_or("");
        let next = lines.get(idx + 1).copied().unwrap_or("");
        if contains_issue(prev) || contains_issue(next) {
            result.push((*line).to_string());
        }
    }

    let mut output_value = if result.is_empty() {
        preserve_unknown_or_empty_summary(
            output,
            "(build output compressed - no warnings or errors)",
        )
    } else {
        result.join("\n")
    };
    if output.ends_with('\n') && !output_value.ends_with('\n') {
        output_value.push('\n');
    }
    output_value
}

fn is_progress_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('[') {
        return false;
    }
    let Some(close) = trimmed.find(']') else {
        return false;
    };
    trimmed[1..close]
        .chars()
        .all(|ch| ch.is_ascii_digit() || ch == '%' || ch == '.' || ch.is_whitespace())
}

fn contains_issue(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("warning") || lower.contains("error")
}
