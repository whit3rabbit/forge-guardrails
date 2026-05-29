use super::preserve_unknown_or_empty_summary;

const ERROR_PATTERNS: &[&str] = &[
    "failed",
    "error",
    "traceback",
    "assertionerror",
    "typeerror",
    "syntaxerror",
    "panic:",
    "fatal:",
];

pub(super) fn filter_test_output(command: &str, output: &str) -> String {
    if command.contains("pytest") || command.contains("py.test") {
        return filter_pytest(output);
    }
    if command.contains("go test") {
        return filter_go_test(output);
    }

    let mut error_lines = Vec::new();
    let mut summary_lines = Vec::new();
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if ERROR_PATTERNS.iter().any(|pattern| lower.contains(pattern)) {
            error_lines.push(line.to_string());
        }
        if contains_summary_signal(&lower) {
            summary_lines.push(line.to_string());
        }
    }

    if error_lines.is_empty() {
        if summary_lines.is_empty() {
            preserve_unknown_or_empty_summary(output, "(all tests passed)")
        } else {
            summary_lines.join("\n")
        }
    } else {
        error_lines
            .into_iter()
            .take(30)
            .chain(summary_lines)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn filter_pytest(output: &str) -> String {
    let mut failures = Vec::new();
    let mut summary = Vec::new();
    let mut in_failure = false;
    let mut failure_lines = Vec::new();

    for line in output.lines() {
        if is_pytest_failure_header(line) {
            if in_failure && !failure_lines.is_empty() {
                failures.push(failure_lines.join("\n"));
            }
            in_failure = true;
            failure_lines = vec![line.to_string()];
        } else if in_failure {
            failure_lines.push(line.to_string());
            if line.starts_with('=') || line.trim().is_empty() {
                in_failure = false;
                failures.push(failure_lines.join("\n"));
                failure_lines.clear();
            }
        }

        if starts_with_count_summary(line)
            || line.starts_with("FAILED ")
            || line.starts_with("ERROR ")
        {
            summary.push(line.to_string());
        }
    }
    if in_failure && !failure_lines.is_empty() {
        failures.push(failure_lines.join("\n"));
    }

    if failures.is_empty() {
        if summary.is_empty() {
            preserve_unknown_or_empty_summary(output, "(all tests passed)")
        } else {
            summary.join("\n")
        }
    } else {
        let mut result = format!("FAILURES ({}):\n\n", failures.len());
        result.push_str(
            &failures
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
        if failures.len() > 5 {
            result.push_str(&format!("\n\n... and {} more", failures.len() - 5));
        }
        if !summary.is_empty() {
            result.push_str("\n\n");
            result.push_str(&summary.join("\n"));
        }
        result
    }
}

fn filter_go_test(output: &str) -> String {
    let mut failures = Vec::new();
    let mut summary = Vec::new();
    for line in output.lines() {
        if line.starts_with("--- FAIL:") || line.starts_with("panic:") {
            failures.push(line.to_string());
        } else if line.starts_with("ok ") || line.starts_with("FAIL ") {
            summary.push(line.to_string());
        }
    }

    if failures.is_empty() {
        if summary.is_empty() {
            preserve_unknown_or_empty_summary(output, "(all tests passed)")
        } else {
            summary.join("\n")
        }
    } else {
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
}

fn is_pytest_failure_header(line: &str) -> bool {
    let trimmed = line.trim_start_matches('_');
    line.starts_with("___") && trimmed.starts_with(' ')
}

fn starts_with_count_summary(line: &str) -> bool {
    let Some((count, rest)) = line.split_once(' ') else {
        return false;
    };
    count.parse::<usize>().is_ok()
        && ["passed", "failed", "error", "skipped", "warning"]
            .iter()
            .any(|prefix| rest.starts_with(prefix))
}

fn contains_summary_signal(lower: &str) -> bool {
    [
        "passed",
        "failed",
        "error",
        "skipped",
        "test suites",
        "tests:",
        "time:",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}
