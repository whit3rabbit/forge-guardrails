const ERROR_PATTERNS: &[&str] = &[
    "error",
    "err!",
    "failed",
    "enoent",
    "eexist",
    "eacces",
    "eperm",
    "module_not_found",
    "syntaxerror",
    "typeerror",
];

pub(super) fn filter_npm_output(command: &str, output: &str) -> String {
    let lower_output = output.to_ascii_lowercase();
    if ERROR_PATTERNS
        .iter()
        .any(|pattern| lower_output.contains(pattern))
    {
        let lines = output.split('\n').collect::<Vec<_>>();
        let error_lines = lines
            .iter()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                ERROR_PATTERNS.iter().any(|pattern| lower.contains(pattern))
            })
            .copied()
            .collect::<Vec<_>>();
        if !error_lines.is_empty() && error_lines.len() * 2 < lines.len() {
            return error_lines.join("\n");
        }
    }

    if command.contains("install") || command.contains(" add") || command.contains(" remove") {
        return filter_npm_install(output);
    }
    if command.contains("test")
        || command.contains("jest")
        || command.contains("mocha")
        || command.contains("vitest")
    {
        return filter_npm_test(output);
    }
    if command.contains("lint") || command.contains("eslint") || command.contains("prettier") {
        return filter_npm_lint(output);
    }
    if output.len() > 10_000 {
        return format!(
            "{}\n... (truncated)",
            output.chars().take(5000).collect::<String>()
        );
    }
    output.to_string()
}

fn filter_npm_install(output: &str) -> String {
    let mut added = 0usize;
    let mut changed = 0usize;
    let mut removed = 0usize;
    let mut warnings = Vec::new();

    for line in output.lines() {
        if line.contains("added ") && line.contains("package") {
            added += 1;
        } else if line.contains("changed ") {
            changed += 1;
        } else if line.contains("removed ") {
            removed += 1;
        } else if line.contains("warn") || line.contains("deprecated") {
            warnings.push(line.trim().to_string());
        }
    }

    let mut result = String::new();
    if added > 0 {
        result.push_str(&format!("Added: {added} packages\n"));
    }
    if changed > 0 {
        result.push_str(&format!("Changed: {changed} packages\n"));
    }
    if removed > 0 {
        result.push_str(&format!("Removed: {removed} packages\n"));
    }
    if !warnings.is_empty() && warnings.len() <= 10 {
        result.push_str(&format!("Warnings:\n{}\n", warnings.join("\n")));
    } else if warnings.len() > 10 {
        result.push_str(&format!("Warnings: {} (truncated)\n", warnings.len()));
    }

    if result.is_empty() {
        output.to_string()
    } else {
        result
    }
}

fn filter_npm_test(output: &str) -> String {
    let mut failures = Vec::new();
    let mut summary = Vec::new();
    let mut in_failure = false;
    let mut failure_block = Vec::new();

    for line in output.lines() {
        let trimmed = line.trim();
        if starts_test_summary(trimmed) {
            summary.push(trimmed.to_string());
        }
        if contains_test_failure_line(line) {
            in_failure = true;
            failure_block.push(line.to_string());
        } else if in_failure && trimmed.is_empty() {
            in_failure = false;
            if !failure_block.is_empty() {
                failures.push(failure_block.join("\n"));
                failure_block.clear();
            }
        }
    }
    if !failure_block.is_empty() {
        failures.push(failure_block.join("\n"));
    }

    if failures.is_empty() {
        return if summary.is_empty() {
            "(all tests passed)".to_string()
        } else {
            summary.join("\n")
        };
    }

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
        result.push_str(&format!("\n\n... and {} more failures", failures.len() - 5));
    }
    if !summary.is_empty() {
        result.push_str("\n\n");
        result.push_str(&summary.join("\n"));
    }
    result
}

fn filter_npm_lint(output: &str) -> String {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for line in output.lines() {
        if line.contains("error") || line.contains("Error") {
            errors.push(line.trim().to_string());
        } else if line.contains("warn") || line.contains("Warning") {
            warnings.push(line.trim().to_string());
        }
    }

    let mut result = String::new();
    if !errors.is_empty() {
        result.push_str(&format!(
            "Errors ({}):\n{}\n",
            errors.len(),
            errors
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        ));
        if errors.len() > 20 {
            result.push_str(&format!("... and {} more\n", errors.len() - 20));
        }
    }
    if !warnings.is_empty() && warnings.len() <= 20 {
        result.push_str(&format!(
            "Warnings ({}):\n{}\n",
            warnings.len(),
            warnings.join("\n")
        ));
    } else if warnings.len() > 20 {
        result.push_str(&format!("Warnings: {} (truncated)\n", warnings.len()));
    }

    if result.is_empty() {
        "(clean)".to_string()
    } else {
        result
    }
}

fn starts_test_summary(line: &str) -> bool {
    ["pass", "fail", "PASS", "FAIL", "Tests:", "Test Suites:"]
        .iter()
        .any(|prefix| line.starts_with(prefix))
}

fn contains_test_failure_line(line: &str) -> bool {
    line.contains("FAIL")
        || line.contains("failed")
        || line.contains("Error:")
        || (line.contains("at ") && line.contains('(') && line.contains(':') && line.contains(')'))
}
