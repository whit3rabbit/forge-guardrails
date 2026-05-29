pub(super) fn filter_pip_output(command: &str, output: &str) -> String {
    let trimmed = command.trim_start();
    if !(trimmed.starts_with("pip install") || trimmed.starts_with("pip3 install")) {
        return output.to_string();
    }

    let mut result = Vec::new();
    let mut satisfied_count = 0usize;
    for line in output.lines() {
        if is_issue(line) || is_success(line) {
            flush_satisfied(&mut result, &mut satisfied_count);
            result.push(line.to_string());
            continue;
        }
        if line.starts_with("Requirement already satisfied:") {
            satisfied_count += 1;
            continue;
        }
        if line.starts_with("Collecting") || line.starts_with("  Downloading") {
            continue;
        }
        flush_satisfied(&mut result, &mut satisfied_count);
        if !line.trim().is_empty() {
            result.push(line.to_string());
        }
    }
    flush_satisfied(&mut result, &mut satisfied_count);

    if result.is_empty() {
        "(pip install output compressed - all requirements already satisfied)".to_string()
    } else {
        result.join("\n")
    }
}

fn flush_satisfied(result: &mut Vec<String>, satisfied_count: &mut usize) {
    if *satisfied_count > 0 {
        result.push(format!(
            "  ... {satisfied_count} requirements already satisfied ..."
        ));
        *satisfied_count = 0;
    }
}

fn is_issue(line: &str) -> bool {
    let upper = line.to_ascii_uppercase();
    upper.starts_with("ERROR") || upper.starts_with("WARNING") || upper.starts_with("FAILED")
}

fn is_success(line: &str) -> bool {
    line.starts_with("Successfully installed") || line.starts_with("Successfully uninstalled")
}
