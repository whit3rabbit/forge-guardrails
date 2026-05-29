const ERROR_PATTERNS: &[&str] = &[
    "error:",
    "fatal:",
    "hint:",
    "warning:",
    "conflict",
    "merge conflict",
    "cannot",
    "failed",
    "rejected",
    "untracked files",
];

pub(super) fn filter_git_output(command: &str, output: &str) -> String {
    let lower = output.to_ascii_lowercase();
    if ERROR_PATTERNS.iter().any(|pattern| lower.contains(pattern)) {
        return output.to_string();
    }

    if command.contains("status") {
        return filter_git_status(output);
    }
    if command.contains("diff") {
        return filter_git_diff(output);
    }
    if command.contains("log") {
        return filter_git_log(output, 10);
    }
    if output.len() > 10_000 {
        return format!(
            "{}\n... (truncated)",
            &output[..safe_char_boundary(output, 5000)]
        );
    }
    output.to_string()
}

fn filter_git_status(output: &str) -> String {
    let mut changed = Vec::new();
    let mut untracked = Vec::new();
    let mut in_untracked = false;

    for line in output.lines() {
        if line.starts_with("Untracked files:") {
            in_untracked = true;
            continue;
        }
        if in_untracked {
            if line.trim().is_empty() || line.starts_with('\t') {
                if !line.trim().is_empty() {
                    untracked.push(line.trim().to_string());
                }
                continue;
            }
            in_untracked = false;
        }

        let trimmed = line.trim();
        if is_short_status_line(line) || is_long_status_line(trimmed) {
            changed.push(trimmed.to_string());
        }
    }

    let mut result = String::new();
    if !changed.is_empty() {
        result.push_str(&changed.join("\n"));
    }
    if !untracked.is_empty() && untracked.len() <= 20 {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&format!("Untracked: {}", untracked.join(", ")));
    } else if untracked.len() > 20 {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&format!("Untracked: {} files", untracked.len()));
    }

    if result.is_empty() {
        "(clean)".to_string()
    } else {
        result
    }
}

fn is_short_status_line(line: &str) -> bool {
    let mut chars = line.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let Some(second) = chars.next() else {
        return false;
    };
    let Some(third) = chars.next() else {
        return false;
    };
    matches!(first, 'A' | 'M' | 'D' | 'R' | 'C' | 'U' | '?' | '!' | ' ')
        && matches!(second, 'A' | 'M' | 'D' | 'R' | 'C' | 'U' | '?' | '!' | ' ')
        && third.is_whitespace()
}

fn is_long_status_line(trimmed: &str) -> bool {
    ["modified:", "added:", "deleted:", "renamed:", "copied:"]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

fn filter_git_diff(output: &str) -> String {
    let mut files = Vec::new();
    let mut hunks = Vec::new();

    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            if let Some((file, _)) = rest.split_once(" b/") {
                files.push(file.to_string());
            }
        } else if line.starts_with("@@") {
            hunks.push(line.to_string());
        } else if line.starts_with('+') || line.starts_with('-') {
            if line.len() > 120 {
                hunks.push(format!("{}...", &line[..safe_char_boundary(line, 120)]));
            } else {
                hunks.push(line.to_string());
            }
        }
    }

    if files.is_empty() {
        return output.to_string();
    }

    let mut result = format!("Files changed: {}\n", files.len());
    result.push_str(
        &files
            .iter()
            .map(|file| format!("  {file}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    if !hunks.is_empty() && hunks.len() <= 50 {
        result.push_str("\n\n");
        result.push_str(&hunks.join("\n"));
    } else if hunks.len() > 50 {
        result.push_str(&format!("\n\n... {} hunk headers (truncated)", hunks.len()));
    }
    result
}

fn filter_git_log(output: &str, max_entries: usize) -> String {
    let mut commits = Vec::new();
    let mut current: Option<String> = None;

    for line in output.lines() {
        if let Some(hash) = line.strip_prefix("commit ") {
            if let Some(commit) = current.take() {
                commits.push(commit);
            }
            if commits.len() >= max_entries {
                break;
            }
            current = Some(format!("commit {}", &hash[..hash.len().min(5)]));
        } else if line.starts_with("Author:") || line.starts_with("Date:") {
        } else if line.starts_with("Merge:") {
            if let Some(commit) = &mut current {
                commit.push_str(" (merge)");
            }
        } else if !line.trim().is_empty() {
            if let Some(commit) = &mut current {
                commit.push(' ');
                commit.push_str(line.trim());
            }
        }
    }
    if let Some(commit) = current {
        if commits.len() < max_entries {
            commits.push(commit);
        }
    }

    if commits.is_empty() {
        "(empty)".to_string()
    } else {
        commits.join("\n")
    }
}

fn safe_char_boundary(value: &str, limit: usize) -> usize {
    if value.len() <= limit {
        return value.len();
    }
    value
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|idx| *idx <= limit)
        .last()
        .unwrap_or(0)
}
