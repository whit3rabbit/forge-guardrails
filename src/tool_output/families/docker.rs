use super::preserve_unknown_or_empty_summary;

pub(super) fn filter_docker_output(command: &str, output: &str) -> String {
    let trimmed = command.trim_start();
    if !(trimmed.starts_with("docker build")
        || trimmed.starts_with("docker pull")
        || trimmed.starts_with("docker push")
        || trimmed.starts_with("podman build")
        || trimmed.starts_with("podman pull")
        || trimmed.starts_with("podman push"))
    {
        return output.to_string();
    }

    let mut result = Vec::new();
    for line in output.lines() {
        if contains_issue(line) || contains_success(line) {
            result.push(line.to_string());
            continue;
        }
        if is_progress_line(line) {
            continue;
        }
        if !line.trim().is_empty() {
            result.push(line.to_string());
        }
    }

    if result.is_empty() {
        preserve_unknown_or_empty_summary(
            output,
            "(docker output compressed - all layers cached, no errors)",
        )
    } else {
        result.join("\n")
    }
}

fn contains_issue(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("error") || lower.contains("fatal") || lower.contains("failed")
}

fn contains_success(line: &str) -> bool {
    line.contains("Successfully built") || line.contains("Successfully tagged")
}

fn is_progress_line(line: &str) -> bool {
    line.starts_with('#')
        || line.contains("Downloading")
        || line.contains("Extracting")
        || line.contains("Waiting")
        || line.trim().is_empty()
}
