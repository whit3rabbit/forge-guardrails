use super::super::filters::is_noise_path;
use indexmap::IndexMap;

pub(super) fn filter_fs_output(command: &str, output: &str) -> String {
    let mut result = if command_starts_with(command, "ls") {
        filter_ls(output)
    } else if command_starts_with(command, "find") {
        filter_find(output)
    } else if command_starts_with(command, "tree") {
        filter_tree(output)
    } else if output.len() > 10_000 {
        format!(
            "{}\n... (truncated)",
            output.chars().take(5000).collect::<String>()
        )
    } else {
        output.to_string()
    };
    result = compress_paths(&result);
    result
}

fn command_starts_with(command: &str, name: &str) -> bool {
    let trimmed = command.trim_start();
    trimmed.starts_with(name) || trimmed.contains(&format!(" {name} "))
}

fn filter_ls(output: &str) -> String {
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        if is_noise_path(line) {
            continue;
        }
        if line.ends_with('/') {
            dirs.push(line.to_string());
        } else {
            files.push(line.to_string());
        }
    }

    let mut result = String::new();
    if !dirs.is_empty() {
        result.push_str(&format!(
            "Dirs ({}):\n{}\n",
            dirs.len(),
            dirs.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
        ));
        if dirs.len() > 30 {
            result.push_str(&format!("... and {} more\n", dirs.len() - 30));
        }
    }
    if !files.is_empty() {
        result.push_str(&format!(
            "Files ({}):\n{}\n",
            files.len(),
            files
                .iter()
                .take(50)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        ));
        if files.len() > 50 {
            result.push_str(&format!("... and {} more", files.len() - 50));
        }
    }
    if result.is_empty() {
        "(empty)".to_string()
    } else {
        result
    }
}

fn filter_find(output: &str) -> String {
    let filtered = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter(|line| !is_noise_path(line))
        .collect::<Vec<_>>();

    if filtered.len() <= 100 {
        return if filtered.is_empty() {
            "(empty)".to_string()
        } else {
            filtered.join("\n")
        };
    }

    let mut groups: IndexMap<String, usize> = IndexMap::new();
    for path in filtered {
        let top = path
            .split('/')
            .next()
            .filter(|part| !part.is_empty())
            .unwrap_or(".");
        *groups.entry(top.to_string()).or_insert(0) += 1;
    }
    let mut sorted = groups.into_iter().collect::<Vec<_>>();
    sorted.sort_by(|(_, left), (_, right)| right.cmp(left));

    let total = sorted.iter().map(|(_, count)| *count).sum::<usize>();
    let mut result = format!("{total} files:\n");
    for (dir, count) in sorted.iter().take(10) {
        result.push_str(&format!("  {dir}/: {count}\n"));
    }
    if sorted.len() > 10 {
        result.push_str(&format!("  ... and {} more directories", sorted.len() - 10));
    }
    result
}

fn filter_tree(output: &str) -> String {
    let lines = output.lines().collect::<Vec<_>>();
    let mut filtered = lines
        .iter()
        .copied()
        .filter(|line| {
            line.find(|ch: char| !ch.is_whitespace())
                .is_some_and(|idx| idx <= 6)
        })
        .collect::<Vec<_>>();

    if filtered.len() > 80 {
        let head = filtered[..40].join("\n");
        let tail = filtered[filtered.len() - 40..].join("\n");
        return format!(
            "{head}\n  ... {} entries omitted ...\n{tail}",
            filtered.len() - 80
        );
    }

    if let Some(summary) = lines
        .iter()
        .find(|line| line.contains(" directories") || line.contains(" files"))
    {
        filtered.push(summary);
    }

    if filtered.is_empty() {
        output.to_string()
    } else {
        filtered.join("\n")
    }
}

fn compress_paths(output: &str) -> String {
    let lines = output.lines().collect::<Vec<_>>();
    if lines.len() < 3 {
        return output.to_string();
    }

    let path_lines = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let path = line.trim();
            (path.contains('/') && !is_noise_path(path)).then_some((idx, path))
        })
        .collect::<Vec<_>>();
    if path_lines.len() < 3 {
        return output.to_string();
    }

    let paths = path_lines.iter().map(|(_, path)| *path).collect::<Vec<_>>();
    let Some(prefix) = common_dir_prefix(&paths) else {
        return output.to_string();
    };
    let suffixes = paths
        .iter()
        .map(|path| path[prefix.len()..].to_string())
        .collect::<Vec<_>>();
    if suffixes.len() < 3 {
        return output.to_string();
    }

    let original_len = suffixes
        .iter()
        .map(|suffix| prefix.len() + suffix.len())
        .sum::<usize>();
    let compressed_len = prefix.len() + 2 + suffixes.join(", ").len() + 2;
    if (compressed_len as f64) >= (original_len as f64 * 0.7) {
        return output.to_string();
    }

    let mut result = Vec::new();
    let mut path_idx = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        if path_idx < path_lines.len() && idx == path_lines[path_idx].0 {
            if path_idx == 0 {
                result.push(format!("{prefix}[{}]", suffixes.join(", ")));
            }
            path_idx += 1;
        } else {
            result.push((*line).to_string());
        }
    }
    result.join("\n")
}

fn common_dir_prefix(paths: &[&str]) -> Option<String> {
    let mut prefix = paths.first()?.to_string();
    for path in paths.iter().skip(1) {
        let mut end = 0usize;
        for ((idx, left), right) in prefix.char_indices().zip(path.chars()) {
            if left != right {
                break;
            }
            end = idx + left.len_utf8();
        }
        prefix.truncate(end);
        if prefix.is_empty() {
            return None;
        }
    }
    let slash = prefix.rfind('/')?;
    (slash > 0).then(|| prefix[..=slash].to_string())
}
