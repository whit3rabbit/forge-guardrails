use super::is_noise_path;
use indexmap::IndexMap;

const MAX_RESULTS: usize = 100;

pub(in crate::tool_output) fn filter_glob_output(output: &str) -> String {
    let paths = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|path| !is_noise_path(path))
        .collect::<Vec<_>>();

    if paths.is_empty() {
        return "(no matches)".to_string();
    }

    if paths.len() <= MAX_RESULTS {
        return paths.join("\n");
    }

    let mut groups: IndexMap<String, Vec<&str>> = IndexMap::new();
    for path in paths {
        let top = path
            .split('/')
            .next()
            .filter(|part| !part.is_empty())
            .unwrap_or(".");
        groups.entry(top.to_string()).or_default().push(path);
    }

    let mut sorted = groups.into_iter().collect::<Vec<_>>();
    sorted.sort_by_key(|(_, paths)| std::cmp::Reverse(paths.len()));

    let total = sorted.iter().map(|(_, paths)| paths.len()).sum::<usize>();
    let mut result = format!("{total} files:\n");
    for (dir, files) in sorted.iter().take(20) {
        result.push_str(&format!("  {dir}/: {} files\n", files.len()));
        for path in files.iter().take(3) {
            result.push_str(&format!("    {path}\n"));
        }
        if files.len() > 3 {
            result.push_str(&format!("    ... and {} more\n", files.len() - 3));
        }
    }
    if sorted.len() > 20 {
        result.push_str(&format!("  ... and {} more directories", sorted.len() - 20));
    }
    result
}
