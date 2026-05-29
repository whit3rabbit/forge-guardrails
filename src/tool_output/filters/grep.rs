use super::is_noise_path;
use indexmap::IndexMap;
use serde_json::Value;

const MAX_MATCHES_PER_FILE: usize = 10;

#[derive(Debug)]
struct GrepMatch {
    file: String,
    line_num: String,
    content: String,
}

pub(in crate::tool_output) fn filter_grep_output(output: &str) -> String {
    let lines = output.lines().collect::<Vec<_>>();
    let mut matches: IndexMap<String, Vec<String>> = IndexMap::new();
    let is_rg_json = lines
        .first()
        .and_then(|line| parse_rg_json_line(line))
        .is_some();

    if is_rg_json {
        for line in lines {
            let Some(parsed) = parse_rg_json_line(line) else {
                continue;
            };
            if is_noise_path(&parsed.file) {
                continue;
            }
            push_match(&mut matches, parsed.file, parsed.line_num, parsed.content);
        }
    } else {
        for line in lines {
            if is_noise_path(line) {
                continue;
            }
            let Some(parsed) = parse_grep_line(line) else {
                continue;
            };
            push_match(&mut matches, parsed.file, parsed.line_num, parsed.content);
        }
    }

    if matches.is_empty() {
        return "(no matches)".to_string();
    }

    let match_count = matches.values().map(Vec::len).sum::<usize>();
    let mut result = format!("{} files, {match_count} matches:\n\n", matches.len());
    for (file, file_matches) in matches {
        result.push_str(&format!("{file}:\n"));
        for line in &file_matches {
            result.push_str(&format!("  {line}\n"));
        }
        if file_matches.len() >= MAX_MATCHES_PER_FILE {
            result.push_str("  ... more matches\n");
        }
        result.push('\n');
    }
    result.trim_end().to_string()
}

fn push_match(
    matches: &mut IndexMap<String, Vec<String>>,
    file: String,
    line_num: String,
    content: String,
) {
    let file_matches = matches.entry(file).or_default();
    if file_matches.len() < MAX_MATCHES_PER_FILE {
        file_matches.push(format!("{line_num}: {}", content.trim()));
    }
}

fn parse_rg_json_line(line: &str) -> Option<GrepMatch> {
    let parsed: Value = serde_json::from_str(line).ok()?;
    if parsed.get("type")?.as_str()? != "match" {
        return None;
    }
    let data = parsed.get("data")?;
    let file = data.get("path")?.get("text")?.as_str()?.to_string();
    let line_num = data.get("line_number")?.as_i64()?.to_string();
    let content = data
        .get("lines")
        .and_then(|lines| lines.get("text"))
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_string)
        .or_else(|| {
            data.get("submatches")
                .and_then(Value::as_array)
                .map(|matches| {
                    matches
                        .iter()
                        .filter_map(|item| item.get("match")?.get("text")?.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
        })
        .unwrap_or_default();
    Some(GrepMatch {
        file,
        line_num,
        content,
    })
}

fn parse_grep_line(line: &str) -> Option<GrepMatch> {
    let (file, rest) = line.split_once(':')?;
    let (line_num, rest) = rest.split_once(':')?;
    if line_num.parse::<usize>().is_err() {
        return None;
    }
    let content = if let Some((column, content)) = rest.split_once(':') {
        if column.parse::<usize>().is_ok() {
            content
        } else {
            rest
        }
    } else {
        rest
    };
    Some(GrepMatch {
        file: file.to_string(),
        line_num: line_num.to_string(),
        content: content.to_string(),
    })
}
