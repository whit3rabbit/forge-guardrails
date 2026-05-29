const MAX_LINES: usize = 80;
const MAX_BYTES: usize = 20 * 1024;
const HEAD_LINES: usize = 20;
const TAIL_LINES: usize = 20;

pub(in crate::tool_output) fn filter_generic_output(output: &str) -> String {
    filter_generic_output_with_limit(output, MAX_BYTES)
}

pub(in crate::tool_output) fn filter_generic_output_with_limit(
    output: &str,
    max_bytes: usize,
) -> String {
    let lines = output.lines().collect::<Vec<_>>();
    let stack_frames = lines.iter().filter(|line| is_stack_frame(line)).count();
    if stack_frames > 5 {
        return preserve_trailing_newline(output, compress_stack_trace(&lines));
    }

    let byte_limit = max_bytes.min(MAX_BYTES);
    if lines.len() <= MAX_LINES && output.len() <= byte_limit {
        return output.to_string();
    }

    if lines.len() <= HEAD_LINES + TAIL_LINES {
        return output.to_string();
    }
    let head = lines[..HEAD_LINES].join("\n");
    let tail = lines[lines.len() - TAIL_LINES..].join("\n");
    format!(
        "{head}\n\n... {} lines omitted ...\n\n{tail}",
        lines.len() - HEAD_LINES - TAIL_LINES
    )
}

fn compress_stack_trace(lines: &[&str]) -> String {
    let mut stack_start = None;
    let mut stack_end = None;
    for (idx, line) in lines.iter().enumerate() {
        if is_stack_frame(line) {
            stack_start.get_or_insert(idx);
            stack_end = Some(idx);
        }
    }

    let (Some(start), Some(end)) = (stack_start, stack_end) else {
        return lines.join("\n");
    };
    if end.saturating_sub(start) < 4 {
        return lines.join("\n");
    }

    let mut result = Vec::new();
    result.extend(lines[..start].iter().map(|line| (*line).to_string()));
    result.push(lines[start].to_string());
    let middle = end - start - 1;
    if middle > 0 {
        result.push(format!("  ... {middle} stack frames omitted ..."));
    }
    result.push(lines[end].to_string());
    result.extend(lines[end + 1..].iter().map(|line| (*line).to_string()));
    result.join("\n")
}

fn is_stack_frame(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("at ") && trimmed.contains('(') && trimmed.contains(')')
}

fn preserve_trailing_newline(original: &str, mut value: String) -> String {
    if original.ends_with('\n') && !value.ends_with('\n') {
        value.push('\n');
    }
    value
}
