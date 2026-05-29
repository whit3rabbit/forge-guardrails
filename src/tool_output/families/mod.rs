mod cargo;
mod docker;
mod fs;
mod generic;
mod git;
mod make;
mod npm;
mod pip;
mod test;

use super::filters::{filter_glob_output, filter_grep_output, filter_read_output};

pub(super) use generic::filter_generic_output;

pub(super) fn preserve_unknown_or_empty_summary(output: &str, empty_summary: &str) -> String {
    if output.trim().is_empty() {
        empty_summary.to_string()
    } else {
        output.to_string()
    }
}

pub(super) fn filter_bash_output(
    family: &str,
    command: &str,
    output: &str,
    max_output_bytes: usize,
) -> String {
    if is_grep_command(command) {
        return filter_grep_output(output);
    }
    match family {
        "git" => git::filter_git_output(command, output),
        "cargo" => cargo::filter_cargo_output(command, output),
        "npm" => npm::filter_npm_output(command, output),
        "test" => test::filter_test_output(command, output),
        "docker" => docker::filter_docker_output(command, output),
        "pip" => pip::filter_pip_output(command, output),
        "make" => make::filter_make_output(output),
        "fs" | "glob" => fs::filter_fs_output(command, output),
        "grep" => filter_grep_output(output),
        "read" => filter_bash_read(command, output),
        _ => generic::filter_generic_output_with_limit(output, max_output_bytes / 2),
    }
}

fn is_grep_command(command: &str) -> bool {
    command
        .split_whitespace()
        .any(|token| matches!(token, "grep" | "rg" | "ag" | "ack"))
}

fn filter_bash_read(command: &str, output: &str) -> String {
    let path = parse_plain_cat_path(command).unwrap_or(command);
    if command.trim_start().starts_with("ls") || command.trim_start().starts_with("find") {
        return filter_glob_output(output);
    }
    filter_read_output(path, output)
}

fn parse_plain_cat_path(command: &str) -> Option<&str> {
    let trimmed = command.trim();
    let rest = trimmed.strip_prefix("cat ")?;
    if rest.starts_with('-')
        || rest.contains('|')
        || rest.contains('&')
        || rest.contains(';')
        || rest.contains('>')
        || rest.contains('<')
        || rest.contains('*')
        || rest.contains('"')
        || rest.split_whitespace().count() != 1
    {
        return None;
    }
    Some(rest)
}
