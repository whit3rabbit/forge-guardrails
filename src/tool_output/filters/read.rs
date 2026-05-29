const SOURCE_EXTENSIONS: &[&str] = &[
    ".ts", ".tsx", ".js", ".jsx", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".rb",
    ".swift", ".kt", ".scala",
];
const CONFIG_EXTENSIONS: &[&str] = &[
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg", ".env", ".conf", ".lock",
];
const DOC_EXTENSIONS: &[&str] = &[".md", ".mdx", ".rst", ".txt"];
const MAX_LINES_PASS: usize = 80;

pub(in crate::tool_output) fn filter_read_output(file_path: &str, content: &str) -> String {
    let lines = content.lines().count();
    if lines <= MAX_LINES_PASS {
        return content.to_string();
    }

    let ext = extension(file_path);
    if CONFIG_EXTENSIONS.contains(&ext.as_str()) {
        return content.to_string();
    }
    if DOC_EXTENSIONS.contains(&ext.as_str()) {
        return filter_markdown(content);
    }
    if SOURCE_EXTENSIONS.contains(&ext.as_str()) {
        return outline_source(file_path, content);
    }
    generic_outline(content)
}

fn extension(path: &str) -> String {
    path.rsplit_once('.')
        .map(|(_, ext)| format!(".{}", ext.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn filter_markdown(content: &str) -> String {
    let mut result = Vec::new();
    let mut in_code_block = false;
    let mut code_block_lines = 0usize;

    for line in content.lines() {
        if line.starts_with("```") {
            if in_code_block {
                result.push(format!("``` [{code_block_lines} lines]"));
                code_block_lines = 0;
                in_code_block = false;
            } else {
                in_code_block = true;
                code_block_lines = 0;
            }
            continue;
        }
        if in_code_block {
            code_block_lines += 1;
            continue;
        }
        if line.starts_with('#') {
            result.push(line.to_string());
        }
    }

    if result.is_empty() {
        content.to_string()
    } else {
        result.join("\n")
    }
}

fn outline_source(file_path: &str, content: &str) -> String {
    let ext = extension(file_path);
    let total_lines = content.split('\n').count();
    let mut symbols = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim_start();
        if let Some(symbol) = symbol_text(&ext, trimmed) {
            symbols.push((idx + 1, symbol));
        }
    }

    let mut result = format!(
        "// {file_path} ({total_lines} lines, {} symbols)\n",
        symbols.len()
    );
    for (line_no, symbol) in symbols.iter().take(50) {
        result.push_str(&format!("  L{line_no}: {symbol}\n"));
    }
    if symbols.len() > 50 {
        result.push_str(&format!("  ... and {} more symbols\n", symbols.len() - 50));
    }
    result.push_str("\n// Use read with line range to see specific sections");
    result
}

fn symbol_text(ext: &str, trimmed: &str) -> Option<String> {
    match ext {
        ".ts" | ".tsx" | ".js" | ".jsx" => starts_with_any(
            trimmed,
            &[
                "export function ",
                "export async function ",
                "function ",
                "async function ",
                "export class ",
                "class ",
                "export interface ",
                "interface ",
                "export type ",
                "type ",
                "export enum ",
                "enum ",
                "export const ",
                "const ",
                "let ",
                "var ",
            ],
        )
        .then(|| prefix_symbol(trimmed)),
        ".py" => starts_with_any(trimmed, &["def ", "async def ", "class "])
            .then(|| prefix_symbol(trimmed)),
        ".rs" => starts_with_any(
            trimmed,
            &[
                "pub fn ",
                "pub async fn ",
                "fn ",
                "async fn ",
                "pub struct ",
                "struct ",
                "pub enum ",
                "enum ",
                "pub trait ",
                "trait ",
                "impl ",
                "pub mod ",
                "mod ",
                "use ",
                "pub use ",
                "type ",
                "pub type ",
                "const ",
                "pub const ",
                "static ",
                "pub static ",
            ],
        )
        .then(|| prefix_symbol(trimmed)),
        ".go" => starts_with_any(trimmed, &["func ", "type ", "var ", "const "])
            .then(|| prefix_symbol(trimmed)),
        ".java" => (trimmed.contains(" class ")
            || trimmed.contains(" interface ")
            || trimmed.contains(" enum ")
            || starts_with_any(
                trimmed,
                &[
                    "class ",
                    "interface ",
                    "enum ",
                    "public ",
                    "private ",
                    "protected ",
                ],
            ))
        .then(|| prefix_symbol(trimmed)),
        _ => starts_with_any(
            trimmed,
            &[
                "pub ", "fn ", "struct ", "enum ", "impl ", "class ", "def ", "import ", "export ",
            ],
        )
        .then(|| prefix_symbol(trimmed)),
    }
}

fn starts_with_any(value: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| value.starts_with(prefix))
}

fn prefix_symbol(trimmed: &str) -> String {
    let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
    let Some(kind_idx) = tokens.iter().position(|token| {
        matches!(
            *token,
            "fn" | "function"
                | "class"
                | "struct"
                | "enum"
                | "trait"
                | "impl"
                | "mod"
                | "use"
                | "type"
                | "const"
                | "static"
                | "def"
                | "interface"
                | "var"
                | "let"
        )
    }) else {
        return trimmed.to_string();
    };
    let Some(raw_name) = tokens.get(kind_idx + 1) else {
        return tokens[..=kind_idx].join(" ");
    };
    let name = raw_name
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>();
    if name.is_empty() {
        tokens[..=kind_idx].join(" ")
    } else {
        format!("{} {name}", tokens[..=kind_idx].join(" "))
    }
}

fn generic_outline(content: &str) -> String {
    let lines = content.lines().collect::<Vec<_>>();
    if lines.len() <= 30 {
        return content.to_string();
    }
    let mut result = lines[..20].join("\n");
    result.push_str(&format!(
        "\n\n... {} lines omitted ...\n\n",
        lines.len().saturating_sub(30)
    ));
    result.push_str(&lines[lines.len().saturating_sub(10)..].join("\n"));
    result
}
