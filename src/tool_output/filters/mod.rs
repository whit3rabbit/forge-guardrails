mod glob;
mod grep;
mod read;

pub(in crate::tool_output) use glob::filter_glob_output;
pub(in crate::tool_output) use grep::filter_grep_output;
pub(in crate::tool_output) use read::filter_read_output;

const NOISE_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    "dist",
    "build",
    ".cache",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "target",
    "coverage",
    ".turbo",
    ".parcel-cache",
    "vendor",
];

pub(in crate::tool_output) fn is_noise_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    NOISE_DIRS.iter().any(|dir| {
        lower.contains(&format!("/{dir}/"))
            || lower.starts_with(&format!("{dir}/"))
            || lower == *dir
            || lower.contains(&format!("/{dir} "))
    }) || lower.contains(".min.")
}
