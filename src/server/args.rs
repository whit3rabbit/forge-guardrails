use std::path::Path;

const ALLOWED_EXTRA_FLAGS_WITH_VALUES: &[&str] = &[
    "--reasoning-budget",
    "--reasoning-format",
    "--chat-template",
    "--chat-template-file",
];

#[allow(clippy::too_many_arguments)]
pub(super) fn build_backend_args(
    backend: &str,
    port: i64,
    gguf_path: &Path,
    mode: &str,
    extra_flags: &[String],
    ctx_override: Option<i64>,
    cache_type_k: Option<&str>,
    cache_type_v: Option<&str>,
    n_slots: Option<i64>,
    kv_unified: bool,
) -> Result<Vec<String>, String> {
    let extra_flags = validate_backend_extra_flags(extra_flags)?;
    let mut args = vec![
        "-m".to_string(),
        gguf_path.to_string_lossy().to_string(),
        "--port".to_string(),
        port.to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "-ngl".to_string(),
        "999".to_string(),
    ];

    if backend == "llamaserver" && mode == "native" {
        args.push("--jinja".to_string());
    }

    if let Some(ctx) = ctx_override {
        args.push("-c".to_string());
        args.push(ctx.to_string());
    }

    if let Some(ck) = cache_type_k {
        args.push("--cache-type-k".to_string());
        args.push(ck.to_string());
    }
    if let Some(cv) = cache_type_v {
        args.push("--cache-type-v".to_string());
        args.push(cv.to_string());
    }
    if let Some(slots) = n_slots {
        args.push("--parallel".to_string());
        args.push(slots.to_string());
    }
    if kv_unified {
        args.push("--kv-unified".to_string());
    }

    args.extend(extra_flags);
    Ok(args)
}

pub(super) fn validate_backend_extra_flags(extra_flags: &[String]) -> Result<Vec<String>, String> {
    let mut normalized = Vec::with_capacity(extra_flags.len());
    let mut index = 0;
    while index < extra_flags.len() {
        let token = &extra_flags[index];
        if token == "--" {
            return Err("--extra-flags cannot contain an internal -- separator".to_string());
        }
        if !token.starts_with('-') {
            return Err(format!("unsupported backend extra flag value: {token}"));
        }

        let (flag, inline_value) = match token.split_once('=') {
            Some((flag, value)) => (flag, Some(value)),
            None => (token.as_str(), None),
        };
        if !ALLOWED_EXTRA_FLAGS_WITH_VALUES.contains(&flag) {
            return Err(format!("unsupported backend extra flag: {flag}"));
        }

        let value = match inline_value {
            Some("") => {
                return Err(format!("{flag} requires a non-empty value"));
            }
            Some(value) => value.to_string(),
            None => {
                index += 1;
                let Some(value) = extra_flags.get(index) else {
                    return Err(format!("{flag} requires a value"));
                };
                if value == "--" || value.starts_with("--") {
                    return Err(format!("{flag} requires a value"));
                }
                value.clone()
            }
        };

        normalized.push(flag.to_string());
        normalized.push(value);
        index += 1;
    }
    Ok(normalized)
}
