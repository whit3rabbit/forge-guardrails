use std::path::Path;

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
) -> Vec<String> {
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

    args.extend(extra_flags.iter().cloned());
    args
}
