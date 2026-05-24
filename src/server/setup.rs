use std::path::Path;

use crate::context::{ContextManager, NoCompact};

use super::budget::BudgetMode;
use super::lifecycle::LifecycleOptions;
use super::manager::ServerManager;

/// Convenience factory that creates a ServerManager and ContextManager.
///
/// Enforces identity rules: ollama requires model name (no file path),
/// llamaserver/llamafile requires file path (no model name).
#[allow(clippy::too_many_arguments)]
pub fn setup_backend(
    backend: &str,
    model: Option<&str>,
    gguf_path: Option<&Path>,
    llamafile_runtime: Option<&Path>,
    budget_mode: BudgetMode,
    manual_tokens: Option<i64>,
    port: i64,
    mode: &str,
    extra_flags: &[String],
    cache_type_k: Option<&str>,
    cache_type_v: Option<&str>,
    n_slots: Option<i64>,
    kv_unified: bool,
) -> Result<(ServerManager, ContextManager), String> {
    // Identity validation.
    match backend {
        "ollama" => {
            let m = model.ok_or("ollama requires a model name")?;
            if gguf_path.is_some() {
                return Err("ollama does not accept a file path".to_string());
            }
            if llamafile_runtime.is_some() {
                return Err("ollama does not accept llamafile_runtime".to_string());
            }
            if Path::new(m).extension().is_some() || m.contains('/') || m.contains('\\') {
                return Err("ollama does not accept a file path".to_string());
            }
        }
        "llamaserver" => {
            if model.is_some() {
                return Err(format!(
                    "{} does not accept a model name; use gguf_path",
                    backend
                ));
            }
            if gguf_path.is_none() {
                return Err(format!("{} requires a file path (gguf_path)", backend));
            }
            if llamafile_runtime.is_some() {
                return Err("llamaserver does not accept llamafile_runtime".to_string());
            }
        }
        "llamafile" => {
            if model.is_some() {
                return Err(format!(
                    "{} does not accept a model name; use gguf_path",
                    backend
                ));
            }
            if gguf_path.is_none() {
                return Err(format!("{} requires a file path (gguf_path)", backend));
            }
            if llamafile_runtime.is_none() {
                return Err("llamafile requires llamafile_runtime".to_string());
            }
        }
        _ => return Err(format!("Unknown backend: {}", backend)),
    }

    let mut mgr = ServerManager::new(backend, port, None);
    if let Some(runtime) = llamafile_runtime {
        mgr = mgr.with_llamafile_runtime(runtime);
    }

    let identity = if backend == "ollama" {
        model.unwrap_or("")
    } else {
        gguf_path.and_then(|path| path.to_str()).unwrap_or("")
    };
    let gguf = gguf_path.unwrap_or_else(|| Path::new(""));
    let budget = mgr.start_with_budget_options(
        identity,
        gguf,
        mode,
        budget_mode,
        manual_tokens,
        extra_flags,
        cache_type_k,
        cache_type_v,
        n_slots,
        kv_unified,
        LifecycleOptions::default(),
    )?;

    let ctx_mgr = ContextManager::new(Box::new(NoCompact), budget, None, None, None);

    Ok((mgr, ctx_mgr))
}
