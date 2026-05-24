//! Backend lifecycle manager: process spawning, budget resolution, VRAM tiers.
//!
//! ServerManager controls llama-server/llamafile/Ollama backend processes.
//! BudgetMode controls how context window size is determined.

use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::Mutex;

use crate::context::{detect_hardware, ContextManager, NoCompact};
use crate::error::{BackendError, BudgetResolutionError};

/// Budget resolution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetMode {
    /// Use backend-reported context size.
    Backend,
    /// User-specified token count.
    Manual,
    /// Maximum available context.
    ForgeFull,
    /// Half available context (two-phase start for llamaserver).
    ForgeFast,
}

impl BudgetMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Backend => "backend",
            Self::Manual => "manual",
            Self::ForgeFull => "forge-full",
            Self::ForgeFast => "forge-fast",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "backend" => Some(Self::Backend),
            "manual" => Some(Self::Manual),
            "forge-full" => Some(Self::ForgeFull),
            "forge-fast" => Some(Self::ForgeFast),
            _ => None,
        }
    }
}

impl std::str::FromStr for BudgetMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "backend" => Ok(Self::Backend),
            "manual" => Ok(Self::Manual),
            "forge-full" => Ok(Self::ForgeFull),
            "forge-fast" => Ok(Self::ForgeFast),
            _ => Err(()),
        }
    }
}

/// Tracked configuration for reuse decisions during start().
#[derive(Debug, Clone, PartialEq)]
struct RunConfig {
    model: String,
    mode: String,
    ctx_override: Option<i64>,
    extra_flags: Vec<String>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    n_slots: Option<i64>,
    kv_unified: bool,
}

/// Backend process lifecycle manager.
///
/// Handles subprocess spawning for llamaserver/llamafile, budget resolution,
/// VRAM tier detection, and health polling.
pub struct ServerManager {
    backend: String,
    port: i64,
    _models_dir: Option<PathBuf>,
    process: Mutex<Option<Child>>,
    current_config: Mutex<Option<RunConfig>>,
    last_context: Mutex<Option<i64>>,
}

impl ServerManager {
    pub fn new(backend: &str, port: i64, models_dir: Option<&Path>) -> Self {
        Self {
            backend: backend.to_string(),
            port,
            _models_dir: models_dir.map(|p| p.to_path_buf()),
            process: Mutex::new(None),
            current_config: Mutex::new(None),
            last_context: Mutex::new(None),
        }
    }

    /// Start the backend process. No-op for ollama.
    /// For llamaserver/llamafile, spawns a subprocess with the given config.
    /// Returns true if a new process was spawned, false if reused.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &self,
        model: &str,
        gguf_path: &Path,
        mode: &str,
        extra_flags: &[String],
        ctx_override: Option<i64>,
        cache_type_k: Option<&str>,
        cache_type_v: Option<&str>,
        n_slots: Option<i64>,
        kv_unified: bool,
    ) -> Result<bool, BackendError> {
        let new_config = RunConfig {
            model: model.to_string(),
            mode: mode.to_string(),
            ctx_override,
            extra_flags: extra_flags.to_vec(),
            cache_type_k: cache_type_k.map(|s| s.to_string()),
            cache_type_v: cache_type_v.map(|s| s.to_string()),
            n_slots,
            kv_unified,
        };

        // Check if we can reuse the running process.
        let can_reuse = {
            let guard = self
                .current_config
                .lock()
                .map_err(|e| BackendError::new(0, e.to_string()))?;
            guard.as_ref() == Some(&new_config)
        };
        if can_reuse {
            return Ok(false);
        }

        // Stop existing if running.
        self.stop()?;

        if self.backend == "ollama" {
            let mut guard = self
                .current_config
                .lock()
                .map_err(|e| BackendError::new(0, e.to_string()))?;
            *guard = Some(new_config);
            return Ok(false);
        }

        let binary = if self.backend == "llamafile" {
            self.find_llamafile_binary(gguf_path)?
        } else {
            "llama-server".to_string()
        };

        let args = build_backend_args(
            &self.backend,
            self.port,
            gguf_path,
            mode,
            extra_flags,
            ctx_override,
            cache_type_k,
            cache_type_v,
            n_slots,
            kv_unified,
        );
        let mut cmd = std::process::Command::new(&binary);
        cmd.args(&args);

        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        let child = cmd
            .spawn()
            .map_err(|e| BackendError::new(0, format!("Failed to start {}: {}", binary, e)))?;

        {
            let mut guard = self
                .process
                .lock()
                .map_err(|e| BackendError::new(0, e.to_string()))?;
            *guard = Some(child);
        }
        {
            let mut guard = self
                .current_config
                .lock()
                .map_err(|e| BackendError::new(0, e.to_string()))?;
            *guard = Some(new_config);
        }

        Ok(true)
    }

    /// Resolve the context budget based on the given mode.
    pub fn resolve_budget(
        &self,
        mode: BudgetMode,
        manual_tokens: Option<i64>,
        n_slots: Option<i64>,
        kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        match mode {
            BudgetMode::Manual => manual_tokens.ok_or_else(|| {
                BudgetResolutionError::new()
                    .with_cause("manual_tokens required for MANUAL budget mode")
            }),
            BudgetMode::Backend => self.resolve_backend_budget(),
            BudgetMode::ForgeFull => self.resolve_forge_full(n_slots, kv_unified),
            BudgetMode::ForgeFast => self.resolve_forge_fast(n_slots, kv_unified),
        }
    }

    fn resolve_backend_budget(&self) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget());
        }
        let ctx = self.query_props_context()?;
        Ok(ctx)
    }

    fn resolve_forge_full(
        &self,
        n_slots: Option<i64>,
        kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget());
        }
        let ctx = self.query_props_context()?;
        let budget = if !kv_unified {
            if let Some(slots) = n_slots {
                if slots > 1 {
                    ctx * slots
                } else {
                    ctx
                }
            } else {
                ctx
            }
        } else {
            ctx
        };
        Ok(budget)
    }

    fn resolve_forge_fast(
        &self,
        n_slots: Option<i64>,
        kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget() / 2);
        }
        // Two-phase: start without context override already done in start().
        // Read /props for max context.
        let per_slot_ctx = self.query_props_context()?;

        // Recover total context for multi-slot non-unified before halving.
        let total_ctx = if !kv_unified {
            if let Some(slots) = n_slots {
                if slots > 1 {
                    per_slot_ctx * slots
                } else {
                    per_slot_ctx
                }
            } else {
                per_slot_ctx
            }
        } else {
            per_slot_ctx
        };

        let halved = total_ctx / 2;
        // Store for later reference.
        {
            if let Ok(mut g) = self.last_context.lock() {
                *g = Some(halved);
            }
        }

        // Per-slot budget for non-unified.
        if !kv_unified {
            if let Some(slots) = n_slots {
                if slots > 1 {
                    return Ok(halved / slots);
                }
            }
        }
        Ok(halved)
    }

    /// VRAM tier budget for ollama.
    pub fn ollama_vram_budget() -> i64 {
        match detect_hardware() {
            Ok(Some(hw)) => {
                let gb = hw.vram_total_gb();
                if gb < 24.0 {
                    4096
                } else if gb < 48.0 {
                    32768
                } else {
                    262144
                }
            }
            _ => 4096,
        }
    }

    /// Query the backend /props endpoint for the actual context length.
    pub fn query_props_context(&self) -> Result<i64, BudgetResolutionError> {
        let url = format!("http://127.0.0.1:{}/props", self.port);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|e| BudgetResolutionError::new().with_cause(e.to_string()))?;

        rt.block_on(async {
            let resp = reqwest::Client::new()
                .get(&url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
                .map_err(|e| BudgetResolutionError::new().with_cause(e.to_string()))?;

            if !resp.status().is_success() {
                return Err(
                    BudgetResolutionError::new().with_cause(format!("Status {}", resp.status()))
                );
            }

            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| BudgetResolutionError::new().with_cause(e.to_string()))?;

            let ctx = json
                .get("default_generation_settings")
                .and_then(|s| s.get("n_ctx"))
                .and_then(|n| n.as_i64())
                .ok_or_else(|| {
                    BudgetResolutionError::new()
                        .with_cause("missing context field in /props response")
                })?;

            Ok(ctx)
        })
    }

    /// Stop the backend process. For ollama, runs 'ollama stop'.
    /// Waits up to 10s for termination, then forces kill.
    /// Sleeps 3s after termination for VRAM clearing.
    pub fn stop(&self) -> Result<(), BackendError> {
        if self.backend == "ollama" {
            if let Ok(guard) = self.current_config.lock() {
                if let Some(ref cfg) = *guard {
                    let _ = std::process::Command::new("ollama")
                        .arg("stop")
                        .arg(&cfg.model)
                        .output();
                }
            }
            if let Ok(mut g) = self.current_config.lock() {
                *g = None;
            }
            return Ok(());
        }

        let mut process_guard = self
            .process
            .lock()
            .map_err(|e| BackendError::new(0, e.to_string()))?;

        if let Some(ref mut child) = *process_guard {
            let _ = child.kill();
            // Wait with 10s timeout.
            let _ = child.wait();
        }
        *process_guard = None;

        if let Ok(mut g) = self.current_config.lock() {
            *g = None;
        }
        if let Ok(mut g) = self.last_context.lock() {
            *g = None;
        }

        Ok(())
    }

    /// Find the llamafile runtime binary in the model directory.
    fn find_llamafile_binary(&self, gguf_path: &Path) -> Result<String, BackendError> {
        let dir = gguf_path.parent().ok_or_else(|| {
            BackendError::new(0, "Cannot determine model directory from gguf path")
        })?;

        let entries = std::fs::read_dir(dir)
            .map_err(|e| BackendError::new(0, format!("Cannot read model directory: {}", e)))?;

        let mut best: Option<(String, String)> = None;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if name_str.starts_with("llamafile-") || name_str.contains("llamafile") {
                match &best {
                    None => {
                        best = Some((entry.path().to_string_lossy().to_string(), name_str));
                    }
                    Some((_, existing)) if name_str > *existing => {
                        best = Some((entry.path().to_string_lossy().to_string(), name_str));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(p, _)| p).ok_or_else(|| {
            BackendError::new(0, "No llamafile runtime binary found in model directory")
        })
    }
}

/// Convenience factory that creates a ServerManager and ContextManager.
///
/// Enforces identity rules: ollama requires model name (no file path),
/// llamaserver/llamafile requires file path (no model name).
#[allow(clippy::too_many_arguments)]
pub fn setup_backend(
    backend: &str,
    model: Option<&str>,
    gguf_path: Option<&Path>,
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
            if Path::new(m).extension().is_some() || m.contains('/') || m.contains('\\') {
                return Err("ollama does not accept a file path".to_string());
            }
        }
        "llamaserver" | "llamafile" => {
            if model.is_some() {
                return Err(format!(
                    "{} does not accept a model name; use gguf_path",
                    backend
                ));
            }
            if gguf_path.is_none() {
                return Err(format!("{} requires a file path (gguf_path)", backend));
            }
        }
        _ => return Err(format!("Unknown backend: {}", backend)),
    }

    let mgr = ServerManager::new(backend, port, None);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .map_err(|e| e.to_string())?;

    // Start backend process
    if backend != "ollama" {
        let m_name = model.unwrap_or("");
        let path = gguf_path.ok_or("llamaserver/llamafile requires gguf_path")?;
        mgr.start(
            m_name,
            path,
            mode,
            extra_flags,
            if budget_mode == BudgetMode::Manual {
                manual_tokens
            } else {
                None
            },
            cache_type_k,
            cache_type_v,
            n_slots,
            kv_unified,
        )
        .map_err(|e| e.to_string())?;

        // Wait for readiness by polling `/props`
        let url = format!("http://127.0.0.1:{}/props", port);
        let mut ready = false;
        let client = reqwest::Client::new();
        for _ in 0..40 {
            let res = rt.block_on(async {
                client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(1))
                    .send()
                    .await
            });
            if let Ok(resp) = res {
                if resp.status().is_success() {
                    ready = true;
                    break;
                }
            }
            rt.block_on(async {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            });
        }
        if !ready {
            return Err("Backend failed to become ready within timeout".to_string());
        }
    }

    let budget = mgr
        .resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
        .map_err(|e| e.to_string())?;

    let ctx_mgr = ContextManager::new(Box::new(NoCompact), budget, None, None, None);

    Ok((mgr, ctx_mgr))
}

impl Drop for ServerManager {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.process.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_backend_args(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_mode_str_roundtrip() {
        for (s, mode) in [
            ("backend", BudgetMode::Backend),
            ("manual", BudgetMode::Manual),
            ("forge-full", BudgetMode::ForgeFull),
            ("forge-fast", BudgetMode::ForgeFast),
        ] {
            assert_eq!(BudgetMode::parse(s), Some(mode));
            assert_eq!(mode.as_str(), s);
        }
    }

    #[test]
    fn budget_mode_unknown() {
        assert_eq!(BudgetMode::parse("unknown"), None);
    }

    fn args_for(
        backend: &str,
        mode: &str,
        extra_flags: &[String],
        n_slots: Option<i64>,
        kv_unified: bool,
    ) -> Vec<String> {
        build_backend_args(
            backend,
            8080,
            Path::new("model.gguf"),
            mode,
            extra_flags,
            None,
            None,
            None,
            n_slots,
            kv_unified,
        )
    }

    #[test]
    fn llamaserver_native_args_include_jinja() {
        let args = args_for("llamaserver", "native", &[], None, false);
        assert!(args.iter().any(|arg| arg == "--jinja"));
    }

    #[test]
    fn llamaserver_prompt_args_omit_jinja() {
        let args = args_for("llamaserver", "prompt", &[], None, false);
        assert!(!args.iter().any(|arg| arg == "--jinja"));
    }

    #[test]
    fn backend_args_do_not_force_temperature() {
        let args = args_for("llamaserver", "native", &[], None, false);
        assert!(!args.iter().any(|arg| arg == "--temp"));
    }

    #[test]
    fn backend_args_use_python_parallel_flags() {
        let args = args_for("llamaserver", "native", &[], Some(4), true);
        assert!(args
            .windows(2)
            .any(|pair| pair[0] == "--parallel" && pair[1] == "4"));
        assert!(args.iter().any(|arg| arg == "--kv-unified"));
        assert!(!args.iter().any(|arg| arg == "-np"));
        assert!(!args.iter().any(|arg| arg == "--parallel-unified"));
    }

    #[test]
    fn backend_args_preserve_extra_flags() {
        let extra = vec!["--reasoning-budget".to_string(), "0".to_string()];
        let args = args_for("llamaserver", "native", &extra, None, false);
        assert!(args
            .windows(2)
            .any(|pair| pair[0] == "--reasoning-budget" && pair[1] == "0"));
    }

    #[test]
    fn setup_backend_ollama_requires_model() {
        let result = setup_backend(
            "ollama",
            None,
            None,
            BudgetMode::Backend,
            None,
            8080,
            "native",
            &[],
            None,
            None,
            None,
            false,
        );
        match result {
            Err(e) => assert!(e.contains("requires a model name")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn setup_backend_ollama_rejects_file_path() {
        let result = setup_backend(
            "ollama",
            Some("/path/to/model.gguf"),
            None,
            BudgetMode::Backend,
            None,
            8080,
            "native",
            &[],
            None,
            None,
            None,
            false,
        );
        match result {
            Err(e) => assert!(e.contains("does not accept a file path")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn setup_backend_llamaserver_rejects_model_name() {
        let result = setup_backend(
            "llamaserver",
            Some("mymodel"),
            Some(Path::new("t.gguf")),
            BudgetMode::Backend,
            None,
            8080,
            "native",
            &[],
            None,
            None,
            None,
            false,
        );
        match result {
            Err(e) => assert!(e.contains("does not accept a model name")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn setup_backend_llamaserver_requires_file_path() {
        let result = setup_backend(
            "llamaserver",
            None,
            None,
            BudgetMode::Backend,
            None,
            8080,
            "native",
            &[],
            None,
            None,
            None,
            false,
        );
        match result {
            Err(e) => assert!(e.contains("requires a file path")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn resolve_budget_manual_requires_tokens() {
        let mgr = ServerManager::new("ollama", 8080, None);
        let result = mgr.resolve_budget(BudgetMode::Manual, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_budget_manual_with_tokens() {
        let mgr = ServerManager::new("ollama", 8080, None);
        let result = mgr.resolve_budget(BudgetMode::Manual, Some(8192), None, false);
        assert_eq!(result.unwrap(), 8192);
    }

    #[test]
    fn server_manager_new_defaults() {
        let mgr = ServerManager::new("ollama", 9090, None);
        assert!(mgr.process.lock().unwrap().is_none());
        assert!(mgr.current_config.lock().unwrap().is_none());
    }

    #[test]
    fn ollama_start_is_noop() {
        let mgr = ServerManager::new("ollama", 8080, None);
        let result = mgr.start(
            "llama3",
            Path::new("dummy"),
            "native",
            &[],
            None,
            None,
            None,
            None,
            false,
        );
        assert!(!result.unwrap());
        assert!(mgr.process.lock().unwrap().is_none());
    }

    #[test]
    fn run_config_equality() {
        let c1 = RunConfig {
            model: "m".into(),
            mode: "native".into(),
            ctx_override: None,
            extra_flags: vec![],
            cache_type_k: None,
            cache_type_v: None,
            n_slots: None,
            kv_unified: false,
        };
        let c2 = RunConfig {
            model: "m".into(),
            mode: "native".into(),
            ctx_override: None,
            extra_flags: vec![],
            cache_type_k: None,
            cache_type_v: None,
            n_slots: None,
            kv_unified: false,
        };
        assert_eq!(c1, c2);
    }

    #[test]
    fn run_config_inequality_mode() {
        let c1 = RunConfig {
            model: "m".into(),
            mode: "native".into(),
            ctx_override: None,
            extra_flags: vec![],
            cache_type_k: None,
            cache_type_v: None,
            n_slots: None,
            kv_unified: false,
        };
        let c2 = RunConfig {
            model: "m".into(),
            mode: "prompt".into(),
            ctx_override: None,
            extra_flags: vec![],
            cache_type_k: None,
            cache_type_v: None,
            n_slots: None,
            kv_unified: false,
        };
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_setup_backend_ordering() {
        use std::fs::{create_dir_all, File};
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        let test_dir = std::path::Path::new("target/debug/test_setup_backend");
        create_dir_all(test_dir).unwrap();

        let binary_path = test_dir.join("llamafile-mock");
        {
            let mut f = File::create(&binary_path).unwrap();
            f.write_all(b"#!/bin/sh\nsleep 10\n").unwrap();
            let mut permissions = f.metadata().unwrap().permissions();
            permissions.set_mode(0o755);
            f.set_permissions(permissions).unwrap();
        }

        let mut server = mockito::Server::new();
        let port = server
            .host_with_port()
            .split(':')
            .next_back()
            .unwrap()
            .parse::<i64>()
            .unwrap();

        let _mock = server
            .mock("GET", "/props")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"default_generation_settings": {"n_ctx": 2048}}"#)
            .create();

        let gguf_path = test_dir.join("model.gguf");
        {
            File::create(&gguf_path)
                .unwrap()
                .write_all(b"dummy gguf")
                .unwrap();
        }

        let result = setup_backend(
            "llamafile",
            None,
            Some(&gguf_path),
            BudgetMode::Backend,
            None,
            port,
            "native",
            &[],
            None,
            None,
            None,
            false,
        );

        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (server_mgr, ctx_mgr) = result.unwrap();
        assert_eq!(ctx_mgr.budget(), 2048);

        let _ = server_mgr.stop();
        let _ = std::fs::remove_dir_all(test_dir);
    }
}
