//! Backend lifecycle manager: process spawning, budget resolution, VRAM tiers.
//!
//! ServerManager controls llama-server/llamafile/Ollama backend processes.
//! BudgetMode controls how context window size is determined.

use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use crate::context::{detect_hardware, ContextManager, NoCompact};
use crate::error::{BackendError, BudgetResolutionError};

#[cfg(unix)]
use nix::sys::signal::{kill, killpg, Signal};
#[cfg(unix)]
use nix::unistd::Pid;

const BACKEND_READY_TIMEOUT: Duration = Duration::from_secs(180);
const BACKEND_READY_POLL_INTERVAL: Duration = Duration::from_secs(2);
const BACKEND_PROPS_TIMEOUT: Duration = Duration::from_secs(5);
const BACKEND_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const BACKEND_VRAM_CLEAR_DELAY: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy)]
struct LifecycleOptions {
    ready_timeout: Duration,
    ready_poll_interval: Duration,
    stop_timeout: Duration,
    vram_clear_delay: Duration,
}

impl Default for LifecycleOptions {
    fn default() -> Self {
        Self {
            ready_timeout: BACKEND_READY_TIMEOUT,
            ready_poll_interval: BACKEND_READY_POLL_INTERVAL,
            stop_timeout: BACKEND_STOP_TIMEOUT,
            vram_clear_delay: BACKEND_VRAM_CLEAR_DELAY,
        }
    }
}

#[derive(Debug)]
enum ManagedChildState {
    Missing,
    Running,
    Exited(ExitStatus),
}

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
    llamafile_runtime: Option<PathBuf>,
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
            llamafile_runtime: None,
            process: Mutex::new(None),
            current_config: Mutex::new(None),
            last_context: Mutex::new(None),
        }
    }

    pub fn with_llamafile_runtime(mut self, path: impl AsRef<Path>) -> Self {
        self.llamafile_runtime = Some(path.as_ref().to_path_buf());
        self
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
        self.start_with_options(
            model,
            gguf_path,
            mode,
            extra_flags,
            ctx_override,
            cache_type_k,
            cache_type_v,
            n_slots,
            kv_unified,
            LifecycleOptions::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_with_options(
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
        options: LifecycleOptions,
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
        if can_reuse
            && (self.backend == "ollama"
                || matches!(self.child_state()?, ManagedChildState::Running))
        {
            return Ok(false);
        }

        // Stop existing if running.
        self.stop_with_options(options.stop_timeout, options.vram_clear_delay)?;

        if self.backend == "ollama" {
            let mut guard = self
                .current_config
                .lock()
                .map_err(|e| BackendError::new(0, e.to_string()))?;
            *guard = Some(new_config);
            return Ok(false);
        }

        let binary = if self.backend == "llamafile" {
            validate_llamafile_runtime_path(self.llamafile_runtime.as_deref().ok_or_else(
                || BackendError::new(0, "llamafile backend requires llamafile_runtime"),
            )?)?
        } else {
            PathBuf::from("llama-server")
        };

        self.ensure_backend_port_available()?;

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
        #[cfg(unix)]
        cmd.process_group(0);

        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        let child = cmd
            .spawn()
            .map_err(|e| BackendError::new(0, format!("Failed to start {:?}: {}", binary, e)))?;

        {
            let mut guard = self
                .process
                .lock()
                .map_err(|e| BackendError::new(0, e.to_string()))?;
            *guard = Some(child);
        }

        if let Err(err) = self.wait_until_ready(options.ready_timeout, options.ready_poll_interval)
        {
            let _ = self.stop_with_options(options.stop_timeout, Duration::ZERO);
            return Err(err);
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

    fn child_state(&self) -> Result<ManagedChildState, BackendError> {
        let mut guard = self
            .process
            .lock()
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        match guard.as_mut() {
            Some(child) => match child
                .try_wait()
                .map_err(|e| BackendError::new(0, format!("Failed to poll backend: {}", e)))?
            {
                Some(status) => Ok(ManagedChildState::Exited(status)),
                None => Ok(ManagedChildState::Running),
            },
            None => Ok(ManagedChildState::Missing),
        }
    }

    fn ensure_backend_port_available(&self) -> Result<(), BackendError> {
        let port: u16 = self
            .port
            .try_into()
            .map_err(|_| BackendError::new(0, format!("Invalid backend port: {}", self.port)))?;
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        if TcpStream::connect_timeout(&addr, Duration::from_millis(50)).is_ok() {
            return Err(BackendError::new(
                0,
                format!("Backend port {} is already accepting connections", port),
            ));
        }
        let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| {
            BackendError::new(
                0,
                format!("Backend port {} is not available on 127.0.0.1: {}", port, e),
            )
        })?;
        drop(listener);
        Ok(())
    }

    fn wait_until_ready(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<(), BackendError> {
        let url = format!("http://127.0.0.1:{}/props", self.port);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        let client = reqwest::Client::new();
        let deadline = Instant::now() + timeout;

        loop {
            match self.child_state()? {
                ManagedChildState::Running => {}
                ManagedChildState::Exited(status) => {
                    return Err(BackendError::new(
                        0,
                        format!("Backend exited before readiness: {}", status),
                    ));
                }
                ManagedChildState::Missing => {
                    return Err(BackendError::new(
                        0,
                        "Backend process disappeared before readiness",
                    ));
                }
            }

            let ready = rt.block_on(async {
                match client.get(&url).timeout(BACKEND_PROPS_TIMEOUT).send().await {
                    Ok(resp) if resp.status().is_success() => resp
                        .json::<serde_json::Value>()
                        .await
                        .ok()
                        .and_then(|json| {
                            json.get("default_generation_settings")
                                .and_then(|settings| settings.as_object())
                                .map(|_| true)
                        })
                        .unwrap_or(false),
                    _ => false,
                }
            });
            if ready {
                return Ok(());
            }

            if Instant::now() >= deadline {
                return Err(BackendError::new(
                    0,
                    format!(
                        "Backend failed to become ready within {}s",
                        timeout.as_secs()
                    ),
                ));
            }
            thread::sleep(poll_interval);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn start_with_budget_options(
        &self,
        model: &str,
        gguf_path: &Path,
        mode: &str,
        budget_mode: BudgetMode,
        manual_tokens: Option<i64>,
        extra_flags: &[String],
        cache_type_k: Option<&str>,
        cache_type_v: Option<&str>,
        n_slots: Option<i64>,
        kv_unified: bool,
        options: LifecycleOptions,
    ) -> Result<i64, String> {
        if budget_mode == BudgetMode::Manual && manual_tokens.is_none() {
            return Err("manual mode requires manual_tokens".to_string());
        }

        if self.backend == "ollama" {
            self.start_with_options(
                model,
                gguf_path,
                mode,
                extra_flags,
                None,
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
                options,
            )
            .map_err(|e| e.to_string())?;
            return self
                .resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
                .map_err(|e| e.to_string());
        }

        if budget_mode == BudgetMode::ForgeFast {
            self.start_with_options(
                model,
                gguf_path,
                mode,
                extra_flags,
                None,
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
                options,
            )
            .map_err(|e| e.to_string())?;
            let reported_ctx = self.query_props_context().map_err(|e| e.to_string())?;
            let total_ctx = if kv_unified || n_slots.is_none_or(|slots| slots <= 1) {
                reported_ctx
            } else {
                reported_ctx * n_slots.unwrap_or(1)
            };
            let half_total = total_ctx / 2;
            if let Ok(mut g) = self.last_context.lock() {
                *g = Some(half_total);
            }
            self.start_with_options(
                model,
                gguf_path,
                mode,
                extra_flags,
                Some(half_total),
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
                options,
            )
            .map_err(|e| e.to_string())?;
            return self
                .resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
                .map_err(|e| e.to_string());
        }

        let ctx_override = if budget_mode == BudgetMode::Manual {
            manual_tokens
        } else {
            None
        };
        self.start_with_options(
            model,
            gguf_path,
            mode,
            extra_flags,
            ctx_override,
            cache_type_k,
            cache_type_v,
            n_slots,
            kv_unified,
            options,
        )
        .map_err(|e| e.to_string())?;
        self.resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
            .map_err(|e| e.to_string())
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
            BudgetMode::Manual => {
                if self.backend == "ollama" {
                    manual_tokens.ok_or_else(|| {
                        BudgetResolutionError::new()
                            .with_cause("manual_tokens required for MANUAL budget mode")
                    })
                } else {
                    self.resolve_backend_budget()
                }
            }
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
        _n_slots: Option<i64>,
        _kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget());
        }
        self.query_props_context()
    }

    fn resolve_forge_fast(
        &self,
        _n_slots: Option<i64>,
        _kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget() / 2);
        }
        self.query_props_context()
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
        self.stop_with_options(BACKEND_STOP_TIMEOUT, BACKEND_VRAM_CLEAR_DELAY)
    }

    fn stop_with_options(
        &self,
        stop_timeout: Duration,
        vram_clear_delay: Duration,
    ) -> Result<(), BackendError> {
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
            let _ = terminate_child(child);
            match wait_for_child_exit(child, stop_timeout) {
                Ok(Some(_)) => {}
                Ok(None) => {
                    let _ = kill_child_now(child);
                    let _ = child.wait();
                }
                Err(_) => {
                    let _ = kill_child_now(child);
                    let _ = child.wait();
                }
            }
        }
        *process_guard = None;

        if let Ok(mut g) = self.current_config.lock() {
            *g = None;
        }
        if let Ok(mut g) = self.last_context.lock() {
            *g = None;
        }

        if !vram_clear_delay.is_zero() {
            thread::sleep(vram_clear_delay);
        }

        Ok(())
    }
}

fn wait_for_child_exit(
    child: &mut Child,
    timeout: Duration,
) -> std::io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn terminate_child(child: &mut Child) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let pid = Pid::from_raw(child.id() as i32);
        let direct = kill(pid, Signal::SIGTERM);
        let group = killpg(pid, Signal::SIGTERM);
        if direct.is_err() && group.is_err() {
            return child.kill();
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        child.kill()
    }
}

fn kill_child_now(child: &mut Child) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        if killpg(Pid::from_raw(child.id() as i32), Signal::SIGKILL).is_err() {
            return child.kill();
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        child.kill()
    }
}

fn validate_llamafile_runtime_path(path: &Path) -> Result<PathBuf, BackendError> {
    if !path.is_absolute() {
        return Err(BackendError::new(
            0,
            "llamafile_runtime must be an absolute path",
        ));
    }

    let canonical = std::fs::canonicalize(path).map_err(|e| {
        BackendError::new(
            0,
            format!(
                "Cannot resolve llamafile_runtime '{}': {}",
                path.display(),
                e
            ),
        )
    })?;
    let metadata = std::fs::metadata(&canonical).map_err(|e| {
        BackendError::new(
            0,
            format!(
                "Cannot read llamafile_runtime '{}': {}",
                canonical.display(),
                e
            ),
        )
    })?;

    if !metadata.is_file() {
        return Err(BackendError::new(
            0,
            format!(
                "llamafile_runtime must be a regular file: {}",
                canonical.display()
            ),
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(BackendError::new(
                0,
                format!(
                    "llamafile_runtime must be executable: {}",
                    canonical.display()
                ),
            ));
        }
    }

    Ok(canonical)
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

impl Drop for ServerManager {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.process.lock() {
            if let Some(ref mut child) = *guard {
                let _ = terminate_child(child);
                let _ = wait_for_child_exit(child, Duration::from_millis(200));
                let _ = kill_child_now(child);
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

    static TEST_PORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn port_test_guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_PORT_LOCK.lock().unwrap()
    }
    use std::net::TcpListener;

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

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::current_dir()
            .unwrap()
            .join("target/debug")
            .join(format!("{}-{}-{}", name, std::process::id(), nanos));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_executable(path: &Path, body: &[u8]) {
        use std::io::Write;

        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(body).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = file.metadata().unwrap().permissions();
            permissions.set_mode(0o755);
            file.set_permissions(permissions).unwrap();
        }
    }

    #[allow(dead_code)]
    fn sh_quote(path: &Path) -> String {
        format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
    }

    #[allow(dead_code)]
    fn free_port() -> i64 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.local_addr().unwrap().port() as i64
    }

    #[allow(dead_code)]
    fn test_lifecycle_options() -> LifecycleOptions {
        LifecycleOptions {
            ready_timeout: Duration::from_secs(5),
            ready_poll_interval: Duration::from_millis(20),
            stop_timeout: Duration::from_millis(200),
            vram_clear_delay: Duration::ZERO,
        }
    }

    #[allow(dead_code)]
    fn write_fake_http_backend(path: &Path, log_path: &Path, fixed_ctx: Option<i64>) {
        let fixed_ctx = fixed_ctx
            .map(|ctx| ctx.to_string())
            .unwrap_or_else(|| "None".to_string());
        let body = format!(
            r#"#!/bin/sh
printf '%s\n' "$*" >> {log}
python3 - "$@" <<'PY'
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
fixed_ctx = {fixed_ctx}
ctx = fixed_ctx
if ctx is None:
    ctx = 8192
    if "-c" in args:
        ctx = int(args[args.index("-c") + 1])

body = json.dumps({{"default_generation_settings": {{"n_ctx": ctx}}}}).encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        pass

signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
"#,
            log = sh_quote(log_path),
            fixed_ctx = fixed_ctx
        );
        write_executable(path, body.as_bytes());
    }

    fn write_stubborn_runtime(path: &Path, term_marker: &Path) {
        let marker = serde_json::to_string(&term_marker.to_string_lossy()).unwrap();
        let body = format!(
            r#"#!/bin/sh
exec python3 - "$@" <<'PY'
import signal
import time

marker = {marker}

def handle_term(_signum, _frame):
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("term")

signal.signal(signal.SIGTERM, handle_term)
while True:
    time.sleep(1)
PY
"#,
            marker = marker
        );
        write_executable(path, body.as_bytes());
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
    fn setup_backend_llamafile_requires_runtime() {
        let result = setup_backend(
            "llamafile",
            None,
            Some(Path::new("t.gguf")),
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
            Err(e) => assert!(e.contains("requires llamafile_runtime")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn setup_backend_ollama_rejects_llamafile_runtime() {
        let result = setup_backend(
            "ollama",
            Some("llama3"),
            None,
            Some(Path::new("/tmp/llamafile")),
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
            Err(e) => assert!(e.contains("does not accept llamafile_runtime")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn setup_backend_llamaserver_rejects_llamafile_runtime() {
        let result = setup_backend(
            "llamaserver",
            None,
            Some(Path::new("t.gguf")),
            Some(Path::new("/tmp/llamafile")),
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
            Err(e) => assert!(e.contains("does not accept llamafile_runtime")),
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
    fn llamafile_requires_explicit_runtime_before_spawn() {
        let test_dir = unique_test_dir("llamafile-requires-runtime");
        let gguf_path = test_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"dummy gguf").unwrap();
        let marker = test_dir.join("marker");
        let adjacent = test_dir.join("zz-llamafile");
        write_executable(
            &adjacent,
            format!("#!/bin/sh\nprintf executed > '{}'\n", marker.display()).as_bytes(),
        );

        let mgr = ServerManager::new("llamafile", 8080, None);
        let result = mgr.start("", &gguf_path, "native", &[], None, None, None, None, false);

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("llamafile_runtime"));
        assert!(!marker.exists());
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn llamafile_runtime_rejects_relative_path() {
        let result = validate_llamafile_runtime_path(Path::new("llamafile"));
        assert!(result.unwrap_err().to_string().contains("absolute path"));
    }

    #[test]
    fn llamafile_runtime_rejects_missing_path() {
        let test_dir = unique_test_dir("llamafile-missing-runtime");
        let missing = test_dir.join("missing-llamafile");
        let result = validate_llamafile_runtime_path(&missing);
        assert!(result.unwrap_err().to_string().contains("Cannot resolve"));
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn llamafile_runtime_rejects_non_file_path() {
        let test_dir = unique_test_dir("llamafile-non-file-runtime");
        let result = validate_llamafile_runtime_path(&test_dir);
        assert!(result.unwrap_err().to_string().contains("regular file"));
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[cfg(unix)]
    #[test]
    fn llamafile_runtime_rejects_non_executable_file() {
        use std::os::unix::fs::PermissionsExt;

        let test_dir = unique_test_dir("llamafile-non-executable-runtime");
        let runtime = test_dir.join("llamafile-runtime");
        std::fs::write(&runtime, b"#!/bin/sh\n").unwrap();
        let mut permissions = std::fs::metadata(&runtime).unwrap().permissions();
        permissions.set_mode(0o644);
        std::fs::set_permissions(&runtime, permissions).unwrap();

        let result = validate_llamafile_runtime_path(&runtime);
        assert!(result.unwrap_err().to_string().contains("executable"));
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn llamafile_runtime_validation_canonicalizes_path() {
        let test_dir = unique_test_dir("llamafile-canonical-runtime");
        let runtime = test_dir.join("llamafile-runtime");
        write_executable(&runtime, b"#!/bin/sh\nsleep 10\n");

        let result = validate_llamafile_runtime_path(&runtime).unwrap();
        assert_eq!(result, runtime.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn managed_port_occupied_fails_before_spawn() {
        let _guard = port_test_guard();
        let test_dir = unique_test_dir("managed-port-occupied");
        let runtime = test_dir.join("llamafile-runtime");
        let log_path = test_dir.join("spawn.log");
        write_fake_http_backend(&runtime, &log_path, None);
        let gguf_path = test_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"dummy gguf").unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port() as i64;
        let mgr = ServerManager::new("llamafile", port, None).with_llamafile_runtime(&runtime);

        let result = mgr.start_with_options(
            "",
            &gguf_path,
            "native",
            &[],
            None,
            None,
            None,
            None,
            false,
            test_lifecycle_options(),
        );

        assert!(result.is_err());
        assert!(!log_path.exists());
        drop(listener);
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn child_exit_during_readiness_fails_start() {
        let _guard = port_test_guard();
        let test_dir = unique_test_dir("managed-child-exits");
        let runtime = test_dir.join("llamafile-runtime");
        write_executable(&runtime, b"#!/bin/sh\nexit 17\n");
        let gguf_path = test_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"dummy gguf").unwrap();
        let port = free_port();
        let mgr = ServerManager::new("llamafile", port, None).with_llamafile_runtime(&runtime);

        let result = mgr.start_with_options(
            "",
            &gguf_path,
            "native",
            &[],
            None,
            None,
            None,
            None,
            false,
            test_lifecycle_options(),
        );

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exited before readiness"));
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn readiness_requires_default_generation_settings() {
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
            .with_body(r#"{"status":"ok"}"#)
            .create();
        let child = std::process::Command::new("sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let mgr = ServerManager::new("llamaserver", port, None);
        *mgr.process.lock().unwrap() = Some(child);

        let result = mgr.wait_until_ready(Duration::from_millis(120), Duration::from_millis(20));

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("failed to become ready"));
        let _ = mgr.stop_with_options(Duration::from_millis(200), Duration::ZERO);
    }

    #[test]
    fn failed_readiness_terminates_child_and_clears_state() {
        let _guard = port_test_guard();
        let test_dir = unique_test_dir("managed-failed-readiness-cleans");
        let runtime = test_dir.join("llamafile-runtime");
        let term_marker = test_dir.join("term");
        write_stubborn_runtime(&runtime, &term_marker);
        let gguf_path = test_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"dummy gguf").unwrap();
        let port = free_port();
        let mgr = ServerManager::new("llamafile", port, None).with_llamafile_runtime(&runtime);

        let result = mgr.start_with_options(
            "",
            &gguf_path,
            "native",
            &[],
            None,
            None,
            None,
            None,
            false,
            test_lifecycle_options(),
        );

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("failed to become ready"));
        assert!(mgr.process.lock().unwrap().is_none());
        assert!(term_marker.exists());
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[cfg(unix)]
    #[test]
    fn stop_terminates_then_kills_stubborn_process_group() {
        let test_dir = unique_test_dir("managed-stop-kill");
        let runtime = test_dir.join("stubborn");
        let term_marker = test_dir.join("term");
        let ready_marker = test_dir.join("ready");
        let term_marker_json = serde_json::to_string(&term_marker.to_string_lossy()).unwrap();
        let ready_marker_json = serde_json::to_string(&ready_marker.to_string_lossy()).unwrap();
        write_executable(
            &runtime,
            format!(
                r#"#!/bin/sh
exec python3 - <<'PY'
import signal
import time

term_marker = {term_marker}
ready_marker = {ready_marker}

def handle_term(_signum, _frame):
    with open(term_marker, "w", encoding="utf-8") as fh:
        fh.write("term")

signal.signal(signal.SIGTERM, handle_term)
with open(ready_marker, "w", encoding="utf-8") as fh:
    fh.write("ready")
while True:
    time.sleep(1)
PY
"#,
                term_marker = term_marker_json,
                ready_marker = ready_marker_json
            )
            .as_bytes(),
        );
        let mut child = {
            let mut command = std::process::Command::new(&runtime);
            command.process_group(0);
            command.spawn().unwrap()
        };
        assert!(child.try_wait().unwrap().is_none());
        let deadline = Instant::now() + Duration::from_secs(2);
        while !ready_marker.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(ready_marker.exists());
        let mgr = ServerManager::new("llamaserver", 8080, None);
        *mgr.process.lock().unwrap() = Some(child);

        mgr.stop_with_options(Duration::from_millis(120), Duration::ZERO)
            .unwrap();

        assert!(mgr.process.lock().unwrap().is_none());
        assert!(term_marker.exists());
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn forge_fast_restarts_with_half_total_context() {
        let _guard = port_test_guard();
        let test_dir = unique_test_dir("managed-forge-fast");
        let runtime = test_dir.join("llamafile-runtime");
        let log_path = test_dir.join("spawn.log");
        write_fake_http_backend(&runtime, &log_path, None);
        let gguf_path = test_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"dummy gguf").unwrap();
        let mgr =
            ServerManager::new("llamafile", free_port(), None).with_llamafile_runtime(&runtime);

        let budget = mgr
            .start_with_budget_options(
                "",
                &gguf_path,
                "native",
                BudgetMode::ForgeFast,
                None,
                &[],
                None,
                None,
                None,
                false,
                test_lifecycle_options(),
            )
            .unwrap();

        assert_eq!(budget, 4096);
        let log = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(!lines[0].contains(" -c "));
        assert!(lines[1].contains(" -c 4096"));
        let _ = mgr.stop_with_options(Duration::from_millis(200), Duration::ZERO);
        let _ = std::fs::remove_dir_all(test_dir);
    }

    #[test]
    fn manual_llamaserver_budget_comes_from_props() {
        let _guard = port_test_guard();
        let test_dir = unique_test_dir("managed-manual-budget");
        let runtime = test_dir.join("llamafile-runtime");
        let log_path = test_dir.join("spawn.log");
        write_fake_http_backend(&runtime, &log_path, Some(2048));
        let gguf_path = test_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"dummy gguf").unwrap();
        let mgr =
            ServerManager::new("llamafile", free_port(), None).with_llamafile_runtime(&runtime);

        let budget = mgr
            .start_with_budget_options(
                "",
                &gguf_path,
                "native",
                BudgetMode::Manual,
                Some(4096),
                &[],
                None,
                None,
                None,
                false,
                test_lifecycle_options(),
            )
            .unwrap();

        assert_eq!(budget, 2048);
        assert!(std::fs::read_to_string(&log_path)
            .unwrap()
            .contains(" -c 4096"));
        let _ = mgr.stop_with_options(Duration::from_millis(200), Duration::ZERO);
        let _ = std::fs::remove_dir_all(test_dir);
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
        let _guard = port_test_guard();
        use std::fs::{create_dir_all, File};
        use std::io::Write;

        let test_dir = unique_test_dir("test-setup-backend");
        let model_dir = test_dir.join("models");
        let runtime_dir = test_dir.join("bin");
        create_dir_all(&model_dir).unwrap();
        create_dir_all(&runtime_dir).unwrap();

        let binary_path = runtime_dir.join("llamafile-mock");
        let log_path = test_dir.join("spawn.log");
        write_fake_http_backend(&binary_path, &log_path, Some(2048));
        let port = free_port();

        let gguf_path = model_dir.join("model.gguf");
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
            Some(&binary_path),
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
