use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::BackendError;

#[cfg(unix)]
use nix::sys::signal::{kill, killpg, Signal};
#[cfg(unix)]
use nix::unistd::Pid;

use super::args::{build_backend_args, validate_backend_extra_flags};
use super::manager::{RunConfig, ServerManager};
use super::runtime::validate_llamafile_runtime_path;

const BACKEND_READY_TIMEOUT: Duration = Duration::from_secs(180);
const BACKEND_READY_POLL_INTERVAL: Duration = Duration::from_secs(2);
const BACKEND_PROPS_TIMEOUT: Duration = Duration::from_secs(5);
const BACKEND_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const BACKEND_VRAM_CLEAR_DELAY: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy)]
pub(super) struct LifecycleOptions {
    pub(super) ready_timeout: Duration,
    pub(super) ready_poll_interval: Duration,
    pub(super) stop_timeout: Duration,
    pub(super) vram_clear_delay: Duration,
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

impl ServerManager {
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
    pub(super) fn start_with_options(
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
        let extra_flags =
            validate_backend_extra_flags(extra_flags).map_err(|e| BackendError::new(0, e))?;
        let new_config = RunConfig {
            model: model.to_string(),
            mode: mode.to_string(),
            ctx_override,
            extra_flags: extra_flags.clone(),
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
            Path::new("llama-server").to_path_buf()
        };

        self.ensure_backend_port_available()?;

        let args = build_backend_args(
            &self.backend,
            self.port,
            gguf_path,
            mode,
            &extra_flags,
            ctx_override,
            cache_type_k,
            cache_type_v,
            n_slots,
            kv_unified,
        )
        .map_err(|e| BackendError::new(0, e))?;
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

        let mut last_err = None;
        for i in 0..10 {
            if i > 0 {
                thread::sleep(Duration::from_millis(50));
            }
            if TcpStream::connect_timeout(&addr, Duration::from_millis(50)).is_ok() {
                last_err = Some(BackendError::new(
                    0,
                    format!("Backend port {} is already accepting connections", port),
                ));
                continue;
            }
            match TcpListener::bind(("127.0.0.1", port)) {
                Ok(listener) => {
                    drop(listener);
                    return Ok(());
                }
                Err(e) => {
                    last_err = Some(BackendError::new(
                        0,
                        format!("Backend port {} is not available on 127.0.0.1: {}", port, e),
                    ));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            BackendError::new(0, format!("Backend port {} is not available", port))
        }))
    }

    pub(super) fn wait_until_ready(
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

    /// Stop the backend process. For ollama, runs 'ollama stop'.
    /// Waits up to 10s for termination, then forces kill.
    /// Sleeps 3s after termination for VRAM clearing.
    pub fn stop(&self) -> Result<(), BackendError> {
        self.stop_with_options(BACKEND_STOP_TIMEOUT, BACKEND_VRAM_CLEAR_DELAY)
    }

    pub(super) fn stop_with_options(
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
