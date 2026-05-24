use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::Mutex;

/// Tracked configuration for reuse decisions during start().
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RunConfig {
    pub(super) model: String,
    pub(super) mode: String,
    pub(super) ctx_override: Option<i64>,
    pub(super) extra_flags: Vec<String>,
    pub(super) cache_type_k: Option<String>,
    pub(super) cache_type_v: Option<String>,
    pub(super) n_slots: Option<i64>,
    pub(super) kv_unified: bool,
}

/// Backend process lifecycle manager.
///
/// Handles subprocess spawning for llamaserver/llamafile, budget resolution,
/// VRAM tier detection, and health polling.
pub struct ServerManager {
    pub(super) backend: String,
    pub(super) port: i64,
    pub(super) _models_dir: Option<PathBuf>,
    pub(super) llamafile_runtime: Option<PathBuf>,
    pub(super) process: Mutex<Option<Child>>,
    pub(super) current_config: Mutex<Option<RunConfig>>,
    pub(super) last_context: Mutex<Option<i64>>,
}

impl ServerManager {
    /// Creates a new `ServerManager` tracking the specified backend and port.
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

    /// Sets the executable path for the llamafile runtime.
    pub fn with_llamafile_runtime(mut self, path: impl AsRef<Path>) -> Self {
        self.llamafile_runtime = Some(path.as_ref().to_path_buf());
        self
    }
}
