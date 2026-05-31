use std::net::TcpListener;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::thread;
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

use super::args::build_backend_args;
use super::lifecycle::LifecycleOptions;
use super::manager::RunConfig;
use super::runtime::validate_llamafile_runtime_path;
use super::*;

static TEST_PORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn port_test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_PORT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
    .expect("backend args")
}

fn args_result(extra_flags: &[String]) -> Result<Vec<String>, String> {
    build_backend_args(
        "llamaserver",
        8080,
        Path::new("model.gguf"),
        "native",
        extra_flags,
        None,
        None,
        None,
        None,
        false,
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

#[cfg(all(unix, not(all(target_os = "macos", target_arch = "x86_64"))))]
fn fake_backend_lifecycle_options() -> LifecycleOptions {
    LifecycleOptions {
        ready_timeout: Duration::from_secs(30),
        ..test_lifecycle_options()
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
fn backend_args_accept_allowlisted_inline_extra_flags() {
    let extra = vec![
        "--reasoning-budget=0".to_string(),
        "--reasoning-format=auto".to_string(),
    ];
    let args = args_for("llamaserver", "native", &extra, None, false);
    assert!(args
        .windows(2)
        .any(|pair| pair[0] == "--reasoning-budget" && pair[1] == "0"));
    assert!(args
        .windows(2)
        .any(|pair| pair[0] == "--reasoning-format" && pair[1] == "auto"));
}

#[test]
fn backend_args_reject_core_override_extra_flags() {
    for flag in ["-m", "--model", "--host", "--port", "-c", "--ctx-size"] {
        let err = args_result(&[flag.to_string(), "unsafe".to_string()]).unwrap_err();
        assert!(err.contains("unsupported backend extra flag"));
    }
}

#[test]
fn backend_args_reject_unsupported_and_malformed_extra_flags() {
    for extra in [
        vec!["--temp".to_string(), "0".to_string()],
        vec!["positional".to_string()],
        vec!["--".to_string(), "--reasoning-budget".to_string()],
        vec!["--reasoning-budget".to_string()],
        vec!["--reasoning-format=".to_string()],
    ] {
        assert!(
            args_result(&extra).is_err(),
            "expected rejection for {extra:?}"
        );
    }
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
fn invalid_extra_flags_fail_before_stopping_existing_backend() {
    let _guard = port_test_guard();
    let test_dir = unique_test_dir("managed-invalid-extra-flags");
    let gguf_path = test_dir.join("model.gguf");
    std::fs::write(&gguf_path, b"dummy gguf").unwrap();
    let child = std::process::Command::new("sleep")
        .arg("5")
        .spawn()
        .unwrap();
    let mgr = ServerManager::new("llamaserver", free_port(), None);
    *mgr.process.lock().unwrap() = Some(child);

    let result = mgr.start_with_options(
        "",
        &gguf_path,
        "native",
        &["--host".to_string(), "0.0.0.0".to_string()],
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
        .contains("unsupported backend extra flag"));
    assert!(mgr
        .process
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .try_wait()
        .unwrap()
        .is_none());
    let _ = mgr.stop_with_options(Duration::from_millis(200), Duration::ZERO);
    let _ = std::fs::remove_dir_all(test_dir);
}

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(all(unix, not(all(target_os = "macos", target_arch = "x86_64"))))]
#[test]
fn forge_fast_restarts_with_half_total_context() {
    let _guard = port_test_guard();
    let test_dir = unique_test_dir("managed-forge-fast");
    let runtime = test_dir.join("llamafile-runtime");
    let log_path = test_dir.join("spawn.log");
    write_fake_http_backend(&runtime, &log_path, None);
    let gguf_path = test_dir.join("model.gguf");
    std::fs::write(&gguf_path, b"dummy gguf").unwrap();
    let mgr = ServerManager::new("llamafile", free_port(), None).with_llamafile_runtime(&runtime);

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
            fake_backend_lifecycle_options(),
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

#[cfg(all(unix, not(all(target_os = "macos", target_arch = "x86_64"))))]
#[test]
fn manual_llamaserver_budget_comes_from_props() {
    let _guard = port_test_guard();
    let test_dir = unique_test_dir("managed-manual-budget");
    let runtime = test_dir.join("llamafile-runtime");
    let log_path = test_dir.join("spawn.log");
    write_fake_http_backend(&runtime, &log_path, Some(2048));
    let gguf_path = test_dir.join("model.gguf");
    std::fs::write(&gguf_path, b"dummy gguf").unwrap();
    let mgr = ServerManager::new("llamafile", free_port(), None).with_llamafile_runtime(&runtime);

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
            fake_backend_lifecycle_options(),
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

#[cfg(all(unix, not(all(target_os = "macos", target_arch = "x86_64"))))]
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
