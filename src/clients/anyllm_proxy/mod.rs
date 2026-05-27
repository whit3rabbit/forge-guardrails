//! Client adapters for anyllm provider routing.
//!
//! This keeps forge-guardrails responsible for interception, validation, and
//! nudging while delegating provider routing and upstream compatibility to
//! anyllm through either its in-process runtime or a running sidecar.

mod call_info;
mod request;
mod response;
mod runtime;
mod sidecar;
mod streaming;
mod usage;

#[cfg(test)]
mod tests;

pub use runtime::AnyLlmRuntimeClient;
pub use sidecar::AnyLlmProxyClient;

/// Default anyllm_proxy sidecar chat completions endpoint.
///
/// This is an upstream hop used by `AnyLlmProxyClient`, not the public
/// forge-guardrails proxy listen port.
pub const DEFAULT_ANYLLM_PROXY_URL: &str = "http://127.0.0.1:3000/v1/chat/completions";
