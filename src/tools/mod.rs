//! Built-in tools and executors for the agent runtime.

/// The respond tool for signaling completion or sending direct messages.
pub mod respond;

pub use respond::{respond_spec, respond_tool, RESPOND_TOOL_NAME};
