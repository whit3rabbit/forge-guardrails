//! Backend lifecycle manager: process spawning, budget resolution, VRAM tiers.
//!
//! ServerManager controls llama-server/llamafile/Ollama backend processes.
//! BudgetMode controls how context window size is determined.

mod args;
mod budget;
mod lifecycle;
mod manager;
mod runtime;
mod setup;

pub use budget::BudgetMode;
pub use manager::ServerManager;
pub use setup::setup_backend;

#[cfg(test)]
mod tests;
