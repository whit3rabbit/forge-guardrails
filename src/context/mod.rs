//! Context tracking and compaction strategies.

/// Hardware memory context and token limits probing.
pub mod hardware;
/// Memory budget manager and compaction orchestration.
pub mod manager;
/// Token budget compaction strategies.
pub mod strategies;

pub use hardware::{detect_hardware, HardwareProfile, MemoryKind};
pub use manager::{default_context_warning, CompactEvent, ContextManager};
pub use strategies::{CompactStrategy, NoCompact, SlidingWindowCompact, TieredCompact};
