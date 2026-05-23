pub mod hardware;
pub mod manager;
pub mod strategies;

pub use hardware::{detect_hardware, HardwareProfile, MemoryKind};
pub use manager::{default_context_warning, CompactEvent, ContextManager};
pub use strategies::{CompactStrategy, NoCompact, SlidingWindowCompact, TieredCompact};
