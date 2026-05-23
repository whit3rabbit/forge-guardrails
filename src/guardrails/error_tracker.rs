/// Tracks two independent counters: consecutive retries and consecutive tool errors.
///
/// Each counter has a configurable exhaustion threshold. Exhaustion is determined
/// by strict greater-than comparison (counter must exceed max, not just equal it).
/// Soft errors are excluded from the tool error counter.
/// Success does not auto-reset counters; only explicit reset methods clear them.
pub struct ErrorTracker {
    consecutive_retries: i32,
    consecutive_tool_errors: i32,
    max_retries: i32,
    max_tool_errors: i32,
}

impl ErrorTracker {
    pub fn new(max_retries: i32, max_tool_errors: i32) -> Self {
        Self {
            consecutive_retries: 0,
            consecutive_tool_errors: 0,
            max_retries,
            max_tool_errors,
        }
    }

    pub fn record_retry(&mut self) {
        self.consecutive_retries += 1;
    }

    pub fn reset_retries(&mut self) {
        self.consecutive_retries = 0;
    }

    /// Record a tool execution result. Soft errors are excluded from the counter.
    /// Success does not reset the counter (only reset_errors does).
    pub fn record_result(&mut self, success: bool, is_soft_error: bool) {
        if !success && !is_soft_error {
            self.consecutive_tool_errors += 1;
        }
    }

    pub fn reset_errors(&mut self) {
        self.consecutive_tool_errors = 0;
    }

    pub fn retries_exhausted(&self) -> bool {
        self.consecutive_retries > self.max_retries
    }

    pub fn tool_errors_exhausted(&self) -> bool {
        self.consecutive_tool_errors > self.max_tool_errors
    }

    pub fn consecutive_retries(&self) -> i32 {
        self.consecutive_retries
    }

    pub fn consecutive_tool_errors(&self) -> i32 {
        self.consecutive_tool_errors
    }

    pub fn max_retries(&self) -> i32 {
        self.max_retries
    }
}
