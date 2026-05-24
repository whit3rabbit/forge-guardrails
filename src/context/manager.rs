//! Context window manager with token tracking, compaction triggering,
//! and threshold callbacks.

use crate::context::hardware::estimate_tokens_heuristic;
use crate::context::strategies::CompactStrategy;
use crate::core::message::{Message, ToolCallInfo};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Immutable record of a compaction event.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactEvent {
    /// The step index at which the compaction occurred.
    pub step_index: i64,
    /// Estimated token count of the context prior to compaction.
    pub tokens_before: i64,
    /// Estimated token count of the context after compaction.
    pub tokens_after: i64,
    /// Total context token budget.
    pub budget_tokens: i64,
    /// Message count in the conversation list prior to compaction.
    pub messages_before: usize,
    /// Message count in the conversation list after compaction.
    pub messages_after: usize,
    /// The compaction phase reached (e.g. 1, 2, 3).
    pub phase_reached: i64,
}

/// Callback type invoked when compaction occurs.
pub type OnCompactFn = Box<dyn Fn(&CompactEvent) + Send + Sync>;

/// Callback type for threshold warnings. Returns an optional warning string.
pub type OnThresholdFn = Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync>;

#[derive(Debug, Clone, Copy)]
struct StoredTokenCount {
    count: i64,
    messages_fingerprint: Option<u64>,
}

impl StoredTokenCount {
    fn matches(self, messages_fingerprint: u64) -> bool {
        self.messages_fingerprint
            .map(|fingerprint| fingerprint == messages_fingerprint)
            .unwrap_or(true)
    }
}

/// Central context budget tracker.
///
/// Wraps a compaction strategy and provides token estimation, threshold
/// checking, and compaction triggering. Tracks a stored token count scoped to
/// the last observed message list when possible.
pub struct ContextManager {
    strategy: Box<dyn CompactStrategy>,
    budget_tokens: i64,
    on_compact: Option<OnCompactFn>,
    context_thresholds: Option<Vec<f64>>,
    on_context_threshold: Option<OnThresholdFn>,
    stored_token_count: Option<StoredTokenCount>,
    last_observed_messages_fingerprint: Option<u64>,
    fired_thresholds: Vec<bool>,
}

impl ContextManager {
    /// Creates a new `ContextManager` with the specified strategy, budget, and callbacks.
    pub fn new(
        strategy: Box<dyn CompactStrategy>,
        budget_tokens: i64,
        on_compact: Option<OnCompactFn>,
        context_thresholds: Option<Vec<f64>>,
        on_context_threshold: Option<OnThresholdFn>,
    ) -> Self {
        let fired = context_thresholds
            .as_ref()
            .map(|t| vec![false; t.len()])
            .unwrap_or_default();
        // Sort thresholds ascending for deterministic processing.
        let sorted_thresholds = context_thresholds.map(|mut t| {
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            t
        });
        Self {
            strategy,
            budget_tokens,
            on_compact,
            context_thresholds: sorted_thresholds,
            on_context_threshold,
            stored_token_count: None,
            last_observed_messages_fingerprint: None,
            fired_thresholds: fired,
        }
    }

    /// Get the context budget in tokens.
    pub fn budget(&self) -> i64 {
        self.budget_tokens
    }

    /// Estimate token count for messages.
    ///
    /// Returns the stored count if `update_token_count` was called for this
    /// message list, otherwise falls back to the character-count heuristic.
    pub fn estimate_tokens(&self, messages: &[Message]) -> i64 {
        let fingerprint = message_fingerprint(messages);
        match self.stored_token_count {
            Some(stored) if stored.matches(fingerprint) => stored.count,
            _ => estimate_tokens_heuristic(messages),
        }
    }

    /// Store an actual token count from the backend.
    ///
    /// The count is tied to the most recent message list observed by
    /// `maybe_compact` or `check_thresholds`. If no message list has been
    /// observed, the count remains unscoped for backwards compatibility.
    pub fn update_token_count(&mut self, count: i64) {
        self.stored_token_count = Some(StoredTokenCount {
            count,
            messages_fingerprint: self.last_observed_messages_fingerprint,
        });
    }

    fn estimate_current_tokens(&mut self, messages: &[Message]) -> i64 {
        let fingerprint = self.observe_messages(messages);
        match self.stored_token_count {
            Some(stored) if stored.matches(fingerprint) => stored.count,
            Some(_) => {
                self.stored_token_count = None;
                estimate_tokens_heuristic(messages)
            }
            None => estimate_tokens_heuristic(messages),
        }
    }

    fn observe_messages(&mut self, messages: &[Message]) -> u64 {
        let fingerprint = message_fingerprint(messages);
        self.last_observed_messages_fingerprint = Some(fingerprint);
        fingerprint
    }

    /// Apply compaction if the strategy deems it necessary.
    ///
    /// Returns the original message slice when no compaction occurs (phase 0),
    /// or a new list when compaction occurs (phase > 0).
    pub fn maybe_compact<'a>(
        &mut self,
        messages: &'a [Message],
        step_index: i64,
        step_hint: Option<&str>,
    ) -> std::borrow::Cow<'a, [Message]> {
        let tokens_before = self.estimate_current_tokens(messages);
        let (compacted, phase) = self
            .strategy
            .compact(messages, self.budget_tokens, step_hint);

        if phase == 0 {
            return std::borrow::Cow::Borrowed(messages);
        }

        let tokens_after = estimate_tokens_heuristic(&compacted);
        let event = CompactEvent {
            step_index,
            tokens_before,
            tokens_after,
            budget_tokens: self.budget_tokens,
            messages_before: messages.len(),
            messages_after: compacted.len(),
            phase_reached: phase,
        };

        if let Some(ref callback) = self.on_compact {
            callback(&event);
        }

        // Clear stored token count so heuristic runs on next estimate.
        self.stored_token_count = None;

        std::borrow::Cow::Owned(compacted)
    }

    /// Check context thresholds and fire the highest unfired threshold
    /// callback if usage crosses it.
    ///
    /// Returns `None` when thresholds or callback are not configured,
    /// budget is zero/negative, or no threshold is newly crossed.
    pub fn check_thresholds(&mut self, messages: &[Message]) -> Option<String> {
        if self.context_thresholds.is_none() || self.on_context_threshold.is_none() {
            return None;
        }

        if self.budget_tokens <= 0 {
            return None;
        }

        let tokens = self.estimate_current_tokens(messages);
        let pct = tokens as f64 / self.budget_tokens as f64;
        let thresholds = self.context_thresholds.as_ref()?;

        // Reset thresholds where usage has dropped below them.
        for (i, &threshold) in thresholds.iter().enumerate() {
            if pct < threshold && self.fired_thresholds[i] {
                self.fired_thresholds[i] = false;
            }
        }

        // Find highest unfired threshold that is crossed.
        // Thresholds are sorted ascending, so iterate in reverse.
        let mut fired_idx: Option<usize> = None;
        for (i, &threshold) in thresholds.iter().enumerate().rev() {
            if pct >= threshold && !self.fired_thresholds[i] {
                fired_idx = Some(i);
                break;
            }
        }

        let idx = fired_idx?;

        self.fired_thresholds[idx] = true;
        let callback = self.on_context_threshold.as_ref()?;
        callback(tokens, self.budget_tokens, pct)
    }
}

fn message_fingerprint(messages: &[Message]) -> u64 {
    let mut hasher = DefaultHasher::new();
    messages.len().hash(&mut hasher);
    for message in messages {
        message.role.hash(&mut hasher);
        message.content.hash(&mut hasher);
        message.metadata.msg_type.hash(&mut hasher);
        message.metadata.step_index.hash(&mut hasher);
        message.metadata.original_type.hash(&mut hasher);
        message.metadata.token_estimate.hash(&mut hasher);
        message.tool_name.hash(&mut hasher);
        message.tool_call_id.hash(&mut hasher);
        hash_tool_calls(&message.tool_calls, &mut hasher);
    }
    hasher.finish()
}

fn hash_tool_calls(tool_calls: &Option<Vec<ToolCallInfo>>, hasher: &mut DefaultHasher) {
    match tool_calls {
        Some(calls) => {
            true.hash(hasher);
            calls.len().hash(hasher);
            for call in calls {
                call.name.hash(hasher);
                call.call_id.hash(hasher);
                match &call.args {
                    Some(args) => {
                        true.hash(hasher);
                        args.len().hash(hasher);
                        for (key, value) in args {
                            key.hash(hasher);
                            value.to_string().hash(hasher);
                        }
                    }
                    None => false.hash(hasher),
                }
            }
        }
        None => false.hash(hasher),
    }
}

/// Default context warning callback.
///
/// Escalating message: >= 80% mentions "nearly full", >= 65% mentions
/// "filling up", otherwise a mild reminder. Always includes percentage,
/// token count, and budget.
pub fn default_context_warning(tokens: i64, budget: i64, pct: f64) -> Option<String> {
    let pct_display = (pct * 100.0) as i64;
    let message = if pct >= 0.80 {
        format!(
            "Context window nearly full: {}% ({} / {} tokens)",
            pct_display, tokens, budget
        )
    } else if pct >= 0.65 {
        format!(
            "Context window filling up: {}% ({} / {} tokens)",
            pct_display, tokens, budget
        )
    } else {
        format!(
            "Context usage at {}% ({} / {} tokens)",
            pct_display, tokens, budget
        )
    };
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::strategies::NoCompact;
    use crate::core::message::{Message, MessageMeta, MessageRole, MessageType};

    #[test]
    fn compact_event_fields() {
        let event = CompactEvent {
            step_index: 5,
            tokens_before: 1000,
            tokens_after: 500,
            budget_tokens: 800,
            messages_before: 10,
            messages_after: 6,
            phase_reached: 2,
        };
        assert_eq!(event.step_index, 5);
        assert_eq!(event.tokens_after, 500);
        assert_eq!(event.phase_reached, 2);
    }

    #[test]
    fn estimate_tokens_heuristic_fallback() {
        let msgs = vec![Message::new(
            MessageRole::User,
            "a".repeat(100),
            MessageMeta::new(MessageType::UserInput),
        )];
        let mgr = ContextManager::new(Box::new(NoCompact), 1000, None, None, None);
        assert_eq!(mgr.estimate_tokens(&msgs), 25);
    }

    #[test]
    fn update_token_count_overrides_heuristic() {
        let msgs = vec![Message::new(
            MessageRole::User,
            "a".repeat(100),
            MessageMeta::new(MessageType::UserInput),
        )];
        let mut mgr = ContextManager::new(Box::new(NoCompact), 1000, None, None, None);
        mgr.update_token_count(500);
        assert_eq!(mgr.estimate_tokens(&msgs), 500);
    }

    #[test]
    fn default_warning_escalates() {
        let w50 = default_context_warning(400, 800, 0.50).unwrap();
        assert!(w50.contains("50%"));
        assert!(!w50.contains("nearly full"));
        assert!(!w50.contains("filling up"));

        let w65 = default_context_warning(520, 800, 0.65).unwrap();
        assert!(w65.contains("filling up"));

        let w80 = default_context_warning(640, 800, 0.80).unwrap();
        assert!(w80.contains("nearly full"));
    }
}
