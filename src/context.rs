//! Context window manager with token tracking, compaction triggering,
//! and threshold callbacks.

use crate::compact::CompactStrategy;
use crate::hardware::estimate_tokens_heuristic;

/// Immutable record of a compaction event.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactEvent {
    pub step_index: i64,
    pub tokens_before: i64,
    pub tokens_after: i64,
    pub budget_tokens: i64,
    pub messages_before: usize,
    pub messages_after: usize,
    pub phase_reached: i64,
}

/// Callback type invoked when compaction occurs.
pub type OnCompactFn = Box<dyn Fn(&CompactEvent)>;

/// Callback type for threshold warnings. Returns an optional warning string.
pub type OnThresholdFn = Box<dyn Fn(i64, i64, f64) -> Option<String>>;

/// Central context budget tracker.
///
/// Wraps a compaction strategy and provides token estimation, threshold
/// checking, and compaction triggering. Tracks a stored token count that
/// overrides the heuristic when set via `update_token_count`.
pub struct ContextManager {
    strategy: Box<dyn CompactStrategy>,
    budget_tokens: i64,
    on_compact: Option<OnCompactFn>,
    context_thresholds: Option<Vec<f64>>,
    on_context_threshold: Option<OnThresholdFn>,
    stored_token_count: Option<i64>,
    fired_thresholds: Vec<bool>,
}

impl ContextManager {
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
            fired_thresholds: fired,
        }
    }

    /// Get the context budget in tokens.
    pub fn budget(&self) -> i64 {
        self.budget_tokens
    }

    /// Estimate token count for messages.
    ///
    /// Returns the stored count if `update_token_count` was called, otherwise
    /// falls back to the character-count heuristic.
    pub fn estimate_tokens(&self, messages: &[crate::Message]) -> i64 {
        match self.stored_token_count {
            Some(count) => count,
            None => estimate_tokens_heuristic(messages),
        }
    }

    /// Store an actual token count from the backend.
    ///
    /// Subsequent `estimate_tokens` calls return this value until the next
    /// update.
    pub fn update_token_count(&mut self, count: i64) {
        self.stored_token_count = Some(count);
    }

    /// Apply compaction if the strategy deems it necessary.
    ///
    /// Returns the original message slice when no compaction occurs (phase 0),
    /// or a new list when compaction occurs (phase > 0).
    pub fn maybe_compact<'a>(
        &mut self,
        messages: &'a [crate::Message],
        step_index: i64,
        step_hint: Option<&str>,
    ) -> std::borrow::Cow<'a, [crate::Message]> {
        let tokens_before = self.estimate_tokens(messages);
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
    pub fn check_thresholds(&mut self, messages: &[crate::Message]) -> Option<String> {
        let thresholds = self.context_thresholds.as_ref()?;
        let callback = self.on_context_threshold.as_ref()?;

        if self.budget_tokens <= 0 {
            return None;
        }

        let tokens = self.estimate_tokens(messages);
        let pct = tokens as f64 / self.budget_tokens as f64;

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
        callback(tokens, self.budget_tokens, pct)
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
    use crate::compact::NoCompact;
    use crate::message::{Message, MessageMeta, MessageRole, MessageType};

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
