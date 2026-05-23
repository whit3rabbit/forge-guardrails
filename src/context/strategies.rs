//! Compaction strategies for context window management.

use crate::context::hardware::estimate_tokens_heuristic;
use crate::core::message::MessageType;

/// Trait for compaction strategies that compress message history.
///
/// Implementations receive the full message list, a token budget, and an
/// optional step hint. They return a new (possibly unchanged) message list
/// and a phase indicator: 0 means no compaction, 1+ means compaction.
pub trait CompactStrategy: Send + Sync {
    fn compact(
        &self,
        messages: &[crate::core::message::Message],
        budget_tokens: i64,
        step_hint: Option<&str>,
    ) -> (Vec<crate::core::message::Message>, i64);
}

/// Passthrough strategy that performs no compaction.
///
/// Returns a shallow clone of the input with phase 0.
pub struct NoCompact;

impl CompactStrategy for NoCompact {
    fn compact(
        &self,
        messages: &[crate::core::message::Message],
        _budget_tokens: i64,
        _step_hint: Option<&str>,
    ) -> (Vec<crate::core::message::Message>, i64) {
        (messages.to_vec(), 0)
    }
}

/// Sliding-window compaction that keeps the system prompt, user input,
/// and the N most recent iterations.
pub struct SlidingWindowCompact {
    pub keep_recent: i64,
    pub compact_threshold: f64,
}

impl SlidingWindowCompact {
    pub fn new(keep_recent: i64) -> Self {
        Self {
            keep_recent,
            compact_threshold: 0.75,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.compact_threshold = threshold;
        self
    }
}

impl CompactStrategy for SlidingWindowCompact {
    fn compact(
        &self,
        messages: &[crate::core::message::Message],
        budget_tokens: i64,
        _step_hint: Option<&str>,
    ) -> (Vec<crate::core::message::Message>, i64) {
        let trigger = (budget_tokens as f64 * self.compact_threshold) as i64;
        let current_tokens = estimate_tokens_heuristic(messages);
        if current_tokens <= trigger {
            return (messages.to_vec(), 0);
        }

        // Messages 0 and 1 are always preserved (system prompt + user input).
        if messages.len() <= 2 {
            return (messages.to_vec(), 1);
        }

        let (header, rest) = messages.split_at(2);
        let protected = find_protected_window(rest, self.keep_recent);

        let mut result = header.to_vec();
        result.extend_from_slice(protected);
        (result, 1)
    }
}

/// Three-phase progressive compaction strategy.
///
/// Phase 1: Drop nudge messages, truncate long tool results.
/// Phase 2: Phase 1 + drop all tool results.
/// Phase 3: Phase 2 + drop reasoning and text_response (tool_call skeleton only).
pub struct TieredCompact {
    pub keep_recent: i64,
    pub compact_threshold: f64,
    pub phase_thresholds: Option<[f64; 3]>,
}

/// Truncation limit for tool results in Phase 1.
const TRUNCATION_LIMIT: usize = 200;

impl TieredCompact {
    pub fn new(keep_recent: i64) -> Self {
        Self {
            keep_recent,
            compact_threshold: 0.75,
            phase_thresholds: None,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.compact_threshold = threshold;
        self
    }

    pub fn with_phase_thresholds(mut self, thresholds: [f64; 3]) -> Self {
        self.phase_thresholds = Some(thresholds);
        self
    }

    fn phase_triggers(&self, budget: i64) -> [i64; 3] {
        match self.phase_thresholds {
            Some(ref t) => [
                (budget as f64 * t[0]) as i64,
                (budget as f64 * t[1]) as i64,
                (budget as f64 * t[2]) as i64,
            ],
            None => [
                (budget as f64 * self.compact_threshold) as i64,
                (budget as f64 * self.compact_threshold) as i64,
                (budget as f64 * self.compact_threshold) as i64,
            ],
        }
    }
}

impl CompactStrategy for TieredCompact {
    fn compact(
        &self,
        messages: &[crate::core::message::Message],
        budget_tokens: i64,
        _step_hint: Option<&str>,
    ) -> (Vec<crate::core::message::Message>, i64) {
        let triggers = self.phase_triggers(budget_tokens);
        let initial_tokens = estimate_tokens_heuristic(messages);

        // Check if any compaction is needed.
        if initial_tokens <= triggers[0] {
            return (messages.to_vec(), 0);
        }

        if messages.len() <= 2 {
            return (
                messages.to_vec(),
                if initial_tokens > triggers[0] { 1 } else { 0 },
            );
        }

        // Apply phase 1.
        let after_p1 = apply_phase1(messages, self.keep_recent);
        let p1_tokens = estimate_tokens_heuristic(&after_p1);
        if p1_tokens <= triggers[1] {
            return (after_p1, 1);
        }

        // Apply phase 2.
        let after_p2 = apply_phase2(&after_p1, self.keep_recent);
        let p2_tokens = estimate_tokens_heuristic(&after_p2);
        if p2_tokens <= triggers[2] {
            return (after_p2, 2);
        }

        // Apply phase 3.
        let after_p3 = apply_phase3(&after_p2, self.keep_recent);
        (after_p3, 3)
    }
}

/// Identify messages belonging to the protected window (last keep_recent
/// iterations). Returns a slice of messages from the rest (after header).
fn find_protected_window(
    rest: &[crate::core::message::Message],
    keep_recent: i64,
) -> &[crate::core::message::Message] {
    let mut step_indices: Vec<i64> = rest.iter().filter_map(|m| m.metadata.step_index).collect();
    step_indices.sort();
    step_indices.dedup();

    if step_indices.is_empty() {
        return rest;
    }

    let keep_from = if step_indices.len() <= keep_recent as usize {
        0
    } else {
        step_indices.len() - keep_recent as usize
    };
    let protected_steps: Vec<i64> = step_indices[keep_from..].to_vec();

    let cutoff = rest
        .iter()
        .position(|m| {
            m.metadata
                .step_index
                .is_some_and(|idx| protected_steps.contains(&idx))
        })
        .unwrap_or(rest.len());

    &rest[cutoff..]
}

/// Phase 1: drop nudge messages from eligible zone, truncate long tool results.
fn apply_phase1(
    messages: &[crate::core::message::Message],
    keep_recent: i64,
) -> Vec<crate::core::message::Message> {
    let header = &messages[..2];
    let rest = &messages[2..];
    let protected = find_protected_window(rest, keep_recent);
    let protected_start = rest.len() - protected.len();

    let mut result = header.to_vec();
    for (i, msg) in rest.iter().enumerate() {
        if i >= protected_start {
            // Protected window: pass through unchanged.
            result.push(msg.clone());
            continue;
        }

        match msg.metadata.msg_type {
            // Drop nudge types entirely.
            MessageType::StepNudge | MessageType::PrerequisiteNudge | MessageType::RetryNudge => {
                continue
            }
            // Truncate long tool results.
            MessageType::ToolResult => {
                if msg.content.len() > TRUNCATION_LIMIT {
                    let truncated: String = msg.content.chars().take(TRUNCATION_LIMIT).collect();
                    let removed = msg.content.len() - TRUNCATION_LIMIT;
                    let mut new_msg = msg.clone();
                    new_msg.content = format!("{}...[{} characters removed]", truncated, removed);
                    result.push(new_msg);
                } else {
                    result.push(msg.clone());
                }
            }
            // Everything else passes through.
            _ => result.push(msg.clone()),
        }
    }
    result
}

/// Phase 2: Phase 1 + drop all tool_result and tool_call messages from eligible zone.
fn apply_phase2(
    messages: &[crate::core::message::Message],
    keep_recent: i64,
) -> Vec<crate::core::message::Message> {
    let header = &messages[..2];
    let rest = &messages[2..];
    let protected = find_protected_window(rest, keep_recent);
    let protected_start = rest.len() - protected.len();

    let mut result = header.to_vec();
    for (i, msg) in rest.iter().enumerate() {
        if i >= protected_start {
            result.push(msg.clone());
            continue;
        }

        match msg.metadata.msg_type {
            MessageType::StepNudge
            | MessageType::PrerequisiteNudge
            | MessageType::RetryNudge
            | MessageType::ToolCall
            | MessageType::ToolResult => continue,
            _ => result.push(msg.clone()),
        }
    }
    result
}

/// Phase 3: Phase 2 + drop reasoning and text_response from eligible zone.
fn apply_phase3(
    messages: &[crate::core::message::Message],
    keep_recent: i64,
) -> Vec<crate::core::message::Message> {
    let header = &messages[..2];
    let rest = &messages[2..];
    let protected = find_protected_window(rest, keep_recent);
    let protected_start = rest.len() - protected.len();

    let mut result = header.to_vec();
    for (i, msg) in rest.iter().enumerate() {
        if i >= protected_start {
            result.push(msg.clone());
            continue;
        }

        match msg.metadata.msg_type {
            MessageType::StepNudge
            | MessageType::PrerequisiteNudge
            | MessageType::RetryNudge
            | MessageType::ToolResult
            | MessageType::Reasoning
            | MessageType::TextResponse => continue,
            _ => result.push(msg.clone()),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::message::{Message, MessageMeta, MessageRole, MessageType};

    fn sys_msg(content: &str) -> Message {
        Message::new(
            MessageRole::System,
            content,
            MessageMeta::new(MessageType::SystemPrompt),
        )
    }

    fn user_msg(content: &str) -> Message {
        Message::new(
            MessageRole::User,
            content,
            MessageMeta::new(MessageType::UserInput),
        )
    }

    fn tool_call_msg(step: i64, content: &str) -> Message {
        Message::new(
            MessageRole::Assistant,
            content,
            MessageMeta::new(MessageType::ToolCall).with_step_index(step),
        )
    }

    fn tool_result_msg(step: i64, content: &str) -> Message {
        Message::new(
            MessageRole::Tool,
            content,
            MessageMeta::new(MessageType::ToolResult).with_step_index(step),
        )
    }

    #[test]
    fn no_compact_returns_clone() {
        let msgs = vec![sys_msg("sys"), user_msg("usr")];
        let (result, phase) = NoCompact.compact(&msgs, 1000, None);
        assert_eq!(phase, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn sliding_window_preserves_header() {
        let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
        for step in 0..5 {
            msgs.push(tool_call_msg(step, "call"));
            msgs.push(tool_result_msg(step, "result"));
        }
        let strategy = SlidingWindowCompact::new(2);
        let (result, _phase) = strategy.compact(&msgs, 1, None);
        assert_eq!(result[0].content, "sys");
        assert_eq!(result[1].content, "usr");
        assert_eq!(result[0].metadata.msg_type, MessageType::SystemPrompt);
        assert_eq!(result[1].metadata.msg_type, MessageType::UserInput);
    }
}
