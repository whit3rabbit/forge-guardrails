//! Integration tests for context tests.

use forge_guardrails::{
    default_context_warning, CompactEvent, ContextManager, Message, MessageMeta, MessageRole,
    MessageType, NoCompact, TieredCompact,
};

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

fn build_6pair() -> Vec<Message> {
    let mut msgs = vec![sys_msg("system prompt"), user_msg("user input")];
    for step in 0..6 {
        msgs.push(tool_call_msg(step, &format!("call_{}", step)));
        msgs.push(tool_result_msg(step, &format!("result_{}", step)));
    }
    msgs
}

// ts-041: Token estimation heuristic: total chars / 4 via integer division.
#[test]
fn estimate_tokens_two_messages() {
    let msgs = vec![
        Message::new(
            MessageRole::User,
            "a".repeat(100),
            MessageMeta::new(MessageType::UserInput),
        ),
        Message::new(
            MessageRole::Assistant,
            "b".repeat(200),
            MessageMeta::new(MessageType::TextResponse),
        ),
    ];
    let mgr = ContextManager::new(Box::new(NoCompact), 10000, None, None, None);
    assert_eq!(mgr.estimate_tokens(&msgs), 75);
}

#[test]
fn estimate_tokens_floor_division() {
    let msgs = vec![Message::new(
        MessageRole::User,
        "a".repeat(41),
        MessageMeta::new(MessageType::UserInput),
    )];
    let mgr = ContextManager::new(Box::new(NoCompact), 10000, None, None, None);
    assert_eq!(mgr.estimate_tokens(&msgs), 10);
}

#[test]
fn estimate_tokens_empty() {
    let msgs: Vec<Message> = vec![];
    let mgr = ContextManager::new(Box::new(NoCompact), 10000, None, None, None);
    assert_eq!(mgr.estimate_tokens(&msgs), 0);
}

// ts-042: update_token_count overrides heuristic.
#[test]
fn update_token_count_overrides() {
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
fn update_token_count_applies_to_same_observed_messages() {
    let msgs = vec![user_msg("small")];
    let mut mgr = ContextManager::new(Box::new(NoCompact), 1000, None, None, None);
    let _ = mgr.maybe_compact(&msgs, 0, None);

    mgr.update_token_count(500);

    assert_eq!(mgr.estimate_tokens(&msgs), 500);
}

#[test]
fn stored_token_count_is_ignored_after_message_mutation() {
    let mut msgs = vec![user_msg("small")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|tokens, _, _| Some(format!("tokens={tokens}")));
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        1000,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    let _ = mgr.maybe_compact(&msgs, 0, None);
    mgr.update_token_count(800);

    msgs.push(user_msg("changed"));

    assert!(mgr.check_thresholds(&msgs).is_none());
    assert!(mgr.estimate_tokens(&msgs) < 500);
}

// ts-043: maybe_compact returns original list reference when phase 0.
#[test]
fn maybe_compact_returns_original_on_phase0() {
    let msgs = vec![sys_msg("sys"), user_msg("usr")];
    let mut mgr = ContextManager::new(Box::new(NoCompact), 10000, None, None, None);
    let result = mgr.maybe_compact(&msgs, 0, None);
    assert!(matches!(result, std::borrow::Cow::Borrowed(_)));
}

// ts-044: maybe_compact calls on_compact with accurate CompactEvent.
#[test]
fn maybe_compact_emits_event() {
    let msgs = build_6pair();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let on_compact: Box<dyn Fn(&CompactEvent) + Send + Sync> = Box::new(move |e: &CompactEvent| {
        events_clone.lock().unwrap().push(e.clone());
    });
    let mut mgr = ContextManager::new(
        Box::new(TieredCompact::new(2).with_threshold(0.0)),
        1,
        Some(on_compact),
        None,
        None,
    );
    let _result = mgr.maybe_compact(&msgs, 5, None);
    let evts = events.lock().unwrap();
    assert_eq!(evts.len(), 1);
    let evt = &evts[0];
    assert!(evt.tokens_before > evt.tokens_after);
    assert_eq!(evt.budget_tokens, 1);
    assert!(evt.messages_before > evt.messages_after);
    assert!(evt.phase_reached >= 1);
    assert_eq!(evt.step_index, 5);
}

// ts-045: maybe_compact with on_compact=None does not error.
#[test]
fn maybe_compact_no_callback() {
    let msgs = build_6pair();
    let mut mgr = ContextManager::new(
        Box::new(TieredCompact::new(2).with_threshold(0.0)),
        1,
        None,
        None,
        None,
    );
    let result = mgr.maybe_compact(&msgs, 0, None);
    assert!(matches!(result, std::borrow::Cow::Owned(_)));
}

// ts-046: CompactEvent is immutable (all fields are non-mutable in Rust).
#[test]
fn compact_event_immutability() {
    let evt = CompactEvent {
        step_index: 1,
        tokens_before: 100,
        tokens_after: 50,
        budget_tokens: 80,
        messages_before: 10,
        messages_after: 5,
        phase_reached: 2,
    };
    // Fields are accessible but not mutable (no &mut methods).
    assert_eq!(evt.phase_reached, 2);
}

// ts-047: default_context_warning at 50%.
#[test]
fn default_warning_50pct() {
    let result = default_context_warning(400, 800, 0.50).unwrap();
    assert!(result.contains("50%"));
}

// ts-048: default_context_warning at 65%.
#[test]
fn default_warning_65pct() {
    let result = default_context_warning(520, 800, 0.65).unwrap();
    assert!(result.contains("filling up"));
}

// ts-049: default_context_warning at 80%.
#[test]
fn default_warning_80pct() {
    let result = default_context_warning(640, 800, 0.80).unwrap();
    assert!(result.contains("nearly full"));
}

// ts-050: default_context_warning escalates with different messages.
#[test]
fn default_warning_escalates() {
    let w50 = default_context_warning(400, 800, 0.50).unwrap();
    let w80 = default_context_warning(640, 800, 0.80).unwrap();
    assert!(!w50.contains("nearly full"));
    assert!(w80.contains("nearly full"));
    assert!(w50 != w80);
}

// ts-051: check_thresholds with no callback returns None.
#[test]
fn check_thresholds_no_callback() {
    let msgs = vec![sys_msg("sys")];
    let mut mgr = ContextManager::new(Box::new(NoCompact), 8000, None, Some(vec![0.5]), None);
    // Clear stored count so heuristic runs.
    mgr.update_token_count(5000);
    assert!(mgr.check_thresholds(&msgs).is_none());
}

// ts-052: check_thresholds with no thresholds returns None.
#[test]
fn check_thresholds_no_thresholds() {
    let msgs = vec![sys_msg("sys")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|_, _, _| Some("warning".to_string()));
    let mut mgr = ContextManager::new(Box::new(NoCompact), 8000, None, None, Some(callback));
    mgr.update_token_count(5000);
    assert!(mgr.check_thresholds(&msgs).is_none());
}

// ts-053: check_thresholds fires when threshold is crossed.
#[test]
fn check_thresholds_fires_on_cross() {
    let msgs = vec![sys_msg("sys")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|tokens, budget, pct| {
            Some(format!("{}% ({}/{})", (pct * 100.0) as i64, tokens, budget))
        });
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        8000,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    mgr.update_token_count(4800); // 60% of 8000
    let result = mgr.check_thresholds(&msgs);
    assert!(result.is_some());
    let msg = result.unwrap();
    assert!(msg.contains("60%"));
}

// ts-054: check_thresholds does not fire below threshold.
#[test]
fn check_thresholds_no_fire_below() {
    let msgs = vec![sys_msg("sys")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|_, _, _| Some("warning".to_string()));
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        8000,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    mgr.update_token_count(3200); // 40% of 8000
    assert!(mgr.check_thresholds(&msgs).is_none());
}

// ts-055: check_thresholds fires once per crossing.
#[test]
fn check_thresholds_fire_once() {
    let msgs = vec![sys_msg("sys")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|_, _, _| Some("warning".to_string()));
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        8000,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    mgr.update_token_count(4800); // 60% of 8000
    let first = mgr.check_thresholds(&msgs);
    assert!(first.is_some());
    let second = mgr.check_thresholds(&msgs);
    assert!(second.is_none());
}

// ts-056: check_thresholds resets after usage drops.
#[test]
fn check_thresholds_reset_on_drop() {
    let msgs = vec![sys_msg("sys")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|_, _, _| Some("warning".to_string()));
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        8000,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    // Cross threshold.
    mgr.update_token_count(4800);
    let first = mgr.check_thresholds(&msgs);
    assert!(first.is_some());

    // Drop below threshold (simulating post-compaction).
    mgr.update_token_count(2400); // 30%
    let _ = mgr.check_thresholds(&msgs); // Should reset.

    // Cross again.
    mgr.update_token_count(4800);
    let second = mgr.check_thresholds(&msgs);
    assert!(second.is_some());
}

// ts-057: check_thresholds fires highest crossed threshold.
#[test]
fn check_thresholds_highest_crossed() {
    let msgs = vec![sys_msg("sys")];
    let fired = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let fired_clone = fired.clone();
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(move |_tokens, _budget, pct| {
            fired_clone.lock().unwrap().push(pct);
            Some(format!("{}%", (pct * 100.0) as i64))
        });
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        8000,
        None,
        Some(vec![0.3, 0.5, 0.7]),
        Some(callback),
    );
    // Jump to 60%: crosses 0.3 and 0.5, but not 0.7.
    // Only highest unfired (0.5) should fire.
    mgr.update_token_count(4800); // 60%
    let result = mgr.check_thresholds(&msgs);
    assert!(result.is_some());
    assert_eq!(fired.lock().unwrap().len(), 1);
    // The fired threshold should be 0.5 (the highest below 0.6).
    let fired_pct = fired.lock().unwrap()[0];
    assert!((fired_pct - 0.6).abs() < f64::EPSILON);
}

// ts-058: Custom callback returning None: threshold fires but returns None.
#[test]
fn check_thresholds_custom_callback_returns_none() {
    let msgs = vec![sys_msg("sys")];
    let called = std::sync::Arc::new(std::sync::Mutex::new(false));
    let called_clone = called.clone();
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(move |_, _, _| {
            *called_clone.lock().unwrap() = true;
            None
        });
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        8000,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    mgr.update_token_count(4800);
    let result = mgr.check_thresholds(&msgs);
    assert!(result.is_none());
    assert!(*called.lock().unwrap());
}

// ts-059: check_thresholds with zero budget returns None.
#[test]
fn check_thresholds_zero_budget() {
    let msgs = vec![sys_msg("sys")];
    let callback: Box<dyn Fn(i64, i64, f64) -> Option<String> + Send + Sync> =
        Box::new(|_, _, _| Some("warning".to_string()));
    let mut mgr = ContextManager::new(
        Box::new(NoCompact),
        0,
        None,
        Some(vec![0.5]),
        Some(callback),
    );
    mgr.update_token_count(100);
    assert!(mgr.check_thresholds(&msgs).is_none());
}

// ts-060: Per-phase thresholds through ContextManager, phase 1 only.
#[test]
fn context_manager_phase1_event() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let on_compact: Box<dyn Fn(&CompactEvent) + Send + Sync> = Box::new(move |e: &CompactEvent| {
        events_clone.lock().unwrap().push(e.clone());
    });
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    // Use enough content so phase 1 compaction (just calls + results, no nudges to drop)
    // is still meaningful. With phase_thresholds that stop after phase 1.
    let big_nudge = "n".repeat(400);
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(Message::new(
            MessageRole::Assistant,
            &big_nudge,
            MessageMeta::new(MessageType::StepNudge).with_step_index(step),
        ));
        msgs.push(tool_result_msg(step, "result"));
    }
    // Same threshold logic as tiered_stops_at_phase1: budget=500, [0.5, 0.5, 1.0]
    let mut mgr = ContextManager::new(
        Box::new(TieredCompact::new(2).with_phase_thresholds([0.5, 0.5, 1.0])),
        500,
        Some(on_compact),
        None,
        None,
    );
    let _result = mgr.maybe_compact(&msgs, 3, None);
    let evts = events.lock().unwrap();
    assert_eq!(evts.len(), 1);
    assert_eq!(evts[0].phase_reached, 1);
}

// ts-061: All phases through ContextManager, phase 3.
#[test]
fn context_manager_phase3_event() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let on_compact: Box<dyn Fn(&CompactEvent) + Send + Sync> = Box::new(move |e: &CompactEvent| {
        events_clone.lock().unwrap().push(e.clone());
    });
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long = "x".repeat(500);
    for step in 0..6 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long));
        msgs.push(Message::new(
            MessageRole::Assistant,
            long.clone(),
            MessageMeta::new(MessageType::Reasoning).with_step_index(step),
        ));
    }
    let mut mgr = ContextManager::new(
        Box::new(TieredCompact::new(2).with_threshold(0.0)),
        1,
        Some(on_compact),
        None,
        None,
    );
    let _result = mgr.maybe_compact(&msgs, 5, None);
    let evts = events.lock().unwrap();
    assert_eq!(evts.len(), 1);
    assert_eq!(evts[0].phase_reached, 3);
}

// ts-062: No compaction through ContextManager when under threshold.
#[test]
fn context_manager_no_compact_under_threshold() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let on_compact: Box<dyn Fn(&CompactEvent) + Send + Sync> = Box::new(move |e: &CompactEvent| {
        events_clone.lock().unwrap().push(e.clone());
    });
    let msgs = vec![sys_msg("sys"), user_msg("usr")];
    let mut mgr = ContextManager::new(
        Box::new(TieredCompact::new(2).with_threshold(0.99)),
        10000,
        Some(on_compact),
        None,
        None,
    );
    let result = mgr.maybe_compact(&msgs, 0, None);
    assert!(matches!(result, std::borrow::Cow::Borrowed(_)));
    assert!(events.lock().unwrap().is_empty());
}
