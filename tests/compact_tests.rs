use forge_guardrails::{
    CompactStrategy, Message, MessageMeta, MessageRole, MessageType, NoCompact,
    SlidingWindowCompact, TieredCompact,
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

fn reasoning_msg(step: i64, content: &str) -> Message {
    Message::new(
        MessageRole::Assistant,
        content,
        MessageMeta::new(MessageType::Reasoning).with_step_index(step),
    )
}

fn text_response_msg(step: i64, content: &str) -> Message {
    Message::new(
        MessageRole::Assistant,
        content,
        MessageMeta::new(MessageType::TextResponse).with_step_index(step),
    )
}

fn nudge_msg(step: i64, msg_type: MessageType) -> Message {
    Message::new(
        MessageRole::Assistant,
        "nudge content",
        MessageMeta::new(msg_type).with_step_index(step),
    )
}

/// Build a standard 6-pair history: system + user + 6 iterations of
/// (tool_call + tool_result), each with enough content to exceed a small
/// budget.
fn build_6pair_history() -> Vec<Message> {
    let mut msgs = vec![sys_msg("system prompt"), user_msg("user input")];
    for step in 0..6 {
        msgs.push(tool_call_msg(step, &format!("call_{}", step)));
        msgs.push(tool_result_msg(step, &format!("result_{}", step)));
    }
    msgs
}

// ts-013: NoCompact returns shallow copy with phase 0.
#[test]
fn no_compact_returns_copy_phase_0() {
    let msgs = vec![sys_msg("sys"), user_msg("usr")];
    let (result, phase) = NoCompact.compact(&msgs, 1000, None);
    assert_eq!(phase, 0);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].content, "sys");
    assert_eq!(result[1].content, "usr");
}

// ts-014: SlidingWindowCompact preserves system and user messages.
#[test]
fn sliding_window_preserves_header() {
    let msgs = build_6pair_history();
    let strategy = SlidingWindowCompact::new(2);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(result[0].metadata.msg_type, MessageType::SystemPrompt);
    assert_eq!(result[1].metadata.msg_type, MessageType::UserInput);
}

// ts-015: SlidingWindowCompact keeps only last N iterations.
#[test]
fn sliding_window_keeps_last_n() {
    let msgs = build_6pair_history(); // 14 messages total
    let strategy = SlidingWindowCompact::new(2);
    let (result, phase) = strategy.compact(&msgs, 1, None);
    assert!(phase > 0);
    // 2 header + 4 from last 2 iterations (2 calls + 2 results)
    assert_eq!(result.len(), 2 + 4);
}

// ts-016: SlidingWindowCompact with short history.
#[test]
fn sliding_window_short_history() {
    let msgs = vec![
        sys_msg("sys"),
        user_msg("usr"),
        tool_call_msg(0, "call"),
        tool_result_msg(0, "result"),
    ];
    let strategy = SlidingWindowCompact::new(3);
    let (result, phase) = strategy.compact(&msgs, 1, None);
    assert!(phase > 0);
    assert_eq!(result.len(), 4);
}

// ts-017: SlidingWindowCompact no compaction under threshold.
#[test]
fn sliding_window_no_compaction_under_threshold() {
    let msgs = vec![sys_msg("sys"), user_msg("usr")];
    let strategy = SlidingWindowCompact::new(2).with_threshold(0.99);
    let (result, phase) = strategy.compact(&msgs, 10000, None);
    assert_eq!(phase, 0);
    assert_eq!(result.len(), 2);
}

// ts-018: SlidingWindowCompact at exact boundary.
#[test]
fn sliding_window_exact_boundary() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..3 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = SlidingWindowCompact::new(3);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(result.len(), msgs.len());
}

// ts-019: Tiered Phase 1 drops nudge messages outside keep_recent.
#[test]
fn tiered_phase1_drops_nudges() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..6 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(nudge_msg(step, MessageType::StepNudge));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    // Count nudges in eligible zone (outside protected window).
    let protected_start = result.len() - 2 * 3; // keep_recent=2, 3 msgs per iteration
    let eligible: Vec<_> = result[2..protected_start].to_vec();
    let nudge_count = eligible
        .iter()
        .filter(|m| {
            matches!(
                m.metadata.msg_type,
                MessageType::StepNudge | MessageType::PrerequisiteNudge | MessageType::RetryNudge
            )
        })
        .count();
    assert_eq!(nudge_count, 0);
}

// ts-020: Tiered Phase 1 truncates long tool results.
//
// Build messages where dropping nudges and truncating brings tokens below
// the phase 2 trigger but above the phase 1 trigger.
#[test]
fn tiered_phase1_truncates_long_tool_results() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long_content = "x".repeat(500);
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long_content));
    }
    // budget=100, phase_thresholds=[0.1, 1.0, 1.0]
    // triggers=[10, 100, 100]
    // Initial tokens: (3 + 3 + 4*5 + 4*500) / 4 = 2020 / 4 = 505. 505 > 10 triggers phase 1.
    // After phase 1: eligible tool results truncated to ~230 chars each (200 + marker text).
    // Approx: (3 + 3 + 4*5 + 2*500 + 2*230) / 4 = 1271 / 4 = 317. 317 <= 100? No.
    // Need phase 2 trigger high enough. Use [0.1, 100.0, 100.0] -> triggers=[10, 10000, 10000].
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.1, 100.0, 100.0]);
    let (result, phase) = strategy.compact(&msgs, 100, None);
    assert_eq!(phase, 1);
    // Check that eligible zone has truncated tool results.
    for msg in &result {
        if msg.metadata.msg_type == MessageType::ToolResult {
            if msg.metadata.step_index.unwrap_or(-1) >= 2 {
                // Protected zone: unchanged.
                assert_eq!(msg.content.len(), long_content.len());
            } else {
                // Eligible zone: should have truncation marker.
                assert!(msg.content.contains("chars removed"));
                assert!(msg.content.len() < long_content.len());
                assert!(msg.content.starts_with(&"x".repeat(200)));
            }
        }
    }
}

// ts-021: Tiered Phase 1 preserves text_response messages.
#[test]
fn tiered_phase1_preserves_text_response() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(text_response_msg(step, "text"));
        msgs.push(tool_result_msg(step, "result"));
    }
    // Use large phase 2 trigger to stop at phase 1.
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 100.0, 100.0]);
    let (result, phase) = strategy.compact(&msgs, 100, None);
    assert_eq!(phase, 1);
    let text_count = result
        .iter()
        .filter(|m| m.metadata.msg_type == MessageType::TextResponse)
        .count();
    assert_eq!(text_count, 4);
}

// ts-022: Tiered Phase 1 preserves recent window untouched.
#[test]
fn tiered_phase1_preserves_protected_window() {
    let msgs = build_6pair_history();
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 100.0, 100.0]);
    let (result, _phase) = strategy.compact(&msgs, 100, None);
    let protected: Vec<_> = result
        .iter()
        .filter(|m| m.metadata.step_index.unwrap_or(-1) >= 4)
        .collect();
    assert_eq!(protected.len(), 4);
}

// ts-023: Tiered Phase 1 does not truncate short tool results.
#[test]
fn tiered_phase1_no_truncation_short_results() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let short_content = "short";
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, short_content));
    }
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 100.0, 100.0]);
    let (result, _phase) = strategy.compact(&msgs, 100, None);
    for msg in &result {
        if msg.metadata.msg_type == MessageType::ToolResult
            && msg.metadata.step_index.unwrap_or(-1) < 2
        {
            assert!(!msg.content.contains("chars removed"));
        }
    }
}

// ts-024: Tiered Phase 2 drops all tool_results from eligible zone.
#[test]
fn tiered_phase2_drops_tool_results() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long_content = "x".repeat(500);
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long_content));
    }
    // Use very large phase 3 trigger to stop at phase 2.
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (result, phase) = strategy.compact(&msgs, 100, None);
    assert!(phase >= 2);
    let eligible: Vec<_> = result
        .iter()
        .filter(|m| {
            m.metadata.step_index.unwrap_or(-1) < 2
                && m.metadata.msg_type == MessageType::ToolResult
        })
        .collect();
    assert_eq!(eligible.len(), 0);
}

// ts-025: Tiered Phase 2 preserves reasoning messages.
#[test]
fn tiered_phase2_preserves_reasoning() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(reasoning_msg(step, "thinking"));
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (result, phase) = strategy.compact(&msgs, 100, None);
    assert!(phase >= 2);
    let reasoning_count = result
        .iter()
        .filter(|m| m.metadata.msg_type == MessageType::Reasoning)
        .count();
    assert_eq!(reasoning_count, 4);
}

// ts-026: Tiered Phase 2 preserves text_response messages.
#[test]
fn tiered_phase2_preserves_text_response() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(text_response_msg(step, "text"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (result, phase) = strategy.compact(&msgs, 100, None);
    assert!(phase >= 2);
    let text_count = result
        .iter()
        .filter(|m| m.metadata.msg_type == MessageType::TextResponse)
        .count();
    assert_eq!(text_count, 4);
}

// ts-027: Tiered Phase 2 preserves eligible tool_call skeletons.
#[test]
fn tiered_phase2_preserves_eligible_tool_calls() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (result, _phase) = strategy.compact(&msgs, 100, None);
    let call_count = result
        .iter()
        .filter(|m| m.metadata.msg_type == MessageType::ToolCall)
        .count();
    assert_eq!(call_count, 4);
}

// ts-028: Tiered Phase 2 preserves recent window.
#[test]
fn tiered_phase2_preserves_protected_window() {
    let msgs = build_6pair_history();
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (result, _phase) = strategy.compact(&msgs, 100, None);
    let protected: Vec<_> = result
        .iter()
        .filter(|m| m.metadata.step_index.unwrap_or(-1) >= 4)
        .collect();
    assert_eq!(protected.len(), 4);
}

// ts-029: Tiered Phase 3 drops reasoning from eligible zone.
#[test]
fn tiered_phase3_drops_reasoning() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(reasoning_msg(step, "thinking"));
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(phase, 3);
    let eligible_reasoning: Vec<_> = result
        .iter()
        .filter(|m| {
            m.metadata.step_index.unwrap_or(-1) < 2 && m.metadata.msg_type == MessageType::Reasoning
        })
        .collect();
    assert_eq!(eligible_reasoning.len(), 0);
}

// ts-030: Tiered Phase 3 drops text_response from eligible zone.
#[test]
fn tiered_phase3_drops_text_response() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(text_response_msg(step, "text"));
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(phase, 3);
    let eligible_text: Vec<_> = result
        .iter()
        .filter(|m| {
            m.metadata.step_index.unwrap_or(-1) < 2
                && m.metadata.msg_type == MessageType::TextResponse
        })
        .collect();
    assert_eq!(eligible_text.len(), 0);
}

// ts-031: Tiered Phase 3 keeps tool_call skeletons.
#[test]
fn tiered_phase3_keeps_tool_calls() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    let call_count = result
        .iter()
        .filter(|m| m.metadata.msg_type == MessageType::ToolCall)
        .count();
    assert_eq!(call_count, 4);
}

// ts-032: Tiered Phase 3 preserves system and user as identical.
#[test]
fn tiered_phase3_preserves_system_and_user() {
    let msgs = build_6pair_history();
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(result[0].content, "system prompt");
    assert_eq!(result[1].content, "user input");
    assert_eq!(result[0].metadata.msg_type, MessageType::SystemPrompt);
    assert_eq!(result[1].metadata.msg_type, MessageType::UserInput);
}

// ts-033: Tiered Phase 3 preserves keep_recent window references.
#[test]
fn tiered_phase3_preserves_protected_window() {
    let msgs = build_6pair_history();
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    let protected: Vec<_> = result
        .iter()
        .filter(|m| m.metadata.step_index.unwrap_or(-1) >= 4)
        .collect();
    assert_eq!(protected.len(), 4);
}

// ts-034: Progressive escalation stops at phase 1.
//
// Build history with nudges. After dropping nudges (phase 1), the remaining
// tokens are low enough that the phase 2 trigger is not exceeded.
#[test]
fn tiered_stops_at_phase1() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    // Add large nudge content to boost initial tokens.
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
    // Eligible (steps 0,1): 2*(5 + 400 + 7) = 824 chars = 206 tokens from nudges+calls+results
    // Protected (steps 2,3): 2*(5 + 400 + 7) = 824 chars = 206 tokens
    // Header: 3+3 = 6 chars = 1 token
    // Initial total: 1 + 206 + 206 = 413 tokens
    // After phase 1: nudges dropped, eligible = 2*(5+7)=24 chars = 6 tokens. Total: 1 + 6 + 206 = 213
    // budget=500, thresholds=[0.5, 0.5, 1.0] -> triggers=[250, 250, 500]
    // 413 > 250 -> phase 1 fires. 213 <= 250 -> stops at phase 1.
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.5, 0.5, 1.0]);
    let (_result, phase) = strategy.compact(&msgs, 500, None);
    assert_eq!(phase, 1);
}

// ts-035: Progressive escalation stops at phase 2.
#[test]
fn tiered_stops_at_phase2() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long_content = "x".repeat(500);
    for step in 0..4 {
        msgs.push(reasoning_msg(step, "thinking about stuff here"));
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long_content));
    }
    // Phase 1+2 trigger, phase 3 set high enough.
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (result, phase) = strategy.compact(&msgs, 100, None);
    assert_eq!(phase, 2);
    let reasoning_count = result
        .iter()
        .filter(|m| m.metadata.msg_type == MessageType::Reasoning)
        .count();
    assert_eq!(reasoning_count, 4);
}

// ts-036: Progressive escalation reaches phase 3.
#[test]
fn tiered_reaches_phase3() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long_content = "x".repeat(500);
    for step in 0..8 {
        msgs.push(reasoning_msg(step, &long_content));
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long_content));
    }
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let (result, phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(phase, 3);
    let eligible: Vec<_> = result[2..]
        .iter()
        .filter(|m| m.metadata.step_index.unwrap_or(-1) < 6)
        .collect();
    let has_reasoning = eligible
        .iter()
        .any(|m| m.metadata.msg_type == MessageType::Reasoning);
    let has_tool_result = eligible
        .iter()
        .any(|m| m.metadata.msg_type == MessageType::ToolResult);
    assert!(!has_reasoning);
    assert!(!has_tool_result);
}

// ts-037: No eligible messages when keep_recent >= iteration count.
#[test]
fn tiered_all_protected() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..3 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, "result"));
    }
    let strategy = TieredCompact::new(5).with_threshold(0.0);
    let (result, _phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(result.len(), msgs.len());
}

// ts-038: Compaction does not mutate input list.
#[test]
fn compaction_no_mutation() {
    let msgs = build_6pair_history();
    let original_len = msgs.len();
    let original_contents: Vec<String> = msgs.iter().map(|m| m.content.clone()).collect();
    let strategy = TieredCompact::new(2).with_threshold(0.0);
    let _ = strategy.compact(&msgs, 1, None);
    assert_eq!(msgs.len(), original_len);
    for (i, msg) in msgs.iter().enumerate() {
        assert_eq!(msg.content, original_contents[i]);
    }
}

// ts-039: No compaction when under threshold.
#[test]
fn tiered_no_compaction_under_threshold() {
    let msgs = vec![sys_msg("sys"), user_msg("usr")];
    let strategy = TieredCompact::new(2).with_threshold(0.99);
    let (result, phase) = strategy.compact(&msgs, 10000, None);
    assert_eq!(phase, 0);
    assert_eq!(result.len(), 2);
}

// ts-040: Per-phase thresholds control escalation independently.
#[test]
fn per_phase_thresholds_phase1_only() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
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
    // Same logic as tiered_stops_at_phase1.
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.5, 0.5, 1.0]);
    let (_result, phase) = strategy.compact(&msgs, 500, None);
    assert_eq!(phase, 1);
}

#[test]
fn per_phase_thresholds_phase2_only() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long = "x".repeat(500);
    for step in 0..4 {
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long));
    }
    // Phase 2 stops because phase 3 trigger is very large.
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (_result, phase) = strategy.compact(&msgs, 100, None);
    assert_eq!(phase, 2);
}

#[test]
fn per_phase_thresholds_phase3() {
    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    let long = "x".repeat(500);
    for step in 0..4 {
        msgs.push(reasoning_msg(step, &long));
        msgs.push(tool_call_msg(step, "call"));
        msgs.push(tool_result_msg(step, &long));
    }
    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 0.0]);
    let (_result, phase) = strategy.compact(&msgs, 1, None);
    assert_eq!(phase, 3);
}

#[test]
fn test_compaction_drops_tool_calls_and_results_together() {
    use forge_guardrails::ToolCallInfo;
    use indexmap::IndexMap;

    let mut msgs = vec![sys_msg("sys"), user_msg("usr")];
    for step in 0..4 {
        let tc_info = ToolCallInfo::new("search", Some(IndexMap::new()), format!("tc_000{}", step));
        let call_msg = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall).with_step_index(step),
        )
        .with_tool_calls(vec![tc_info]);
        let result_msg = Message::new(
            MessageRole::Tool,
            "result",
            MessageMeta::new(MessageType::ToolResult).with_step_index(step),
        )
        .with_tool_name("search")
        .with_tool_call_id(format!("tc_000{}", step));

        msgs.push(call_msg);
        msgs.push(result_msg);
    }

    let strategy = TieredCompact::new(2).with_phase_thresholds([0.0, 0.0, 100.0]);
    let (compacted_msgs, phase) = strategy.compact(&msgs, 100, None);
    assert!(phase >= 2);

    let mut call_ids = std::collections::HashSet::new();
    let mut result_ids = std::collections::HashSet::new();

    for m in &compacted_msgs {
        if let Some(tcs) = &m.tool_calls {
            for tc in tcs {
                call_ids.insert(tc.call_id.clone());
            }
        }
        if m.metadata.msg_type == MessageType::ToolResult {
            if let Some(id) = &m.tool_call_id {
                result_ids.insert(id.to_string());
            }
        }
    }

    let expected: std::collections::HashSet<String> =
        ["tc_0002".to_string(), "tc_0003".to_string()]
            .into_iter()
            .collect();
    assert_eq!(call_ids, expected);
    assert_eq!(result_ids, expected);
}
