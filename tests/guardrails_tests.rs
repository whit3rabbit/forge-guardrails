//! Tests for the guardrails safety layer.
//! Covers all 15 test scenarios from behavior-spec-unit-004 (ts-001 through ts-015),
//! plus additional edge cases from observable_behaviors and edge_cases.

use forge_guardrails::guardrails::{
    ErrorTracker, GuardAction, Guardrails, Nudge, ResponseValidator, RetryNudgeFn, StepEnforcer,
    StepPrerequisite, TerminalTool,
};
use forge_guardrails::streaming::{LLMResponse, TextResponse, ToolCall};
use indexmap::{IndexMap, IndexSet};
use std::sync::{Arc, Mutex};

fn make_tool_call(tool: &str) -> ToolCall {
    ToolCall::new(tool, IndexMap::new())
}

fn make_tool_call_with_args(tool: &str, args: &[(&str, &str)]) -> ToolCall {
    let mut map = IndexMap::new();
    for (k, v) in args {
        map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    ToolCall::new(tool, map)
}

fn make_guardrails(max_retries: i32, required_steps: Option<Vec<&str>>) -> Guardrails {
    let tools: Vec<String> = vec!["search".into(), "analyze".into(), "respond".into()];
    Guardrails::new(
        tools,
        TerminalTool::Single("respond".into()),
        required_steps.map(|s| s.into_iter().map(|x| x.into()).collect()),
        None,
        max_retries,
        2,
        true,
        3,
        None,
    )
}

fn make_args(pairs: &[(&str, &str)]) -> IndexMap<String, serde_json::Value> {
    let mut map = IndexMap::new();
    for (k, v) in pairs {
        map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    map
}

// ts-001
#[test]
fn error_tracker_retries_exhaustion_and_reset() {
    let mut tracker = ErrorTracker::new(2, 2);
    tracker.record_retry();
    assert!(!tracker.retries_exhausted());
    tracker.record_retry();
    assert!(!tracker.retries_exhausted());
    tracker.record_retry();
    assert!(tracker.retries_exhausted());
    tracker.reset_retries();
    assert!(!tracker.retries_exhausted());
    assert_eq!(tracker.consecutive_retries(), 0);
}

// ts-002
#[test]
fn error_tracker_soft_errors_excluded() {
    let mut tracker = ErrorTracker::new(3, 1);
    tracker.record_result(false, true);
    tracker.record_result(false, true);
    tracker.record_result(false, true);
    assert_eq!(tracker.consecutive_tool_errors(), 0);
    assert!(!tracker.tool_errors_exhausted());
}

// ob-002, ob-003, ob-004
#[test]
fn error_tracker_success_does_not_reset_errors() {
    let mut tracker = ErrorTracker::new(3, 2);
    tracker.record_result(false, false);
    tracker.record_result(true, false);
    assert_eq!(tracker.consecutive_tool_errors(), 1);
}

#[test]
fn error_tracker_counters_independent() {
    let mut tracker = ErrorTracker::new(3, 2);
    tracker.record_retry();
    tracker.record_retry();
    assert_eq!(tracker.consecutive_retries(), 2);
    assert_eq!(tracker.consecutive_tool_errors(), 0);
    tracker.record_result(false, false);
    assert_eq!(tracker.consecutive_retries(), 2);
    assert_eq!(tracker.consecutive_tool_errors(), 1);
}

// ts-003
#[test]
fn validator_rescue_raw_json() {
    let validator = ResponseValidator::new(vec!["search".into()], true, None);
    let text = TextResponse::new(r#"{"tool": "search", "args": {"query": "test"}}"#);
    let response = LLMResponse::Text(text);
    let result = validator.validate(&response);
    assert!(!result.needs_retry);
    let calls = result.tool_calls.expect("should have tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool, "search");
}

// ts-004
#[test]
fn validator_rescue_code_fenced_json() {
    let validator = ResponseValidator::new(vec!["search".into()], true, None);
    let text = TextResponse::new("```json\n{\"tool\": \"search\", \"args\": {\"q\": \"hi\"}}\n```");
    let response = LLMResponse::Text(text);
    let result = validator.validate(&response);
    assert!(!result.needs_retry);
    let calls = result.tool_calls.expect("should have tool_calls");
    assert_eq!(calls[0].tool, "search");
}

// ts-005
#[test]
fn validator_unknown_tool_in_rescue_falls_through() {
    let validator = ResponseValidator::new(vec!["search".into()], true, None);
    let text = TextResponse::new(r#"{"tool": "nonexistent", "args": {}}"#);
    let response = LLMResponse::Text(text);
    let result = validator.validate(&response);
    assert!(result.needs_retry);
    assert_eq!(result.nudge.expect("nudge").kind, "retry");
}

// ob-007
#[test]
fn validator_unknown_tool_rejected() {
    let validator = ResponseValidator::new(vec!["search".into()], true, None);
    let calls = vec![make_tool_call("bogus_tool")];
    let response = LLMResponse::ToolCalls(calls);
    let result = validator.validate(&response);
    assert!(result.needs_retry);
    let nudge = result.nudge.expect("should have nudge");
    assert_eq!(nudge.kind, "unknown_tool");
    assert!(nudge.content.contains("bogus_tool"));
}

// ob-008
#[test]
fn validator_empty_tool_calls_valid() {
    let validator = ResponseValidator::new(vec!["search".into()], true, None);
    let response = LLMResponse::ToolCalls(Vec::new());
    let result = validator.validate(&response);
    assert!(!result.needs_retry);
    let calls = result.tool_calls.expect("should have tool_calls");
    assert!(calls.is_empty());
}

// ec-005
#[test]
fn validator_rescue_disabled_text_is_retry() {
    let validator = ResponseValidator::new(vec!["search".into()], false, None);
    let text = TextResponse::new(r#"{"tool": "search", "args": {}}"#);
    let response = LLMResponse::Text(text);
    let result = validator.validate(&response);
    assert!(result.needs_retry);
    assert_eq!(result.nudge.expect("nudge").kind, "retry");
}

// ts-006
#[test]
fn step_enforcer_escalating_tiers() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    let calls = vec![make_tool_call("respond")];
    assert_eq!(
        enforcer.check(&calls).nudge.as_ref().expect("nudge").tier,
        1
    );
    assert_eq!(
        enforcer.check(&calls).nudge.as_ref().expect("nudge").tier,
        2
    );
    assert_eq!(
        enforcer.check(&calls).nudge.as_ref().expect("nudge").tier,
        3
    );
    assert_eq!(
        enforcer.check(&calls).nudge.as_ref().expect("nudge").tier,
        3
    );
}

// ts-007
#[test]
fn step_enforcer_terminal_after_steps() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    enforcer.record("search", None);
    let calls = vec![make_tool_call("respond")];
    assert!(!enforcer.check(&calls).needs_nudge);
}

// ts-008
#[test]
fn step_enforcer_arg_matched_blocks_different_value() {
    let mut enforcer = StepEnforcer::new(
        vec![],
        IndexSet::from(["respond".into()]),
        Some({
            let mut map: IndexMap<String, Vec<StepPrerequisite>> = IndexMap::new();
            map.insert(
                "analyze".into(),
                vec![StepPrerequisite::ArgMatched {
                    tool: "search".into(),
                    match_arg: "topic".into(),
                }],
            );
            map
        }),
        3,
        2,
    );
    let args = make_args(&[("topic", "rust")]);
    enforcer.record("search", Some(&args));
    let analyze_call = make_tool_call_with_args("analyze", &[("topic", "python")]);
    let result = enforcer.check_prerequisites(&[analyze_call]);
    assert!(result.needs_nudge);
    assert_eq!(result.nudge.as_ref().expect("nudge").kind, "prerequisite");
}

// ts-009
#[test]
fn step_enforcer_batch_blocks_prereq_and_tool_together() {
    let mut enforcer = StepEnforcer::new(
        vec![],
        IndexSet::from(["respond".into()]),
        Some({
            let mut map: IndexMap<String, Vec<StepPrerequisite>> = IndexMap::new();
            map.insert(
                "analyze".into(),
                vec![StepPrerequisite::NameOnly("search".into())],
            );
            map
        }),
        3,
        2,
    );
    let batch = vec![make_tool_call("search"), make_tool_call("analyze")];
    assert!(enforcer.check_prerequisites(&batch).needs_nudge);
}

// ob-013 / ec-003
#[test]
fn step_enforcer_no_required_steps_satisfied() {
    let mut enforcer = StepEnforcer::new(vec![], IndexSet::from(["respond".into()]), None, 3, 2);
    assert!(enforcer.is_satisfied());
    let calls = vec![make_tool_call("respond")];
    assert!(!enforcer.check(&calls).needs_nudge);
}

// ob-014
#[test]
fn step_enforcer_duplicate_recording_idempotent() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    enforcer.record("search", None);
    enforcer.record("search", None);
    assert!(enforcer.is_satisfied());
    assert_eq!(enforcer.completed_steps().len(), 1);
}

#[test]
fn step_enforcer_non_required_tool_no_effect() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    enforcer.record("other_tool", None);
    assert!(!enforcer.is_satisfied());
    assert_eq!(enforcer.pending(), vec!["search"]);
}

// ob-022
#[test]
fn step_enforcer_premature_exhaustion_strict_gt() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    let calls = vec![make_tool_call("respond")];
    enforcer.check(&calls);
    enforcer.check(&calls);
    enforcer.check(&calls);
    assert_eq!(enforcer.premature_attempts(), 3);
    assert!(!enforcer.premature_exhausted());
    enforcer.check(&calls);
    assert!(enforcer.premature_exhausted());
}

// ob-023 / ob-024
#[test]
fn step_enforcer_reset_premature_restarts_escalation() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    let calls = vec![make_tool_call("respond")];
    enforcer.check(&calls);
    enforcer.check(&calls);
    assert_eq!(enforcer.premature_attempts(), 2);
    enforcer.reset_premature();
    assert_eq!(enforcer.premature_attempts(), 0);
    let r = enforcer.check(&calls);
    assert_eq!(r.nudge.as_ref().expect("nudge").tier, 1);
}

#[test]
fn step_enforcer_reset_prereq_violations() {
    let mut enforcer = StepEnforcer::new(vec![], IndexSet::new(), None, 3, 2);
    enforcer.reset_prereq_violations();
    assert_eq!(enforcer.prereq_violations(), 0);
}

// ob-025
#[test]
fn step_enforcer_summary_hint_and_completed_steps() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into(), "analyze".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    assert_eq!(enforcer.summary_hint(), "[No steps completed yet]");
    enforcer.record("search", None);
    assert!(enforcer.summary_hint().contains("search"));
    let completed = enforcer.completed_steps();
    assert_eq!(completed.len(), 1);
    assert!(completed.contains_key("search"));
    assert!(!completed.contains_key("analyze"));
}

// ob-021
#[test]
fn step_enforcer_multiple_terminal_tools() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into(), "submit".into()]),
        None,
        3,
        2,
    );
    let r1 = enforcer.check(&[make_tool_call("respond")]);
    assert!(r1.needs_nudge);
    enforcer.reset_premature();
    let r2 = enforcer.check(&[make_tool_call("submit")]);
    assert!(r2.needs_nudge);
    enforcer.record("search", None);
    assert!(enforcer.is_satisfied());
    let r3 = enforcer.check(&[make_tool_call("submit")]);
    assert!(!r3.needs_nudge);
}

// ts-010
#[test]
fn guardrails_fatal_after_exhausted_retries() {
    let mut g = make_guardrails(2, None);
    let text = LLMResponse::Text(TextResponse::new("just text"));
    // counter=1, 1>2=false -> Retry
    assert_eq!(g.check(&text).action, GuardAction::Retry);
    // counter=2, 2>2=false -> Retry
    assert_eq!(g.check(&text).action, GuardAction::Retry);
    // counter=3, 3>2=true -> Fatal
    let r3 = g.check(&text);
    assert_eq!(r3.action, GuardAction::Fatal);
    assert!(r3
        .reason
        .as_ref()
        .expect("reason")
        .contains("bad responses"));
}

// ts-011
#[test]
fn guardrails_valid_resets_retry_counter() {
    let mut g = make_guardrails(2, None);
    let text = LLMResponse::Text(TextResponse::new("just text"));
    g.check(&text);
    g.check(&text);
    g.check(&text);
    let valid = LLMResponse::ToolCalls(vec![make_tool_call("search")]);
    assert_eq!(g.check(&valid).action, GuardAction::Execute);
    // Counter was reset, so next text is retry (not fatal).
    assert_eq!(g.check(&text).action, GuardAction::Retry);
}

// ts-012
#[test]
fn guardrails_record_true_when_terminal_and_satisfied() {
    let mut g = make_guardrails(3, Some(vec!["search"]));
    // search is not terminal.
    assert!(!g.record(&["search"]));
    // respond is terminal, search is now satisfied.
    assert!(g.record(&["respond"]));
}

#[test]
fn guardrails_record_false_non_terminal_even_if_satisfied() {
    let mut g = make_guardrails(3, Some(vec!["search"]));
    g.record(&["search"]);
    assert!(!g.record(&["analyze"]));
}

// ec-001
#[test]
fn guardrails_record_terminal_unsatisfied_returns_false() {
    let mut g = make_guardrails(3, Some(vec!["search", "analyze"]));
    assert!(!g.record(&["respond"]));
}

// ts-013
#[test]
fn guardrails_custom_retry_nudge_receives_text() {
    let captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let captured_clone = captured.clone();
    let custom_fn: RetryNudgeFn = Box::new(move |text: &str| {
        let mut guard = captured_clone.lock().unwrap();
        *guard = text.to_string();
        format!("Custom: {}", text)
    });
    let tools: Vec<String> = vec!["search".into()];
    let mut g = Guardrails::new(
        tools,
        TerminalTool::Single("respond".into()),
        None,
        None,
        3,
        2,
        false,
        3,
        Some(custom_fn),
    );
    let text = LLMResponse::Text(TextResponse::new("hello world"));
    let result = g.check(&text);
    assert_eq!(result.action, GuardAction::Retry);
    let nudge = result.nudge.expect("should have nudge");
    assert!(nudge.content.contains("Custom:"));
    assert!(nudge.content.contains("hello world"));
    assert_eq!(captured.lock().unwrap().as_str(), "hello world");
}

// ts-014
#[test]
fn step_enforcer_multi_terminal_all_trigger() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into(), "submit".into()]),
        None,
        3,
        2,
    );
    assert!(enforcer.check(&[make_tool_call("respond")]).needs_nudge);
    enforcer.reset_premature();
    assert!(enforcer.check(&[make_tool_call("submit")]).needs_nudge);
}

// ts-015
#[test]
fn guardrails_rescue_code_fenced_text() {
    let mut g = make_guardrails(3, None);
    let text = LLMResponse::Text(TextResponse::new(
        "```json\n{\"tool\": \"search\", \"args\": {\"query\": \"rust\"}}\n```",
    ));
    let result = g.check(&text);
    assert_eq!(result.action, GuardAction::Execute);
    let calls = result.tool_calls.expect("should have tool_calls");
    assert_eq!(calls[0].tool, "search");
}

// ob-015
#[test]
fn guardrails_valid_but_step_blocked() {
    let mut g = make_guardrails(3, Some(vec!["search"]));
    let calls = LLMResponse::ToolCalls(vec![make_tool_call("respond")]);
    assert_eq!(g.check(&calls).action, GuardAction::StepBlocked);
}

// ob-016
#[test]
fn guardrails_fatal_from_premature_exhaustion() {
    let mut g = make_guardrails(3, Some(vec!["search"]));
    let calls = LLMResponse::ToolCalls(vec![make_tool_call("respond")]);
    for _ in 0..3 {
        assert_eq!(g.check(&calls.clone()).action, GuardAction::StepBlocked);
    }
    let r = g.check(&calls);
    assert_eq!(r.action, GuardAction::Fatal);
    assert!(r
        .reason
        .as_ref()
        .expect("reason")
        .contains("skipped required steps"));
}

// ob-017
#[test]
fn guardrails_record_resets_counters() {
    let mut g = make_guardrails(2, Some(vec!["search"]));
    let text = LLMResponse::Text(TextResponse::new("text"));
    g.check(&text);
    g.check(&text);
    g.record(&["search"]);
    // Retry counter was reset by record().
    assert_eq!(g.check(&text).action, GuardAction::Retry);
}

// ob-018 / ec-008
#[test]
fn guardrails_retry_resets_on_valid_even_if_step_blocked() {
    let mut g = make_guardrails(2, Some(vec!["search"]));
    let text = LLMResponse::Text(TextResponse::new("text"));
    g.check(&text);
    g.check(&text);
    let calls = LLMResponse::ToolCalls(vec![make_tool_call("respond")]);
    assert_eq!(g.check(&calls).action, GuardAction::StepBlocked);
    // Retry counter was reset by valid validation. Next text is retry, not fatal.
    assert_eq!(g.check(&text).action, GuardAction::Retry);
}

// ob-020
#[test]
fn guardrails_terminal_tool_single_string() {
    let mut g = make_guardrails(3, None);
    assert!(!g.record(&["search"]));
}

#[test]
fn guardrails_terminal_tool_set() {
    let tools: Vec<String> = vec!["search".into(), "respond".into(), "submit".into()];
    let mut g = Guardrails::new(
        tools,
        TerminalTool::Multiple(IndexSet::from(["respond".into(), "submit".into()])),
        None,
        None,
        3,
        2,
        true,
        3,
        None,
    );
    assert!(g.record(&["respond"]));
}

// neg-005
#[test]
fn nudge_accepts_any_strings() {
    let n = Nudge::new("custom_role", "content", "custom_kind");
    assert_eq!(n.role, "custom_role");
    assert_eq!(n.kind, "custom_kind");
    assert_eq!(n.tier, 0);
}

#[test]
fn nudge_with_tier_returns_new_instance() {
    let n1 = Nudge::new("system", "content", "retry");
    let n2 = n1.with_tier(3);
    assert_eq!(n2.tier, 3);
}

// ob-011, ob-012, ec-006, ec-007
#[test]
fn step_enforcer_mixed_prerequisites() {
    let mut enforcer = StepEnforcer::new(
        vec![],
        IndexSet::from(["respond".into()]),
        Some({
            let mut map: IndexMap<String, Vec<StepPrerequisite>> = IndexMap::new();
            map.insert(
                "analyze".into(),
                vec![
                    StepPrerequisite::NameOnly("setup".into()),
                    StepPrerequisite::ArgMatched {
                        tool: "search".into(),
                        match_arg: "topic".into(),
                    },
                ],
            );
            map
        }),
        3,
        2,
    );
    enforcer.record("setup", None);
    let args = make_args(&[("topic", "rust")]);
    enforcer.record("search", Some(&args));
    let analyze_call = make_tool_call_with_args("analyze", &[("topic", "rust")]);
    assert!(!enforcer.check_prerequisites(&[analyze_call]).needs_nudge);
}

#[test]
fn step_enforcer_arg_matched_different_value_blocks() {
    let mut enforcer = StepEnforcer::new(
        vec![],
        IndexSet::from(["respond".into()]),
        Some({
            let mut map: IndexMap<String, Vec<StepPrerequisite>> = IndexMap::new();
            map.insert(
                "analyze".into(),
                vec![StepPrerequisite::ArgMatched {
                    tool: "search".into(),
                    match_arg: "topic".into(),
                }],
            );
            map
        }),
        3,
        2,
    );
    let args_rust = make_args(&[("topic", "rust")]);
    enforcer.record("search", Some(&args_rust));
    let analyze_call = make_tool_call_with_args("analyze", &[("topic", "python")]);
    assert!(enforcer.check_prerequisites(&[analyze_call]).needs_nudge);
}

// ec-004
#[test]
fn step_enforcer_mixed_batch_triggers_premature() {
    let mut enforcer = StepEnforcer::new(
        vec!["search".into()],
        IndexSet::from(["respond".into()]),
        None,
        3,
        2,
    );
    let batch = vec![make_tool_call("search"), make_tool_call("respond")];
    assert!(enforcer.check(&batch).needs_nudge);
}
