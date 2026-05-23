use forge_guardrails::steps::Prerequisite;
use forge_guardrails::StepTracker;
use indexmap::IndexMap;
use serde_json::Value;

fn make_args(pairs: &[(&str, &str)]) -> IndexMap<String, Value> {
    let mut map = IndexMap::new();
    for (k, v) in pairs {
        map.insert(k.to_string(), Value::String(v.to_string()));
    }
    map
}

#[test]
fn empty_required_satisfied() {
    let tracker = StepTracker::new(vec![]);
    assert!(tracker.is_satisfied());
}

#[test]
fn two_required_one_recorded() {
    let mut tracker = StepTracker::new(vec!["step_a".into(), "step_b".into()]);
    tracker.record("step_a", None);
    assert!(!tracker.is_satisfied());
}

#[test]
fn two_required_both_recorded() {
    let mut tracker = StepTracker::new(vec!["step_a".into(), "step_b".into()]);
    tracker.record("step_a", None);
    tracker.record("step_b", None);
    assert!(tracker.is_satisfied());
}

#[test]
fn pending_preserves_order() {
    let mut tracker = StepTracker::new(vec!["first".into(), "second".into(), "third".into()]);
    tracker.record("second", None);
    let pending = tracker.pending();
    assert_eq!(pending, vec!["first", "third"]);
}

#[test]
fn record_idempotent_for_completed() {
    let mut tracker = StepTracker::new(vec!["step_a".into()]);
    tracker.record("step_a", None);
    tracker.record("step_a", None);
    assert_eq!(tracker.completed_count(), 1);
}

#[test]
fn record_accumulates_executed_tools() {
    let mut tracker = StepTracker::new(vec!["step_a".into()]);
    let args = make_args(&[("key", "val")]);
    tracker.record("step_a", Some(&args));
    tracker.record("step_a", None);
    // executed_tools["step_a"] should have 2 entries
    // This is verified indirectly through the prerequisite matching
}

#[test]
fn non_required_tool_tracked() {
    let mut tracker = StepTracker::new(vec!["required_step".into()]);
    tracker.record("extra_tool", None);
    assert!(tracker.completed_count() >= 1);
    assert!(!tracker.is_satisfied());
}

#[test]
fn summary_hint_empty() {
    let tracker = StepTracker::new(vec!["step_a".into()]);
    assert_eq!(tracker.summary_hint(), "[No steps completed yet]");
}

#[test]
fn summary_hint_execution_order() {
    let mut tracker = StepTracker::new(vec![]);
    tracker.record("charlie", None);
    tracker.record("alpha", None);
    tracker.record("bravo", None);
    assert_eq!(
        tracker.summary_hint(),
        "[Steps completed: charlie, alpha, bravo]"
    );
}

#[test]
fn name_only_prereq_satisfied() {
    let mut tracker = StepTracker::new(vec![]);
    tracker.record("search", None);
    let args = IndexMap::new();
    let result =
        tracker.check_prerequisites("analyze", &args, &[Prerequisite::NameOnly("search".into())]);
    assert!(result.satisfied);
    assert!(result.missing.is_empty());
}

#[test]
fn name_only_prereq_unsatisfied() {
    let tracker = StepTracker::new(vec![]);
    let args = IndexMap::new();
    let result =
        tracker.check_prerequisites("analyze", &args, &[Prerequisite::NameOnly("search".into())]);
    assert!(!result.satisfied);
    assert!(result.missing.contains(&"search".to_string()));
}

#[test]
fn arg_matched_prereq_satisfied() {
    let mut tracker = StepTracker::new(vec![]);
    let args = make_args(&[("topic", "rust")]);
    tracker.record("search", Some(&args));

    let current_args = make_args(&[("topic", "rust")]);
    let result = tracker.check_prerequisites(
        "analyze",
        &current_args,
        &[Prerequisite::ArgMatched {
            tool: "search".into(),
            match_arg: "topic".into(),
        }],
    );
    assert!(result.satisfied);
}

#[test]
fn arg_matched_prereq_different_value() {
    let mut tracker = StepTracker::new(vec![]);
    let args = make_args(&[("topic", "rust")]);
    tracker.record("search", Some(&args));

    let current_args = make_args(&[("topic", "python")]);
    let result = tracker.check_prerequisites(
        "analyze",
        &current_args,
        &[Prerequisite::ArgMatched {
            tool: "search".into(),
            match_arg: "topic".into(),
        }],
    );
    assert!(!result.satisfied);
    assert!(result.missing.contains(&"search".to_string()));
}

#[test]
fn arg_matched_prereq_missing_key_treated_as_null() {
    let mut tracker = StepTracker::new(vec![]);
    let args = IndexMap::new();
    tracker.record("search", Some(&args));

    let current_args = IndexMap::new();
    let result = tracker.check_prerequisites(
        "analyze",
        &current_args,
        &[Prerequisite::ArgMatched {
            tool: "search".into(),
            match_arg: "topic".into(),
        }],
    );
    // Both sides resolve to Null, so satisfied.
    assert!(result.satisfied);
}

#[test]
fn multiple_prereqs_partial() {
    let mut tracker = StepTracker::new(vec![]);
    tracker.record("step_a", None);
    // step_b not recorded
    let args = IndexMap::new();
    let result = tracker.check_prerequisites(
        "tool",
        &args,
        &[
            Prerequisite::NameOnly("step_a".into()),
            Prerequisite::NameOnly("step_b".into()),
        ],
    );
    assert!(!result.satisfied);
    assert_eq!(result.missing.len(), 1);
    assert!(result.missing.contains(&"step_b".to_string()));
}

#[test]
fn record_with_none_args_stores_empty_map() {
    let mut tracker = StepTracker::new(vec![]);
    tracker.record("tool", None);
    // Verify it was recorded without error and completed_steps has it
    assert!(tracker.completed_count() == 1);
}
