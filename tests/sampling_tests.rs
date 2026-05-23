use forge_guardrails::{
    apply_sampling_defaults, get_sampling_defaults, UnsupportedModelError, MODEL_SAMPLING_DEFAULTS,
};

// -- get_sampling_defaults tests --

#[test]
fn known_model_returns_copy() {
    let d1 = get_sampling_defaults("qwen3:8b-q4_K_M");
    let d2 = get_sampling_defaults("qwen3:8b-q4_K_M");
    assert_eq!(d1, d2);
    assert!(!d1.is_empty());
}

#[test]
fn mutating_result_does_not_affect_map() {
    let mut d1 = get_sampling_defaults("qwen3:8b-q4_K_M");
    let original = d1.get("temperature").cloned();
    d1.insert("temperature".to_string(), serde_json::json!(99.0));
    let d2 = get_sampling_defaults("qwen3:8b-q4_K_M");
    assert_eq!(d2.get("temperature"), original.as_ref());
}

#[test]
fn unknown_model_returns_empty_dict() {
    let d = get_sampling_defaults("totally-unknown-model");
    assert!(d.is_empty());
}

#[test]
fn qwen35_uses_general_tasks_with_presence_penalty() {
    let d = get_sampling_defaults("qwen3.5:27b-q4_K_M");
    assert_eq!(d.get("presence_penalty"), Some(&serde_json::json!(1.5)));
}

#[test]
fn qwen3_coder_repeat_penalty_no_min_p_or_presence() {
    let d = get_sampling_defaults("qwen3-coder:30b-a3b-instruct-q4_K_M");
    assert_eq!(d.get("repeat_penalty"), Some(&serde_json::json!(1.05)));
    assert!(d.get("min_p").is_none());
    assert!(d.get("presence_penalty").is_none());
}

#[test]
fn get_defaults_does_not_log_for_unknown() {
    // Pure function: no side effects to observe directly, but calling it
    // with an unknown model must not panic or emit output.
    let d = get_sampling_defaults("no-such-model");
    assert!(d.is_empty());
}

// -- apply_sampling_defaults tests --

#[test]
fn strict_known_returns_dict_copy() {
    let result = apply_sampling_defaults("qwen3:8b-q4_K_M", true);
    assert!(result.is_ok());
    let d = result.expect("ok");
    assert!(!d.is_empty());
    assert_eq!(d.get("temperature"), Some(&serde_json::json!(0.6)));
}

#[test]
fn strict_unknown_raises_unsupported_model_error() {
    let result = apply_sampling_defaults("no-such-model", true);
    assert!(result.is_err());
    let err = result.expect_err("err");
    assert_eq!(err.model, "no-such-model");
    // Verify it is the correct error type
    let _: &UnsupportedModelError = &err;
}

#[test]
fn non_strict_known_returns_empty_and_logs() {
    let result = apply_sampling_defaults("qwen3:8b-q4_K_M", false);
    assert!(result.is_ok());
    let d = result.expect("ok");
    assert!(d.is_empty());
}

#[test]
fn non_strict_unknown_returns_empty_silent() {
    let result = apply_sampling_defaults("no-such-model", false);
    assert!(result.is_ok());
    let d = result.expect("ok");
    assert!(d.is_empty());
}

#[test]
fn apply_strict_mutation_does_not_affect_map() {
    let result = apply_sampling_defaults("qwen3:8b-q4_K_M", true);
    let mut d = result.expect("ok");
    let original = d.get("temperature").cloned();
    d.insert("temperature".to_string(), serde_json::json!(99.0));
    let d2 = apply_sampling_defaults("qwen3:8b-q4_K_M", true).expect("ok");
    assert_eq!(d2.get("temperature"), original.as_ref());
}

// -- Model-specific sampling verification --

#[test]
fn granite_40_greedy() {
    let d = get_sampling_defaults("granite-4.0:h-micro-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&serde_json::json!(0.0)));
    assert_eq!(d.get("top_p"), Some(&serde_json::json!(1.0)));
    assert_eq!(d.get("top_k"), Some(&serde_json::json!(0)));
}

#[test]
fn granite_41_mirrors_40() {
    let d = get_sampling_defaults("granite4.1:8b-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&serde_json::json!(0.0)));
    assert_eq!(d.get("top_p"), Some(&serde_json::json!(1.0)));
    assert_eq!(d.get("top_k"), Some(&serde_json::json!(0)));
}

#[test]
fn mistral_small_4_reasoning_effort() {
    let d = get_sampling_defaults("mistral-small-4:119b-2603-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&serde_json::json!(0.7)));
    assert!(d.get("top_p").is_none());
    assert!(d.get("top_k").is_none());
    let kwargs = d.get("chat_template_kwargs").expect("exists");
    assert_eq!(kwargs["reasoning_effort"], "high");
}

#[test]
fn mistral_nemo_temperature_0_3() {
    let d = get_sampling_defaults("mistral-nemo:12b-instruct-2407-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&serde_json::json!(0.3)));
}

#[test]
fn ministral_3_instruct_low_temperature() {
    let d = get_sampling_defaults("ministral-3:8b-instruct-2512-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&serde_json::json!(0.05)));
}

#[test]
fn gpt_oss_no_repeat_or_presence_penalty() {
    let d = get_sampling_defaults("gpt-oss:120b-q4_K_M");
    assert!(d.get("repeat_penalty").is_none());
    assert!(d.get("presence_penalty").is_none());
}

#[test]
fn nemotron_super_chat_template_kwargs() {
    let d = get_sampling_defaults("nemotron-3-super:120b-a12b-q4_K_M");
    let kwargs = d.get("chat_template_kwargs").expect("exists");
    assert_eq!(kwargs["enable_thinking"], true);
    assert_eq!(kwargs["low_effort"], true);
    assert_eq!(kwargs["force_nonempty_content"], true);
}

#[test]
fn model_defaults_map_has_sixtynine_entries() {
    assert_eq!(MODEL_SAMPLING_DEFAULTS.len(), 69);
}
