use super::*;
use serde_json::json;

#[test]
fn get_defaults_known_model_returns_copy() {
    let d1 = get_sampling_defaults("qwen3:8b-q4_K_M");
    let d2 = get_sampling_defaults("qwen3:8b-q4_K_M");
    assert_eq!(d1, d2);
    assert!(!d1.is_empty());
}

#[test]
fn get_defaults_mutation_does_not_affect_map() {
    let mut d1 = get_sampling_defaults("qwen3:8b-q4_K_M");
    let original_temp = d1.get("temperature").cloned();
    d1.insert("temperature".to_string(), json!(99.0));
    let d2 = get_sampling_defaults("qwen3:8b-q4_K_M");
    assert_eq!(d2.get("temperature"), original_temp.as_ref());
}

#[test]
fn get_defaults_unknown_model_returns_empty() {
    let d = get_sampling_defaults("nonexistent-model");
    assert!(d.is_empty());
}

#[test]
fn get_defaults_qwen35_has_presence_penalty() {
    let d = get_sampling_defaults("qwen3.5:27b-q4_K_M");
    assert_eq!(d.get("presence_penalty"), Some(&json!(1.5)));
}

#[test]
fn get_defaults_qwen3_coder_has_repeat_penalty() {
    let d = get_sampling_defaults("qwen3-coder:30b-a3b-instruct-q4_K_M");
    assert_eq!(d.get("repeat_penalty"), Some(&json!(1.05)));
    assert!(d.get("min_p").is_none());
    assert!(d.get("presence_penalty").is_none());
}

#[test]
fn apply_strict_known_returns_copy() {
    let result = apply_sampling_defaults("qwen3:8b-q4_K_M", true);
    assert!(result.is_ok());
    let d = result.expect("ok");
    assert!(!d.is_empty());
    assert_eq!(d.get("temperature"), Some(&json!(0.6)));
}

#[test]
fn apply_strict_unknown_raises() {
    let result = apply_sampling_defaults("nonexistent-model", true);
    assert!(result.is_err());
    let err = result.expect_err("should be err");
    assert_eq!(err.model, "nonexistent-model");
}

#[test]
fn apply_non_strict_known_returns_empty() {
    let result = apply_sampling_defaults("qwen3:8b-q4_K_M", false);
    assert!(result.is_ok());
    let d = result.expect("ok");
    assert!(d.is_empty());
}

#[test]
fn apply_non_strict_unknown_returns_empty_silent() {
    let result = apply_sampling_defaults("nonexistent-model", false);
    assert!(result.is_ok());
    let d = result.expect("ok");
    assert!(d.is_empty());
}

#[test]
fn apply_mutation_does_not_affect_map() {
    let result = apply_sampling_defaults("qwen3:8b-q4_K_M", true);
    let mut d = result.expect("ok");
    let original = d.get("temperature").cloned();
    d.insert("temperature".to_string(), json!(99.0));
    let d2 = apply_sampling_defaults("qwen3:8b-q4_K_M", true).expect("ok");
    assert_eq!(d2.get("temperature"), original.as_ref());
}

#[test]
fn granite_40_greedy_decoding() {
    let d = get_sampling_defaults("granite-4.0:h-micro-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&json!(0.0)));
    assert_eq!(d.get("top_p"), Some(&json!(1.0)));
    assert_eq!(d.get("top_k"), Some(&json!(0)));
}

#[test]
fn mistral_small_4_has_chat_template_kwargs() {
    let d = get_sampling_defaults("mistral-small-4:119b-2603-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&json!(0.7)));
    let kwargs = d.get("chat_template_kwargs").expect("should exist");
    assert_eq!(kwargs["reasoning_effort"], "high");
}

#[test]
fn gpt_oss_has_reasoning_effort_medium() {
    let d = get_sampling_defaults("gpt-oss:120b-q4_K_M");
    assert_eq!(d.get("temperature"), Some(&json!(1.0)));
    let kwargs = d.get("chat_template_kwargs").expect("should exist");
    assert_eq!(kwargs["reasoning_effort"], "medium");
}

#[test]
fn nemotron_nano_deterministic_preset() {
    let d = get_sampling_defaults("Nemotron-3-Nano-30B-A3B-Q4_K_M");
    assert_eq!(d.get("temperature"), Some(&json!(0.6)));
    assert_eq!(d.get("top_p"), Some(&json!(0.95)));
    let kwargs = d.get("chat_template_kwargs").expect("should exist");
    assert_eq!(kwargs["enable_thinking"], true);
}

#[test]
fn all_sixtynine_models_present() {
    assert_eq!(MODEL_SAMPLING_DEFAULTS.len(), 69);
}

#[test]
fn one_shot_info_fires_once() {
    // Reset the log tracking state for this test.
    {
        let mut guard = INFO_LOGGED.lock().expect("mutex");
        *guard = None;
    }
    // First call should log.
    fire_one_shot_info("test-one-shot-model");
    // Second call should not panic or fail.
    fire_one_shot_info("test-one-shot-model");
    // Verify it was recorded.
    let guard = INFO_LOGGED.lock().expect("mutex");
    let logged = guard.as_ref().expect("set");
    assert!(logged.contains("test-one-shot-model"));
}
