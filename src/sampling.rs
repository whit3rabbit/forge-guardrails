//! Per-model sampling defaults sourced from HuggingFace model cards.
//!
//! `MODEL_SAMPLING_DEFAULTS` is a static map of recommended sampling parameters.
//! Each entry is keyed by the model identity string callers use. All forms
//! point at independent rows so vendor-specific guidance can diverge.
//!
//! Two functions operate on the map:
//! - `get_sampling_defaults`: pure lookup, no side effects.
//! - `apply_sampling_defaults`: policy layer with strict flag, logging, errors.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::sync::Mutex;

use serde_json::{json, Map, Value};

use crate::error::UnsupportedModelError;

/// Per-model recommended sampling parameters sourced from HuggingFace model
/// cards. Keyed by model identity string. Each entry is independent even when
/// multiple identity forms refer to the same underlying model.
pub static MODEL_SAMPLING_DEFAULTS: LazyLock<HashMap<&str, Map<String, Value>>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();

        // Qwen3 thinking-mode
        m.insert(
            "qwen3:8b-q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        );

        // Qwen3 instruct variant
        m.insert(
            "qwen3:4b-instruct-2507-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        );

        // Qwen3.5/3.6 general-tasks
        m.insert(
            "qwen3.5:27b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        );

        // Qwen3-Coder
        m.insert(
            "qwen3-coder:30b-a3b-instruct-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("repeat_penalty", json!(1.05)),
            ]),
        );

        // Qwen3-Coder-Next
        m.insert(
            "qwen3-coder-next:80b-a3b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(40)),
            ]),
        );

        // Gemma 4
        m.insert(
            "gemma4:31b-it-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        );

        // Mistral Small 4
        m.insert(
            "mistral-small-4:119b-2603-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("chat_template_kwargs", json!({"reasoning_effort": "high"})),
            ]),
        );

        // Mistral Small 3.2 / Devstral Small 2
        m.insert(
            "mistral-small-3.2:24b-instruct-2506-q4_K_M",
            params(&[("temperature", json!(0.15))]),
        );

        // Ministral-3 Instruct
        m.insert(
            "ministral-3:8b-instruct-2512-q4_K_M",
            params(&[("temperature", json!(0.05))]),
        );

        // Ministral-3 Reasoning
        m.insert(
            "ministral-3:8b-reasoning-2512-q4_K_M",
            params(&[("temperature", json!(0.7))]),
        );

        // Mistral Nemo
        m.insert(
            "mistral-nemo:12b-instruct-2407-q4_K_M",
            params(&[("temperature", json!(0.3))]),
        );

        // Granite 4.0
        m.insert(
            "granite-4.0:h-micro-q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        );

        // Granite 4.1
        m.insert(
            "granite4.1:8b-q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        );

        // GPT-OSS 120B
        m.insert(
            "gpt-oss:120b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
                ("min_p", json!(0.0)),
                (
                    "chat_template_kwargs",
                    json!({"reasoning_effort": "medium"}),
                ),
            ]),
        );

        // NVIDIA Nemotron-3-Super
        m.insert(
            "nemotron-3-super:120b-a12b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                (
                    "chat_template_kwargs",
                    json!({
                        "enable_thinking": true,
                        "low_effort": true,
                        "force_nonempty_content": true
                    }),
                ),
            ]),
        );

        // NVIDIA Nemotron-3-Nano
        m.insert(
            "Nemotron-3-Nano-30B-A3B-Q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("chat_template_kwargs", json!({"enable_thinking": true})),
            ]),
        );

        m
    });

/// Helper to build a Map from key-value pairs.
fn params(pairs: &[(&str, Value)]) -> Map<String, Value> {
    let mut m = Map::new();
    for &(k, ref v) in pairs {
        m.insert(k.to_string(), v.clone());
    }
    m
}

/// Internal tracking for one-shot INFO log per (model, process) pair.
static INFO_LOGGED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Pure lookup function for recommended sampling params.
///
/// Returns a fresh `Map` copy of the map entry for known models, or an empty
/// map for unknown models. No logging, no raising, no side effects.
pub fn get_sampling_defaults(model: &str) -> Map<String, Value> {
    match MODEL_SAMPLING_DEFAULTS.get(model) {
        Some(entry) => entry.clone(),
        None => Map::new(),
    }
}

/// Policy-layer function called by client constructors at instantiation time.
///
/// Implements a four-quadrant behavior based on the `strict` flag and model
/// presence in the defaults map:
///
/// | strict | known  | result                           |
/// |--------|--------|----------------------------------|
/// | true   | yes    | fresh copy of the map entry      |
/// | true   | no     | `UnsupportedModelError` raised   |
/// | false  | yes    | empty map, one-shot INFO log     |
/// | false  | no     | empty map, silent                |
pub fn apply_sampling_defaults(
    model: &str,
    strict: bool,
) -> Result<Map<String, Value>, UnsupportedModelError> {
    let defaults = get_sampling_defaults(model);
    let known = !defaults.is_empty();

    if strict {
        if known {
            Ok(defaults)
        } else {
            Err(UnsupportedModelError::new(model))
        }
    } else if known {
        fire_one_shot_info(model);
        Ok(Map::new())
    } else {
        Ok(Map::new())
    }
}

/// Fire a one-shot INFO log per (model, process) pair.
///
/// Uses a process-global Mutex to track which models have already been
/// logged. The log fires once and is silent on subsequent calls for the
/// same model within the same process.
fn fire_one_shot_info(model: &str) {
    // If the mutex is poisoned, skip logging rather than panic.
    let Ok(mut guard) = INFO_LOGGED.lock() else {
        return;
    };
    let logged = guard.get_or_insert_with(HashSet::new);
    if !logged.contains(model) {
        log::info!(
            "Model '{}' has recommended sampling defaults. \
             Consider opting in with strict mode for optimal behavior.",
            model
        );
        logged.insert(model.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn all_sixteen_models_present() {
        assert_eq!(MODEL_SAMPLING_DEFAULTS.len(), 16);
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
}
