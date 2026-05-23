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

        // Qwen3 — thinking-mode values (card also lists non-thinking variant; forge
        // runs these in thinking mode by default via --reasoning-format auto).
        m.insert(
            "qwen3:4b-instruct-2507-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
        m.insert(
            "qwen3:4b-thinking-2507-q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
        m.insert(
            "qwen3:8b-q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-8B
        m.insert(
            "Qwen3-8B-Q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-8B
        m.insert(
            "qwen3:8b-q8_0",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-8B
        m.insert(
            "Qwen3-8B-Q8_0",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-8B
        m.insert(
            "qwen3:14b-q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-14B
        m.insert(
            "Qwen3-14B-Q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-14B

        // Qwen3.5/3.6 — thinking-mode general-tasks profile
        m.insert(
            "qwen3.5:27b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.5-27B
        m.insert(
            "Qwen3.5-27B-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.5-27B
        m.insert(
            "qwen3.5:35b-a3b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.5-35B-A3B
        m.insert(
            "Qwen3.5-35B-A3B-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.5-35B-A3B
        m.insert(
            "qwen3.6:35b-a3b-ud-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.6-35B-A3B
        m.insert(
            "Qwen3.6-35B-A3B-UD-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
                ("presence_penalty", json!(1.5)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.6-35B-A3B

        // Qwen3-Coder — non-thinking instruct
        m.insert(
            "qwen3-coder:30b-a3b-instruct-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("repeat_penalty", json!(1.05)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
        m.insert(
            "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("repeat_penalty", json!(1.05)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct

        // Qwen3-Next 80B-A3B-Instruct
        m.insert(
            "qwen3-next:80b-a3b-instruct-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
        m.insert(
            "Qwen3-Next-80B-A3B-Instruct-Q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
                ("min_p", json!(0.0)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct

        // Qwen3-Coder-Next
        m.insert(
            "qwen3-coder-next:80b-a3b-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(40)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-Coder-Next
        m.insert(
            "Qwen3-Coder-Next-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(40)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3-Coder-Next

        // Gemma 4
        m.insert(
            "gemma4:31b-it-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-31b-it
        m.insert(
            "gemma-4-31B-it-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-31b-it
        m.insert(
            "gemma4:26b-a4b-it-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-26b-a4b-it
        m.insert(
            "gemma-4-26B-A4B-it-UD-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-26b-a4b-it
        m.insert(
            "gemma4:26b-a4b-it-q8_0",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-26b-a4b-it
        m.insert(
            "gemma-4-26B-A4B-it-Q8_0",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-26b-a4b-it
        m.insert(
            "gemma4:e4b-it-q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-e4b-it
        m.insert(
            "gemma-4-E4B-it-Q4_K_M",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-e4b-it
        m.insert(
            "gemma4:e4b-it-q8_0",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-e4b-it
        m.insert(
            "gemma-4-E4B-it-Q8_0",
            params(&[
                ("temperature", json!(1.0)),
                ("top_p", json!(0.95)),
                ("top_k", json!(64)),
            ]),
        ); // https://huggingface.co/google/gemma-4-e4b-it

        // Mistral Small 4
        m.insert(
            "mistral-small-4:119b-2603-q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("chat_template_kwargs", json!({"reasoning_effort": "high"})),
            ]),
        ); // https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
        m.insert(
            "Mistral-Small-4-119B-2603-UD-Q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("chat_template_kwargs", json!({"reasoning_effort": "high"})),
            ]),
        ); // https://huggingface.co/mistralai/Mistral-Small-4-119B-2603

        // Qwen3.5-122B-A10B
        m.insert(
            "Qwen3.5-122B-A10B-Q4_K_M",
            params(&[
                ("temperature", json!(0.7)),
                ("top_p", json!(0.8)),
                ("top_k", json!(20)),
            ]),
        ); // https://huggingface.co/Qwen/Qwen3.5-122B-A10B

        // gpt-oss-120b
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
        ); // https://huggingface.co/openai/gpt-oss-120b
        m.insert(
            "gpt-oss-120b-Q4_K_M",
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
        ); // https://huggingface.co/openai/gpt-oss-120b

        // NVIDIA Nemotron-3-Super-120B-A12B
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
        ); // https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
        m.insert(
            "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M",
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
        ); // https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

        // NVIDIA Nemotron-3-Nano-30B-A3B
        m.insert(
            "Nemotron-3-Nano-30B-A3B-Q4_K_M",
            params(&[
                ("temperature", json!(0.6)),
                ("top_p", json!(0.95)),
                ("chat_template_kwargs", json!({"enable_thinking": true})),
            ]),
        ); // https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

        // Mistral Small 3.2 & Devstral Small 2
        m.insert(
            "mistral-small-3.2:24b-instruct-2506-q4_K_M",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
        m.insert(
            "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
        m.insert(
            "mistral-small-3.2:24b-instruct-2506-q8_0",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
        m.insert(
            "Mistral-Small-3.2-24B-Instruct-2506-Q8_0",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
        m.insert(
            "devstral-small-2:24b-instruct-2512-q4_K_M",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
        m.insert(
            "Devstral-Small-2-24B-Instruct-2512-Q4_K_M",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
        m.insert(
            "devstral-small-2:24b-instruct-2512-q8_0",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
        m.insert(
            "Devstral-Small-2-24B-Instruct-2512-Q8_0",
            params(&[("temperature", json!(0.15))]),
        ); // https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512

        // Ministral-3 Instruct
        m.insert(
            "ministral-3:8b-instruct-2512-q4_K_M",
            params(&[("temperature", json!(0.05))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
        m.insert(
            "Ministral-3-8B-Instruct-2512-Q4_K_M",
            params(&[("temperature", json!(0.05))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
        m.insert(
            "ministral-3:8b-instruct-2512-q8_0",
            params(&[("temperature", json!(0.05))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
        m.insert(
            "Ministral-3-8B-Instruct-2512-Q8_0",
            params(&[("temperature", json!(0.05))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
        m.insert(
            "ministral-3:14b-instruct-2512-q4_K_M",
            params(&[("temperature", json!(0.05))]),
        ); // https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
        m.insert(
            "Ministral-3-14B-Instruct-2512-Q4_K_M",
            params(&[("temperature", json!(0.05))]),
        ); // https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

        // Ministral-3 Reasoning
        m.insert(
            "ministral-3:8b-reasoning-2512-q4_K_M",
            params(&[("temperature", json!(0.7))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
        m.insert(
            "Ministral-3-8B-Reasoning-2512-Q4_K_M",
            params(&[("temperature", json!(0.7))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
        m.insert(
            "ministral-3:8b-reasoning-2512-q8_0",
            params(&[("temperature", json!(0.7))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
        m.insert(
            "Ministral-3-8B-Reasoning-2512-Q8_0",
            params(&[("temperature", json!(0.7))]),
        ); // https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
        m.insert(
            "ministral-3:14b-reasoning-2512-q4_K_M",
            params(&[("temperature", json!(1.0))]),
        ); // https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512
        m.insert(
            "Ministral-3-14B-Reasoning-2512-Q4_K_M",
            params(&[("temperature", json!(1.0))]),
        ); // https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512

        // Mistral Nemo
        m.insert(
            "mistral-nemo:12b-instruct-2407-q4_K_M",
            params(&[("temperature", json!(0.3))]),
        ); // https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        m.insert(
            "Mistral-Nemo-Instruct-2407-Q4_K_M",
            params(&[("temperature", json!(0.3))]),
        ); // https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        m.insert(
            "Mistral-Nemo-Instruct-2407.Q4_K_M",
            params(&[("temperature", json!(0.3))]),
        ); // https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407

        // Granite 4.0
        m.insert(
            "granite-4.0:h-micro-q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0
        m.insert(
            "granite-4.0-h-micro-Q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0
        m.insert(
            "granite-4.0:h-tiny-q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0
        m.insert(
            "granite-4.0-h-tiny-Q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0

        // Granite 4.1
        m.insert(
            "granite4.1:8b-q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // unconfirmed; mirrors granite-4.0 IBM convention
        m.insert(
            "granite-4.1-8b-Q4_K_M",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // unconfirmed; mirrors granite-4.0 IBM convention
        m.insert(
            "granite4.1:8b-q8_0",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // unconfirmed; mirrors granite-4.0 IBM convention
        m.insert(
            "granite-4.1-8b-Q8_0",
            params(&[
                ("temperature", json!(0.0)),
                ("top_p", json!(1.0)),
                ("top_k", json!(0)),
            ]),
        ); // unconfirmed; mirrors granite-4.0 IBM convention
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
}
