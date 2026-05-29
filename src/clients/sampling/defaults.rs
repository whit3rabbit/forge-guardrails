use std::collections::HashMap;
use std::sync::LazyLock;

use serde_json::{json, Map, Value};

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
