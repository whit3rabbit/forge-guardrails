//! Helper functions for Ollama client configuration.

use serde_json::{json, Map, Value};

use super::OllamaClient;
use crate::clients::base::SamplingParams;

/// Keywords that indicate a model supports thinking/reasoning mode.
const THINK_KEYWORDS: &[&str] = &["think", "reasoning", "r1", "deepseek", "qwq"];

pub(super) fn detect_think_mode(model: &str) -> (bool, bool) {
    let lower = model.to_lowercase();
    let matches = THINK_KEYWORDS.iter().any(|kw| lower.contains(kw));
    (matches, false)
}

impl OllamaClient {
    /// Builds Ollama options Map merging sampling defaults, instance fields, and per-call overrides.
    pub fn build_options(&self, sampling: Option<&SamplingParams>) -> Map<String, Value> {
        let mut opts = Map::new();
        if let Some(ref defaults) = self.sampling_defaults {
            for (k, v) in defaults {
                opts.insert(k.clone(), v.clone());
            }
        }
        if let Some(t) = self.temperature {
            opts.insert("temperature".into(), json!(t));
        }
        if let Some(t) = self.top_p {
            opts.insert("top_p".into(), json!(t));
        }
        if let Some(k) = self.top_k {
            opts.insert("top_k".into(), json!(k));
        }
        if let Some(m) = self.min_p {
            opts.insert("min_p".into(), json!(m));
        }
        if let Some(r) = self.repeat_penalty {
            opts.insert("repeat_penalty".into(), json!(r));
        }
        if let Some(p) = self.presence_penalty {
            opts.insert("presence_penalty".into(), json!(p));
        }
        if let Some(sp) = sampling {
            for (k, v) in sp {
                if matches!(
                    k.as_str(),
                    "temperature"
                        | "top_p"
                        | "top_k"
                        | "min_p"
                        | "repeat_penalty"
                        | "presence_penalty"
                        | "seed"
                ) {
                    opts.insert(k.clone(), v.clone());
                }
            }
        }
        if let Ok(guard) = self.num_ctx.lock() {
            if let Some(ctx) = *guard {
                opts.insert("num_ctx".into(), json!(ctx));
            }
        }
        opts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_heuristic_match() {
        let (t, _) = detect_think_mode("deepseek-r1:8b");
        assert!(t);
    }

    #[test]
    fn think_heuristic_qwq() {
        let (t, _) = detect_think_mode("qwq:32b");
        assert!(t);
    }

    #[test]
    fn think_heuristic_no_match() {
        let (t, _) = detect_think_mode("llama3:8b");
        assert!(!t);
    }

    #[test]
    fn think_explicit_true() {
        let c = OllamaClient::new("llama3").with_think(Some(true));
        assert!(c.is_think_enabled());
        assert!(c.is_think_resolved());
    }

    #[test]
    fn think_explicit_false() {
        let c = OllamaClient::new("deepseek-r1").with_think(Some(false));
        assert!(!c.is_think_enabled());
        assert!(c.is_think_resolved());
    }

    #[test]
    fn options_no_ctx() {
        let c = OllamaClient::new("llama3");
        assert!(c.build_options(None).is_empty());
    }

    #[test]
    fn options_with_ctx() {
        let c = OllamaClient::new("llama3");
        c.set_num_ctx(Some(8192));
        assert_eq!(c.build_options(None).get("num_ctx"), Some(&json!(8192)));
    }

    #[test]
    fn options_all_params() {
        let c = OllamaClient::new("llama3")
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_top_k(40)
            .with_min_p(0.05)
            .with_repeat_penalty(1.1)
            .with_presence_penalty(0.5);
        let o = c.build_options(None);
        assert_eq!(o.get("temperature"), Some(&json!(0.7)));
        assert_eq!(o.get("top_k"), Some(&json!(40)));
    }

    #[test]
    fn per_call_override() {
        let c = OllamaClient::new("llama3").with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        assert_eq!(
            c.build_options(Some(&sp)).get("temperature"),
            Some(&json!(0.9))
        );
    }

    #[test]
    fn per_call_none_uses_instance() {
        let c = OllamaClient::new("llama3").with_temperature(0.5);
        assert_eq!(c.build_options(None).get("temperature"), Some(&json!(0.5)));
    }

    #[test]
    fn per_call_instance_immutability() {
        let c = OllamaClient::new("llama3").with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let _ = c.build_options(Some(&sp));
        assert_eq!(c.build_options(None).get("temperature"), Some(&json!(0.5)));
    }

    #[test]
    fn temp_default_absent() {
        assert!(OllamaClient::new("llama3")
            .build_options(None)
            .get("temperature")
            .is_none());
    }

    #[test]
    fn temp_explicit() {
        assert_eq!(
            OllamaClient::new("llama3")
                .with_temperature(0.3)
                .build_options(None)
                .get("temperature"),
            Some(&json!(0.3))
        );
    }

    #[test]
    fn ctx_none_default() {
        let c = OllamaClient::new("llama3");
        assert!(c.num_ctx.lock().expect("l").is_none());
    }

    #[test]
    fn ctx_set_and_clear() {
        let c = OllamaClient::new("llama3");
        c.set_num_ctx(Some(4096));
        assert_eq!(*c.num_ctx.lock().expect("l"), Some(4096));
        c.set_num_ctx(None);
        assert!(c.num_ctx.lock().expect("l").is_none());
    }

    #[test]
    fn rec_unknown() {
        assert!(OllamaClient::new("unknown")
            .with_recommended_sampling(true)
            .sampling_defaults
            .is_none());
    }

    #[test]
    fn rec_known() {
        assert!(OllamaClient::new("qwen3:8b-q4_K_M")
            .with_recommended_sampling(true)
            .sampling_defaults
            .is_some());
    }

    #[test]
    fn rec_explicit_override() {
        let c = OllamaClient::new("qwen3:8b-q4_K_M")
            .with_recommended_sampling(true)
            .with_temperature(0.99);
        assert_eq!(c.build_options(None).get("temperature"), Some(&json!(0.99)));
    }
}
