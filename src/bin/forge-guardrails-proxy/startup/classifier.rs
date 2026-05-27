use std::sync::Arc;

#[cfg(feature = "classifier")]
use forge_guardrails::{
    FinalResponseContext, FinalResponseScore, ScoringContext, ToolCall, ToolCallScore,
};
use forge_guardrails::{FinalResponseScorer, ScorerMode, ToolCallScorer};

use crate::config::ProxyConfig;

pub(super) fn build_classifier_scorer(
    config: &ProxyConfig,
) -> Result<Option<Arc<dyn ToolCallScorer>>, String> {
    if config.classifier_mode == ScorerMode::Disabled {
        return Ok(None);
    }
    let Some(dir) = config.classifier_dir.as_deref() else {
        return Ok(None);
    };

    #[cfg(feature = "classifier")]
    {
        let scorer = forge_guardrails::OnnxToolCallScorer::from_dir_with_model(
            dir,
            Some(config.classifier_mode),
            config.classifier_model,
        )
        .map_err(|err| format!("failed to load classifier artifact: {err}"))?;
        let scorer: Arc<dyn ToolCallScorer> = Arc::new(scorer);
        Ok(Some(wrap_tool_latency_warning(
            scorer,
            config.classifier_max_latency_ms,
        )))
    }

    #[cfg(not(feature = "classifier"))]
    {
        let _ = dir;
        Err("classifier support requires building with --features classifier".to_string())
    }
}

pub(super) fn build_final_response_classifier_scorer(
    config: &ProxyConfig,
) -> Result<Option<Arc<dyn FinalResponseScorer>>, String> {
    if config.final_response_classifier_mode == ScorerMode::Disabled {
        return Ok(None);
    }
    let Some(dir) = config.final_response_classifier_dir.as_deref() else {
        return Ok(None);
    };

    #[cfg(feature = "classifier")]
    {
        let scorer = forge_guardrails::OnnxFinalResponseScorer::from_dir_with_model(
            dir,
            Some(config.final_response_classifier_mode),
            config.final_response_classifier_model,
        )
        .map_err(|err| format!("failed to load final-response classifier artifact: {err}"))?;
        let scorer: Arc<dyn FinalResponseScorer> = Arc::new(scorer);
        Ok(Some(wrap_final_response_latency_warning(
            scorer,
            config.final_response_classifier_max_latency_ms,
        )))
    }

    #[cfg(not(feature = "classifier"))]
    {
        let _ = dir;
        Err(
            "final-response classifier support requires building with --features classifier"
                .to_string(),
        )
    }
}

#[cfg(feature = "classifier")]
struct ToolCallLatencyWarningScorer {
    inner: Arc<dyn ToolCallScorer>,
    max_latency_ms: u64,
}

#[cfg(feature = "classifier")]
impl ToolCallScorer for ToolCallLatencyWarningScorer {
    fn score(&self, ctx: &ScoringContext, candidate: &ToolCall) -> anyhow::Result<ToolCallScore> {
        let score = self.inner.score(ctx, candidate)?;
        if score.latency_ms > self.max_latency_ms as f64 {
            tracing::warn!(
                target: "forge.classifier",
                tool = %candidate.tool,
                latency_ms = score.latency_ms,
                max_latency_ms = self.max_latency_ms,
                "tool-call classifier latency exceeded configured warning limit"
            );
        }
        Ok(score)
    }
}

#[cfg(feature = "classifier")]
fn wrap_tool_latency_warning(
    scorer: Arc<dyn ToolCallScorer>,
    max_latency_ms: Option<u64>,
) -> Arc<dyn ToolCallScorer> {
    match max_latency_ms {
        Some(max_latency_ms) => Arc::new(ToolCallLatencyWarningScorer {
            inner: scorer,
            max_latency_ms,
        }),
        None => scorer,
    }
}

#[cfg(feature = "classifier")]
struct FinalResponseLatencyWarningScorer {
    inner: Arc<dyn FinalResponseScorer>,
    max_latency_ms: u64,
}

#[cfg(feature = "classifier")]
impl FinalResponseScorer for FinalResponseLatencyWarningScorer {
    fn score(&self, ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore> {
        let score = self.inner.score(ctx)?;
        if score.latency_ms > self.max_latency_ms as f64 {
            tracing::warn!(
                target: "forge.classifier",
                latency_ms = score.latency_ms,
                max_latency_ms = self.max_latency_ms,
                "final-response classifier latency exceeded configured warning limit"
            );
        }
        Ok(score)
    }
}

#[cfg(feature = "classifier")]
fn wrap_final_response_latency_warning(
    scorer: Arc<dyn FinalResponseScorer>,
    max_latency_ms: Option<u64>,
) -> Arc<dyn FinalResponseScorer> {
    match max_latency_ms {
        Some(max_latency_ms) => Arc::new(FinalResponseLatencyWarningScorer {
            inner: scorer,
            max_latency_ms,
        }),
        None => scorer,
    }
}
