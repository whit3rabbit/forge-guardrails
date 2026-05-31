//! ONNX Runtime-backed tool-call scorer.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use anyhow::Context;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{TensorElementType, ValueType};

pub(crate) mod cache;
pub(crate) mod final_response;
#[cfg(test)]
mod tests;
pub(crate) mod tool_call;

pub use final_response::OnnxFinalResponseScorer;
pub use tool_call::OnnxToolCallScorer;

/// Maximum number of ONNX sessions a scorer may hold.
pub const MAX_ONNX_SESSION_POOL_SIZE: usize = 4;

/// Default number of serialized scorer inputs cached per ONNX scorer.
pub(crate) const DEFAULT_ONNX_SCORE_CACHE_CAPACITY: usize = 1024;

/// Runtime options for ONNX-backed semantic scorers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxScorerOptions {
    /// Number of ONNX Runtime sessions to load for concurrent scoring.
    pub session_pool_size: usize,
    /// ONNX Runtime intra-op thread count per session.
    pub intra_threads: usize,
}

impl Default for OnnxScorerOptions {
    fn default() -> Self {
        Self {
            session_pool_size: 1,
            intra_threads: 1,
        }
    }
}

impl OnnxScorerOptions {
    /// Validate that option values stay inside bounded runtime limits.
    pub fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (1..=MAX_ONNX_SESSION_POOL_SIZE).contains(&self.session_pool_size),
            "ONNX session pool size must be between 1 and {}, got {}",
            MAX_ONNX_SESSION_POOL_SIZE,
            self.session_pool_size
        );
        anyhow::ensure!(
            self.intra_threads > 0,
            "ONNX intra-op thread count must be positive"
        );
        Ok(self)
    }
}

pub(crate) fn build_session_pool(
    model_path: &Path,
    options: OnnxScorerOptions,
    label_count: usize,
) -> anyhow::Result<(Vec<Mutex<Session>>, HashSet<String>)> {
    let mut sessions = Vec::with_capacity(options.session_pool_size);
    let mut expected_input_names = None;
    for _ in 0..options.session_pool_size {
        let session = build_session(model_path, options.intra_threads)?;
        let input_names = validate_session_inputs(&session)?;
        validate_session_outputs(&session, label_count)?;
        if let Some(expected) = &expected_input_names {
            anyhow::ensure!(
                expected == &input_names,
                "ONNX session pool input names differ across sessions"
            );
        } else {
            expected_input_names = Some(input_names.clone());
        }
        sessions.push(Mutex::new(session));
    }
    let input_names = expected_input_names.context("ONNX session pool is empty")?;
    Ok((sessions, input_names))
}

fn build_session(model_path: &Path, intra_threads: usize) -> anyhow::Result<Session> {
    let mut builder = Session::builder()?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!("failed to set ONNX optimization level: {err}"))?;
    builder = builder
        .with_intra_threads(intra_threads)
        .map_err(|err| anyhow::anyhow!("failed to set ONNX intra-op threads: {err}"))?;
    builder
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load ONNX model {}", model_path.display()))
}

fn validate_session_inputs(session: &Session) -> anyhow::Result<HashSet<String>> {
    let mut names = HashSet::new();
    for input in session.inputs() {
        let name = input.name().to_string();
        anyhow::ensure!(
            input.dtype().tensor_type() == Some(TensorElementType::Int64),
            "classifier input '{}' must be int64 tensor, got {}",
            name,
            input.dtype()
        );
        match name.as_str() {
            "input_ids" | "attention_mask" | "token_type_ids" => {
                names.insert(name);
            }
            other => {
                anyhow::bail!("unsupported classifier ONNX input '{other}'");
            }
        }
    }
    anyhow::ensure!(
        names.contains("input_ids"),
        "classifier ONNX model is missing input_ids"
    );
    anyhow::ensure!(
        names.contains("attention_mask"),
        "classifier ONNX model is missing attention_mask"
    );
    Ok(names)
}

fn validate_session_outputs(session: &Session, label_count: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !session.outputs().is_empty(),
        "classifier ONNX model has no outputs"
    );
    let output = session
        .outputs()
        .iter()
        .find(|output| output.name() == "logits")
        .unwrap_or(&session.outputs()[0]);
    anyhow::ensure!(
        matches!(
            output.dtype(),
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                ..
            }
        ),
        "classifier output '{}' must be float32 tensor, got {}",
        output.name(),
        output.dtype()
    );
    if let Some(shape) = output.dtype().tensor_shape() {
        if let Some(&last_dim) = shape.last() {
            anyhow::ensure!(
                last_dim < 0 || last_dim as usize == label_count,
                "classifier output '{}' last dimension must be {}, got {}",
                output.name(),
                label_count,
                last_dim
            );
        }
    }
    Ok(())
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|value| value / sum).collect()
}
