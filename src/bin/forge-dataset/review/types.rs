use serde_json::Value;

use super::client::JsonLlmClient;
use super::progress::ReviewProgress;
use super::resume::ResumeState;

#[derive(Debug)]
pub(crate) struct CaptureJob {
    pub(crate) row_index: usize,
    pub(crate) capture: Value,
}

pub(crate) struct CaptureBatchRun<'a> {
    pub(crate) batch: &'a mut Vec<CaptureJob>,
    pub(crate) reviewer: &'a JsonLlmClient,
    pub(crate) verifier: &'a JsonLlmClient,
    pub(crate) reject_path: &'a std::path::Path,
    pub(crate) output_path: &'a str,
    pub(crate) resume_state: &'a mut ResumeState,
    pub(crate) progress: &'a mut ReviewProgress,
    pub(crate) alternative_proposals: &'a mut Vec<AlternativeProposal>,
}

#[derive(Debug, Default)]
pub(crate) struct CaptureReviewOutcome {
    pub(crate) writes: Vec<CaptureWrite>,
    pub(crate) alternative_proposals: Vec<AlternativeProposal>,
}

impl CaptureReviewOutcome {
    pub(crate) fn training(&mut self, row: Value) {
        self.writes.push(CaptureWrite::Training(row));
    }

    pub(crate) fn reject(
        &mut self,
        reason: &str,
        detail: impl Into<String>,
        capture: &Value,
        raw_response: Option<Value>,
    ) {
        self.writes.push(CaptureWrite::Reject(RejectRecord {
            reason: reason.to_string(),
            detail: detail.into(),
            capture: capture.clone(),
            raw_response,
        }));
    }
}

#[derive(Debug)]
pub(crate) enum CaptureWrite {
    Training(Value),
    Reject(RejectRecord),
}

#[derive(Debug)]
pub(crate) struct RejectRecord {
    pub(crate) reason: String,
    pub(crate) detail: String,
    pub(crate) capture: Value,
    pub(crate) raw_response: Option<Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct ReviewDecision {
    pub(crate) label: String,
    pub(crate) confidence: f64,
    pub(crate) rationale: String,
    pub(crate) corrected_candidate_call: Option<Value>,
    pub(crate) raw: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct VerifierDecision {
    pub(crate) accepted: bool,
    pub(crate) rationale: String,
    pub(crate) raw: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct AlternativeProposal {
    pub(crate) capture: Value,
    pub(crate) example_group_id: String,
    pub(crate) candidate_call: Value,
    pub(crate) label: String,
}

pub(crate) struct AlternativeReviewRun<'a> {
    pub(crate) reviewer: &'a JsonLlmClient,
    pub(crate) verifier: &'a JsonLlmClient,
    pub(crate) proposals: Vec<AlternativeProposal>,
    pub(crate) alternative_cap: usize,
    pub(crate) max_per_group: usize,
    pub(crate) reject_path: &'a std::path::Path,
    pub(crate) output_path: &'a str,
    pub(crate) resume_state: &'a mut ResumeState,
    pub(crate) progress: &'a mut ReviewProgress,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReviewRole {
    Reviewer,
    Verifier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderConfig {
    pub(crate) provider: String,
    pub(crate) chat_url: String,
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
}
