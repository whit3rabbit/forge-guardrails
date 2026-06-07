use serde_json::Value;

use super::types::AlternativeProposal;

#[derive(Debug)]
pub(crate) struct ReviewProgress {
    pub(crate) total: usize,
    pub(crate) seen: usize,
    pub(crate) skipped: usize,
    pub(crate) accepted_real: usize,
    pub(crate) accepted_corrected: usize,
    pub(crate) accepted_alternatives: usize,
    pub(crate) rejected: usize,
    pub(crate) alternatives_seen: usize,
    pub(crate) alternatives_skipped: usize,
}

impl ReviewProgress {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            total,
            seen: 0,
            skipped: 0,
            accepted_real: 0,
            accepted_corrected: 0,
            accepted_alternatives: 0,
            rejected: 0,
            alternatives_seen: 0,
            alternatives_skipped: 0,
        }
    }

    pub(crate) fn log_skip_if_needed(&self) {
        if self.skipped == 1 || self.skipped.is_multiple_of(250) {
            eprintln!(
                "review resume skipped={} seen={}/{}",
                self.skipped, self.seen, self.total
            );
        }
    }

    pub(crate) fn log_reviewing(&self, capture: &Value) {
        eprintln!(
            "review row {}/{} group={} domain={} scenario={} tool={}",
            self.seen,
            self.total,
            value_str(capture.get("example_group_id")),
            trace_str(capture, "domain"),
            trace_str(capture, "scenario"),
            candidate_tool(capture.get("candidate_call"))
        );
    }

    pub(crate) fn log_accepted(&self, row_index: usize, label: &str, source_bucket: &str) {
        eprintln!(
            "review row {}/{} accepted label={} source_bucket={} accepted_real={} accepted_corrected={} rejected={}",
            row_index,
            self.total,
            label,
            source_bucket,
            self.accepted_real,
            self.accepted_corrected,
            self.rejected
        );
    }

    pub(crate) fn log_rejected(&self, row_index: usize, reason: &str) {
        eprintln!(
            "review row {}/{} rejected reason={} accepted_real={} accepted_corrected={} rejected={}",
            row_index,
            self.total,
            reason,
            self.accepted_real,
            self.accepted_corrected,
            self.rejected
        );
    }

    pub(crate) fn log_alternative_skip_if_needed(&self) {
        if self.alternatives_skipped == 1 || self.alternatives_skipped.is_multiple_of(250) {
            eprintln!(
                "review alternative skipped={} seen={}",
                self.alternatives_skipped, self.alternatives_seen
            );
        }
    }

    pub(crate) fn log_alternative_reviewing(&self, proposal: &AlternativeProposal) {
        eprintln!(
            "review alternative {} group={} label={} tool={}",
            self.alternatives_seen,
            proposal.example_group_id,
            proposal.label,
            candidate_tool(Some(&proposal.candidate_call))
        );
    }

    pub(crate) fn log_alternative_accepted(&self, label: &str) {
        eprintln!(
            "review alternative {} accepted label={} accepted_alternatives={} rejected={}",
            self.alternatives_seen, label, self.accepted_alternatives, self.rejected
        );
    }

    pub(crate) fn log_alternative_rejected(&self, reason: &str) {
        eprintln!(
            "review alternative {} rejected reason={} accepted_alternatives={} rejected={}",
            self.alternatives_seen, reason, self.accepted_alternatives, self.rejected
        );
    }
}

pub(crate) fn value_str(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("unknown")
}

pub(crate) fn trace_str<'a>(capture: &'a Value, key: &str) -> &'a str {
    capture
        .get("proxy_trace")
        .and_then(|trace| trace.get(key))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

pub(crate) fn candidate_tool(candidate: Option<&Value>) -> &str {
    candidate
        .and_then(|candidate| candidate.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}
