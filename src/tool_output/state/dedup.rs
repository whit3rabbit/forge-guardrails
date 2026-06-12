use indexmap::IndexMap;
use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use super::memo::{MemoLookup, MemoRecord, MemoState};

/// Bounded in-memory dedup and memo state for compressed tool outputs.
#[derive(Debug, Default)]
pub struct ToolOutputCompressionState {
    inner: Mutex<DedupState>,
    memo: Mutex<MemoState>,
}

#[derive(Debug, Default)]
struct DedupState {
    sessions: IndexMap<String, VecDeque<DedupRecord>>,
    session_order: VecDeque<String>,
}

#[derive(Debug, Clone)]
struct DedupRecord {
    hash: u64,
    output_len: usize,
    tool_name: String,
    tool_call_id: String,
}

impl ToolOutputCompressionState {
    /// Create empty bounded compression state.
    pub fn new() -> Self {
        Self::default()
    }

    pub(in crate::tool_output) fn deduplicate(
        &self,
        session_id: &str,
        tool_call_id: &str,
        tool_name: &str,
        output: &str,
        max_sessions: usize,
        max_entries_per_session: usize,
    ) -> Option<String> {
        if session_id.is_empty() || tool_call_id.is_empty() || output.is_empty() {
            return None;
        }
        let max_sessions = max_sessions.max(1);
        let max_entries_per_session = max_entries_per_session.max(1);
        let hash = hash_output(output);
        let mut state = self.inner.lock().expect("tool output dedup lock");

        let output_len = output.len();
        if let Some(records) = state.sessions.get(session_id) {
            if let Some(record) = records.iter().find(|record| {
                record.tool_name == tool_name
                    && record.hash == hash
                    && record.output_len == output_len
            }) {
                // The same call re-sent in a later request must keep its
                // content; only a different call with identical output is a
                // true duplicate.
                if record.tool_call_id == tool_call_id {
                    return None;
                }
                // Collapse the LATER duplicate, not the earlier one. The
                // earlier message anchors the prompt-cache prefix; rewriting
                // it on every resend would bust the prefix on every subsequent
                // request. The later message is cheapest to replace: it has
                // not yet been cached. Near-duplicate delta encoding was
                // considered and rejected because it breaks determinism when
                // the base entry is evicted under FIFO pressure.
                return Some(format!(
                    "[Duplicate of {} ({}); see earlier result]",
                    record.tool_call_id, record.tool_name
                ));
            }
        }

        if !state.sessions.contains_key(session_id) {
            state.session_order.push_back(session_id.to_string());
            state
                .sessions
                .insert(session_id.to_string(), VecDeque::new());
        }

        let records = state
            .sessions
            .get_mut(session_id)
            .expect("session inserted above");
        records.push_back(DedupRecord {
            hash,
            output_len,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
        });
        while records.len() > max_entries_per_session {
            records.pop_front();
        }

        while state.sessions.len() > max_sessions {
            let Some(oldest) = state.session_order.pop_front() else {
                break;
            };
            if oldest != session_id {
                state.sessions.shift_remove(&oldest);
            } else {
                state.session_order.push_back(oldest);
                break;
            }
        }

        None
    }

    pub(in crate::tool_output) fn lookup_memo(
        &self,
        session_id: &str,
        tool_call_id: &str,
        input_hash: u64,
        input_len: usize,
        config_fp: u64,
    ) -> MemoLookup {
        self.memo.lock().expect("tool output memo lock").lookup(
            session_id,
            tool_call_id,
            input_hash,
            input_len,
            config_fp,
        )
    }

    pub(in crate::tool_output) fn store_memo(
        &self,
        session_id: &str,
        tool_call_id: &str,
        record: MemoRecord,
        max_sessions: usize,
    ) {
        self.memo.lock().expect("tool output memo lock").store(
            session_id,
            tool_call_id,
            record,
            max_sessions,
        );
    }
}

fn hash_output(output: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    output.hash(&mut hasher);
    hasher.finish()
}
