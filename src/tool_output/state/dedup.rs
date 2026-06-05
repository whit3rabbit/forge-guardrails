use indexmap::IndexMap;
use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

/// Bounded in-memory dedup state for compressed tool outputs.
#[derive(Debug, Default)]
pub struct ToolOutputCompressionState {
    inner: Mutex<DedupState>,
}

#[derive(Debug, Default)]
struct DedupState {
    sessions: IndexMap<String, VecDeque<DedupRecord>>,
    session_order: VecDeque<String>,
    next_call_index: u64,
}

#[derive(Debug, Clone)]
struct DedupRecord {
    hash: u64,
    output_len: usize,
    tool_name: String,
    call_index: u64,
}

impl ToolOutputCompressionState {
    /// Create empty bounded compression state.
    pub fn new() -> Self {
        Self::default()
    }

    pub(in crate::tool_output) fn deduplicate(
        &self,
        session_id: &str,
        tool_name: &str,
        output: &str,
        max_sessions: usize,
        max_entries_per_session: usize,
    ) -> Option<String> {
        if session_id.is_empty() || output.is_empty() {
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
                return Some(format!(
                    "[Duplicate of call #{} ({}) - see earlier result]",
                    record.call_index, record.tool_name
                ));
            }
        }

        if !state.sessions.contains_key(session_id) {
            state.session_order.push_back(session_id.to_string());
            state
                .sessions
                .insert(session_id.to_string(), VecDeque::new());
        }

        state.next_call_index = state.next_call_index.saturating_add(1);
        let call_index = state.next_call_index;
        let records = state
            .sessions
            .get_mut(session_id)
            .expect("session inserted above");
        records.push_back(DedupRecord {
            hash,
            output_len,
            tool_name: tool_name.to_string(),
            call_index,
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
}

fn hash_output(output: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    output.hash(&mut hasher);
    hasher.finish()
}
