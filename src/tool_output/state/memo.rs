use super::super::ToolOutputCompressionConfig;
use indexmap::IndexMap;
use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::hash::{Hash, Hasher};

/// Maximum bytes of one memoized compressed output.
pub(in crate::tool_output) const MAX_MEMO_OUTPUT_BYTES: usize = 16 * 1024;
/// Maximum total memoized output bytes retained per session.
pub(in crate::tool_output) const MAX_MEMO_BYTES_PER_SESSION: usize = 512 * 1024;

/// Bounded per-session memo of final compressed outputs keyed by tool call
/// id. Reusing the memoized bytes on a full-history resend keeps prior
/// messages byte-stable across requests, which preserves upstream
/// prompt-cache prefixes and skips recompression work.
#[derive(Debug, Default)]
pub(in crate::tool_output) struct MemoState {
    sessions: IndexMap<String, MemoSession>,
    session_order: VecDeque<String>,
}

#[derive(Debug, Default)]
struct MemoSession {
    entries: IndexMap<String, MemoRecord>,
    order: VecDeque<String>,
    total_bytes: usize,
}

#[derive(Debug, Clone)]
pub(in crate::tool_output) struct MemoRecord {
    pub(in crate::tool_output) input_hash: u64,
    pub(in crate::tool_output) input_len: usize,
    pub(in crate::tool_output) config_fingerprint: u64,
    pub(in crate::tool_output) output: String,
}

pub(in crate::tool_output) enum MemoLookup {
    /// Identical input and config seen before; reuse these exact bytes.
    Hit(String),
    /// An entry exists for this call id but input or config differ; the
    /// compressed form of an already-sent message is about to change.
    Changed,
    Miss,
}

impl MemoState {
    pub(in crate::tool_output) fn lookup(
        &self,
        session_id: &str,
        tool_call_id: &str,
        input_hash: u64,
        input_len: usize,
        config_fingerprint: u64,
    ) -> MemoLookup {
        let Some(session) = self.sessions.get(session_id) else {
            return MemoLookup::Miss;
        };
        let Some(record) = session.entries.get(tool_call_id) else {
            return MemoLookup::Miss;
        };
        if record.input_hash == input_hash
            && record.input_len == input_len
            && record.config_fingerprint == config_fingerprint
        {
            MemoLookup::Hit(record.output.clone())
        } else {
            MemoLookup::Changed
        }
    }

    pub(in crate::tool_output) fn store(
        &mut self,
        session_id: &str,
        tool_call_id: &str,
        record: MemoRecord,
        max_sessions: usize,
    ) {
        let max_sessions = max_sessions.max(1);
        let oversized = record.output.len() > MAX_MEMO_OUTPUT_BYTES;
        if !self.sessions.contains_key(session_id) {
            if oversized {
                return;
            }
            self.session_order.push_back(session_id.to_string());
            self.sessions
                .insert(session_id.to_string(), MemoSession::default());
        }
        let session = self
            .sessions
            .get_mut(session_id)
            .expect("session inserted above");
        session.remove(tool_call_id);
        if !oversized {
            session.total_bytes += record.output.len();
            session.order.push_back(tool_call_id.to_string());
            session.entries.insert(tool_call_id.to_string(), record);
            while session.total_bytes > MAX_MEMO_BYTES_PER_SESSION {
                let Some(oldest) = session.order.pop_front() else {
                    break;
                };
                if let Some(old) = session.entries.shift_remove(&oldest) {
                    session.total_bytes -= old.output.len();
                }
            }
        }

        while self.sessions.len() > max_sessions {
            let Some(oldest) = self.session_order.pop_front() else {
                break;
            };
            if oldest != session_id {
                self.sessions.shift_remove(&oldest);
            } else {
                self.session_order.push_back(oldest);
                break;
            }
        }
    }

    #[allow(dead_code)]
    pub(in crate::tool_output) fn remove(&mut self, session_id: &str, tool_call_id: &str) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.remove(tool_call_id);
        }
    }
}

impl MemoSession {
    fn remove(&mut self, tool_call_id: &str) {
        if let Some(old) = self.entries.shift_remove(tool_call_id) {
            self.total_bytes -= old.output.len();
            self.order.retain(|id| id != tool_call_id);
        }
    }
}

/// Fingerprint of the config fields that influence compressed output; a
/// mismatch invalidates the memo entry instead of replaying stale bytes.
pub(in crate::tool_output) fn config_fingerprint(config: &ToolOutputCompressionConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.mode.as_str().hash(&mut hasher);
    config.method.as_str().hash(&mut hasher);
    config.redact_secrets.hash(&mut hasher);
    config.enable_dedup.hash(&mut hasher);
    config.max_output_bytes.hash(&mut hasher);
    hasher.finish()
}
