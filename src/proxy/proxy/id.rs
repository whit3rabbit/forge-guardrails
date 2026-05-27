use std::sync::atomic::{AtomicU64, Ordering};

static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random-ish completion ID.
pub(super) fn generate_completion_id() -> String {
    format!("chatcmpl-{}", uuid_prefix())
}

/// Generate a random-ish call ID.
pub(super) fn generate_call_id() -> String {
    let id = CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{id:016x}")
}

pub(crate) fn openai_stream_completion_id() -> String {
    generate_completion_id()
}

/// Short hex prefix for IDs (8 chars).
fn uuid_prefix() -> String {
    use std::time::SystemTime;
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:08x}", (t as u32).wrapping_mul(2654435761))
}
