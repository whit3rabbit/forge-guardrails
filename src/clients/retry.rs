//! Shared retry policy for transient upstream HTTP failures.
//!
//! The backend adapters that talk to OpenAI-compatible upstreams over reqwest
//! (the anyllm sidecar client and the Anthropic client) can hit HTTP 429 when
//! rate-limited, 5xx on transient provider faults, or a refused/reset connection
//! that proves the request never reached the server. The backoff, jitter,
//! `Retry-After` parsing, and status policy all live in [`anyllm_client::retry`];
//! this module is the Forge-side glue. It resolves the env-configured retry
//! count once and runs one shared send loop (with Forge's optional auth,
//! per-request timeout, and stderr-visible warnings) so each adapter does not
//! re-implement the policy.

use anyllm_client::retry::{
    backoff_delay, is_quota_exhausted, is_retryable, parse_retry_after, RetryPolicy,
};

use crate::error::BackendError;

/// Default retries beyond the first attempt (3 total attempts), preserving the
/// historical conservative default when `FORGE_UPSTREAM_MAX_RETRIES` is unset.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// Pure parse of the `FORGE_UPSTREAM_MAX_RETRIES` value (kept separate from the
/// env read so it is testable without mutating process-global state).
fn parse_max_retries(raw: Option<&str>) -> u32 {
    match raw {
        Some(value) => value.trim().parse::<u32>().unwrap_or(DEFAULT_MAX_RETRIES),
        None => DEFAULT_MAX_RETRIES,
    }
}

/// Resolves the shared upstream retry count from `FORGE_UPSTREAM_MAX_RETRIES`.
///
/// The env var is a retry count (attempts beyond the first), so total attempts
/// are `retries + 1`. An unset or unparseable value falls back to
/// [`DEFAULT_MAX_RETRIES`]. A value of `0` disables retries (one attempt).
pub fn max_retries_from_env() -> u32 {
    parse_max_retries(std::env::var("FORGE_UPSTREAM_MAX_RETRIES").ok().as_deref())
}

/// The default upstream retry policy: the env-configured retry count, retrying
/// 408/429/5xx statuses plus connect-only transport errors.
///
/// Connect errors are retried because a refused/reset connection proves the
/// upstream never received the POST, so a retry cannot duplicate a (billable,
/// non-idempotent) completion. Read/response timeouts are never retried, because
/// the server may have already processed the request; that split is enforced
/// inside [`anyllm_client::retry`].
pub fn upstream_retry_policy() -> RetryPolicy {
    RetryPolicy::new(max_retries_from_env()).with_transport_retries(true)
}

/// Sends a POST request with retry on 408/429/5xx (and connect transport
/// errors), reusing [`anyllm_client::retry`]'s backoff and `Retry-After` policy.
///
/// `build` is invoked once per attempt because reqwest consumes the
/// `RequestBuilder`; callers serialize the body once and reuse it inside the
/// closure (for example via `.body(bytes.clone())`) so retries do not re-encode
/// it. Returns the upstream response on a 2xx, or a [`BackendError`] carrying the
/// terminal status code (or `0` for an unretried transport error).
pub async fn send_post_with_retry<F>(
    mut build: F,
    policy: &RetryPolicy,
    label: &str,
) -> Result<reqwest::Response, BackendError>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let max_retries = policy.max_retries;
    for attempt in 0..=max_retries {
        let resp = match build().send().await {
            Ok(resp) => resp,
            Err(e) => {
                // Only connect errors are safe to retry: the request was never
                // sent, so re-sending cannot duplicate a completion.
                if policy.retry_transport_errors && attempt < max_retries && e.is_connect() {
                    let delay = backoff_delay(attempt, None);
                    eprintln!(
                        "warning: {label} transport error (attempt {}/{}); retrying in {}ms",
                        attempt + 1,
                        max_retries + 1,
                        delay.as_millis()
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return Err(BackendError::new(0, e.to_string()));
            }
        };

        if resp.status().is_success() {
            return Ok(resp);
        }

        let status = resp.status().as_u16();
        if attempt < max_retries && is_retryable(status) {
            // Read Retry-After before the body is consumed.
            let retry_after = parse_retry_after(resp.headers());
            if status == 429 {
                // A 429 is either a transient rate limit (retry) or a hard
                // quota/credit exhaustion carrying OpenAI's `insufficient_quota`
                // code, which will not clear by waiting. Peek the body via
                // anyllm's structured detector and fast-fail on exhaustion
                // instead of burning backoff cycles; this matches anyllm's own
                // retry loop. Reading the body also returns the connection to
                // the pool.
                let body_text = resp.text().await.unwrap_or_default();
                if is_quota_exhausted(&body_text) {
                    return Err(BackendError::new(i64::from(status), body_text));
                }
            } else {
                // Drain the body so the connection returns to the pool before sleeping.
                let _ = resp.bytes().await;
            }
            let delay = backoff_delay(attempt, retry_after);
            // The proxy installs no tracing subscriber, so stderr is the
            // operator-visible channel for transient upstream retries.
            eprintln!(
                "warning: {label} upstream status {status} (attempt {}/{}); retrying in {}ms",
                attempt + 1,
                max_retries + 1,
                delay.as_millis()
            );
            tokio::time::sleep(delay).await;
            continue;
        }

        let body_text = resp.text().await.unwrap_or_default();
        return Err(BackendError::new(i64::from(status), body_text));
    }
    unreachable!("loop runs max_retries + 1 times and always returns")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_max_retries_handles_values() {
        assert_eq!(parse_max_retries(None), DEFAULT_MAX_RETRIES);
        assert_eq!(parse_max_retries(Some("0")), 0);
        assert_eq!(parse_max_retries(Some(" 4 ")), 4);
        assert_eq!(parse_max_retries(Some("bogus")), DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn default_policy_enables_connect_retries() {
        // Independent of the env-derived retry count, the shared policy always
        // opts into connect-only transport retries.
        assert!(upstream_retry_policy().retry_transport_errors);
    }
}
