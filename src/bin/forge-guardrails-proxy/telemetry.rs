use std::sync::Arc;

use sentry::protocol::{Event, Frame, Stacktrace};

const FORGE_SENTRY_ENABLED: &str = "FORGE_SENTRY_ENABLED";
const FORGE_SENTRY_DSN: &str =
    "https://7da11bb6d89104cb93803a37d77bb86c@o4511520494190592.ingest.us.sentry.io/4511525462999040";

pub(crate) fn init_from_env() -> Result<Option<sentry::ClientInitGuard>, String> {
    if !parse_enabled(std::env::var(FORGE_SENTRY_ENABLED).ok().as_deref())? {
        return Ok(None);
    }

    Ok(Some(sentry::init((
        FORGE_SENTRY_DSN,
        sentry::ClientOptions {
            release: sentry::release_name!(),
            send_default_pii: false,
            max_request_body_size: sentry::MaxRequestBodySize::None,
            auto_session_tracking: false,
            before_send: Some(Arc::new(|event| Some(scrub_event(event)))),
            ..Default::default()
        },
    ))))
}

pub(crate) fn capture_proxy_exit_error() {
    if !is_enabled() {
        return;
    }
    let mut event = Event::new();
    event.level = sentry::Level::Error;
    event.message = Some("forge proxy exited with error".to_string());
    event
        .tags
        .insert("forge.event".to_string(), "proxy_exit_error".to_string());
    sentry::capture_event(event);
}

fn is_enabled() -> bool {
    parse_enabled(std::env::var(FORGE_SENTRY_ENABLED).ok().as_deref()).unwrap_or(false)
}

pub(crate) fn parse_enabled(raw: Option<&str>) -> Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(false);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "{FORGE_SENTRY_ENABLED} must be true or false, got '{raw}'"
        )),
    }
}

pub(crate) fn scrub_event(mut event: Event<'static>) -> Event<'static> {
    event.user = None;
    event.request = None;
    event.breadcrumbs = Default::default();
    event.extra.clear();
    event.debug_meta = Default::default();
    event.contexts.clear();
    event.modules.clear();
    event.server_name = None;
    event.transaction = None;
    event.culprit = None;
    event.logentry = None;
    event.template = None;

    if !event.exception.values.is_empty() {
        event.message = Some("redacted panic".to_string());
    }
    for exception in &mut event.exception.values {
        exception.value = None;
        if let Some(stacktrace) = exception.stacktrace.as_mut() {
            scrub_stacktrace(stacktrace);
        }
        if let Some(stacktrace) = exception.raw_stacktrace.as_mut() {
            scrub_stacktrace(stacktrace);
        }
    }
    if let Some(stacktrace) = event.stacktrace.as_mut() {
        scrub_stacktrace(stacktrace);
    }
    for thread in &mut event.threads.values {
        thread.name = None;
        if let Some(stacktrace) = thread.stacktrace.as_mut() {
            scrub_stacktrace(stacktrace);
        }
        if let Some(stacktrace) = thread.raw_stacktrace.as_mut() {
            scrub_stacktrace(stacktrace);
        }
    }

    event
}

fn scrub_stacktrace(stacktrace: &mut Stacktrace) {
    stacktrace.registers.clear();
    for frame in &mut stacktrace.frames {
        scrub_frame(frame);
    }
}

fn scrub_frame(frame: &mut Frame) {
    frame.filename = None;
    frame.abs_path = None;
    frame.package = None;
    frame.pre_context.clear();
    frame.context_line = None;
    frame.post_context.clear();
    frame.vars.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentry::protocol::{
        Breadcrumb, DebugImage, DebugMeta, Event, Exception, Frame, Request, Stacktrace, User,
        Values,
    };
    use serde_json::json;
    use std::borrow::Cow;

    #[test]
    fn parse_enabled_accepts_unset_and_false_values() {
        assert!(!parse_enabled(None).expect("unset"));
        assert!(!parse_enabled(Some("")).expect("empty"));
        assert!(!parse_enabled(Some("false")).expect("false"));
        assert!(!parse_enabled(Some("0")).expect("zero"));
        assert!(!parse_enabled(Some("off")).expect("off"));
    }

    #[test]
    fn parse_enabled_accepts_true_values() {
        assert!(parse_enabled(Some("true")).expect("true"));
        assert!(parse_enabled(Some("1")).expect("one"));
        assert!(parse_enabled(Some("yes")).expect("yes"));
        assert!(parse_enabled(Some("on")).expect("on"));
    }

    #[test]
    fn parse_enabled_rejects_invalid_values() {
        let err = parse_enabled(Some("maybe")).expect_err("invalid");
        assert!(err.contains("FORGE_SENTRY_ENABLED must be true or false"));
    }

    #[test]
    fn scrub_event_removes_private_event_fields_and_frame_data() {
        let mut event = Event::new();
        event.user = Some(User {
            email: Some("secret@example.com".to_string()),
            ..Default::default()
        });
        event.request = Some(Request {
            data: Some("secret body".to_string()),
            ..Default::default()
        });
        event.breadcrumbs = Values::from(vec![Breadcrumb {
            message: Some("secret breadcrumb".to_string()),
            ..Default::default()
        }]);
        event.extra.insert("secret".to_string(), json!("value"));
        event.server_name = Some(Cow::Borrowed("private-host"));
        event.transaction = Some("POST /private".to_string());
        event.culprit = Some("private culprit".to_string());
        event
            .modules
            .insert("private-module".to_string(), "1".to_string());
        event.debug_meta = Cow::Owned(DebugMeta {
            images: vec![DebugImage::Proguard(sentry::protocol::ProguardDebugImage {
                uuid: sentry::protocol::Uuid::nil(),
            })],
            ..Default::default()
        });
        event.exception = Values::from(vec![Exception {
            value: Some("panic secret".to_string()),
            stacktrace: Some(Stacktrace {
                frames: vec![Frame {
                    filename: Some("private.rs".to_string()),
                    abs_path: Some("/Users/secret/private.rs".to_string()),
                    package: Some("/private/bin".to_string()),
                    pre_context: vec!["secret before".to_string()],
                    context_line: Some("secret line".to_string()),
                    post_context: vec!["secret after".to_string()],
                    vars: [("token".to_string(), json!("secret"))]
                        .into_iter()
                        .collect(),
                    function: Some("safe_function_name".to_string()),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        }]);

        let scrubbed = scrub_event(event);
        let encoded = serde_json::to_string(&scrubbed).expect("event json");

        assert!(scrubbed.user.is_none());
        assert!(scrubbed.request.is_none());
        assert!(scrubbed.breadcrumbs.is_empty());
        assert!(scrubbed.extra.is_empty());
        assert!(scrubbed.debug_meta.is_empty());
        assert!(scrubbed.server_name.is_none());
        assert!(scrubbed.transaction.is_none());
        assert!(scrubbed.culprit.is_none());
        assert!(scrubbed.modules.is_empty());
        assert!(!encoded.contains("secret"));
        assert!(encoded.contains("safe_function_name"));
    }
}
