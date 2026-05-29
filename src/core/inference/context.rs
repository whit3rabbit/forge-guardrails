use crate::context::manager::ContextManager;
use crate::core::message::Message;

pub(super) enum ContextAccess<'a> {
    Direct(&'a mut ContextManager),
    Shared(&'a tokio::sync::Mutex<ContextManager>),
}

impl ContextAccess<'_> {
    pub(super) async fn maybe_compact(
        &mut self,
        messages: &[Message],
        step_index: i64,
        step_hint: Option<&str>,
    ) -> Option<Vec<Message>> {
        match self {
            Self::Direct(context) => match context.maybe_compact(messages, step_index, step_hint) {
                std::borrow::Cow::Owned(new_msgs) => Some(new_msgs),
                std::borrow::Cow::Borrowed(_) => None,
            },
            Self::Shared(context) => {
                let mut context = context.lock().await;
                match context.maybe_compact(messages, step_index, step_hint) {
                    std::borrow::Cow::Owned(new_msgs) => Some(new_msgs),
                    std::borrow::Cow::Borrowed(_) => None,
                }
            }
        }
    }

    pub(super) async fn check_thresholds(&mut self, messages: &[Message]) -> Option<String> {
        match self {
            Self::Direct(context) => context.check_thresholds(messages),
            Self::Shared(context) => {
                let mut context = context.lock().await;
                context.check_thresholds(messages)
            }
        }
    }

    pub(super) async fn update_token_count(&mut self, count: i64) {
        match self {
            Self::Direct(context) => context.update_token_count(count),
            Self::Shared(context) => {
                let mut context = context.lock().await;
                context.update_token_count(count);
            }
        }
    }
}
