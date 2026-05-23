//! Slot worker: serializes workflow execution via priority queuing.
//!
//! SlotWorker processes tasks one at a time from a priority queue.
//! Higher-priority tasks (lower integer) can preempt the running task.

use super::runner::WorkflowRunner;
use super::workflow::Workflow;
use crate::clients::base::LLMClient;
use crate::error::ForgeError;
use indexmap::IndexMap;
use serde_json::Value;
use std::cmp::Ordering;
use std::sync::Arc;
use tokio::sync::{watch, Notify};

/// A pending or running task in the slot worker queue.
struct SlotTask {
    workflow: Arc<Workflow>,
    user_message: String,
    priority: i32,
    prompt_vars: Option<IndexMap<String, String>>,
    cancel_tx: watch::Sender<bool>,
    result_tx: tokio::sync::oneshot::Sender<Result<Value, ForgeError>>,
}

impl Eq for SlotTask {}

impl PartialEq for SlotTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl PartialOrd for SlotTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SlotTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority integer = higher priority (min-heap: smallest first).
        self.priority.cmp(&other.priority)
    }
}

/// Serializes workflow execution on a single inference slot with priority
/// queuing and auto-preemption.
pub struct SlotWorker<C: LLMClient> {
    runner: Arc<WorkflowRunner<C>>,
    queue: Arc<tokio::sync::Mutex<Vec<SlotTask>>>,
    notify: Arc<Notify>,
    current_cancel: Arc<tokio::sync::Mutex<Option<watch::Sender<bool>>>>,
    current_priority: Arc<tokio::sync::Mutex<Option<i32>>>,
}

impl<C: LLMClient> SlotWorker<C> {
    pub fn new(runner: Arc<WorkflowRunner<C>>) -> Self {
        Self {
            runner,
            queue: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            notify: Arc::new(Notify::new()),
            current_cancel: Arc::new(tokio::sync::Mutex::new(None)),
            current_priority: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Submit a workflow for execution.
    ///
    /// Returns a oneshot receiver that resolves to the terminal tool result.
    /// If the submitted task has strictly higher priority (lower integer) than
    /// the currently running task, the running task is cancelled.
    pub async fn submit(
        &self,
        workflow: Arc<Workflow>,
        user_message: String,
        priority: i32,
        prompt_vars: Option<IndexMap<String, String>>,
    ) -> tokio::sync::oneshot::Receiver<Result<Value, ForgeError>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let (cancel_tx, _cancel_rx) = watch::channel(false);

        // Check preemption: cancel current if new task has higher priority.
        {
            let cur_priority = self.current_priority.lock().await;
            if let Some(cur_p) = *cur_priority {
                if priority < cur_p {
                    let cancel = self.current_cancel.lock().await;
                    if let Some(ref tx) = *cancel {
                        let _ = tx.send(true);
                    }
                }
            }
        }

        let task = SlotTask {
            workflow,
            user_message,
            priority,
            prompt_vars,
            cancel_tx,
            result_tx,
        };

        {
            let mut queue = self.queue.lock().await;
            // Insert sorted by priority (min-heap: lowest integer first).
            let pos = queue
                .iter()
                .position(|t| task.cmp(t) == Ordering::Less)
                .unwrap_or(queue.len());
            queue.insert(pos, task);
        }

        self.notify.notify_one();
        result_rx
    }

    /// Run the worker loop. Processes tasks from the queue sequentially.
    pub async fn run(&self) {
        loop {
            // Wait for a task.
            {
                let queue = self.queue.lock().await;
                if queue.is_empty() {
                    drop(queue);
                    self.notify.notified().await;
                    continue;
                }
            }

            let task = {
                let mut queue = self.queue.lock().await;
                if queue.is_empty() {
                    continue;
                }
                queue.remove(0)
            };

            // Check if the receiver has already been cancelled.
            if task.result_tx.is_closed() {
                continue;
            }

            // Set current state.
            {
                let mut cur_cancel = self.current_cancel.lock().await;
                *cur_cancel = Some(task.cancel_tx.clone());
                let mut cur_priority = self.current_priority.lock().await;
                *cur_priority = Some(task.priority);
            }

            let cancel_rx = task.cancel_tx.subscribe();

            let result = self
                .runner
                .run(
                    &task.workflow,
                    &task.user_message,
                    task.prompt_vars.as_ref(),
                    None,
                    Some(cancel_rx),
                )
                .await;

            // Clear current state.
            {
                let mut cur_cancel = self.current_cancel.lock().await;
                *cur_cancel = None;
                let mut cur_priority = self.current_priority.lock().await;
                *cur_priority = None;
            }

            let _ = task.result_tx.send(result);
        }
    }

    /// Cancel the currently running task.
    pub async fn cancel_current(&self) {
        let cancel = self.current_cancel.lock().await;
        if let Some(ref tx) = *cancel {
            let _ = tx.send(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_task_ordering() {
        let (tx1, _) = watch::channel(false);
        let (tx2, _) = watch::channel(false);
        let (rx1, _) = tokio::sync::oneshot::channel();
        let (rx2, _) = tokio::sync::oneshot::channel();

        let low = SlotTask {
            workflow: Arc::new(dummy_workflow()),
            user_message: "low".to_string(),
            priority: 10,
            prompt_vars: None,
            cancel_tx: tx1,
            result_tx: rx1,
        };
        let high = SlotTask {
            workflow: Arc::new(dummy_workflow()),
            user_message: "high".to_string(),
            priority: 1,
            prompt_vars: None,
            cancel_tx: tx2,
            result_tx: rx2,
        };
        assert!(high < low, "lower priority integer should sort first");
    }

    fn dummy_workflow() -> Workflow {
        use crate::core::workflow::{TerminalToolInput, ToolDef};
        use crate::tools::respond::respond_tool;
        use indexmap::IndexMap;

        let mut tools: IndexMap<String, ToolDef> = IndexMap::new();
        tools.insert("respond".to_string(), respond_tool());
        Workflow::new(
            "test",
            "test workflow",
            tools,
            vec![],
            TerminalToolInput::Single("respond".to_string()),
            "You are a helper.",
        )
        .expect("valid workflow")
    }
}
