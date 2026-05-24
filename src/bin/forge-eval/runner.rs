use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use forge_guardrails::{
    CompactEvent, ContextManager, LLMClient, Message, NoCompact, StreamChunk, WorkflowRunner,
};
use tokio::sync::Mutex;

use crate::ablation::parse_ablation;
use crate::cli::Cli;
use crate::counting_client::CountingClient;
use crate::report::{row_for_result, write_row};
use crate::scenarios::build_scenario;

pub(crate) async fn run_with_client<C: LLMClient + 'static>(
    client: C,
    cli: &Cli,
    model: &str,
) -> Result<(), String> {
    let client = Arc::new(CountingClient::new(client));
    let scenario_names = if cli.scenarios.is_empty() {
        vec![
            "basic_2step".to_string(),
            "sequential_3step".to_string(),
            "error_recovery".to_string(),
        ]
    } else {
        cli.scenarios.clone()
    };
    let ablation = parse_ablation(&cli.ablation)?;

    for scenario_name in &scenario_names {
        for run_idx in 1..=cli.runs {
            let scenario = build_scenario(scenario_name, ablation.use_required_steps)?;
            let emitted: Arc<StdMutex<Vec<Message>>> = Arc::new(StdMutex::new(Vec::new()));
            let emitted_cb = emitted.clone();
            let compactions: Arc<StdMutex<Vec<CompactEvent>>> = Arc::new(StdMutex::new(Vec::new()));
            let compactions_cb = compactions.clone();
            let context = Arc::new(Mutex::new(ContextManager::new(
                Box::new(NoCompact),
                cli.num_ctx,
                Some(Box::new(move |event: &CompactEvent| {
                    compactions_cb
                        .lock()
                        .expect("compaction capture lock")
                        .push(event.clone());
                })),
                None,
                None,
            )));
            let runner = WorkflowRunner::new(
                client.clone(),
                context,
                15,
                ablation.max_retries,
                2,
                cli.stream,
                Some(Box::new(|_chunk: &StreamChunk| {})),
                Some(Box::new(move |message: &Message| {
                    emitted_cb
                        .lock()
                        .expect("message capture lock")
                        .push(message.clone());
                })),
                ablation.rescue_enabled,
                None,
            );

            let before_calls = client.calls();
            let start = Instant::now();
            let result = runner
                .run(&scenario.workflow, &scenario.user_message, None, None, None)
                .await;
            let elapsed = start.elapsed().as_secs_f64();
            let iterations = client.calls() - before_calls;
            let messages = emitted.lock().expect("message capture lock").clone();
            let compaction_events = compactions.lock().expect("compaction capture lock").len();
            let row = row_for_result(
                &cli.backend,
                model,
                &cli.ablation,
                cli,
                &scenario,
                run_idx,
                iterations,
                elapsed,
                result,
                &messages,
                compaction_events,
            );
            write_row(cli.output.as_deref(), &row)?;
        }
    }

    Ok(())
}
