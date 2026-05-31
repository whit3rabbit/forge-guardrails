//! Main entry point for the forge-guardrails-proxy standalone server binary.

mod cli;
mod client;
mod config;
mod response;
pub mod routes;
mod startup;
mod upstream;

use clap::Parser;

use cli::Cli;

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run_main(cli) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run_main(cli: Cli) -> Result<(), String> {
    if cli.classify_download {
        return startup::download_classifier_shortcut(&cli);
    }

    upstream::apply_litellm_env_aliases();
    let startup = startup::build_startup(cli)?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("failed to build tokio runtime: {err}"))?;

    runtime.block_on(routes::serve(
        startup.config,
        startup.client_factory,
        startup.managed_server,
        startup.scorer,
        startup.final_response_scorer,
    ))
}
