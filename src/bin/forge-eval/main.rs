//! Native Rust smoke runner for forge scenarios and evaluations.

mod ablation;
mod cli;
mod counting_client;
mod report;
mod runner;
mod scenarios;
mod startup;

use std::env;

use cli::{parse_args, print_help};
use startup::run_cli;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let cli = match parse_args(env::args().skip(1)) {
        Ok(cli) => cli,
        Err(message) if message == "__help__" => {
            print_help();
            return;
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("run `forge-eval --help` for usage");
            std::process::exit(2);
        }
    };

    if let Err(err) = run_cli(cli).await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
