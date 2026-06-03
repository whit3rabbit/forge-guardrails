//! Private dataset capture and review tool for Forge tool-call verifier rows.

mod agent_logs;
mod assemble;
mod capture;
mod cli;
mod prompts;
mod review;
mod schema;
mod stub_tools;
mod validate;

use std::env;

use cli::{parse_args, print_help, Command};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let command = match parse_args(env::args().skip(1)) {
        Ok(command) => command,
        Err(message) if message == "__help__" => {
            print_help();
            return;
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("run `forge-dataset --help` for usage");
            std::process::exit(2);
        }
    };

    let result = match command {
        Command::Prompts(cli) => prompts::run(cli),
        Command::Capture(cli) => capture::run(cli).await,
        Command::Review(cli) => review::run(*cli).await,
        Command::AgentLogs(cli) => agent_logs::run(*cli),
        Command::Assemble(cli) => assemble::run(cli),
        Command::Validate(cli) => validate::run(cli),
    };

    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
