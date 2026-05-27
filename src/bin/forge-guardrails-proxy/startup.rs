use std::sync::Arc;

use forge_guardrails::{FinalResponseScorer, ServerManager, ToolCallScorer};

use crate::cli::Cli;
use crate::client::ClientFactory;
use crate::config::ProxyConfig;

mod classifier;
mod managed;
mod modes;

#[cfg(test)]
mod tests;

pub(crate) struct Startup {
    pub(crate) config: ProxyConfig,
    pub(crate) client_factory: ClientFactory,
    pub(crate) managed_server: Option<ServerManager>,
    pub(crate) scorer: Option<Arc<dyn ToolCallScorer>>,
    pub(crate) final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
}

pub(crate) fn build_startup(cli: Cli) -> Result<Startup, String> {
    let mut startup = if cli.backend_url.is_none() && cli.backend.is_none() {
        modes::build_env_startup(&cli)?
    } else if let Some(backend_url) = cli.backend_url.as_deref() {
        modes::build_external_startup(&cli, backend_url)?
    } else {
        let backend = cli.backend.expect("backend checked above");
        managed::build_managed_startup(&cli, backend)?
    };
    classifier::prepare_classifier_artifact(&startup.config)?;
    startup.scorer = classifier::build_classifier_scorer(&startup.config)?;
    startup.final_response_scorer =
        classifier::build_final_response_classifier_scorer(&startup.config)?;
    Ok(startup)
}

pub(crate) fn download_classifier_shortcut(cli: &Cli) -> Result<(), String> {
    classifier::download_classifier_shortcut(cli)
}
