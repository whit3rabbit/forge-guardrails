use std::collections::HashMap;

use super::types::{ProviderConfig, ReviewRole};
use crate::cli::{default_minimax_model, default_openrouter_model, ReviewCli};

#[derive(Debug, Clone, Default)]
pub(crate) struct EnvFile {
    pub(crate) values: HashMap<String, String>,
}

impl EnvFile {
    pub(crate) fn load(path: &str) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        let mut values = HashMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((key, value)) = trimmed.split_once('=') else {
                continue;
            };
            values.insert(key.trim().to_string(), unquote_env_value(value.trim()));
        }
        Self { values }
    }
}

pub(crate) fn lookup_env(env_file: &EnvFile, names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        if let Some(value) = env_file.values.get(*name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

pub(crate) fn unquote_env_value(raw: &str) -> String {
    let value = raw.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

pub(crate) fn resolve_provider_config(
    cli: &ReviewCli,
    env_file: &EnvFile,
    role: ReviewRole,
) -> Result<ProviderConfig, String> {
    let provider = cli.provider.clone();
    resolve_provider_config_for(cli, env_file, role, &provider)
}

pub(crate) fn resolve_provider_config_for(
    cli: &ReviewCli,
    env_file: &EnvFile,
    role: ReviewRole,
    requested_provider: &str,
) -> Result<ProviderConfig, String> {
    let manual_base_url = match role {
        ReviewRole::Reviewer => cli.reviewer_base_url.as_str(),
        ReviewRole::Verifier => cli.verifier_base_url.as_str(),
    };
    let manual_model = match role {
        ReviewRole::Reviewer => cli.reviewer_model.as_str(),
        ReviewRole::Verifier => cli.verifier_model.as_str(),
    };
    let manual_key = match role {
        ReviewRole::Reviewer => cli.reviewer_api_key.clone(),
        ReviewRole::Verifier => cli.verifier_api_key.clone(),
    };

    if !manual_base_url.trim().is_empty() && !manual_model.trim().is_empty() {
        return Ok(ProviderConfig {
            provider: requested_provider.to_string(),
            chat_url: manual_base_url.to_string(),
            model: manual_model.to_string(),
            api_key: manual_key.or_else(|| lookup_env(env_file, &role_api_key_names(&role))),
        });
    }

    let provider = if requested_provider == "auto" {
        if lookup_env(env_file, &["MINIMAX_API_KEY"]).is_some() {
            "minimax"
        } else if lookup_env(env_file, &["OPENROUTER_API_KEY"]).is_some() {
            "openrouter"
        } else {
            return Err(format!(
                "{} review requires MINIMAX_API_KEY, OPENROUTER_API_KEY, or manual base URL/model",
                role_name(&role)
            ));
        }
    } else {
        requested_provider
    };

    match provider {
        "minimax" => {
            let api_key = manual_key
                .or_else(|| lookup_env(env_file, &["MINIMAX_API_KEY"]))
                .ok_or_else(|| "MINIMAX_API_KEY is required for provider minimax".to_string())?;
            Ok(ProviderConfig {
                provider: "minimax".to_string(),
                chat_url: "https://api.minimax.io/v1/chat/completions".to_string(),
                model: if manual_model.trim().is_empty() {
                    if cli.minimax_model.trim().is_empty() {
                        lookup_env(
                            env_file,
                            &[
                                "FORGE_DATASET_MINIMAX_MODEL",
                                "GENERATETD_MINIMAX_MODEL",
                                "MINIMAX_MODEL",
                            ],
                        )
                        .unwrap_or_else(|| default_minimax_model().to_string())
                    } else {
                        cli.minimax_model.clone()
                    }
                } else {
                    manual_model.to_string()
                },
                api_key: Some(api_key),
            })
        }
        "openrouter" => {
            let api_key = manual_key
                .or_else(|| lookup_env(env_file, &["OPENROUTER_API_KEY"]))
                .ok_or_else(|| {
                    "OPENROUTER_API_KEY is required for provider openrouter".to_string()
                })?;
            Ok(ProviderConfig {
                provider: "openrouter".to_string(),
                chat_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
                model: if manual_model.trim().is_empty() {
                    if cli.openrouter_model.trim().is_empty() {
                        lookup_env(
                            env_file,
                            &[
                                "FORGE_DATASET_OPENROUTER_MODEL",
                                "GENERATETD_OPENROUTER_MODEL",
                                "OPENROUTER_MODEL",
                            ],
                        )
                        .unwrap_or_else(|| default_openrouter_model().to_string())
                    } else {
                        cli.openrouter_model.clone()
                    }
                } else {
                    manual_model.to_string()
                },
                api_key: Some(api_key),
            })
        }
        other => Err(format!("unknown provider: {other}")),
    }
}

pub(crate) fn role_api_key_names(role: &ReviewRole) -> [&'static str; 3] {
    match role {
        ReviewRole::Reviewer => [
            "FORGE_DATASET_REVIEWER_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ],
        ReviewRole::Verifier => [
            "FORGE_DATASET_VERIFIER_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ],
    }
}

pub(crate) fn role_name(role: &ReviewRole) -> &'static str {
    match role {
        ReviewRole::Reviewer => "reviewer",
        ReviewRole::Verifier => "verifier",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_model_flag_overrides_env_file_model() {
        let cli = ReviewCli {
            input: "capture.jsonl".to_string(),
            output: "training.jsonl".to_string(),
            env_file: ".env".to_string(),
            provider: "openrouter".to_string(),
            verifier_provider: "same".to_string(),
            minimax_model: String::new(),
            openrouter_model: "openrouter/free".to_string(),
            reviewer_base_url: String::new(),
            reviewer_model: String::new(),
            reviewer_api_key: None,
            verifier_base_url: String::new(),
            verifier_model: String::new(),
            verifier_api_key: None,
            max_alternatives_per_group: 2,
            max_alternative_ratio: 1.0 / 3.0,
            concurrency: 1,
            chunk_size: None,
            resume: false,
        };
        let env_file = EnvFile {
            values: HashMap::from([
                (
                    "OPENROUTER_API_KEY".to_string(),
                    "test-openrouter-key".to_string(),
                ),
                (
                    "GENERATETD_OPENROUTER_MODEL".to_string(),
                    "openrouter/owl-alpha".to_string(),
                ),
            ]),
        };

        let config =
            resolve_provider_config_for(&cli, &env_file, ReviewRole::Reviewer, "openrouter")
                .expect("config");

        assert_eq!(config.model, "openrouter/free");
    }

    #[test]
    fn openrouter_env_file_model_is_used_when_flag_is_absent() {
        let cli = ReviewCli {
            input: "capture.jsonl".to_string(),
            output: "training.jsonl".to_string(),
            env_file: ".env".to_string(),
            provider: "openrouter".to_string(),
            verifier_provider: "same".to_string(),
            minimax_model: String::new(),
            openrouter_model: String::new(),
            reviewer_base_url: String::new(),
            reviewer_model: String::new(),
            reviewer_api_key: None,
            verifier_base_url: String::new(),
            verifier_model: String::new(),
            verifier_api_key: None,
            max_alternatives_per_group: 2,
            max_alternative_ratio: 1.0 / 3.0,
            concurrency: 1,
            chunk_size: None,
            resume: false,
        };
        let env_file = EnvFile {
            values: HashMap::from([
                (
                    "OPENROUTER_API_KEY".to_string(),
                    "test-openrouter-key".to_string(),
                ),
                (
                    "GENERATETD_OPENROUTER_MODEL".to_string(),
                    "openrouter/owl-alpha".to_string(),
                ),
            ]),
        };

        let config =
            resolve_provider_config_for(&cli, &env_file, ReviewRole::Reviewer, "openrouter")
                .expect("config");

        assert_eq!(config.model, "openrouter/owl-alpha");
    }
}
