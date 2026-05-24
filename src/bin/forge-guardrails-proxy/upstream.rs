use std::env;

use anyllm_providers::ProviderProtocol;
use reqwest::Url;

pub(crate) struct DirectOpenAiUpstream {
    pub(crate) base_url: String,
    pub(crate) api_key: Option<String>,
}

pub(crate) fn apply_litellm_env_aliases() {
    for (key, value) in anyllm_proxy::config::env_aliases::compute_env_aliases() {
        // SAFETY: this runs before any tokio runtime or application thread is
        // started, matching anyllm_proxy's own startup rule.
        unsafe {
            env::set_var(key, value);
        }
    }
}

pub(crate) fn direct_local_openai_upstream_from_env() -> Option<DirectOpenAiUpstream> {
    direct_local_openai_upstream(
        env::var("OPENAI_BASE_URL").ok().as_deref(),
        env::var("BACKEND").ok().as_deref(),
        env::var("PROXY_CONFIG").is_ok(),
    )
}

fn direct_local_openai_upstream(
    openai_base_url: Option<&str>,
    backend: Option<&str>,
    proxy_config_set: bool,
) -> Option<DirectOpenAiUpstream> {
    if proxy_config_set {
        return None;
    }

    let backend = backend
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("openai");

    if backend.eq_ignore_ascii_case("openai") {
        let raw = openai_base_url?.trim();
        return local_openai_upstream(raw, None);
    }

    let provider = anyllm_providers::get_provider(&backend.to_ascii_lowercase())?;
    if provider.protocol != ProviderProtocol::OpenAICompat {
        return None;
    }

    let base_url = openai_base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(provider.default_base_url);
    local_openai_upstream(base_url, Some(provider.env_vars))
}

fn local_openai_upstream(
    base_url: &str,
    provider_env_vars: Option<&[&str]>,
) -> Option<DirectOpenAiUpstream> {
    if !is_exact_local_url(base_url) {
        return None;
    }
    Some(DirectOpenAiUpstream {
        base_url: base_url.to_string(),
        api_key: direct_openai_api_key(provider_env_vars.unwrap_or(&[])),
    })
}

pub(crate) fn direct_openai_api_key(provider_env_vars: &[&str]) -> Option<String> {
    std::iter::once("OPENAI_API_KEY")
        .chain(provider_env_vars.iter().copied())
        .find_map(|key| {
            env::var(key)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn is_exact_local_url(raw: &str) -> bool {
    let Ok(parsed) = Url::parse(raw) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    matches!(
        host.to_ascii_lowercase().as_str(),
        "host.docker.internal" | "localhost" | "127.0.0.1" | "::1" | "[::1]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_local_upstream_allows_host_docker_internal() {
        assert_eq!(
            direct_local_openai_upstream(Some("http://host.docker.internal:11434/v1"), None, false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://host.docker.internal:11434/v1")
        );
    }

    #[test]
    fn direct_local_upstream_does_not_override_proxy_config() {
        assert!(direct_local_openai_upstream(
            Some("http://host.docker.internal:11434/v1"),
            None,
            true
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_does_not_override_non_openai_backend() {
        assert!(direct_local_openai_upstream(
            Some("http://host.docker.internal:11434/v1"),
            Some("anthropic"),
            false
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_rejects_host_suffix_trick() {
        assert!(direct_local_openai_upstream(
            Some("http://host.docker.internal.example.com/v1"),
            None,
            false
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_uses_ollama_catalog_default() {
        assert_eq!(
            direct_local_openai_upstream(None, Some("ollama"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn direct_local_upstream_uses_ollama_local_override() {
        assert_eq!(
            direct_local_openai_upstream(
                Some("http://host.docker.internal:11434/v1"),
                Some("ollama"),
                false
            )
            .map(|upstream| upstream.base_url)
            .as_deref(),
            Some("http://host.docker.internal:11434/v1")
        );
    }

    #[test]
    fn direct_local_upstream_uses_lm_studio_catalog_default() {
        assert_eq!(
            direct_local_openai_upstream(None, Some("lm_studio"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:1234/v1")
        );
    }

    #[test]
    fn direct_local_upstream_uses_hosted_vllm_catalog_default() {
        assert_eq!(
            direct_local_openai_upstream(None, Some("hosted_vllm"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:8000/v1")
        );
    }

    #[test]
    fn direct_local_upstream_keeps_public_provider_default_on_runtime_path() {
        assert!(direct_local_openai_upstream(None, Some("groq"), false).is_none());
    }

    #[test]
    fn direct_local_upstream_allows_public_provider_when_overridden_to_local() {
        assert_eq!(
            direct_local_openai_upstream(Some("http://localhost:9999/v1"), Some("groq"), false)
                .map(|upstream| upstream.base_url)
                .as_deref(),
            Some("http://localhost:9999/v1")
        );
    }

    #[test]
    fn direct_local_upstream_rejects_non_openai_compat_provider() {
        assert!(direct_local_openai_upstream(
            Some("http://localhost:9999/v1"),
            Some("anthropic"),
            false
        )
        .is_none());
    }

    #[test]
    fn direct_local_upstream_rejects_malformed_url() {
        assert!(direct_local_openai_upstream(Some("not a url"), None, false).is_none());
    }

    #[test]
    fn exact_local_url_allows_ipv6_loopback() {
        assert!(is_exact_local_url("http://[::1]:11434/v1"));
    }
}
