use std::path::Path;

use crate::context::detect_hardware;
use crate::error::BudgetResolutionError;

use super::lifecycle::LifecycleOptions;
use super::manager::ServerManager;

/// Budget resolution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetMode {
    /// Use backend-reported context size.
    Backend,
    /// User-specified token count.
    Manual,
    /// Maximum available context.
    ForgeFull,
    /// Half available context (two-phase start for llamaserver).
    ForgeFast,
}

impl BudgetMode {
    /// Returns the string representation of the budget mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Backend => "backend",
            Self::Manual => "manual",
            Self::ForgeFull => "forge-full",
            Self::ForgeFast => "forge-fast",
        }
    }

    /// Parses a string representation into a `BudgetMode` if valid.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "backend" => Some(Self::Backend),
            "manual" => Some(Self::Manual),
            "forge-full" => Some(Self::ForgeFull),
            "forge-fast" => Some(Self::ForgeFast),
            _ => None,
        }
    }
}

impl std::str::FromStr for BudgetMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "backend" => Ok(Self::Backend),
            "manual" => Ok(Self::Manual),
            "forge-full" => Ok(Self::ForgeFull),
            "forge-fast" => Ok(Self::ForgeFast),
            _ => Err(()),
        }
    }
}

impl ServerManager {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_with_budget_options(
        &self,
        model: &str,
        gguf_path: &Path,
        mode: &str,
        budget_mode: BudgetMode,
        manual_tokens: Option<i64>,
        extra_flags: &[String],
        cache_type_k: Option<&str>,
        cache_type_v: Option<&str>,
        n_slots: Option<i64>,
        kv_unified: bool,
        options: LifecycleOptions,
    ) -> Result<i64, String> {
        if budget_mode == BudgetMode::Manual && manual_tokens.is_none() {
            return Err("manual mode requires manual_tokens".to_string());
        }

        if self.backend == "ollama" {
            self.start_with_options(
                model,
                gguf_path,
                mode,
                extra_flags,
                None,
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
                options,
            )
            .map_err(|e| e.to_string())?;
            return self
                .resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
                .map_err(|e| e.to_string());
        }

        if budget_mode == BudgetMode::ForgeFast {
            self.start_with_options(
                model,
                gguf_path,
                mode,
                extra_flags,
                None,
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
                options,
            )
            .map_err(|e| e.to_string())?;
            let reported_ctx = self.query_props_context().map_err(|e| e.to_string())?;
            let total_ctx = if kv_unified || n_slots.is_none_or(|slots| slots <= 1) {
                reported_ctx
            } else {
                reported_ctx * n_slots.unwrap_or(1)
            };
            let half_total = total_ctx / 2;
            if let Ok(mut g) = self.last_context.lock() {
                *g = Some(half_total);
            }
            self.start_with_options(
                model,
                gguf_path,
                mode,
                extra_flags,
                Some(half_total),
                cache_type_k,
                cache_type_v,
                n_slots,
                kv_unified,
                options,
            )
            .map_err(|e| e.to_string())?;
            return self
                .resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
                .map_err(|e| e.to_string());
        }

        let ctx_override = if budget_mode == BudgetMode::Manual {
            manual_tokens
        } else {
            None
        };
        self.start_with_options(
            model,
            gguf_path,
            mode,
            extra_flags,
            ctx_override,
            cache_type_k,
            cache_type_v,
            n_slots,
            kv_unified,
            options,
        )
        .map_err(|e| e.to_string())?;
        self.resolve_budget(budget_mode, manual_tokens, n_slots, kv_unified)
            .map_err(|e| e.to_string())
    }

    /// Resolve the context budget based on the given mode.
    pub fn resolve_budget(
        &self,
        mode: BudgetMode,
        manual_tokens: Option<i64>,
        n_slots: Option<i64>,
        kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        match mode {
            BudgetMode::Manual => {
                if self.backend == "ollama" {
                    manual_tokens.ok_or_else(|| {
                        BudgetResolutionError::new()
                            .with_cause("manual_tokens required for MANUAL budget mode")
                    })
                } else {
                    self.resolve_backend_budget()
                }
            }
            BudgetMode::Backend => self.resolve_backend_budget(),
            BudgetMode::ForgeFull => self.resolve_forge_full(n_slots, kv_unified),
            BudgetMode::ForgeFast => self.resolve_forge_fast(n_slots, kv_unified),
        }
    }

    fn resolve_backend_budget(&self) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget());
        }
        let ctx = self.query_props_context()?;
        Ok(ctx)
    }

    fn resolve_forge_full(
        &self,
        _n_slots: Option<i64>,
        _kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget());
        }
        self.query_props_context()
    }

    fn resolve_forge_fast(
        &self,
        _n_slots: Option<i64>,
        _kv_unified: bool,
    ) -> Result<i64, BudgetResolutionError> {
        if self.backend == "ollama" {
            return Ok(Self::ollama_vram_budget() / 2);
        }
        self.query_props_context()
    }

    /// VRAM tier budget for ollama.
    pub fn ollama_vram_budget() -> i64 {
        match detect_hardware() {
            Ok(Some(hw)) => {
                let gb = hw.vram_total_gb();
                if gb < 24.0 {
                    4096
                } else if gb < 48.0 {
                    32768
                } else {
                    262144
                }
            }
            _ => 4096,
        }
    }

    /// Query the backend /props endpoint for the actual context length.
    pub fn query_props_context(&self) -> Result<i64, BudgetResolutionError> {
        let url = format!("http://127.0.0.1:{}/props", self.port);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|e| BudgetResolutionError::new().with_cause(e.to_string()))?;

        rt.block_on(async {
            let resp = reqwest::Client::new()
                .get(&url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
                .map_err(|e| BudgetResolutionError::new().with_cause(e.to_string()))?;

            if !resp.status().is_success() {
                return Err(
                    BudgetResolutionError::new().with_cause(format!("Status {}", resp.status()))
                );
            }

            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| BudgetResolutionError::new().with_cause(e.to_string()))?;

            let ctx = json
                .get("default_generation_settings")
                .and_then(|s| s.get("n_ctx"))
                .and_then(|n| n.as_i64())
                .ok_or_else(|| {
                    BudgetResolutionError::new()
                        .with_cause("missing context field in /props response")
                })?;

            Ok(ctx)
        })
    }
}
