#[derive(Debug, Clone, Copy)]
pub(crate) struct Ablation {
    pub(crate) rescue_enabled: bool,
    pub(crate) max_retries: i32,
    pub(crate) use_required_steps: bool,
}

pub(crate) fn parse_ablation(name: &str) -> Result<Ablation, String> {
    let ablation = match name {
        "reforged" => Ablation {
            rescue_enabled: true,
            max_retries: 5,
            use_required_steps: true,
        },
        "no_rescue" => Ablation {
            rescue_enabled: false,
            max_retries: 5,
            use_required_steps: true,
        },
        "no_steps" => Ablation {
            rescue_enabled: true,
            max_retries: 5,
            use_required_steps: false,
        },
        "no_recovery" | "no_nudge" => Ablation {
            rescue_enabled: false,
            max_retries: 0,
            use_required_steps: true,
        },
        "bare" => Ablation {
            rescue_enabled: false,
            max_retries: 0,
            use_required_steps: false,
        },
        "no_compact" => Ablation {
            rescue_enabled: true,
            max_retries: 5,
            use_required_steps: true,
        },
        other => return Err(format!("unsupported ablation: {other}")),
    };
    Ok(ablation)
}
