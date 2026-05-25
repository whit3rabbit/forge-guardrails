//! Nudge message generators for guardrail enforcement.
//!
//! Each function produces a corrective prompt string for a specific
//! enforcement scenario. All functions are pure and side-effect-free.

/// Generate a retry nudge when the model responds with free text instead
/// of a tool call. The raw response argument is accepted for signature
/// compatibility but is not incorporated into the output.
pub fn retry_nudge(_raw_response: &str) -> String {
    "Your previous response was not a valid tool call. \
     You must respond with a tool call, not free text. \
     Please try again with a valid tool call."
        .to_string()
}

/// Generate a nudge when the model calls a tool that does not exist.
/// Lists all available tools as a comma-separated list.
pub fn unknown_tool_nudge(called_tool: &str, available_tools: &[&str]) -> String {
    let tools_list = available_tools.join(", ");
    format!(
        "Tool '{}' does not exist. Available tools: {}. Call one of them.",
        called_tool, tools_list
    )
}

/// Generate a step-enforcement nudge with tiered escalation.
///
/// Tier is clamped to [1, 3]:
/// - Tier 1: polite, mentions terminal tool and required steps
/// - Tier 2: direct, lists only required steps
/// - Tier 3: aggressive imperative demanding a specific tool call
pub fn step_nudge(terminal_tool: &str, pending_steps: &[&str], tier: i32) -> String {
    let clamped = tier.clamp(1, 3);
    let steps_str = pending_steps.join(", ");

    match clamped {
        1 => format!(
            "You cannot call {} yet. \
             You must first complete these required steps: {}. \
             Call one of them now.",
            terminal_tool, steps_str
        ),
        2 => format!(
            "You must call one of these tools now: {}. Pick one.",
            steps_str
        ),
        3 => format!(
            "STOP. You MUST call one of: {}. Do NOT call {}. \
             Your next response MUST be a tool call to one of: {}.",
            steps_str, terminal_tool, steps_str
        ),
        _ => unreachable!("tier clamped to [1,3]"),
    }
}

/// Generate a nudge when a tool is called without its prerequisites.
/// Lists the missing prerequisite tool names as comma-separated.
pub fn prerequisite_nudge(tool_name: &str, missing_prereqs: &[&str]) -> String {
    let prereqs_str = missing_prereqs.join(", ");
    format!(
        "You cannot call {} yet. \
         You must first call: {}. \
         Call the prerequisite tool now.",
        tool_name, prereqs_str
    )
}

/// Generate a nudge when a terminal call is mixed with non-terminal work.
pub fn unsafe_batch_nudge(allowed_next_tools: &[&str], blocked_tools: &[&str]) -> String {
    let allowed = allowed_next_tools.join(", ");
    let blocked = blocked_tools.join(", ");
    format!(
        "Do not combine terminal and non-terminal tools in the same response. \
         Allowed next tool calls: {}. \
         Blocked until later: {}. \
         Retry with only an allowed tool call.",
        allowed, blocked
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_nudge_returns_nonempty() {
        let result = retry_nudge("some text");
        assert!(!result.is_empty());
    }

    #[test]
    fn retry_nudge_does_not_echo_input() {
        let input = "This is some raw model output";
        let result = retry_nudge(input);
        assert!(!result.contains(input));
    }

    #[test]
    fn unknown_tool_nudge_contains_tool_names() {
        let result = unknown_tool_nudge("bad_tool", &["tool_a", "tool_b"]);
        assert!(result.contains("bad_tool"));
        assert!(result.contains("tool_a, tool_b"));
    }

    #[test]
    fn step_nudge_tier1_polite() {
        let result = step_nudge("respond", &["search"], 1);
        assert!(result.contains("respond"));
        assert!(result.contains("search"));
        assert!(!result.contains("STOP"));
    }

    #[test]
    fn step_nudge_tier2_direct() {
        let result = step_nudge("respond", &["search", "analyze"], 2);
        assert!(result.contains("search, analyze"));
        assert!(!result.contains("STOP"));
    }

    #[test]
    fn step_nudge_tier3_aggressive() {
        let result = step_nudge("respond", &["search"], 3);
        assert!(result.contains("STOP"));
        assert!(result.contains("respond"));
        assert!(result.contains("search"));
    }

    #[test]
    fn step_nudge_tier_clamped_low() {
        let a = step_nudge("respond", &["search"], 0);
        let b = step_nudge("respond", &["search"], 1);
        assert_eq!(a, b);
    }

    #[test]
    fn step_nudge_tier_clamped_high() {
        let a = step_nudge("respond", &["search"], 5);
        let b = step_nudge("respond", &["search"], 3);
        assert_eq!(a, b);
    }

    #[test]
    fn prerequisite_nudge_lists_prereqs() {
        let result = prerequisite_nudge("finalize", &["search", "analyze"]);
        assert!(result.contains("finalize"));
        assert!(result.contains("search, analyze"));
    }

    #[test]
    fn unsafe_batch_nudge_lists_allowed_and_blocked() {
        let result = unsafe_batch_nudge(&["search"], &["respond"]);
        assert!(result.contains("search"));
        assert!(result.contains("respond"));
        assert!(result.contains("Do not combine"));
    }
}
