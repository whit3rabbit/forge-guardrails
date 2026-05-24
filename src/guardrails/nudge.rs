/// Frozen correction message carrying role, content, kind, and escalation tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Nudge {
    /// The target message role (typically user).
    pub role: String,
    /// The nudge prompt text content.
    pub content: String,
    /// The type of nudge (e.g. step, prerequisite, retry).
    pub kind: String,
    /// The escalation tier level (0-3).
    pub tier: i32,
}

impl Nudge {
    /// Creates a new `Nudge` with the given role, content, and kind.
    pub fn new(
        role: impl Into<String>,
        content: impl Into<String>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            kind: kind.into(),
            tier: 0,
        }
    }

    /// Sets the escalation tier for the nudge.
    pub fn with_tier(mut self, tier: i32) -> Self {
        self.tier = tier;
        self
    }
}
