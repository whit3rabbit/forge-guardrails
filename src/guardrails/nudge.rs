/// Frozen correction message carrying role, content, kind, and escalation tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Nudge {
    pub role: String,
    pub content: String,
    pub kind: String,
    pub tier: i32,
}

impl Nudge {
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

    pub fn with_tier(mut self, tier: i32) -> Self {
        self.tier = tier;
        self
    }
}
