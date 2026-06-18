use std::fmt;
use std::str::FromStr;

/// Default maximum tool-output bytes retained before safe capping.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
/// Default maximum dedup records kept per session.
pub const DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION: usize = 128;
/// Default maximum sessions kept in dedup memory.
pub const DEFAULT_MAX_DEDUP_SESSIONS: usize = 64;

/// Header prefixed to LZW dictionary-compressed tool outputs.
pub const LZW_DICTIONARY_HEADER: &str = "[Forge LZW Dictionary]";
/// Header prefixed to RePair dictionary-compressed tool outputs.
pub const REPAIR_DICTIONARY_HEADER: &str = "[Forge RePair Dictionary]";
/// Maximum size of a compression dictionary.
pub const DICTIONARY_MAX_DICT_SIZE: usize = 20;
/// Maximum input bytes allowed for dictionary compression.
pub const DICTIONARY_MAX_INPUT_BYTES: usize = 50_000;
/// Minimum occurrences of a pattern required to be added to the dictionary.
pub const DICTIONARY_MIN_OCCURRENCES: usize = 3;
/// Minimum net savings in bytes required to accept dictionary compression.
pub const DICTIONARY_MIN_NET_SAVINGS_BYTES: usize = 32;
/// Minimum per-entry savings in bytes for a dictionary entry to be committed.
pub const DICTIONARY_MIN_ENTRY_SAVINGS_BYTES: usize = 16;
/// Minimum net savings in percentage required to accept dictionary compression.
pub const DICTIONARY_MIN_NET_SAVINGS_PERCENT: usize = 3;

/// Returns true if the output has been compressed with an LZW or RePair dictionary.
pub fn is_dictionary_compressed_output(output: &str) -> bool {
    output.starts_with(LZW_DICTIONARY_HEADER) || output.starts_with(REPAIR_DICTIONARY_HEADER)
}

/// Returns true if the dictionary compression yields meaningful size savings.
pub fn dictionary_has_meaningful_savings(original_len: usize, savings: usize) -> bool {
    savings >= DICTIONARY_MIN_NET_SAVINGS_BYTES
        && savings.saturating_mul(100) / original_len.max(1) >= DICTIONARY_MIN_NET_SAVINGS_PERCENT
}

/// Default compression level for tool outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ToolOutputCompressionMode {
    /// No compression or mutation.
    Disabled,
    /// Safety-only transforms: redaction, ANSI stripping, binary suppression, capping.
    Safe,
    /// Safe transforms plus conservative structural and tool-family compaction.
    #[default]
    Standard,
    /// Standard transforms plus explicitly lossy/high-aggression transforms.
    Aggressive,
}

impl ToolOutputCompressionMode {
    /// Return the stable lowercase mode name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Safe => "safe",
            Self::Standard => "standard",
            Self::Aggressive => "aggressive",
        }
    }

    /// Return true if this mode can change tool output.
    pub fn enabled(self) -> bool {
        self != Self::Disabled
    }
}

impl fmt::Display for ToolOutputCompressionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ToolOutputCompressionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "disabled" | "off" | "false" | "none" => Ok(Self::Disabled),
            "safe" => Ok(Self::Safe),
            "standard" | "on" | "true" => Ok(Self::Standard),
            "aggressive" => Ok(Self::Aggressive),
            other => Err(format!(
                "tool output compression must be disabled, safe, standard, or aggressive, got '{other}'"
            )),
        }
    }
}

/// Dictionary algorithm used by aggressive tool-output compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolOutputCompressionMethod {
    /// LZW-style repeated substring dictionary compression.
    #[default]
    Lzw,
    /// RePair-style repeated adjacent token-pair grammar compression.
    Repair,
    /// Run bounded dictionary methods and keep the smallest valid result.
    Auto,
}

impl ToolOutputCompressionMethod {
    /// Return the stable lowercase method name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lzw => "lzw",
            Self::Repair => "repair",
            Self::Auto => "auto",
        }
    }
}

impl fmt::Display for ToolOutputCompressionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ToolOutputCompressionMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "lzw" => Ok(Self::Lzw),
            "repair" | "re-pair" => Ok(Self::Repair),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "tool output compression method must be lzw, repair, or auto, got '{other}'"
            )),
        }
    }
}

/// Configuration for one tool-output compression pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutputCompressionConfig {
    /// Compression mode.
    pub mode: ToolOutputCompressionMode,
    /// Dictionary method used by aggressive compression.
    pub method: ToolOutputCompressionMethod,
    /// Whether secret-looking values are redacted before other transforms.
    pub redact_secrets: bool,
    /// Whether repeated compressed outputs are replaced with bounded references.
    pub enable_dedup: bool,
    /// Whether per-call compressed output is memoized for byte-stable resends.
    pub enable_memo: bool,
    /// Optional client/session key used for dedup.
    pub session_id: Option<String>,
    /// Maximum bytes retained before safe capping.
    pub max_output_bytes: usize,
    /// Maximum dedup records per session.
    pub max_dedup_entries_per_session: usize,
    /// Maximum dedup sessions retained.
    pub max_dedup_sessions: usize,
}

impl Default for ToolOutputCompressionConfig {
    fn default() -> Self {
        Self {
            mode: ToolOutputCompressionMode::Standard,
            method: ToolOutputCompressionMethod::Lzw,
            redact_secrets: true,
            enable_dedup: true,
            enable_memo: true,
            session_id: None,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_dedup_entries_per_session: DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION,
            max_dedup_sessions: DEFAULT_MAX_DEDUP_SESSIONS,
        }
    }
}

impl ToolOutputCompressionConfig {
    /// Return a disabled compression configuration.
    pub fn disabled() -> Self {
        Self {
            mode: ToolOutputCompressionMode::Disabled,
            ..Self::default()
        }
    }

    /// Build a configuration from a mode with safe defaults.
    pub fn from_mode(mode: ToolOutputCompressionMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// Return true if this configuration can change tool output.
    pub fn enabled(&self) -> bool {
        self.mode.enabled()
    }
}

/// Internal metrics and output for a compression pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutputCompressionResult {
    /// Final output to send to the upstream model.
    pub output: String,
    /// Heuristic token estimate before compression.
    pub before_tokens: i64,
    /// Heuristic token estimate after compression.
    pub after_tokens: i64,
    /// Estimated tokens saved.
    pub saved_tokens: i64,
    /// Estimated percentage saved.
    pub saved_pct: i64,
    /// Canonical tool family: bash, read, grep, glob, or generic.
    pub canonical_tool: String,
    /// Content or command family used by filters.
    pub family: String,
    /// Compression mode used.
    pub mode: ToolOutputCompressionMode,
    /// Whether a secret redaction changed output.
    pub redacted: bool,
    /// Whether binary or oversized output was suppressed/capped.
    pub capped: bool,
    /// Whether output was replaced by a dedup reference.
    pub deduped: bool,
    /// Whether memoized compressed bytes were reused unchanged.
    pub memo_reused: bool,
    /// Whether an existing memo entry was invalidated by changed input or config.
    pub memo_changed: bool,
    /// Names of transforms that changed output.
    pub strategies: Vec<String>,
}
