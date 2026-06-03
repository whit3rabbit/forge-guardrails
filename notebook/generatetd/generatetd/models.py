from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TOOL_LABELS = {
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid",
}

FINAL_RESPONSE_LABELS = {
    "valid_final_response",
    "missing_tool_fact",
    "contradicts_tool_result",
    "unsupported_claim",
    "failed_to_acknowledge_data_gap",
}

DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash:free"


@dataclass
class ToolObservation:
    source: str
    session_id: str
    cwd: str
    timestamp: str
    user_request: str
    tool_name: str
    arguments: dict[str, Any]
    output: str
    call_id: str
    source_path: Path
    line_no: int
    exit_code: int | None = None
    is_error: bool = False
    effective_arguments: dict[str, Any] | None = None
    hook_rewrite: dict[str, Any] | None = None

    def candidate_arguments(self) -> dict[str, Any]:
        return self.effective_arguments if self.effective_arguments is not None else self.arguments


@dataclass
class FinalResponseObservation:
    source: str
    session_id: str
    cwd: str
    timestamp: str
    user_request: str
    final_text: str
    tool_trace: list[str]
    tool_results: list[dict[str, str]]
    source_path: Path
    line_no: int


@dataclass
class ExtractionResult:
    tool_observations: list[ToolObservation] = field(default_factory=list)
    final_observations: list[FinalResponseObservation] = field(default_factory=list)
    files_seen: int = 0
    records_seen: int = 0


@dataclass
class ReviewDecision:
    disposition: str
    label: str
    confidence: float
    rationale: str
    corrected_candidate_call: dict[str, Any] | None = None
    corrected_final_response: str | None = None
    required_facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    privacy_warnings: list[str] = field(default_factory=list)


@dataclass
class ReviewVerification:
    approve_training_row: bool
    confidence: float
    rationale: str
    corrected_disposition: str | None = None
    corrected_label: str | None = None
    privacy_warnings: list[str] = field(default_factory=list)


@dataclass
class GenerateOptions:
    out: Path
    include_codex: bool = True
    include_claude: bool = True
    provider: str = "auto"
    llm_review: bool = False
    verify_review: bool = False
    verifier_provider: str = "same"
    no_api: bool = False
    serializer: str = "v1"
    limit: int | None = None
    since: str | None = None
    project: str | None = None
    emit_notebook_adapter: bool = True
    fail_on_private_public_export: bool = True
    codex_root: Path = Path.home() / ".codex"
    claude_root: Path = Path.home() / ".claude"
    minimax_model: str = DEFAULT_MINIMAX_MODEL
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    api_max_attempts: int = 4
    api_backoff_seconds: float = 1.0
    synthetic_missing_argument: int = 0
    synthetic_wrong_tool: int = 0
    synthetic_tool_not_needed: int = 0
    tool_calls_only: bool = False
    progress: bool = True
