from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import ValidationError

from .discover import claude_history_path, claude_log_paths, codex_log_paths
from .models import (
    FINAL_RESPONSE_LABELS,
    TOOL_LABELS,
    ExtractionResult,
    FinalResponseObservation,
    GenerateOptions,
    ReviewDecision,
    ReviewVerification,
    ToolObservation,
)
from .parsers import _load_claude_history, parse_claude_file, parse_codex_file
from .providers import ProviderConfig, ProviderError, build_review_client
from .sanitizer import (
    privacy_findings,
    sanitize_final_observation,
    sanitize_path,
    sanitize_text,
    sanitize_tool_observation,
)
from .serialization import (
    FINAL_RESPONSE_INPUT_SCHEMA_VERSION,
    TOOLCALL_INPUT_SCHEMA_VERSION_V1,
    TOOLCALL_INPUT_SCHEMA_VERSION_V2,
    build_tool_spec,
    scorer_input_hash,
    scoring_metadata,
    stable_id,
)
from .validator import validate_final_response_row, validate_tool_call_row

MIN_NON_VALID_TRAINING_CONFIDENCE = 0.85


def generate(options: GenerateOptions) -> dict[str, Any]:
    options.out.mkdir(parents=True, exist_ok=True)
    client = None
    verifier_client = None
    provider_used = "none"
    review_api = {
        "api": "none",
        "endpoint": None,
        "model": None,
        "max_attempts": options.api_max_attempts,
        "backoff_seconds": options.api_backoff_seconds,
    }
    verifier_api = {
        "api": "none",
        "endpoint": None,
        "model": None,
        "max_attempts": options.api_max_attempts,
        "backoff_seconds": options.api_backoff_seconds,
    }
    if options.llm_review and not options.no_api:
        progress(options, f"initializing review provider={options.provider}")
        client = build_review_client(
            ProviderConfig(
                provider=options.provider,
                minimax_model=options.minimax_model,
                openrouter_model=options.openrouter_model,
                max_attempts=options.api_max_attempts,
                backoff_seconds=options.api_backoff_seconds,
                on_retry=lambda message: progress(options, message),
            )
        )
        provider_used = type(client).__name__ if client is not None else "none"
        details = client.describe() if client is not None else "api=none"
        progress(options, f"using review client={provider_used} {details} max_attempts={options.api_max_attempts}")
        if client is not None:
            review_api.update({
                "api": client.provider_name,
                "endpoint": client.endpoint,
                "model": client.model,
            })
        if options.verify_review and client is not None:
            verifier_provider = options.provider if options.verifier_provider == "same" else options.verifier_provider
            if verifier_provider == options.provider or options.verifier_provider == "same":
                verifier_client = client
            elif verifier_provider != "none":
                progress(options, f"initializing verifier provider={verifier_provider}")
                verifier_client = build_review_client(
                    ProviderConfig(
                        provider=verifier_provider,
                        minimax_model=options.minimax_model,
                        openrouter_model=options.openrouter_model,
                        max_attempts=options.api_max_attempts,
                        backoff_seconds=options.api_backoff_seconds,
                        on_retry=lambda message: progress(options, message),
                    )
                )
            verifier_details = verifier_client.describe() if verifier_client is not None else "api=none"
            progress(options, f"using verifier client={type(verifier_client).__name__ if verifier_client else 'none'} {verifier_details}")
            if verifier_client is not None:
                verifier_api.update({
                    "api": verifier_client.provider_name,
                    "endpoint": verifier_client.endpoint,
                    "model": verifier_client.model,
                })
    elif options.no_api:
        progress(options, "API review disabled by --no-api")
    else:
        progress(options, "LLM review disabled; failed/ambiguous calls will be quarantined")

    progress(options, "extracting tool calls from local logs")
    extraction = extract(options)
    progress(
        options,
        (
            "extracted "
            f"tools={len(extraction.tool_observations)} "
            f"finals={len(extraction.final_observations)} "
            f"files_seen={extraction.files_seen} records_seen={extraction.records_seen}"
        ),
    )

    tool_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    tool_candidates = _limited(extraction.tool_observations, options.limit)
    for idx, obs in enumerate(tool_candidates, 1):
        progress(options, f"reviewing tool {idx}/{len(tool_candidates)} source={obs.source} tool={obs.tool_name}")
        try:
            row = build_tool_row(obs, options, client, verifier_client)
        except Quarantine as exc:
            quarantine.append(exc.record)
            progress(
                options,
                (
                    f"quarantined tool {idx}/{len(tool_candidates)} "
                    f"reason={exc.record['reason']}{quarantine_detail_suffix(exc.record)}"
                ),
            )
            continue
        tool_rows.append(row)
        progress(options, f"accepted tool {idx}/{len(tool_candidates)} label={row['label']}")

    if options.tool_calls_only:
        progress(options, "skipping final-response rows by --tool-calls-only")
    else:
        remaining = None if options.limit is None else max(0, options.limit - len(tool_rows) - len(quarantine))
        final_candidates = _limited(extraction.final_observations, remaining)
        for idx, obs in enumerate(final_candidates, 1):
            progress(options, f"reviewing final response {idx}/{len(final_candidates)} source={obs.source}")
            try:
                row = build_final_response_row(obs, options, client, verifier_client)
            except Quarantine as exc:
                quarantine.append(exc.record)
                progress(
                    options,
                    (
                        f"quarantined final response {idx}/{len(final_candidates)} "
                        f"reason={exc.record['reason']}{quarantine_detail_suffix(exc.record)}"
                    ),
                )
                continue
            final_rows.append(row)
            progress(options, f"accepted final response {idx}/{len(final_candidates)} label={row['label']}")

    synthetic_rows = generate_synthetic_tool_rows(tool_rows, options)
    if synthetic_rows:
        progress(options, f"generated synthetic tool rows={len(synthetic_rows)}")
        tool_rows.extend(synthetic_rows)

    progress(options, "deduping rows")
    tool_rows, tool_conflicts = dedupe_rows(tool_rows, options.serializer, "tool_call")
    final_rows, final_conflicts = dedupe_rows(final_rows, options.serializer, "final_response")
    conflicts = tool_conflicts + final_conflicts

    if options.fail_on_private_public_export:
        public_rows = [
            row for row in tool_rows + final_rows
            if row.get("review", {}).get("public_export_allowed") is True
        ]
        if public_rows:
            raise RuntimeError("log-derived rows must not be marked public_export_allowed=true")

    progress(options, f"writing outputs to {options.out}")
    write_jsonl(options.out / "tool_call_training.jsonl", tool_rows)
    write_jsonl(options.out / "final_response_training.jsonl", final_rows)
    write_jsonl(options.out / "quarantine.jsonl", quarantine)
    write_jsonl(options.out / "conflicts.jsonl", conflicts)
    if options.emit_notebook_adapter:
        write_jsonl(options.out / "agent_training.notebook.jsonl", notebook_adapter_rows(tool_rows, final_rows))
    split_counts = write_splits(options.out / "splits", tool_rows, final_rows)

    manifest = {
        "schema_version": "generatetd-manifest/v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "provider": provider_used,
        "review_api": review_api,
        "verifier_api": verifier_api,
        "serializer": options.serializer,
        "sources": {
            "codex": options.include_codex,
            "claude": options.include_claude,
            "files_seen": extraction.files_seen,
            "records_seen": extraction.records_seen,
        },
        "counts": {
            "tool_rows": len(tool_rows),
            "final_response_rows": len(final_rows),
            "quarantine": len(quarantine),
            "conflicts": len(conflicts),
        },
        "labels": dict(Counter(row["label"] for row in tool_rows + final_rows)),
        "synthetic": synthetic_counts(tool_rows + final_rows),
        "splits": split_counts,
    }
    (options.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    progress(
        options,
        (
            "done "
            f"tool_rows={len(tool_rows)} final_rows={len(final_rows)} "
            f"quarantine={len(quarantine)} conflicts={len(conflicts)}"
        ),
    )
    return manifest


def extract(options: GenerateOptions) -> ExtractionResult:
    merged = ExtractionResult()
    target = options.limit
    if options.include_codex:
        codex_paths = codex_log_paths(options.codex_root)
        progress(options, f"found codex log files={len(codex_paths)}")
        for idx, path in enumerate(codex_paths, 1):
            if not _path_allowed(path, options):
                continue
            add_filtered_result(merged, parse_codex_file(path), options)
            if idx % 100 == 0:
                progress(options, f"parsed codex files={idx} candidates={candidate_count(merged)}")
            if target is not None and candidate_count(merged) >= target:
                progress(options, f"stopping codex extraction at requested limit={target}")
                return merged
    if options.include_claude:
        history = _load_claude_history(claude_history_path(options.claude_root))
        claude_paths = claude_log_paths(options.claude_root)
        progress(options, f"found claude log files={len(claude_paths)}")
        for idx, path in enumerate(claude_paths, 1):
            if not _path_allowed(path, options):
                continue
            add_filtered_result(merged, parse_claude_file(path, history), options)
            if idx % 100 == 0:
                progress(options, f"parsed claude files={idx} candidates={candidate_count(merged)}")
            if target is not None and candidate_count(merged) >= target:
                progress(options, f"stopping claude extraction at requested limit={target}")
                return merged
    return merged


def add_filtered_result(merged: ExtractionResult, result: ExtractionResult, options: GenerateOptions) -> None:
    merged.files_seen += result.files_seen
    merged.records_seen += result.records_seen
    merged.tool_observations.extend(obs for obs in result.tool_observations if observation_allowed(obs, options))
    merged.final_observations.extend(obs for obs in result.final_observations if observation_allowed(obs, options))


def observation_allowed(obs: ToolObservation | FinalResponseObservation, options: GenerateOptions) -> bool:
    if options.project:
        needle = options.project.lower()
        if needle not in obs.cwd.lower() and needle not in str(obs.source_path).lower():
            return False
    if options.since and not _since_allowed(obs.timestamp, options.since):
        return False
    return True


def candidate_count(result: ExtractionResult) -> int:
    return len(result.tool_observations) + len(result.final_observations)


def build_tool_row(
    obs: ToolObservation,
    options: GenerateOptions,
    client: Any,
    verifier_client: Any = None,
) -> dict[str, Any]:
    safe = sanitize_tool_observation(obs)
    payload = tool_review_payload(safe, options.serializer)
    findings = privacy_findings(payload)
    if findings:
        raise Quarantine("privacy_findings_after_sanitize", payload, findings)

    if safe.is_error and client is None:
        raise Quarantine("needs_llm_review_for_failed_tool", payload, [])

    decision = default_tool_decision(safe)
    if client is not None:
        try:
            decision = client.review_tool_call(payload)
        except (ProviderError, ValidationError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise Quarantine("llm_review_failed", payload, [str(exc)]) from exc
    if decision.disposition != "training_row" or decision.label not in TOOL_LABELS:
        raise Quarantine("review_not_training_tool_row", payload, asdict(decision))
    if decision.label != "valid" and decision.confidence < MIN_NON_VALID_TRAINING_CONFIDENCE:
        raise Quarantine("low_confidence_non_valid_review", payload, asdict(decision))
    verification = verify_decision(payload, decision, verifier_client)
    if verification is not None and not verification.approve_training_row:
        raise Quarantine(
            "review_verifier_rejected",
            payload,
            {"decision": asdict(decision), "verification": asdict(verification)},
        )

    input_obj = payload["input"]
    review = base_review(safe)
    review.update({
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "private_agent_log": True,
        "public_export_allowed": False,
    })
    if verification is not None:
        review.update(verification_review_metadata(verification, verifier_client))
    row: dict[str, Any] = {
        "schema_version": "toolcall-verifier-training/v1",
        "input": input_obj,
        "label": decision.label,
        "review": review,
    }
    if decision.corrected_candidate_call:
        row["corrected_positive"] = {"candidate_call": decision.corrected_candidate_call}
    validate_tool_call_row(row)
    row["id"] = stable_id("tool", scorer_input_hash(input_obj, options.serializer, "tool_call"), decision.label)
    return row


def build_final_response_row(
    obs: FinalResponseObservation,
    options: GenerateOptions,
    client: Any,
    verifier_client: Any = None,
) -> dict[str, Any]:
    safe = sanitize_final_observation(obs)
    payload = final_review_payload(safe)
    findings = privacy_findings(payload)
    if findings:
        raise Quarantine("privacy_findings_after_sanitize", payload, findings)
    if client is None:
        raise Quarantine("needs_llm_review_for_final_response", payload, [])
    try:
        decision = client.review_final_response(payload)
    except (ProviderError, ValidationError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise Quarantine("llm_review_failed", payload, [str(exc)]) from exc
    if decision.disposition != "training_row" or decision.label not in FINAL_RESPONSE_LABELS:
        raise Quarantine("review_not_training_final_row", payload, asdict(decision))
    if decision.label != "valid_final_response" and decision.confidence < MIN_NON_VALID_TRAINING_CONFIDENCE:
        raise Quarantine("low_confidence_non_valid_review", payload, asdict(decision))
    verification = verify_decision(payload, decision, verifier_client)
    if verification is not None and not verification.approve_training_row:
        raise Quarantine(
            "review_verifier_rejected",
            payload,
            {"decision": asdict(decision), "verification": asdict(verification)},
        )

    input_obj = payload["input"]
    input_obj["required_facts"] = decision.required_facts
    review = base_review(safe)
    review.update({
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "private_agent_log": True,
        "public_export_allowed": False,
    })
    if verification is not None:
        review.update(verification_review_metadata(verification, verifier_client))
    row: dict[str, Any] = {
        "schema_version": "final-response-verifier-training/v1",
        "input": input_obj,
        "label": decision.label,
        "review": review,
    }
    if decision.corrected_final_response:
        row["corrected_positive"] = {"candidate_final_response": decision.corrected_final_response}
    validate_final_response_row(row)
    row["id"] = stable_id("final", scorer_input_hash(input_obj, options.serializer, "final_response"), decision.label)
    return row


def verify_decision(payload: dict[str, Any], decision: ReviewDecision, verifier_client: Any) -> ReviewVerification | None:
    if verifier_client is None:
        return None
    try:
        return verifier_client.verify_review(payload, asdict(decision))
    except (ProviderError, ValidationError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise Quarantine("review_verifier_failed", payload, {"decision": asdict(decision), "error": str(exc)}) from exc


def verification_review_metadata(verification: ReviewVerification, verifier_client: Any) -> dict[str, Any]:
    return {
        "verified_by": type(verifier_client).__name__,
        "verifier_api": getattr(verifier_client, "provider_name", "unknown"),
        "verifier_model": getattr(verifier_client, "model", "unknown"),
        "verifier_confidence": verification.confidence,
        "verifier_rationale": verification.rationale,
        "verifier_privacy_warnings": verification.privacy_warnings,
    }


def generate_synthetic_tool_rows(tool_rows: list[dict[str, Any]], options: GenerateOptions) -> list[dict[str, Any]]:
    valid_rows = [row for row in tool_rows if row.get("label") == "valid"]
    synthetic: list[dict[str, Any]] = []
    synthetic.extend(
        _synthetic_rows_for_type(
            valid_rows,
            options.synthetic_missing_argument,
            "missing_argument",
            options,
            mutate_missing_argument,
        )
    )
    synthetic.extend(
        _synthetic_rows_for_type(
            valid_rows,
            options.synthetic_wrong_tool,
            "wrong_tool",
            options,
            mutate_wrong_tool,
        )
    )
    synthetic.extend(
        _synthetic_rows_for_type(
            valid_rows,
            options.synthetic_tool_not_needed,
            "tool_not_needed",
            options,
            mutate_tool_not_needed,
        )
    )
    return synthetic


def _synthetic_rows_for_type(
    valid_rows: list[dict[str, Any]],
    requested: int,
    mutation_type: str,
    options: GenerateOptions,
    mutator: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if requested <= 0:
        return rows
    for row in valid_rows:
        if len(rows) >= requested:
            break
        mutated = mutator(row, mutation_type, options)
        if mutated is not None:
            rows.append(mutated)
    return rows


def mutate_missing_argument(row: dict[str, Any], mutation_type: str, options: GenerateOptions) -> dict[str, Any] | None:
    input_obj = deepcopy(row["input"])
    args = input_obj["candidate_call"].get("arguments")
    if not isinstance(args, dict) or not args:
        return None
    key = sorted(args, key=str)[0]
    removed = args.pop(key)
    rationale = f"Synthetic hard negative: removed semantically necessary argument {key!r} from a valid call."
    return synthetic_tool_row(
        row,
        input_obj,
        "wrong_arguments_semantic",
        mutation_type,
        rationale,
        options,
        {"removed_argument": key, "removed_value_type": type(removed).__name__},
    )


def mutate_wrong_tool(row: dict[str, Any], mutation_type: str, options: GenerateOptions) -> dict[str, Any] | None:
    # TODO: implement wrong-tool mutation using real competing tools from multi-tool contexts.
    # Returning None skips this mutation type entirely (no synthetic wrong_tool rows generated).
    return None


# A pool of varied no-tool user requests used to create diverse tool_not_needed hard negatives.
# Using a single hardcoded string produces near-duplicate rows with identical scorer_input_hashes
# across all source rows (only the candidate_call differs), which dilutes training signal.
_TOOL_NOT_NEEDED_REQUESTS: list[str] = [
    "Reply briefly to a simple greeting. No tool call is needed.",
    "What does the acronym API stand for?",
    "Say hello back to the user.",
    "Tell me a short fun fact about penguins.",
    "What is 17 multiplied by 6?",
    "Summarize what a REST API is in one sentence.",
    "What is the capital of France?",
    "Explain the difference between a list and a tuple in Python.",
    "How do I say 'thank you' in Japanese?",
    "What color is the sky on a clear day?",
    "Give me a one-line definition of machine learning.",
    "Respond to the user's farewell message.",
]


def mutate_tool_not_needed(row: dict[str, Any], mutation_type: str, options: GenerateOptions) -> dict[str, Any] | None:
    input_obj = deepcopy(row["input"])
    # Pick a request from the pool deterministically based on the source row id so
    # reruns are stable, but different source rows get different requests.
    source_id = row.get("id") or ""
    idx = int(stable_id(source_id, "tool_not_needed_pool", length=4), 16) % len(_TOOL_NOT_NEEDED_REQUESTS)
    input_obj["user_request"] = _TOOL_NOT_NEEDED_REQUESTS[idx]
    rationale = "Synthetic hard negative: preserved an unnecessary tool call for a direct no-tool user request."
    return synthetic_tool_row(row, input_obj, "tool_not_needed", mutation_type, rationale, options, {"pool_index": idx})


def synthetic_tool_row(
    source_row: dict[str, Any],
    input_obj: dict[str, Any],
    label: str,
    mutation_type: str,
    rationale: str,
    options: GenerateOptions,
    mutation_metadata: dict[str, Any],
) -> dict[str, Any]:
    review = deepcopy(source_row["review"])
    review.update({
        "confidence": 0.99,
        "rationale": rationale,
        "synthetic": True,
        "synthetic_type": mutation_type,
        "synthetic_from_id": source_row.get("id"),
        "synthetic_mutation": mutation_metadata,
        "private_agent_log": True,
        "public_export_allowed": False,
    })
    row = {
        "schema_version": "toolcall-verifier-training/v1",
        "input": input_obj,
        "label": label,
        "review": review,
    }
    validate_tool_call_row(row)
    row["id"] = stable_id(
        "synthetic-tool",
        mutation_type,
        source_row.get("id"),
        scorer_input_hash(input_obj, options.serializer, "tool_call"),
        label,
    )
    return row


def synthetic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("review", {}).get("synthetic_type"))
        for row in rows
        if row.get("review", {}).get("synthetic") is True
    )
    return dict(counts)


def tool_review_payload(obs: ToolObservation, serializer: str) -> dict[str, Any]:
    args = obs.candidate_arguments()
    tool = build_tool_spec(obs.tool_name, args)
    schema_version = TOOLCALL_INPUT_SCHEMA_VERSION_V2 if serializer == "v2" else TOOLCALL_INPUT_SCHEMA_VERSION_V1
    input_obj: dict[str, Any] = {
        "schema_version": schema_version,
        "user_request": obs.user_request or "Tool call in agent session.",
        "workflow_state": {
            "required_steps": [],
            "completed_steps": [],
            "pending_steps": [],
            "terminal_tools": [],
            "recent_errors": [obs.output] if obs.is_error and obs.output else [],
        },
        "available_tools": [tool],
        "candidate_call": {"name": obs.tool_name, "arguments": args},
    }
    if serializer == "v2":
        input_obj["metadata"] = scoring_metadata()
    return {
        "kind": "tool_call",
        "input": input_obj,
        "tool_output": obs.output,
        "exit_code": obs.exit_code,
        "is_error": obs.is_error,
        "hook_rewrite": obs.hook_rewrite,
        "source": base_review(obs),
    }


def final_review_payload(obs: FinalResponseObservation) -> dict[str, Any]:
    input_obj = {
        "schema_version": FINAL_RESPONSE_INPUT_SCHEMA_VERSION,
        "user_request": obs.user_request or "Terminal response in agent session.",
        "workflow_state": {
            "required_steps": [],
            "completed_steps": obs.tool_trace,
            "pending_steps": [],
            "terminal_tools": [],
            "recent_errors": [],
        },
        "required_facts": [],
        "tool_trace": obs.tool_trace,
        "tool_results": obs.tool_results,
        "candidate_final_response": obs.final_text,
        "metadata": scoring_metadata(),
    }
    return {"kind": "final_response", "input": input_obj, "source": base_review(obs)}


def default_tool_decision(obs: ToolObservation) -> ReviewDecision:
    return ReviewDecision(
        disposition="training_row",
        label="valid",
        confidence=0.65,
        rationale="Successful sanitized agent tool call; no semantic objection inferred without LLM review.",
    )


def base_review(obs: ToolObservation | FinalResponseObservation) -> dict[str, Any]:
    group = stable_id(obs.source, obs.session_id, obs.cwd, obs.user_request)
    return {
        "source": f"{obs.source}-log",
        "session_id": obs.session_id,
        "task_group_id": group,
        "cwd": obs.cwd,
        "source_path": sanitize_path(obs.source_path),
        "line_no": obs.line_no,
        "timestamp": obs.timestamp,
    }


def dedupe_rows(rows: list[dict[str, Any]], serializer: str, kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for row in rows:
        key = scorer_input_hash(row["input"], serializer, kind)
        row["review"]["scorer_input_hash"] = key
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            continue
        if existing["label"] != row["label"]:
            conflicts.append({"scorer_input_hash": key, "rows": [existing, row]})
            continue
        provenance = existing["review"].setdefault("provenance", [])
        provenance.append(row["review"])
    return list(by_key.values()), conflicts


def notebook_adapter_rows(tool_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tool_rows:
        input_obj = row["input"]
        rows.append({
            "kind": "tool_call",
            "label": row["label"],
            "user_request": input_obj["user_request"],
            "workflow_state": input_obj["workflow_state"],
            "available_tools": input_obj["available_tools"],
            "candidate_call": input_obj["candidate_call"],
            "metadata": notebook_metadata(row),
            "rank_score": row["review"].get("confidence", 1.0),
        })
    for row in final_rows:
        input_obj = row["input"]
        rows.append({
            "kind": "final_response",
            "label": row["label"],
            "user_request": input_obj["user_request"],
            "workflow_state": input_obj["workflow_state"],
            "required_facts": input_obj["required_facts"],
            "tool_trace": input_obj["tool_trace"],
            "tool_results": input_obj["tool_results"],
            "candidate_final_response": input_obj["candidate_final_response"],
            "metadata": notebook_metadata(row),
            "rank_score": row["review"].get("confidence", 1.0),
        })
    return rows


def notebook_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row["input"].get("metadata") or scoring_metadata())
    metadata.update({
        "generator": "agent_training",
        "private_agent_log": True,
        "public_export_allowed": False,
        "example_group_id": row["review"].get("task_group_id"),
        "source": row["review"].get("source"),
    })
    if row["review"].get("synthetic") is True:
        metadata.update({
            "synthetic": True,
            "synthetic_type": row["review"].get("synthetic_type"),
            "synthetic_from_id": row["review"].get("synthetic_from_id"),
        })
    return metadata


def write_splits(out_dir: Path, tool_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in tool_rows + final_rows:
        group = str(row.get("review", {}).get("task_group_id") or row.get("id"))
        bucket = int(stable_id(group, length=8), 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        splits[split].append(row)
    for split, rows in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    return {split: len(rows) for split, rows in splits.items()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _limited(items: list[Any], limit: int | None) -> list[Any]:
    return items if limit is None else items[:limit]


def _path_allowed(path: Path, options: GenerateOptions) -> bool:
    if options.since is None:
        return True
    return _since_allowed(path.name, options.since)


def _since_allowed(value: str, since: str) -> bool:
    try:
        return _parse_dateish(value) >= _parse_dateish(since)
    except ValueError:
        return True


def _parse_dateish(value: str) -> datetime:
    if len(value) >= 10:
        value = value[:10]
    return datetime.fromisoformat(value)


def progress(options: GenerateOptions, message: str) -> None:
    if options.progress:
        print(f"[generatetd] {message}", file=sys.stderr, flush=True)


def quarantine_detail_suffix(record: dict[str, Any]) -> str:
    details = record.get("details")
    if isinstance(details, list) and details:
        text = str(details[0])
    elif details:
        text = str(details)
    else:
        return ""
    text = " ".join(text.split())
    if len(text) > 220:
        text = text[:220] + "...<truncated>"
    return f" detail={text}"


class Quarantine(Exception):
    def __init__(self, reason: str, payload: dict[str, Any], details: Any) -> None:
        super().__init__(reason)
        self.record = {
            "reason": reason,
            "details": details,
            "payload": payload,
        }
