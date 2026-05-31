#!/usr/bin/env python3
"""Summarize proxy eval JSONL without relying on upstream report wording."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REDACTED_TERMINAL_TEXT = "[REDACTED]"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def _load_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.extend(_load_rows(path))
    return rows


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{(count / total) * 100:.1f}%"


def _classification(row: dict[str, Any]) -> str | None:
    existing = row.get("proxy_failure_classification")
    if existing and str(existing) != "accuracy_false":
        return str(existing)
    if not row.get("completeness"):
        return str(row.get("error_type") or "incomplete")
    if _required_step_mismatch(row):
        return "proxy_contract_mismatch"
    if row.get("accuracy") is not False:
        return None
    if _has_redacted_terminal_content(row):
        return "terminal_redacted"
    return "accuracy_false"


def _has_redacted_terminal_content(row: dict[str, Any]) -> bool:
    if _is_redacted_terminal_text(row.get("final_text")):
        return True

    if row.get("proxy_terminal_source") == "tool_call":
        tool_args = row.get("tool_args")
        if isinstance(tool_args, list) and tool_args:
            last_args = tool_args[-1]
            if isinstance(last_args, dict):
                return any(_is_redacted_terminal_text(value) for value in last_args.values())

    return False


def _is_redacted_terminal_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == REDACTED_TERMINAL_TEXT


def _missing_required_steps(row: dict[str, Any]) -> list[Any]:
    missing = row.get("missing_required_steps")
    if missing is None:
        missing = row.get("proxy_missing_required_steps")
    return missing if isinstance(missing, list) else []


def _required_step_mismatch(row: dict[str, Any]) -> bool:
    if "required_step_mismatch" in row:
        return bool(row["required_step_mismatch"])
    if row.get("proxy_required_steps_satisfied") is False:
        return True
    return bool(_missing_required_steps(row))


def _scenario_stats(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        scenario = str(row.get("scenario", "<unknown>"))
        classification = _classification(row)
        stats[scenario]["total"] += 1
        if row.get("completeness"):
            stats[scenario]["complete"] += 1
        if bool(row.get("success")):
            stats[scenario]["success"] += 1
        if row.get("completeness") and row.get("accuracy") is False:
            stats[scenario]["completed_inaccurate"] += 1
            if classification:
                stats[scenario][f"classification:{classification}"] += 1
        if _missing_required_steps(row):
            stats[scenario]["missing_required_steps"] += 1
        if classification == "proxy_contract_mismatch":
            stats[scenario]["failed_contract_mismatch"] += 1
    return stats


def _oracle_by_user_message(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        request = row.get("user_message")
        if not isinstance(request, str) or not request:
            continue
        stats[request]["rows"] += 1
        if bool(row.get("success")):
            stats[request]["success"] += 1
        classification = _classification(row)
        if classification:
            stats[request][classification] += 1
    return stats


def _classifier_request(row: dict[str, Any]) -> str | None:
    for key in ("initial_user_request", "user_request"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def print_classifier_summary(
    rows: list[dict[str, Any]],
    classifier_rows: list[dict[str, Any]],
) -> None:
    """Summarize and print classifier telemetry match results."""
    if not classifier_rows:
        return

    oracle_by_request = _oracle_by_user_message(rows)
    labels = Counter(str(row.get("label", "<missing>")) for row in classifier_rows)
    actions = Counter(str(row.get("action", "<missing>")) for row in classifier_rows)
    by_oracle: dict[str, Counter[str]] = defaultdict(Counter)
    unmatched = 0

    for row in classifier_rows:
        request = _classifier_request(row)
        oracle = oracle_by_request.get(request or "")
        if not oracle:
            unmatched += 1
            bucket = "unmatched"
        elif oracle["accuracy_false"]:
            bucket = "accuracy_false_request"
        else:
            bucket = "passing_request"
        by_oracle[bucket][str(row.get("label", "<missing>"))] += 1

    print("  Classifier telemetry:")
    print(f"    Events: {len(classifier_rows)}")
    print(
        "    Labels: "
        + ", ".join(f"{label}={count}" for label, count in sorted(labels.items()))
    )
    print(
        "    Actions: "
        + ", ".join(f"{action}={count}" for action, count in sorted(actions.items()))
    )
    if unmatched:
        print(f"    Unmatched to oracle user_message: {unmatched}")
    for bucket in ("passing_request", "accuracy_false_request", "unmatched"):
        if by_oracle[bucket]:
            parts = [
                f"{label}={count}"
                for label, count in sorted(by_oracle[bucket].items())
            ]
            print(f"    {bucket}: {', '.join(parts)}")


def print_summary(
    rows: list[dict[str, Any]],
    classifier_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Calculate and print aggregate completeness and accuracy results."""
    total = len(rows)
    complete = sum(1 for row in rows if row.get("completeness"))
    success = sum(1 for row in rows if bool(row.get("success")))
    completed_inaccurate = sum(
        1 for row in rows if row.get("completeness") and row.get("accuracy") is False
    )
    incomplete = total - complete
    classifications = Counter(
        c for row in rows if (c := _classification(row)) is not None
    )

    print("Proxy Eval Summary")
    print(f"  Rows: {total}")
    print(f"  Completeness: {complete}/{total} ({_rate(complete, total)})")
    print(f"  Success: {success}/{total} ({_rate(success, total)})")
    print(f"  Completed but inaccurate: {completed_inaccurate}/{total}")
    print(f"  Incomplete/protocol/tool-loop failures: {incomplete}/{total}")

    if classifications:
        parts = [
            f"{name}={count}"
            for name, count in sorted(classifications.items())
        ]
        print(f"  Classifications: {', '.join(parts)}")

    stats = _scenario_stats(rows)
    completeness_weak = []
    accuracy_weak = []
    missing_required_steps = []
    failed_contract_mismatch = []
    for scenario, counter in sorted(stats.items()):
        scenario_total = counter["total"]
        if counter["complete"] < scenario_total:
            completeness_weak.append(
                f"{scenario}={_rate(counter['complete'], scenario_total)}"
            )
        if counter["completed_inaccurate"]:
            classification_parts = [
                f"{key.removeprefix('classification:')}={count}"
                for key, count in sorted(counter.items())
                if key.startswith("classification:")
            ]
            suffix = (
                f" ({', '.join(classification_parts)})"
                if classification_parts
                else ""
            )
            accuracy_weak.append(
                f"{scenario}={counter['completed_inaccurate']}/{scenario_total}{suffix}"
            )
        if counter["missing_required_steps"]:
            missing_required_steps.append(
                f"{scenario}={counter['missing_required_steps']}/{scenario_total}"
            )
        if counter["failed_contract_mismatch"]:
            failed_contract_mismatch.append(
                f"{scenario}={counter['failed_contract_mismatch']}/{scenario_total}"
            )

    print(
        "  Completeness weak: "
        + (", ".join(completeness_weak) if completeness_weak else "none")
    )
    print(
        "  Accuracy weak: "
        + (", ".join(accuracy_weak) if accuracy_weak else "none")
    )
    print(
        "  Missing required steps: "
        + (", ".join(missing_required_steps) if missing_required_steps else "none")
    )
    print(
        "  Failed proxy contract mismatches: "
        + (
            ", ".join(failed_contract_mismatch)
            if failed_contract_mismatch
            else "none"
        )
    )
    print_classifier_summary(rows, classifier_rows or [])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the summary script."""
    parser = argparse.ArgumentParser(
        description="Summarize proxy eval JSONL by completeness and accuracy"
    )
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--classifier-jsonl",
        nargs="*",
        type=Path,
        default=[],
        help="Optional proxy classifier telemetry JSONL files to summarize",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Main entrypoint for parsing options and printing the summary."""
    args = parse_args()
    print_summary(_load_rows(args.jsonl), _load_many(args.classifier_jsonl))


if __name__ == "__main__":
    main()
