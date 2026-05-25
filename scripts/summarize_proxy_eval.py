#!/usr/bin/env python3
"""Summarize proxy eval JSONL without relying on upstream report wording."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{(count / total) * 100:.1f}%"


def _classification(row: dict[str, Any]) -> str | None:
    existing = row.get("proxy_failure_classification")
    if existing:
        return str(existing)
    if not row.get("completeness"):
        return str(row.get("error_type") or "incomplete")
    if row.get("accuracy") is not False:
        return None
    if row.get("proxy_required_steps_satisfied") is False:
        return "proxy_contract_mismatch"
    return "accuracy_false"


def _scenario_stats(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        scenario = str(row.get("scenario", "<unknown>"))
        stats[scenario]["total"] += 1
        if row.get("completeness"):
            stats[scenario]["complete"] += 1
        if bool(row.get("success")):
            stats[scenario]["success"] += 1
        if row.get("completeness") and row.get("accuracy") is False:
            stats[scenario]["completed_inaccurate"] += 1
        if row.get("proxy_missing_required_steps"):
            stats[scenario]["missing_required_steps"] += 1
        if _classification(row) == "proxy_contract_mismatch":
            stats[scenario]["failed_contract_mismatch"] += 1
    return stats


def print_summary(rows: list[dict[str, Any]]) -> None:
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
            accuracy_weak.append(
                f"{scenario}={counter['completed_inaccurate']}/{scenario_total}"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize proxy eval JSONL by completeness and accuracy"
    )
    parser.add_argument("jsonl", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print_summary(_load_rows(args.jsonl))


if __name__ == "__main__":
    main()
