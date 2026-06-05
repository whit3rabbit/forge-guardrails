#!/usr/bin/env python3
"""Compare disabled and compressed proxy eval JSONL outputs."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenTotals:
    baseline: int
    compressed: int
    rows: int

    @property
    def saved(self) -> int:
        return self.baseline - self.compressed


@dataclass
class TelemetryCoverage:
    row_keys: set[tuple[str, str, str, str]]
    scenarios: set[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("scenario", "")),
        str(row.get("run", "")),
        str(row.get("stream", "")),
        str(row.get("budget_tokens", "")),
    )


def request_key(request: dict[str, Any]) -> tuple[str, str, str, str] | None:
    required = ("scenario", "run", "stream", "budget_tokens")
    if any(field not in request for field in required):
        return None
    return (
        str(request.get("scenario", "")),
        str(request.get("run", "")),
        str(request.get("stream", "")),
        str(request.get("budget_tokens", "")),
    )


def pair_rows(
    baseline: list[dict[str, Any]],
    compressed: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int, int]:
    baseline_by_key: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    compressed_by_key: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in baseline:
        baseline_by_key[row_key(row)].append(row)
    for row in compressed:
        compressed_by_key[row_key(row)].append(row)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    baseline_unpaired = 0
    compressed_unpaired = 0
    for key in sorted(set(baseline_by_key) | set(compressed_by_key)):
        baseline_rows = baseline_by_key.get(key, [])
        compressed_rows = compressed_by_key.get(key, [])
        paired = min(len(baseline_rows), len(compressed_rows))
        pairs.extend(zip(baseline_rows[:paired], compressed_rows[:paired]))
        baseline_unpaired += len(baseline_rows) - paired
        compressed_unpaired += len(compressed_rows) - paired
    return pairs, baseline_unpaired, compressed_unpaired


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def token_totals(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *fields: str,
) -> TokenTotals:
    baseline_total = 0
    compressed_total = 0
    rows = 0
    for baseline, compressed in pairs:
        baseline_values = [as_int(baseline.get(field)) for field in fields]
        compressed_values = [as_int(compressed.get(field)) for field in fields]
        if any(value is None for value in baseline_values + compressed_values):
            continue
        baseline_total += sum(value for value in baseline_values if value is not None)
        compressed_total += sum(value for value in compressed_values if value is not None)
        rows += 1
    return TokenTotals(baseline_total, compressed_total, rows)


def expand_jsonl_paths(values: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for value in values or []:
        matches = [Path(match) for match in glob.glob(value)]
        paths.extend(matches or [Path(value)])
    return paths


def default_compression_jsonl_paths(compressed_jsonl: Path) -> list[Path]:
    return sorted(compressed_jsonl.parent.glob("proxy_tool_output_compression_*.jsonl"))


def telemetry_token_totals(paths: list[Path]) -> TokenTotals:
    before_total = 0
    after_total = 0
    events = 0
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row.get("kind") != "tool_output_compression":
                continue
            before = as_int(row.get("before_tokens"))
            after = as_int(row.get("after_tokens"))
            if before is None or after is None:
                continue
            before_total += before
            after_total += after
            events += 1
    return TokenTotals(before_total, after_total, events)


def telemetry_savings_by_scenario(paths: list[Path]) -> dict[str, TokenTotals]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            request = row.get("request")
            if not isinstance(request, dict):
                continue
            scenario = request.get("scenario")
            if not isinstance(scenario, str) or not scenario:
                continue
            before = as_int(row.get("before_tokens"))
            after = as_int(row.get("after_tokens"))
            if before is None or after is None:
                continue
            totals[scenario][0] += before
            totals[scenario][1] += after
            totals[scenario][2] += 1
    return {
        scenario: TokenTotals(values[0], values[1], values[2])
        for scenario, values in totals.items()
    }


def telemetry_coverage(paths: list[Path]) -> TelemetryCoverage:
    row_keys: set[tuple[str, str, str, str]] = set()
    scenarios: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row.get("kind") != "tool_output_compression":
                continue
            request = row.get("request")
            if not isinstance(request, dict):
                continue
            scenario = request.get("scenario")
            if isinstance(scenario, str) and scenario:
                scenarios.add(scenario)
            key = request_key(request)
            if key is not None:
                row_keys.add(key)
    return TelemetryCoverage(row_keys=row_keys, scenarios=scenarios)


def compression_touched_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    coverage: TelemetryCoverage,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    if coverage.row_keys:
        touched = [
            (baseline, compressed)
            for baseline, compressed in pairs
            if row_key(baseline) in coverage.row_keys
        ]
        if touched:
            return touched, "rows with compression telemetry"
    if coverage.scenarios:
        touched = [
            (baseline, compressed)
            for baseline, compressed in pairs
            if str(baseline.get("scenario", "")) in coverage.scenarios
        ]
        if touched:
            return touched, "scenarios with compression telemetry"
    return pairs, "all paired rows"


def has_behavior_change(
    baseline: dict[str, Any],
    compressed: dict[str, Any],
) -> bool:
    return (
        bool(baseline.get("success")) != bool(compressed.get("success"))
        or bool(baseline.get("completeness")) != bool(compressed.get("completeness"))
        or baseline.get("accuracy") != compressed.get("accuracy")
    )


def behavior_changes(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[str]:
    changes: list[str] = []
    for baseline, compressed in pairs:
        if not has_behavior_change(baseline, compressed):
            continue
        key = row_key(baseline)
        compressed_class = compressed.get("proxy_failure_classification")
        compressed_error = compressed.get("error_type")
        detail = (
            f"{key[0]} run {key[1]} stream={key[2]} budget={key[3]}: "
            f"success {bool(baseline.get('success'))}->{bool(compressed.get('success'))}, "
            f"complete {bool(baseline.get('completeness'))}->{bool(compressed.get('completeness'))}, "
            f"accuracy {baseline.get('accuracy')}->{compressed.get('accuracy')}"
        )
        if compressed_class:
            detail += f", compressed_class={compressed_class}"
        if compressed_error:
            detail += f", compressed_error={compressed_error}"
        changes.append(detail)
    return changes


def count_true(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if bool(row.get(field)))


def count_accuracy_false(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("accuracy") is False)


def pct(saved: int, baseline: int) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{saved / baseline * 100:.1f}%"


def print_token_line(label: str, totals: TokenTotals, row_label: str = "rows") -> None:
    if totals.rows == 0:
        print(f"  {label}: unavailable; no paired rows reported usage")
        return
    print(
        f"  {label}: baseline {totals.baseline}, compressed {totals.compressed}, "
        f"saved {totals.saved} ({pct(totals.saved, totals.baseline)}), "
        f"{row_label} {totals.rows}"
    )


def scenario_savings(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[str, TokenTotals, int, int, int]]:
    by_scenario: dict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for baseline, compressed in pairs:
        scenario = str(baseline.get("scenario", "<unknown>"))
        by_scenario[scenario].append((baseline, compressed))

    results = []
    for scenario, scenario_pairs in sorted(by_scenario.items()):
        totals = token_totals(scenario_pairs, "input_tokens")
        baseline_success = count_true(
            [baseline for baseline, _ in scenario_pairs],
            "success",
        )
        compressed_success = count_true(
            [compressed for _, compressed in scenario_pairs],
            "success",
        )
        results.append((
            scenario,
            totals,
            baseline_success,
            compressed_success,
            len(scenario_pairs),
        ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare disabled and compressed local proxy eval JSONL outputs"
    )
    parser.add_argument("baseline_jsonl", type=Path)
    parser.add_argument("compressed_jsonl", type=Path)
    parser.add_argument(
        "--min-input-token-savings",
        type=int,
        default=0,
        help="Minimum aggregate prompt-token savings required for success",
    )
    parser.add_argument(
        "--allow-behavior-regression",
        action="store_true",
        help="Report behavior regressions without failing",
    )
    parser.add_argument(
        "--compression-jsonl",
        action="append",
        help=(
            "Compression telemetry JSONL path or glob. Defaults to "
            "proxy_tool_output_compression_*.jsonl next to the compressed oracle."
        ),
    )
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline_jsonl)
    compressed_rows = load_jsonl(args.compressed_jsonl)
    pairs, baseline_unpaired, compressed_unpaired = pair_rows(
        baseline_rows,
        compressed_rows,
    )
    baseline_paired = [baseline for baseline, _ in pairs]
    compressed_paired = [compressed for _, compressed in pairs]

    input_totals = token_totals(pairs, "input_tokens")
    output_totals = token_totals(pairs, "output_tokens")
    total_totals = token_totals(pairs, "input_tokens", "output_tokens")
    telemetry_paths = expand_jsonl_paths(args.compression_jsonl)
    if not telemetry_paths:
        telemetry_paths = default_compression_jsonl_paths(args.compressed_jsonl)
    telemetry_totals = telemetry_token_totals(telemetry_paths)
    telemetry_by_scenario = telemetry_savings_by_scenario(telemetry_paths)
    coverage = telemetry_coverage(telemetry_paths)
    behavior_pairs, behavior_scope = compression_touched_pairs(pairs, coverage)

    baseline_success = count_true(baseline_paired, "success")
    compressed_success = count_true(compressed_paired, "success")
    baseline_complete = count_true(baseline_paired, "completeness")
    compressed_complete = count_true(compressed_paired, "completeness")
    baseline_accuracy_false = count_accuracy_false(baseline_paired)
    compressed_accuracy_false = count_accuracy_false(compressed_paired)
    behavior_baseline = [baseline for baseline, _ in behavior_pairs]
    behavior_compressed = [compressed for _, compressed in behavior_pairs]
    behavior_baseline_success = count_true(behavior_baseline, "success")
    behavior_compressed_success = count_true(behavior_compressed, "success")
    behavior_baseline_complete = count_true(behavior_baseline, "completeness")
    behavior_compressed_complete = count_true(behavior_compressed, "completeness")
    behavior_baseline_accuracy_false = count_accuracy_false(behavior_baseline)
    behavior_compressed_accuracy_false = count_accuracy_false(behavior_compressed)

    failures: list[str] = []
    warnings: list[str] = []
    if not pairs:
        failures.append("no comparable rows found")
    if baseline_unpaired:
        warnings.append(f"{baseline_unpaired} baseline rows were not paired")
    if compressed_unpaired:
        warnings.append(f"{compressed_unpaired} compressed rows were not paired")
    behavior_keys = {row_key(baseline) for baseline, _ in behavior_pairs}
    untouched_changes = [
        (baseline, compressed)
        for baseline, compressed in pairs
        if row_key(baseline) not in behavior_keys and has_behavior_change(baseline, compressed)
    ]
    if untouched_changes:
        warnings.append(
            f"{len(untouched_changes)} behavior changes occurred outside "
            f"{behavior_scope}; reported but not used as compression failures"
        )

    if not args.allow_behavior_regression:
        if behavior_compressed_success < behavior_baseline_success:
            failures.append(
                f"success regressed on {behavior_scope} from "
                f"{behavior_baseline_success}/{len(behavior_pairs)} to "
                f"{behavior_compressed_success}/{len(behavior_pairs)}"
            )
        if behavior_compressed_complete < behavior_baseline_complete:
            failures.append(
                f"completeness regressed on {behavior_scope} from "
                f"{behavior_baseline_complete}/{len(behavior_pairs)} to "
                f"{behavior_compressed_complete}/{len(behavior_pairs)}"
            )
        if behavior_compressed_accuracy_false > behavior_baseline_accuracy_false:
            failures.append(
                f"accuracy_false increased on {behavior_scope} from "
                f"{behavior_baseline_accuracy_false} to "
                f"{behavior_compressed_accuracy_false}"
            )

    savings_totals = input_totals if input_totals.rows else telemetry_totals
    savings_source = "input token" if input_totals.rows else "compression telemetry input estimate"
    if savings_totals.rows == 0 and args.min_input_token_savings > 0:
        failures.append(
            "input token savings unavailable because no paired rows reported usage "
            "and no compression telemetry was found"
        )
    elif savings_totals.saved < args.min_input_token_savings:
        failures.append(
            f"{savings_source} savings {savings_totals.saved} below required "
            f"{args.min_input_token_savings}"
        )

    print("Compression Eval Summary")
    print(f"  Baseline:   {args.baseline_jsonl}")
    print(f"  Compressed: {args.compressed_jsonl}")
    print(
        f"  Rows: baseline {len(baseline_rows)}, compressed {len(compressed_rows)}, "
        f"paired {len(pairs)}"
    )
    print(
        f"  Success: baseline {baseline_success}/{len(pairs)}, "
        f"compressed {compressed_success}/{len(pairs)}"
    )
    print(
        f"  Completeness: baseline {baseline_complete}/{len(pairs)}, "
        f"compressed {compressed_complete}/{len(pairs)}"
    )
    print(
        f"  Accuracy false: baseline {baseline_accuracy_false}, "
        f"compressed {compressed_accuracy_false}"
    )
    print(
        f"  Behavior gate scope: {behavior_scope}, "
        f"paired {len(behavior_pairs)}/{len(pairs)}"
    )
    print_token_line("Input tokens", input_totals)
    print_token_line("Output tokens", output_totals)
    print_token_line("Total tokens", total_totals)
    if input_totals.rows == 0:
        print_token_line(
            "Compression telemetry input estimate",
            telemetry_totals,
            row_label="events",
        )

    scenario_rows = scenario_savings(pairs)
    if scenario_rows:
        print("  Per-scenario input savings:")
        for scenario, totals, baseline_s, compressed_s, pair_count in scenario_rows:
            if totals.rows == 0:
                telemetry = telemetry_by_scenario.get(scenario)
                if telemetry is not None:
                    savings = (
                        f"telemetry saved {telemetry.saved} "
                        f"({pct(telemetry.saved, telemetry.baseline)}), "
                        f"events {telemetry.rows}"
                    )
                elif telemetry_by_scenario:
                    savings = "no compression telemetry events"
                else:
                    savings = "usage unavailable"
            else:
                savings = (
                    f"saved {totals.saved} ({pct(totals.saved, totals.baseline)})"
                )
            print(
                f"    {scenario}: {savings}, "
                f"success {baseline_s}/{pair_count} -> "
                f"{compressed_s}/{pair_count}"
            )

    changed_rows = behavior_changes(pairs)
    if changed_rows:
        print("  Behavior changes:")
        for detail in changed_rows[:20]:
            print(f"    {detail}")
        if len(changed_rows) > 20:
            print(f"    ... {len(changed_rows) - 20} more")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if failures:
        sys.stdout.flush()
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nCompression comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
