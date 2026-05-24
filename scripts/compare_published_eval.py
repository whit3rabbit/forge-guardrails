#!/usr/bin/env python3
"""Compare local eval JSONL against published Forge leaderboard rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLISHED = ROOT / "forge" / "docs" / "results" / "raw" / "native-vs-prompt.md"


@dataclass
class PublishedRow:
    model: str
    backend_mode: str
    score: float
    accuracy: float
    completeness: float
    n: int
    scenarios: dict[str, float]


def parse_percent(raw: str) -> float:
    return float(raw.rstrip("%")) / 100.0


def parse_legend(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if "rel=relevance_detection" not in line:
            continue
        return dict(re.findall(r"([A-Za-z0-9_]+)=([A-Za-z0-9_]+)", line))
    raise SystemExit("published results legend not found")


def parse_published_row(path: Path, model: str, backend_mode: str) -> PublishedRow:
    text = path.read_text()
    legend = parse_legend(text)
    header_abbrevs: list[str] | None = None
    row_re = re.compile(r"^(.+?)\s+(LS/[NP])\s+\[reforged\]\s+(.+)$")

    for line in text.splitlines():
        if line.startswith("Model/Backend"):
            parts = line.split()
            try:
                header_abbrevs = parts[parts.index("N") + 1 :]
            except ValueError:
                header_abbrevs = None
            continue

        match = row_re.match(line)
        if not match:
            continue
        row_model, row_backend_mode, metrics = match.groups()
        if row_model != model or row_backend_mode != backend_mode:
            continue
        if header_abbrevs is None:
            raise SystemExit("published row found before table header")

        parts = metrics.split()
        if len(parts) < 7 + len(header_abbrevs):
            raise SystemExit(f"published row has unexpected shape: {line}")

        scenario_values = parts[7 : 7 + len(header_abbrevs)]
        scenarios = {
            legend[abbr]: float(value) / 100.0
            for abbr, value in zip(header_abbrevs, scenario_values)
            if abbr in legend
        }
        return PublishedRow(
            model=row_model,
            backend_mode=row_backend_mode,
            score=parse_percent(parts[0]),
            accuracy=parse_percent(parts[1]),
            completeness=parse_percent(parts[2]),
            n=int(parts[6]),
            scenarios=scenarios,
        )

    raise SystemExit(f"published row not found for {model} {backend_mode} [reforged]")


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


def select_local_rows(
    rows: list[dict[str, Any]],
    scenarios: set[str],
    local_model: str | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("scenario") not in scenarios:
            continue
        if local_model is not None and row.get("model") != local_model:
            continue
        if row.get("ablation", "reforged") != "reforged":
            continue
        selected.append(row)
    return selected


def local_metrics(
    selected: list[dict[str, Any]],
    scenarios: set[str],
) -> tuple[float, float, dict[str, float], dict[str, int], list[str]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in scenarios}
    for row in selected:
        by_scenario[row["scenario"]].append(row)

    missing = sorted(scenario for scenario, values in by_scenario.items() if not values)
    total = len(selected)
    if total == 0:
        return 0.0, 0.0, {}, {}, missing

    successes = sum(1 for row in selected if bool(row.get("success")))
    completed = sum(1 for row in selected if bool(row.get("completeness")))
    per_scenario = {
        scenario: sum(1 for row in values if bool(row.get("success"))) / len(values)
        for scenario, values in by_scenario.items()
        if values
    }
    counts = {scenario: len(values) for scenario, values in by_scenario.items() if values}
    return successes / total, completed / total, per_scenario, counts, missing


def pp(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare local eval JSONL against published Forge results"
    )
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--published", type=Path, default=DEFAULT_PUBLISHED)
    parser.add_argument("--model", required=True, help="Published model identity")
    parser.add_argument(
        "--backend-mode",
        choices=["LS/N", "LS/P"],
        default="LS/N",
        help="Published leaderboard backend/mode row",
    )
    parser.add_argument("--local-model", help="Local JSONL model identity filter")
    parser.add_argument("--score-tolerance-pp", type=float, default=15.0)
    parser.add_argument("--completeness-tolerance-pp", type=float, default=5.0)
    parser.add_argument("--scenario-tolerance-pp", type=float, default=30.0)
    parser.add_argument("--strict-scenarios", action="store_true")
    parser.add_argument(
        "--force-proxy-compare",
        action="store_true",
        help="Compare proxy rows to direct published rows despite backend/mode mismatch",
    )
    args = parser.parse_args()

    published = parse_published_row(args.published, args.model, args.backend_mode)
    rows = load_jsonl(args.jsonl)
    published_scenarios = set(published.scenarios)
    selected = select_local_rows(rows, published_scenarios, args.local_model)
    proxy_rows = [
        row
        for row in selected
        if row.get("mode") == "proxy" or row.get("eval_target_backend") == "openai-proxy"
    ]
    if proxy_rows and not args.force_proxy_compare:
        proxy_modes = sorted({
            f"{row.get('backend', 'unknown')}/{row.get('mode', 'unknown')}"
            for row in proxy_rows
        })
        print(f"Published baseline: {published.model} {published.backend_mode} [reforged]")
        print(f"Local rows:         {len(selected)}")
        print(f"Local modes:        {', '.join(proxy_modes)}")
        print(
            "\nPublished comparison skipped: local rows are proxy-mode rows, "
            "not direct LS/N rows. Pass --force-proxy-compare to compare anyway."
        )
        return 0

    local_score, local_cmp, local_scenarios, counts, missing = local_metrics(
        selected,
        published_scenarios,
    )

    failures: list[str] = []
    warnings: list[str] = []
    if missing:
        failures.append("missing scenarios: " + ", ".join(missing))

    score_floor = published.score - args.score_tolerance_pp / 100.0
    cmp_floor = published.completeness - args.completeness_tolerance_pp / 100.0
    if local_score < score_floor:
        failures.append(
            f"score {pp(local_score)} below published {pp(published.score)} "
            f"minus {args.score_tolerance_pp:.1f}pp"
        )
    if local_cmp < cmp_floor:
        failures.append(
            f"completeness {pp(local_cmp)} below published {pp(published.completeness)} "
            f"minus {args.completeness_tolerance_pp:.1f}pp"
        )

    scenario_tol = args.scenario_tolerance_pp / 100.0
    for scenario, published_score in sorted(published.scenarios.items()):
        if scenario not in local_scenarios:
            continue
        local = local_scenarios[scenario]
        if local < published_score - scenario_tol:
            message = (
                f"{scenario}: {pp(local)} below published {pp(published_score)} "
                f"minus {args.scenario_tolerance_pp:.1f}pp"
            )
            if args.strict_scenarios:
                failures.append(message)
            else:
                warnings.append(message)

    print(f"Published baseline: {published.model} {published.backend_mode} [reforged]")
    print(f"Published N:        {published.n}")
    print(f"Local rows:         {sum(counts.values())}")
    print(f"Local scenario N:   {dict(sorted(counts.items()))}")
    print(f"Score:              local {pp(local_score)} vs published {pp(published.score)}")
    print(
        f"Completeness:       local {pp(local_cmp)} vs published {pp(published.completeness)}"
    )
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
    print("\nPublished comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
