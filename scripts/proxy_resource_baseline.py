#!/usr/bin/env python3
"""Sample proxy eval CPU/RSS and build a baseline report.

This script is intentionally stdlib-only. It observes the local eval process
tree from outside the Rust proxy so resource benchmarking does not alter proxy
request handling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROLES = ("proxy", "backend", "wrapper", "other")
REPORT_SECTIONS = (
    ("proxy", "Proxy (headline)"),
    ("backend", "Backend"),
    ("combined", "Combined tree"),
)
STOP_REQUESTED = False


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    cpu_percent: float
    rss_kib: int
    command: str


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return safe.strip("._-") or "resource"


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def classify_role(process: ProcessInfo, root_pid: int) -> str:
    command = process.command
    argv0 = command.split(None, 1)[0] if command.strip() else ""
    basename = Path(argv0).name

    if process.pid == root_pid or "start_llamaserver_proxy.sh" in command:
        return "wrapper"
    if basename == "forge-guardrails-proxy" or "forge-guardrails-proxy" in command:
        return "proxy"
    if basename in {"llama-server", "llamafile", "ollama"}:
        return "backend"
    if "llama-server" in command or "llamafile" in command:
        return "backend"
    return "other"


def read_ps_rows() -> list[ProcessInfo]:
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,pcpu=,rss=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"failed to read process table with ps: {exc}") from exc

    rows: list[ProcessInfo] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            cpu_percent = float(parts[2])
            rss_kib = int(parts[3])
        except ValueError:
            continue
        command = parts[4] if len(parts) >= 5 else ""
        rows.append(ProcessInfo(pid, ppid, cpu_percent, rss_kib, command))
    return rows


def descendants_with_pgrep(root_pid: int) -> set[int] | None:
    if shutil.which("pgrep") is None:
        return None

    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        completed = subprocess.run(
            ["pgrep", "-P", str(parent)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in {0, 1}:
            return None
        if completed.returncode == 1:
            continue
        for raw in completed.stdout.split():
            try:
                child = int(raw)
            except ValueError:
                continue
            if child in descendants:
                continue
            descendants.add(child)
            pending.append(child)
    return descendants


def descendants_from_rows(root_pid: int, rows: list[ProcessInfo]) -> set[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        children[row.ppid].append(row.pid)

    descendants: set[int] = set()
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, []))
    return descendants


def tracked_pids(root_pid: int, rows: list[ProcessInfo]) -> set[int]:
    discovered = descendants_from_rows(root_pid, rows)
    pgrep_descendants = descendants_with_pgrep(root_pid)
    if pgrep_descendants is not None:
        discovered.update(pgrep_descendants)
    return {root_pid, *discovered}


def build_sample(
    *,
    label: str,
    root_pid: int,
    rows: list[ProcessInfo],
    started_at: float,
    now: float | None = None,
) -> dict[str, Any] | None:
    now = time.time() if now is None else now
    process_ids = tracked_pids(root_pid, rows)
    processes = [row for row in rows if row.pid in process_ids]
    if not processes:
        return None

    roles = {
        role: {"process_count": 0, "cpu_percent": 0.0, "rss_kib": 0}
        for role in ROLES
    }
    process_rows = []
    for process in processes:
        role = classify_role(process, root_pid)
        roles[role]["process_count"] += 1
        roles[role]["cpu_percent"] += process.cpu_percent
        roles[role]["rss_kib"] += process.rss_kib
        process_rows.append(
            {
                "pid": process.pid,
                "ppid": process.ppid,
                "role": role,
                "cpu_percent": round(process.cpu_percent, 2),
                "rss_kib": process.rss_kib,
                "command": process.command[:500],
            }
        )

    for role in ROLES:
        roles[role]["cpu_percent"] = round(float(roles[role]["cpu_percent"]), 2)

    combined = {
        "process_count": sum(int(values["process_count"]) for values in roles.values()),
        "cpu_percent": round(
            sum(float(values["cpu_percent"]) for values in roles.values()), 2
        ),
        "rss_kib": sum(int(values["rss_kib"]) for values in roles.values()),
    }
    return {
        "label": label,
        "timestamp": round(now, 3),
        "elapsed_s": round(now - started_at, 3),
        "root_pid": root_pid,
        "roles": roles,
        "combined": combined,
        "processes": sorted(process_rows, key=lambda row: (row["role"], row["pid"])),
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percent / 100.0 * len(ordered)) - 1))
    return ordered[index]


def series_stats(cpu_values: list[float], rss_kib_values: list[int]) -> dict[str, Any]:
    rss_mib_values = [value / 1024.0 for value in rss_kib_values]
    sample_count = len(cpu_values)
    return {
        "samples": sample_count,
        "cpu_percent": {
            "mean": round(sum(cpu_values) / sample_count, 2) if sample_count else 0.0,
            "max": round(max(cpu_values), 2) if sample_count else 0.0,
            "p95": round(percentile(cpu_values, 95), 2),
        },
        "rss_mib": {
            "mean": round(sum(rss_mib_values) / sample_count, 2) if sample_count else 0.0,
            "max": round(max(rss_mib_values), 2) if sample_count else 0.0,
            "p95": round(percentile(rss_mib_values, 95), 2),
            "first": round(rss_mib_values[0], 2) if sample_count else 0.0,
            "last": round(rss_mib_values[-1], 2) if sample_count else 0.0,
            "delta": (
                round(rss_mib_values[-1] - rss_mib_values[0], 2)
                if sample_count
                else 0.0
            ),
        },
    }


def summarize_samples(
    label: str,
    sample_path: Path,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    role_stats: dict[str, Any] = {}
    for role in ROLES:
        role_stats[role] = series_stats(
            [float(sample["roles"][role]["cpu_percent"]) for sample in samples],
            [int(sample["roles"][role]["rss_kib"]) for sample in samples],
        )

    combined_stats = series_stats(
        [float(sample["combined"]["cpu_percent"]) for sample in samples],
        [int(sample["combined"]["rss_kib"]) for sample in samples],
    )
    started_at = float(samples[0]["timestamp"]) if samples else None
    ended_at = float(samples[-1]["timestamp"]) if samples else None
    duration_s = (
        round(ended_at - started_at, 3)
        if started_at is not None and ended_at is not None
        else 0.0
    )

    return {
        "label": label,
        "sample_count": len(samples),
        "duration_s": duration_s,
        "started_at": started_at,
        "ended_at": ended_at,
        "samples_path": str(sample_path),
        "roles": role_stats,
        "combined": combined_stats,
    }


def load_summaries(output_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("resource_summary_*.json")):
        try:
            summaries.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    return sorted(summaries, key=lambda summary: str(summary.get("label", "")))


def _fmt_stat(stats: dict[str, Any], key: str) -> str:
    return f"{float(stats[key]):.2f}"


def format_report(summaries: list[dict[str, Any]]) -> str:
    lines = ["Proxy Resource Baseline", ""]
    if not summaries:
        lines.append("No resource summary files found.")
        return "\n".join(lines) + "\n"

    for index, summary in enumerate(summaries):
        if index:
            lines.append("")
        lines.append(f"Label: {summary['label']}")
        lines.append(
            f"  Samples: {summary.get('sample_count', 0)}, "
            f"duration: {float(summary.get('duration_s', 0.0)):.1f}s"
        )
        for key, title in REPORT_SECTIONS:
            stats = summary["combined"] if key == "combined" else summary["roles"][key]
            cpu = stats["cpu_percent"]
            rss = stats["rss_mib"]
            lines.append(f"  {title}:")
            lines.append(
                "    CPU %: "
                f"mean {_fmt_stat(cpu, 'mean')}, "
                f"max {_fmt_stat(cpu, 'max')}, "
                f"p95 {_fmt_stat(cpu, 'p95')}"
            )
            lines.append(
                "    RSS MiB: "
                f"mean {_fmt_stat(rss, 'mean')}, "
                f"max {_fmt_stat(rss, 'max')}, "
                f"p95 {_fmt_stat(rss, 'p95')}, "
                f"first {_fmt_stat(rss, 'first')}, "
                f"last {_fmt_stat(rss, 'last')}, "
                f"delta {_fmt_stat(rss, 'delta')}"
            )
    return "\n".join(lines) + "\n"


def _handle_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_sample(args: argparse.Namespace) -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    label_for_path = safe_label(args.label)
    sample_path = output_dir / f"resource_samples_{label_for_path}.jsonl"
    summary_path = output_dir / f"resource_summary_{label_for_path}.json"
    samples: list[dict[str, Any]] = []
    started_at = time.time()

    with sample_path.open("w") as handle:
        while not STOP_REQUESTED:
            try:
                sample = build_sample(
                    label=args.label,
                    root_pid=args.root_pid,
                    rows=read_ps_rows(),
                    started_at=started_at,
                )
            except RuntimeError as exc:
                print(f"warning: {exc}", file=sys.stderr)
                break
            if sample is None:
                break
            samples.append(sample)
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            handle.flush()
            time.sleep(args.interval)

    summary = summarize_samples(args.label, sample_path, samples)
    write_text_atomic(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def run_report(args: argparse.Namespace) -> int:
    summaries = load_summaries(args.output_dir)
    text = format_report(summaries)
    write_text_atomic(args.report, text)
    print(text, end="")
    return 0


def positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and report CPU/RSS for local proxy eval process trees"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="sample one eval process tree")
    sample.add_argument("--root-pid", type=int, required=True)
    sample.add_argument("--label", required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--interval", type=positive_float, default=1.0)
    sample.set_defaults(func=run_sample)

    report = subparsers.add_parser("report", help="write a baseline report")
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--report", type=Path, required=True)
    report.set_defaults(func=run_report)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
