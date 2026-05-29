from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tests.eval.ablation import ABLATION_PRESETS
from tests.eval.scenarios import ALL_SCENARIOS, EvalScenario

from .client import OpenAIProxyClient
from .runner import run_proxy_scenario, _result_row


def _print_early_help() -> None:
    parser = argparse.ArgumentParser(
        description="Run upstream Python eval scenarios against an OpenAI proxy"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--scenario", nargs="*")
    parser.add_argument("--tags", nargs="*")
    parser.add_argument(
        "--ablation",
        choices=[
            "bare",
            "no_compact",
            "no_nudge",
            "no_recovery",
            "no_rescue",
            "no_steps",
            "reforged",
        ],
        default="reforged",
    )
    parser.add_argument("--output")
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--backend-label", default="openai-proxy")
    parser.add_argument("--mode-label", default="proxy")
    parser.add_argument("--proxy-backend-mode", choices=["native", "prompt"])
    parser.add_argument("--eval-target-backend", default="openai-proxy")
    parser.add_argument("--no-recommended-sampling", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.print_help()


def _select_scenarios(names: list[str] | None, tags: list[str] | None) -> list[EvalScenario]:
    scenarios = ALL_SCENARIOS
    if names:
        wanted = set(names)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]
        missing = wanted - {scenario.name for scenario in scenarios}
        if missing:
            raise SystemExit(f"unknown scenarios: {', '.join(sorted(missing))}")
    if tags:
        scenarios = [
            scenario
            for scenario in scenarios
            if any(tag in scenario.tags for tag in tags)
        ]
        if not scenarios:
            raise SystemExit(f"no scenarios match tags: {', '.join(tags)}")
    return scenarios


async def main_async(args: argparse.Namespace) -> None:
    scenarios = _select_scenarios(args.scenario, args.tags)
    ablation = ABLATION_PRESETS[args.ablation]
    client = OpenAIProxyClient(
        args.base_url,
        args.model,
        timeout=args.timeout,
        recommended_sampling=not args.no_recommended_sampling,
    )

    output = Path(args.output) if args.output else None
    total_runs = len(scenarios) * args.runs
    completed_runs = 0
    print(
        f"Python oracle: {len(scenarios)} scenarios x {args.runs} runs "
        f"({total_runs} total)",
        file=sys.stderr,
        flush=True,
    )
    for scenario in scenarios:
        for run_idx in range(1, args.runs + 1):
            ordinal = completed_runs + 1
            print(
                f"[{ordinal}/{total_runs}] {scenario.name} "
                f"run {run_idx}/{args.runs}...",
                file=sys.stderr,
                flush=True,
            )
            result = await run_proxy_scenario(
                client,
                scenario,
                stream=args.stream,
                budget_tokens=args.budget_tokens,
                ablation=ablation,
                verbose=args.verbose,
            )
            row = _result_row(
                result,
                scenario,
                run_idx,
                args.model,
                args.stream,
                args.ablation,
                args.budget_tokens,
                args.backend_label,
                args.mode_label,
                args.eval_target_backend,
                args.proxy_backend_mode,
            )
            line = json.dumps(row, separators=(",", ":"))
            if output:
                with output.open("a") as handle:
                    handle.write(line + "\n")
            else:
                print(line)
            completed_runs += 1
            if not result.completeness:
                status = f"FAIL ({result.error_type})"
            elif result.accuracy is False:
                status = "OK (incorrect)"
            else:
                status = "OK"
            print(
                f"[{completed_runs}/{total_runs}] {scenario.name} "
                f"run {run_idx}/{args.runs}: {status}, "
                f"{result.iterations_used} iterations, "
                f"{result.elapsed_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upstream Python eval scenarios against an OpenAI proxy"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--scenario", nargs="*")
    parser.add_argument("--tags", nargs="*")
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATION_PRESETS.keys()),
        default="reforged",
    )
    parser.add_argument("--output")
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--backend-label", default="openai-proxy")
    parser.add_argument("--mode-label", default="proxy")
    parser.add_argument("--proxy-backend-mode", choices=["native", "prompt"])
    parser.add_argument("--eval-target-backend", default="openai-proxy")
    parser.add_argument("--no-recommended-sampling", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def main() -> None:
    asyncio.run(main_async(parse_args()))
