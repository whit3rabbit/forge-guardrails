from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .env import env_default, load_env_file
from .models import DEFAULT_MINIMAX_MODEL, DEFAULT_OPENROUTER_MODEL, GenerateOptions
from .pipeline import generate
from .providers import ProviderError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="Generate Forge verifier training data from agent logs.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="extract, sanitize, review, validate, dedupe, and split rows")
    gen.add_argument("--out", required=True, type=Path)
    gen.add_argument("--provider", choices=["auto", "minimax", "openrouter", "none"], default="auto")
    gen.add_argument("--llm-review", action="store_true")
    gen.add_argument("--verify-review", action="store_true", help="run a second LLM gate before accepting reviewed rows")
    gen.add_argument(
        "--verifier-provider",
        choices=["same", "auto", "minimax", "openrouter", "none"],
        default="same",
        help="provider for --verify-review; default reuses --provider",
    )
    gen.add_argument("--no-api", action="store_true")
    gen.add_argument("--serializer", choices=["v1", "v2"], default="v1")
    gen.add_argument("--limit", type=int)
    gen.add_argument("--since")
    gen.add_argument("--project")
    gen.add_argument("--include-codex", dest="include_codex", action="store_true", default=True)
    gen.add_argument("--no-codex", dest="include_codex", action="store_false")
    gen.add_argument("--include-claude", dest="include_claude", action="store_true", default=True)
    gen.add_argument("--no-claude", dest="include_claude", action="store_false")
    gen.add_argument("--quiet", action="store_true", help="suppress progress messages on stderr")
    gen.add_argument("--emit-notebook-adapter", dest="emit_notebook_adapter", action="store_true", default=True)
    gen.add_argument("--no-notebook-adapter", dest="emit_notebook_adapter", action="store_false")
    gen.add_argument("--fail-on-private-public-export", action="store_true", default=True)
    gen.add_argument("--allow-private-public-export", dest="fail_on_private_public_export", action="store_false")
    gen.add_argument("--codex-root", type=Path, default=Path.home() / ".codex")
    gen.add_argument("--claude-root", type=Path, default=Path.home() / ".claude")
    gen.add_argument(
        "--minimax-model",
        default=env_default("GENERATETD_MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL, "MINIMAX_MODEL"),
    )
    gen.add_argument(
        "--openrouter-model",
        default=env_default("GENERATETD_OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL, "OPENROUTER_MODEL"),
    )
    gen.add_argument("--api-max-attempts", type=int, default=4)
    gen.add_argument("--api-backoff-seconds", type=float, default=1.0)
    gen.add_argument(
        "--synthetic-balanced",
        type=int,
        default=0,
        help="total synthetic hard negatives, split evenly across current synthetic types",
    )
    gen.add_argument("--synthetic-missing-argument", type=int, default=0)
    gen.add_argument("--synthetic-wrong-tool", type=int, default=0)
    gen.add_argument("--synthetic-tool-not-needed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        provider = "none" if args.no_api else args.provider
        try:
            synthetic_missing_argument, synthetic_wrong_tool, synthetic_tool_not_needed = resolve_synthetic_counts(args)
        except ValueError as exc:
            print(f"generatetd: {exc}", file=sys.stderr)
            return 2
        try:
            manifest = generate(
                GenerateOptions(
                    out=args.out,
                    include_codex=args.include_codex,
                    include_claude=args.include_claude,
                    provider=provider,
                    llm_review=args.llm_review,
                    verify_review=args.verify_review,
                    verifier_provider=args.verifier_provider,
                    no_api=args.no_api,
                    serializer=args.serializer,
                    limit=args.limit,
                    since=args.since,
                    project=args.project,
                    emit_notebook_adapter=args.emit_notebook_adapter,
                    fail_on_private_public_export=args.fail_on_private_public_export,
                    codex_root=args.codex_root,
                    claude_root=args.claude_root,
                    minimax_model=args.minimax_model,
                    openrouter_model=args.openrouter_model,
                    api_max_attempts=args.api_max_attempts,
                    api_backoff_seconds=args.api_backoff_seconds,
                    synthetic_missing_argument=synthetic_missing_argument,
                    synthetic_wrong_tool=synthetic_wrong_tool,
                    synthetic_tool_not_needed=synthetic_tool_not_needed,
                    progress=not args.quiet,
                )
            )
        except ProviderError as exc:
            print(f"generatetd: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


def resolve_synthetic_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    per_type = [
        max(0, args.synthetic_missing_argument),
        max(0, args.synthetic_wrong_tool),
        max(0, args.synthetic_tool_not_needed),
    ]
    balanced = max(0, args.synthetic_balanced)
    if balanced and any(per_type):
        raise ValueError("--synthetic-balanced cannot be combined with per-type synthetic count flags")
    if not balanced:
        return tuple(per_type)  # type: ignore[return-value]

    base = balanced // 3
    remainder = balanced % 3
    counts = [base, base, base]
    for index in range(remainder):
        counts[index] += 1
    return counts[0], counts[1], counts[2]
