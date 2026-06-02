from __future__ import annotations

import argparse

from generatetd.cli import resolve_synthetic_counts


def test_resolve_synthetic_balanced_splits_total_evenly():
    args = argparse.Namespace(
        synthetic_balanced=10,
        synthetic_missing_argument=0,
        synthetic_wrong_tool=0,
        synthetic_tool_not_needed=0,
    )
    assert resolve_synthetic_counts(args) == (5, 0, 5)


def test_resolve_synthetic_wrong_tool_is_disabled():
    args = argparse.Namespace(
        synthetic_balanced=0,
        synthetic_missing_argument=2,
        synthetic_wrong_tool=2,
        synthetic_tool_not_needed=2,
    )
    assert resolve_synthetic_counts(args) == (2, 0, 2)


def test_resolve_synthetic_counts_rejects_balanced_with_overrides():
    args = argparse.Namespace(
        synthetic_balanced=10,
        synthetic_missing_argument=1,
        synthetic_wrong_tool=0,
        synthetic_tool_not_needed=0,
    )
    try:
        resolve_synthetic_counts(args)
    except ValueError as exc:
        assert "--synthetic-balanced cannot be combined" in str(exc)
    else:
        raise AssertionError("expected ValueError")
