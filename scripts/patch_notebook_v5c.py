#!/usr/bin/env python3
"""
patch_notebook_v5c.py — hotfixes for the first v5b Colab run, applied in-place to
toolcall_verifier_training_production_colab_v5.ipynb.

Run after patch_notebook_v5b.py (one-shot; refuses to re-run).

Observed failures in the run:
  1. Dataset audit crash: the v5 contrastive builder tags its VALID rows with
     negative_type="contrastive_valid", which the suspicious-valid audit rejects
     (no allowlisted token). 20 rows -> RuntimeError in the dataframe cell.
  2. wrong_tool_semantic collapse: 1542 rows total, xLAM=0, glaive=12 (prior runs
     ~10k). Root cause: WRONG_TOOL_MIN_COMPAT_SCORE gates EMISSION, and on
     arg-name-disjoint tool sets (most xLAM/glaive examples) every distractor
     scores below it, so no wrong-tool negative is generated at all.

Changes:
  [H1] cell 14: negative_type="contrastive_valid" -> pair_role="contrastive_valid"
       (a positive row must not carry negative_type; nothing consumed the old tag)
  [H2] cell 19: min-compat no longer suppresses emission. The best non-ambiguous
       distractor is always emitted with args remapped onto its schema; compat only
       decides the tag: wrong_tool_schema_compatible vs wrong_tool_low_overlap.
       Per-source telemetry now reports both buckets.
"""
import json
import os
import sys

NB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "notebook", "toolcall_verifier_training_production_colab_v5.ipynb",
)
NB_PATH = os.path.normpath(NB_PATH)


def load_nb():
    with open(NB_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_nb(nb):
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def cell_src(cell):
    return "".join(cell["source"])


def set_cell_src(cell, new_src):
    lines = new_src.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    cell["source"] = lines


def find_cell_by_marker(cells, marker, cell_type=None):
    for i, c in enumerate(cells):
        if cell_type and c.get("cell_type") != cell_type:
            continue
        if marker in cell_src(c):
            return i
    return None


def replace_once(src, old, new, label):
    count = src.count(old)
    assert count == 1, f"[{label}] expected exactly 1 occurrence of anchor, found {count}"
    return src.replace(old, new, 1)


# ---------------------------------------------------------------------------
# H1: contrastive valid rows must not carry negative_type (cell 14)
# ---------------------------------------------------------------------------
H1_OLD = 'dict(base_meta, negative_type="contrastive_valid"),'
H1_NEW = 'dict(base_meta, pair_role="contrastive_valid"),'


def patch_h1(cells):
    idx = find_cell_by_marker(cells, H1_OLD)
    assert idx is not None, "[H1] contrastive builder cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, H1_OLD, H1_NEW, "H1")
    set_cell_src(cells[idx], src)
    print(f"  [H1] contrastive valid metadata fixed in cell {idx}")


# ---------------------------------------------------------------------------
# H2: compat tags instead of gating emission (cell 19)
# ---------------------------------------------------------------------------
H2_OLD_TAIL = r'''    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    compat, _, _, tool, ambiguity = scored[0]
    if compat < WRONG_TOOL_MIN_COMPAT_SCORE:
        return None
    candidate = {
        "name": tool.get("name", "unknown_tool"),
        "arguments": _remap_args_onto_tool(gold_args, tool, _WRONG_TOOL_RNG),
    }
    info = {
        "ambiguity_score": round(float(ambiguity), 4),
        "compat_score": round(float(compat), 4),
        "distractor_tool": tool.get("name", "unknown_tool"),
    }
    return candidate, info'''

H2_NEW_TAIL = r'''    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    compat, _, _, tool, ambiguity = scored[0]
    # Low compat must not suppress emission: on arg-name-disjoint tool sets (most of
    # xLAM/glaive) a hard gate zeroed the wrong_tool class in the first v5b run.
    # The best non-ambiguous distractor is still a legitimate wrong tool; remapped
    # args keep it schema-plausible so the model cannot shortcut on schema mismatch.
    candidate = {
        "name": tool.get("name", "unknown_tool"),
        "arguments": _remap_args_onto_tool(gold_args, tool, _WRONG_TOOL_RNG),
    }
    info = {
        "ambiguity_score": round(float(ambiguity), 4),
        "compat_score": round(float(compat), 4),
        "distractor_tool": tool.get("name", "unknown_tool"),
        "schema_compatible": bool(compat >= WRONG_TOOL_MIN_COMPAT_SCORE),
    }
    return candidate, info'''

H2_OLD_DOC = r'''    Near-duplicate tools (high description overlap or identical schemas) are skipped.
    Name similarity is deliberately NOT filtered: get_/set_ and approve_/reject_
    pairs are exactly the hard negatives the classifier needs."""'''

H2_NEW_DOC = r'''    Near-duplicate tools (high description overlap or identical schemas) are skipped.
    Name similarity is deliberately NOT filtered: get_/set_ and approve_/reject_
    pairs are exactly the hard negatives the classifier needs.

    Returns None only when every distractor is ambiguity-filtered (or none exist).
    WRONG_TOOL_MIN_COMPAT_SCORE decides the metadata tag (schema_compatible vs
    low_overlap), not whether a negative is emitted."""'''

H2_OLD_CALLSITE = r'''                wrong, wrong_info = wrong_pair
                wrong_tool_counts.setdefault(source, Counter())["kept"] += 1'''

H2_NEW_CALLSITE = r'''                wrong, wrong_info = wrong_pair
                if wrong_info.pop("schema_compatible"):
                    wt_negative_type = "wrong_tool_schema_compatible"
                    wrong_tool_counts.setdefault(source, Counter())["kept_schema_compatible"] += 1
                else:
                    wt_negative_type = "wrong_tool_low_overlap"
                    wrong_tool_counts.setdefault(source, Counter())["kept_low_overlap"] += 1'''

H2_OLD_META = r'''                    metadata={"generator": "hard_negative", "negative_type": "wrong_tool_schema_compatible", "gold_tool": call["name"], **wrong_info},'''

H2_NEW_META = r'''                    metadata={"generator": "hard_negative", "negative_type": wt_negative_type, "gold_tool": call["name"], **wrong_info},'''

H2_OLD_PRINT = r'''    print("Wrong-tool distractor selection by source (ambiguity-filtered):")
    for src_name in sorted(wrong_tool_counts):
        counts = wrong_tool_counts[src_name]
        total = counts["kept"] + counts["skipped"]
        skip_rate = counts["skipped"] / total if total else 0.0
        print(f"  {src_name}: kept={counts['kept']} skipped={counts['skipped']} skip_rate={skip_rate:.1%}")'''

H2_NEW_PRINT = r'''    print("Wrong-tool distractor selection by source (ambiguity-filtered):")
    for src_name in sorted(wrong_tool_counts):
        counts = wrong_tool_counts[src_name]
        total = sum(counts.values())
        skip_rate = counts["skipped"] / total if total else 0.0
        print(
            f"  {src_name}: schema_compatible={counts['kept_schema_compatible']} "
            f"low_overlap={counts['kept_low_overlap']} skipped={counts['skipped']} "
            f"skip_rate={skip_rate:.1%}"
        )'''


def patch_h2(cells):
    idx = find_cell_by_marker(cells, "def schema_compatible_wrong_tool_candidate(")
    assert idx is not None, "[H2] selector cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, H2_OLD_TAIL, H2_NEW_TAIL, "H2 tail")
    src = replace_once(src, H2_OLD_DOC, H2_NEW_DOC, "H2 docstring")
    src = replace_once(src, H2_OLD_CALLSITE, H2_NEW_CALLSITE, "H2 callsite")
    src = replace_once(src, H2_OLD_META, H2_NEW_META, "H2 metadata")
    src = replace_once(src, H2_OLD_PRINT, H2_NEW_PRINT, "H2 print")
    set_cell_src(cells[idx], src)
    print(f"  [H2] compat fallback emission installed in cell {idx}")


# ---------------------------------------------------------------------------
# Selector harness shared by the pre-apply repro and post-apply smoke
# ---------------------------------------------------------------------------
XLAM_LIKE_GOLD = {
    "name": "get_stock_price",
    "description": "Fetches the latest stock price for a ticker.",
    "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
}
XLAM_LIKE_OTHERS = [
    {"name": "currency_converter", "description": "Converts an amount between two currencies.",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
                    "required": ["amount", "from", "to"]}},
    {"name": "company_news", "description": "Retrieves recent news articles about a company.",
     "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["q"]}},
]
NEAR_DUPLICATE = {
    "name": "fetch_stock_price",
    "description": "Fetches the latest stock price for a ticker.",
    "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
}
COMPATIBLE_SIBLING = {
    "name": "get_stock_dividends",
    "description": "Returns the dividend history for a listed company.",
    "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
}


def load_selector(cells):
    import random
    from typing import Any, Dict, List, Optional, Tuple

    idx = find_cell_by_marker(cells, "def schema_compatible_wrong_tool_candidate(")
    assert idx is not None, "selector cell not found"
    src = cell_src(cells[idx])
    start = src.index("# v5b: schema-compatible wrong-tool selection")
    end = src.index("def missing_arg_candidate(")
    ns = {"random": random, "SEED": 42, "Any": Any, "Dict": Dict, "List": List,
          "Optional": Optional, "Tuple": Tuple}
    exec(
        "def required_args_for_tool(tool):\n"
        "    params = tool.get('parameters') or {}\n"
        "    req = params.get('required') or []\n"
        "    return [str(x) for x in req] if isinstance(req, list) else []\n"
        "def tool_by_name(tools, name):\n"
        "    for t in tools:\n"
        "        if t.get('name') == name:\n"
        "            return t\n"
        "    return None\n",
        ns,
    )
    exec(src[start:end], ns)
    return ns["schema_compatible_wrong_tool_candidate"]


def repro_pre_apply(cells):
    """Assert the bug exists before patching (fails loudly if anchors drifted)."""
    select = load_selector(cells)
    result = select([XLAM_LIKE_GOLD] + XLAM_LIKE_OTHERS, "get_stock_price", {"symbol": "AAPL"})
    assert result is None, "expected pre-patch selector to drop arg-disjoint distractors"
    print("  [repro] pre-patch selector returns None on xLAM-like tools (bug confirmed)")


def smoke_post_apply(cells):
    select = load_selector(cells)
    # Arg-disjoint case now emits a remapped low-overlap negative.
    result = select([XLAM_LIKE_GOLD] + XLAM_LIKE_OTHERS, "get_stock_price", {"symbol": "AAPL"})
    assert result is not None, "post-patch selector must emit on arg-disjoint tools"
    candidate, info = result
    assert info["schema_compatible"] is False, info
    assert candidate["name"] in {"currency_converter", "company_news"}
    picked = next(t for t in XLAM_LIKE_OTHERS if t["name"] == candidate["name"])
    required = set(picked["parameters"]["required"])
    assert required <= set(candidate["arguments"]), "required args must be filled, not verbatim gold args"
    assert "symbol" not in candidate["arguments"], "gold-only args must not leak into a disjoint schema"
    # Schema-compatible sibling still preferred and tagged compatible.
    result2 = select([XLAM_LIKE_GOLD, COMPATIBLE_SIBLING] + XLAM_LIKE_OTHERS,
                     "get_stock_price", {"symbol": "AAPL"})
    candidate2, info2 = result2
    assert candidate2["name"] == "get_stock_dividends", candidate2
    assert info2["schema_compatible"] is True, info2
    assert candidate2["arguments"] == {"symbol": "AAPL"}
    # Ambiguous-only case still skipped.
    result3 = select([XLAM_LIKE_GOLD, NEAR_DUPLICATE], "get_stock_price", {"symbol": "AAPL"})
    assert result3 is None, "near-duplicate-only tool set must still be skipped"
    print("  [smoke] H2 selector: low-overlap emitted+remapped, sibling tagged compatible, near-dup skipped")


def smoke_audit_rule(cells):
    """Run the actual audit predicate from cell 20 against the fixed contrastive metadata."""
    idx = find_cell_by_marker(cells, "def suspicious_valid_hard_negative_reason(")
    assert idx is not None
    src = cell_src(cells[idx])
    start = src.index("def suspicious_valid_hard_negative_reason(")
    end = src.index("def extract_candidate_payload_for_audit(")
    ns = {
        "Optional": type(None), "Any": object,
        "metadata_dict": lambda md: dict(md or {}),
        "is_corrected_positive_metadata": lambda md: False,
    }
    exec("from typing import Any, Optional", ns)
    exec(src[start:end], ns)

    class Row(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    fixed = Row(label="valid", metadata={"generator": "contrastive_pair", "pair_role": "contrastive_valid"})
    reason = ns["suspicious_valid_hard_negative_reason"](fixed)
    assert reason is None, f"fixed contrastive valid row still flagged: {reason}"
    broken = Row(label="valid", metadata={"generator": "contrastive_pair", "negative_type": "contrastive_valid"})
    assert ns["suspicious_valid_hard_negative_reason"](broken) is not None, "audit predicate sanity check failed"
    print("  [smoke] H1 audit rule: pair_role passes, old negative_type tag would still be flagged")


# ---------------------------------------------------------------------------
def main():
    print(f"Loading notebook: {NB_PATH}")
    nb = load_nb()
    cells = nb["cells"]

    full_src = "\n".join(cell_src(c) for c in cells)
    for m in ("wrong_tool_low_overlap", 'pair_role="contrastive_valid"'):
        assert m not in full_src, f"v5c marker {m!r} already present; refusing to re-run"
    for m in ("schema_compatible_wrong_tool_candidate", "WRONG_TOOL_MIN_COMPAT_SCORE"):
        assert m in full_src, f"v5b prerequisite marker {m!r} missing; run patch_notebook_v5b.py first"

    print("\nReproducing bugs against current notebook code...")
    repro_pre_apply(cells)

    print("\nApplying patches...")
    patch_h1(cells)
    patch_h2(cells)

    save_nb(nb)
    print(f"\nNotebook saved: {NB_PATH}")

    print("\nVerifying...")
    nb2 = load_nb()
    cells2 = nb2["cells"]
    full_src2 = "\n".join(cell_src(c) for c in cells2)
    checks = [
        ('pair_role="contrastive_valid"', "contrastive valid pair_role"),
        ("wrong_tool_low_overlap", "low-overlap negative tag"),
        ('"schema_compatible": bool(compat >= WRONG_TOOL_MIN_COMPAT_SCORE)', "compat tag in selector"),
        ("kept_schema_compatible", "split telemetry buckets"),
    ]
    all_ok = True
    for marker, label in checks:
        ok = marker in full_src2
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
        all_ok = all_ok and ok
    assert all_ok, "verification failed"
    assert 'negative_type="contrastive_valid"' not in full_src2, "old contrastive tag still present"

    print("\nFunctional smoke tests on patched notebook code...")
    smoke_post_apply(cells2)
    smoke_audit_rule(cells2)

    print("\nCompile-checking patched cells...")
    for marker in ('pair_role="contrastive_valid"', "def schema_compatible_wrong_tool_candidate("):
        i = find_cell_by_marker(cells2, marker, cell_type="code")
        assert i is not None, f"patched cell with {marker!r} not found"
        try:
            compile(cell_src(cells2[i]), f"cell_{i}", "exec")
        except SyntaxError as exc:
            print(f"  [FAIL] cell {i}: {exc}")
            sys.exit(1)
        print(f"  [OK ] cell {i}")
    print("\nDone. All v5c patches applied and verified.")


if __name__ == "__main__":
    main()
