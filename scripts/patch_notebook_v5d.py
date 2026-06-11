#!/usr/bin/env python3
"""
patch_notebook_v5d.py — fixes for the second v5b/v5c Colab run, applied in-place to
toolcall_verifier_training_production_colab_v5.ipynb.

Run after patch_notebook_v5c.py (one-shot; refuses to re-run).

Observed failures in the run:
  1. Checkpoint selection picked a degenerate epoch-1 model (wrong_args recall 0.08,
     valid precision 0.47): epochs 2-3 were discarded with gate_deficit_score=-inf
     because their false objection exceeded the 2.5x ceiling (epoch 3 by 0.0008, under
     one row of noise at 824 valid rows), and the -inf metric also made
     EarlyStoppingCallback stop training at epoch 3/5 while false objection was
     trending down (0.027 -> 0.013).
  2. EXCLUDE_DETERMINISTIC_INVALID_FROM_ML_TRAINING was a no-op: the filter matched
     the post-mapping name "deterministic_invalid" against raw labels
     (missing_required_args/unknown_tool/...) before normalize_label ran, so 3971 DI
     rows trained and 495 polluted test (70 DI->wrong_tool predictions dragged
     wrong_tool precision to 0.82).
  3. The quantized ONNX parity gate raised and killed the run even though fp32 parity
     passed at 1.000 and the report already records the graceful outcome
     (failed_shadow_only / use fp32 for replay).

Changes:
  [D1] cell 28: tiered checkpoint_selection_score() helper replaces the -inf discard.
       Tier 0: fails competence floor (wrong_args recall < 0.5 or present F1 < 0.7).
       Tier 1: competent but false objection above the ceiling (score rises as it falls).
       Tier 2: competent and within ceiling (lexicographic). All scores finite/monotone
       so early stopping tracks real improvement.
  [D2] cell 20: DI exclusion moved after normalize_label, filtering the dataframe for
       both training and eval splits (deterministic rules own DI at runtime).
  [D3] cell 39: quantized parity gate downgraded to a warning unless
       QUANTIZED_PARITY_HARD_FAIL is set; fp32/PyTorch parity stays fatal.
  [D4] cell 45: checkpoint-selection description updated to the tiered rule.
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
# D1: tiered checkpoint selection (cell 28)
# ---------------------------------------------------------------------------
D1_CONST_ANCHOR = r'''CHECKPOINT_VALID_RECALL_GATE = 0.94
CHECKPOINT_FALSE_OBJECTION_90_GATE = 0.005
CHECKPOINT_WRONG_TOOL_PRECISION_GATE = 0.90


def compute_metrics(eval_pred):'''

D1_HELPER = r'''CHECKPOINT_VALID_RECALL_GATE = 0.94
CHECKPOINT_FALSE_OBJECTION_90_GATE = 0.005
CHECKPOINT_WRONG_TOOL_PRECISION_GATE = 0.90
CHECKPOINT_FALSE_OBJECTION_DISCARD_CEILING = 2.5 * CHECKPOINT_FALSE_OBJECTION_90_GATE
# Competence floor: a checkpoint below either bound is the undertrained "everything is
# valid" pathology — its high valid_recall and near-zero false objections are vacuous.
CHECKPOINT_MIN_WRONG_ARGS_RECALL = 0.50
CHECKPOINT_MIN_PRESENT_MACRO_F1 = 0.70


def checkpoint_selection_score(
    valid_recall: float,
    wrong_tool_precision: float,
    wrong_tool_recall: float,
    present_f1: float,
    wrong_args_recall: float,
    valid_false_objection_90: float,
) -> Tuple[float, bool]:
    """Tiered checkpoint selection, replacing the -inf hard discard that picked a
    degenerate epoch-1 model and starved EarlyStoppingCallback of finite improvements
    while false objection was still trending down across epochs.

    Tier 0, fails competence floor: score in [-100, -99].
    Tier 1, competent but false objection above the discard ceiling: score < ~1.12,
        rising as quality improves and false objection falls.
    Tier 2, competent and within the ceiling: score >= 100, lexicographic ordering
        (valid_recall, then wrong_tool precision, then wrong_tool recall, then F1).
    Tiers cannot overlap, so any competent checkpoint beats any degenerate one and any
    within-ceiling checkpoint beats any over-ceiling one."""
    lexicographic = (
        valid_recall
        + 0.1 * wrong_tool_precision
        + 0.01 * wrong_tool_recall
        + 0.001 * present_f1
    )
    competent = bool(
        wrong_args_recall >= CHECKPOINT_MIN_WRONG_ARGS_RECALL
        and present_f1 >= CHECKPOINT_MIN_PRESENT_MACRO_F1
    )
    fo_ok = bool(valid_false_objection_90 <= CHECKPOINT_FALSE_OBJECTION_DISCARD_CEILING)
    if not competent:
        score = -100.0 + present_f1
    elif not fo_ok:
        score = lexicographic - 10.0 * (
            valid_false_objection_90 - CHECKPOINT_FALSE_OBJECTION_DISCARD_CEILING
        )
    else:
        score = 100.0 + lexicographic - 10.0 * max(
            0.0, valid_false_objection_90 - CHECKPOINT_FALSE_OBJECTION_90_GATE
        )
    return float(score), bool(competent and fo_ok)


def compute_metrics(eval_pred):'''

D1_OLD_BLOCK = r'''    # Constrained lexicographic checkpoint selection.
    # 1. Discard checkpoints with false_objection > 2.5x gate ceiling (non-promotable, score=-inf).
    # 2. Among passing: maximize valid_recall, then wrong_tool_precision, then wrong_tool_recall, then macro_f1.
    # Prevents the blended gate_deficit from selecting low-recall epochs that rarely make high-conf objections.
    CHECKPOINT_FALSE_OBJECTION_DISCARD_CEILING = 2.5 * CHECKPOINT_FALSE_OBJECTION_90_GATE
    valid_recall_deficit = max(0.0, CHECKPOINT_VALID_RECALL_GATE - valid_recall) / CHECKPOINT_VALID_RECALL_GATE
    wrong_tool_precision_deficit = max(0.0, CHECKPOINT_WRONG_TOOL_PRECISION_GATE - wrong_tool_precision) / CHECKPOINT_WRONG_TOOL_PRECISION_GATE
    false_objection_excess = max(0.0, valid_false_objection_90 - CHECKPOINT_FALSE_OBJECTION_90_GATE) / CHECKPOINT_FALSE_OBJECTION_90_GATE
    # Keep legacy gate_deficit for telemetry backward-compat.
    gate_deficit = float(
        valid_recall_deficit
        + wrong_tool_precision_deficit
        + 5.0 * false_objection_excess
        + 0.5 * valid_to_wrong_args_rate
    )
    constrained_promotable = bool(valid_false_objection_90 <= CHECKPOINT_FALSE_OBJECTION_DISCARD_CEILING)
    if not constrained_promotable:
        gate_deficit_score = float("-inf")
    else:
        # Lexicographic scoring: valid_recall is primary (range [0,1]),
        # wrong_tool_precision secondary ([0,0.1]), wrong_tool_recall tertiary ([0,0.01]),
        # macro_f1 quaternary ([0,0.001]).
        gate_deficit_score = (
            valid_recall
            + 0.1 * wrong_tool_precision
            + 0.01 * wrong_tool_recall
            + 0.001 * present_f1
            - 10.0 * max(0.0, valid_false_objection_90 - CHECKPOINT_FALSE_OBJECTION_90_GATE)
        )'''

D1_NEW_BLOCK = r'''    valid_recall_deficit = max(0.0, CHECKPOINT_VALID_RECALL_GATE - valid_recall) / CHECKPOINT_VALID_RECALL_GATE
    wrong_tool_precision_deficit = max(0.0, CHECKPOINT_WRONG_TOOL_PRECISION_GATE - wrong_tool_precision) / CHECKPOINT_WRONG_TOOL_PRECISION_GATE
    false_objection_excess = max(0.0, valid_false_objection_90 - CHECKPOINT_FALSE_OBJECTION_90_GATE) / CHECKPOINT_FALSE_OBJECTION_90_GATE
    # Keep legacy gate_deficit for telemetry backward-compat.
    gate_deficit = float(
        valid_recall_deficit
        + wrong_tool_precision_deficit
        + 5.0 * false_objection_excess
        + 0.5 * valid_to_wrong_args_rate
    )
    gate_deficit_score, constrained_promotable = checkpoint_selection_score(
        valid_recall=valid_recall,
        wrong_tool_precision=wrong_tool_precision,
        wrong_tool_recall=wrong_tool_recall,
        present_f1=present_f1,
        wrong_args_recall=wrong_args_recall,
        valid_false_objection_90=valid_false_objection_90,
    )'''


def patch_d1(cells):
    idx = find_cell_by_marker(cells, "def compute_metrics(eval_pred):")
    assert idx is not None, "[D1] compute_metrics cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, D1_CONST_ANCHOR, D1_HELPER, "D1 helper")
    src = replace_once(src, D1_OLD_BLOCK, D1_NEW_BLOCK, "D1 block")
    set_cell_src(cells[idx], src)
    print(f"  [D1] tiered checkpoint selection installed in cell {idx}")


# ---------------------------------------------------------------------------
# D2: DI exclusion after label mapping (cell 20)
# ---------------------------------------------------------------------------
D2_OLD_DEAD = r'''# Filter deterministic_invalid rows from ML training when enabled.
if _EXCLUDE_DI:
    _before_di = len(all_rows) if 'all_rows' in dir() else 0
    all_rows = [r for r in all_rows if r.label != "deterministic_invalid"] if 'all_rows' in dir() else []
    _removed_di = _before_di - len(all_rows)
    if _removed_di > 0:
        print(f"[EXCLUDE_DI] Removed {_removed_di} deterministic_invalid rows from ML training.")
    del _before_di, _removed_di'''

D2_NEW_DEAD = r'''# deterministic_invalid filtering happens after normalize_label maps raw labels onto the
# collapsed bucket (a previous filter here matched the post-mapping name against raw
# labels such as missing_required_args, so it silently removed nothing).'''

D2_MAP_ANCHOR = r'''df["raw_label"] = df["label"]
df["label"] = df["label"].map(normalize_label)'''

D2_MAP_NEW = r'''df["raw_label"] = df["label"]
df["label"] = df["label"].map(normalize_label)
if _EXCLUDE_DI:
    _di_rows = int((df["label"] == "deterministic_invalid").sum())
    if _di_rows:
        df = df[df["label"] != "deterministic_invalid"].reset_index(drop=True)
        print(f"[EXCLUDE_DI] Removed {_di_rows} deterministic_invalid rows from ML training/eval (deterministic rules own them).")'''


def patch_d2(cells):
    idx = find_cell_by_marker(cells, "_EXCLUDE_DI")
    assert idx is not None, "[D2] DI filter cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, D2_OLD_DEAD, D2_NEW_DEAD, "D2 dead filter")
    src = replace_once(src, D2_MAP_ANCHOR, D2_MAP_NEW, "D2 mapped filter")
    set_cell_src(cells[idx], src)
    print(f"  [D2] post-mapping DI exclusion installed in cell {idx}")


# ---------------------------------------------------------------------------
# D3: quantized parity gate non-fatal (cell 39)
# ---------------------------------------------------------------------------
D3_FLAG_ANCHOR = r'''ONNX_PARITY_ROWS_PER_SPLIT = 512  #@param {type:"integer"}'''

D3_FLAG_NEW = r'''ONNX_PARITY_ROWS_PER_SPLIT = 512  #@param {type:"integer"}
# Quantized parity failure marks the int8 artifact failed_shadow_only instead of killing
# the run: fp32 ONNX (gated separately, and fatally) remains the deployable artifact.
QUANTIZED_PARITY_HARD_FAIL = False  #@param {type:"boolean"}'''

D3_OLD_RAISE = r'''if onnx_parity_report.get("quantized_present") and not onnx_parity_report.get("fp32_quantized_gate_passed"):
    raise RuntimeError(
        "FP32/quantized ONNX top-label agreement "
        f"{onnx_parity_report['fp32_quantized_top_label_agreement']:.4f} is below gate "
        f"{FP32_QUANTIZED_TOP_LABEL_AGREEMENT_GATE:.4f}; use FP32 ONNX for replay and keep quantized shadow-only."
    )'''

D3_NEW_RAISE = r'''if onnx_parity_report.get("quantized_present") and not onnx_parity_report.get("fp32_quantized_gate_passed"):
    _quant_parity_msg = (
        "FP32/quantized ONNX top-label agreement "
        f"{onnx_parity_report['fp32_quantized_top_label_agreement']:.4f} is below gate "
        f"{FP32_QUANTIZED_TOP_LABEL_AGREEMENT_GATE:.4f}; use FP32 ONNX for replay and keep quantized shadow-only."
    )
    if QUANTIZED_PARITY_HARD_FAIL:
        raise RuntimeError(_quant_parity_msg)
    print("WARNING:", _quant_parity_msg)
    print(
        "Continuing: the quantized artifact is marked failed_shadow_only in "
        "onnx_parity_report.json and must not be deployed; fp32 ONNX passed parity."
    )'''


def patch_d3(cells):
    idx = find_cell_by_marker(cells, D3_OLD_RAISE)
    assert idx is not None, "[D3] parity gate cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, D3_FLAG_ANCHOR, D3_FLAG_NEW, "D3 flag")
    src = replace_once(src, D3_OLD_RAISE, D3_NEW_RAISE, "D3 raise")
    set_cell_src(cells[idx], src)
    print(f"  [D3] quantized parity soft-fail installed in cell {idx}")


# ---------------------------------------------------------------------------
# D4: ablation markdown (cell 45)
# ---------------------------------------------------------------------------
D4_OLD = "Early stopping remains enabled. The saved tool-call model is selected by a constrained lexicographic rule: checkpoints whose valid false objection at 0.90 exceeds 2.5x the gate ceiling are discarded outright; among the survivors the selector maximizes valid recall, then wrong_tool_semantic precision, then wrong_tool_semantic recall, then macro F1. The validation/test promotion gates remain the release stop sign."

D4_NEW = "Early stopping remains enabled. The saved tool-call model is selected by a tiered rule. Checkpoints failing the competence floor (wrong_arguments recall >= 0.5 and present-label macro F1 >= 0.7) rank below everything else: their high valid recall is the vacuous everything-is-valid pathology of early epochs. Competent checkpoints whose valid false objection at 0.90 exceeds 2.5x the gate ceiling rank next, with finite scores that keep rising as false objection falls, so early stopping tracks real improvement instead of stopping on -inf. Competent checkpoints within the ceiling rank highest, ordered by valid recall, then wrong_tool_semantic precision, then wrong_tool_semantic recall, then macro F1. The validation/test promotion gates remain the release stop sign."


def patch_d4(cells):
    idx = find_cell_by_marker(cells, "## 13. Recommended ablation matrix", cell_type="markdown")
    assert idx is not None, "[D4] ablation markdown cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, D4_OLD, D4_NEW, "D4")
    set_cell_src(cells[idx], src)
    print(f"  [D4] selection description updated in cell {idx}")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
# Observed validation metrics from the failed run (epochs 1-3).
OBSERVED_EPOCHS = [
    # (valid_recall, wrong_tool_precision, wrong_tool_recall, present_f1, wrong_args_recall, fo_90)
    (0.9550970874, 0.8111587983, 0.8852459016, 0.5308444585, 0.0839694656, 0.0012135922),
    (0.9356796117, 0.9818181818, 0.8852459016, 0.7598799283, 0.9720101781, 0.0266990291),
    (0.9684466019, 0.9603960396, 0.9086651054, 0.9201119436, 0.9643765903, 0.0133495146),
]


def smoke_test_selection():
    """Exec the patched helper and replay the failed run's eval history."""
    from typing import Tuple
    ns = {"Tuple": Tuple}
    exec(D1_HELPER.replace("def compute_metrics(eval_pred):", ""), ns)
    score = ns["checkpoint_selection_score"]
    results = [score(*epoch) for epoch in OBSERVED_EPOCHS]
    scores = [s for s, _ in results]
    promotables = [p for _, p in results]
    assert scores[0] < -99.0, f"epoch 1 must land in tier 0, got {scores[0]}"
    assert promotables == [False, False, False], promotables
    assert scores[0] < scores[1] < scores[2], f"scores must increase monotonically: {scores}"
    assert max(range(3), key=lambda i: scores[i]) == 2, "epoch 3 must be selected"
    # Synthetic tier-2 checkpoint outranks all observed ones and is promotable.
    s_good, p_good = score(0.95, 0.95, 0.92, 0.92, 0.95, 0.004)
    assert p_good and s_good > 100.0 and s_good > scores[2]
    print(f"  [smoke] D1 selection: epoch scores {['%.3f' % s for s in scores]}, "
          f"epoch 3 selected, tier-2 synthetic promotable at {s_good:.3f}")


def smoke_test_di_filter():
    """Replay the mapping + filter sequence on a tiny frame with raw labels."""
    try:
        import pandas as pd
    except ImportError:
        print("  [smoke] D2 skipped (pandas not installed locally)")
        return
    semantic_map = {
        "valid": "valid",
        "missing_required_args": "deterministic_invalid",
        "unknown_tool": "deterministic_invalid",
        "wrong_tool_semantic": "wrong_tool_semantic",
    }
    df = pd.DataFrame({"label": ["valid", "missing_required_args", "unknown_tool", "wrong_tool_semantic"]})
    ns = {"pd": pd, "df": df, "_EXCLUDE_DI": True, "normalize_label": lambda x: semantic_map.get(x, x)}
    exec(D2_MAP_NEW, ns)
    out = ns["df"]
    assert "deterministic_invalid" not in set(out["label"]), out["label"].tolist()
    assert len(out) == 2, f"expected 2 surviving rows, got {len(out)}"
    # And the old dead filter really was a no-op on raw labels.
    raw_labels = ["valid", "missing_required_args", "unknown_tool"]
    survivors = [x for x in raw_labels if x != "deterministic_invalid"]
    assert survivors == raw_labels, "sanity: old filter matched nothing on raw labels"
    print("  [smoke] D2 DI filter: post-mapping filter removes DI rows; old raw-label filter was a no-op")


# ---------------------------------------------------------------------------
def main():
    print(f"Loading notebook: {NB_PATH}")
    nb = load_nb()
    cells = nb["cells"]

    full_src = "\n".join(cell_src(c) for c in cells)
    for m in ("checkpoint_selection_score", "QUANTIZED_PARITY_HARD_FAIL", "CHECKPOINT_MIN_WRONG_ARGS_RECALL"):
        assert m not in full_src, f"v5d marker {m!r} already present; refusing to re-run"
    for m in ("wrong_tool_low_overlap", 'pair_role="contrastive_valid"', "constrained_promotable"):
        assert m in full_src, f"prerequisite marker {m!r} missing; run patch_notebook_v5b/v5c first"

    print("\nRunning pre-apply smoke tests on patch constants...")
    smoke_test_selection()
    smoke_test_di_filter()

    print("\nApplying patches...")
    patch_d1(cells)
    patch_d2(cells)
    patch_d3(cells)
    patch_d4(cells)

    save_nb(nb)
    print(f"\nNotebook saved: {NB_PATH}")

    print("\nVerifying...")
    nb2 = load_nb()
    cells2 = nb2["cells"]
    full_src2 = "\n".join(cell_src(c) for c in cells2)
    checks = [
        ("def checkpoint_selection_score(", "tiered selection helper"),
        ("CHECKPOINT_MIN_WRONG_ARGS_RECALL", "competence floor constant"),
        ("gate_deficit_score, constrained_promotable = checkpoint_selection_score(", "compute_metrics uses helper"),
        ('_di_rows = int((df["label"] == "deterministic_invalid").sum())', "post-mapping DI filter"),
        ("QUANTIZED_PARITY_HARD_FAIL", "quantized parity flag"),
        ("failed_shadow_only in", "soft-fail continuation message"),
        ("tiered rule", "markdown selection description"),
    ]
    all_ok = True
    for marker, label in checks:
        ok = marker in full_src2
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
        all_ok = all_ok and ok
    assert all_ok, "verification failed"
    assert 'gate_deficit_score = float("-inf")' not in full_src2, "-inf discard still present"

    print("\nCompile-checking patched code cells...")
    for marker in ("def checkpoint_selection_score(", "_di_rows", "QUANTIZED_PARITY_HARD_FAIL"):
        i = find_cell_by_marker(cells2, marker, cell_type="code")
        assert i is not None, f"patched cell with {marker!r} not found"
        try:
            compile(cell_src(cells2[i]), f"cell_{i}", "exec")
        except SyntaxError as exc:
            print(f"  [FAIL] cell {i}: {exc}")
            sys.exit(1)
        print(f"  [OK ] cell {i}")
    print("\nDone. All v5d patches applied and verified.")


if __name__ == "__main__":
    main()
