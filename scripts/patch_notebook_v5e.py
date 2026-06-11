#!/usr/bin/env python3
"""
patch_notebook_v5e.py — promotion-gating single-source-of-truth fixes plus
diagnostic-only gate/threshold reporting, applied in-place to
toolcall_verifier_training_production_colab_v5.ipynb.

Run after patch_notebook_v5d.py (one-shot; refuses to re-run).

Observed failure in the run:
  test_metrics.json carried eval_constrained_promotable=true while
  promotion_gate_report.json failed the test split: the checkpoint boolean used the
  relaxed 2.5x discard ceiling (0.0125) while the gate report applied the hard 0.005
  gate, so valid_false_objection_at_0_90 = 18/1569 = 0.01147 produced contradictory
  promotability signals across exported artifacts.

Changes:
  [E1]  cell 28: checkpoint_selection_score keeps the tiered 2.5x score (selection and
        early stopping unchanged) but returns a strict boolean (fo <= 0.005, valid
        recall >= 0.94, wrong_tool precision >= 0.90) so a per-eval flag can never
        claim promotability that the gate report would deny.
  [E2]  cell 28: compute_metrics exports checkpoint_constrained_promotable instead of
        constrained_promotable; Trainer's eval_ prefix removes the ambiguous
        eval_constrained_promotable key from test_metrics/eval_history at the source.
  [E3]  cell 29: score_split adds a margin_top1_top2 column (top1 - top2 probability).
  [E4]  cell 29: diagnostic helpers — threshold_sweep_diagnostics (0.80-0.99 valid
        false-block + per-label objection precision), confidence_margin_diagnostics
        (conf x margin false-objection grid), per_source_diagnostic_gates (>=100-row
        sources vs accuracy/FO/wrong-tool-recall gates, diagnostic-only), and
        derive_promotion_status (single source of truth: "blocked" or
        "promotable_pending_replay" plus blocked_reasons[]).
  [E5]  cell 29: the test-only valid_false_block_rates sweep is replaced by the new
        diagnostics on both validation and test; valid_false_block_rates survives as a
        backward-compat alias derived from the test sweep.
  [E6]  cell 29: promotion_gate_report embeds threshold_sweep,
        confidence_margin_diagnostics, per_source_diagnostics, and the derived
        promotion_status / blocked_reasons / artifact_promotable.
  [E7]  cell 29: test_metrics.json carries the authoritative status fields, and
        assert_promotion_consistency raises if any exported field claims
        promotable=True while the gate report is blocked.
  [E8]  cell 29: high_confidence_mistakes.jsonl exports wrong predictions at
        confidence >= 0.99 for manual audit; training_run_summary carries the
        authoritative status fields.
  [E9]  cell 31: thresholds.json and candidate_thresholds.json carry the authoritative
        status fields.
  [E10] cell 38: artifact_manifest.json carries the authoritative status fields and is
        checked by assert_promotion_consistency (manifest and embedded run summary).

All new gates/sweeps are diagnostic-only: the blocking gate list in
evaluate_promotion_gates is unchanged.
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
# E1/E2: strict checkpoint boolean + unambiguous metric key (cell 28)
# ---------------------------------------------------------------------------
E1_OLD = r'''    return float(score), bool(competent and fo_ok)'''

E1_NEW = r'''    # The tiered score keeps the 2.5x discard ceiling so checkpoint selection and
    # early stopping are unchanged; the boolean uses the strict gate set so a per-eval
    # flag can never claim promotability that promotion_gate_report would deny.
    strict_promotable = bool(
        valid_false_objection_90 <= CHECKPOINT_FALSE_OBJECTION_90_GATE
        and valid_recall >= CHECKPOINT_VALID_RECALL_GATE
        and wrong_tool_precision >= CHECKPOINT_WRONG_TOOL_PRECISION_GATE
    )
    return float(score), strict_promotable'''

E2_OLD = r'''        "constrained_promotable": constrained_promotable,'''

E2_NEW = r'''        "checkpoint_constrained_promotable": constrained_promotable,'''


def patch_e1_e2(cells):
    idx = find_cell_by_marker(cells, "def compute_metrics(eval_pred):")
    assert idx is not None, "[E1/E2] compute_metrics cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, E1_OLD, E1_NEW, "E1")
    src = replace_once(src, E2_OLD, E2_NEW, "E2")
    set_cell_src(cells[idx], src)
    print(f"  [E1] strict checkpoint boolean installed in cell {idx}")
    print(f"  [E2] checkpoint_constrained_promotable key rename in cell {idx}")


# ---------------------------------------------------------------------------
# E3: top1-top2 margin column in score_split (cell 29)
# ---------------------------------------------------------------------------
E3_OLD = r'''    raw["confidence"] = probs.max(axis=-1)'''

E3_NEW = r'''    raw["confidence"] = probs.max(axis=-1)
    _sorted_probs = np.sort(probs, axis=-1)
    raw["margin_top1_top2"] = _sorted_probs[:, -1] - _sorted_probs[:, -2]'''


# ---------------------------------------------------------------------------
# E4: diagnostic helpers + promotion status derivation (cell 29)
# ---------------------------------------------------------------------------
E4_OLD = r'''def evaluate_promotion_gates(scored: pd.DataFrame, split_name: str) -> Dict[str, Any]:'''

E4_HELPERS = r'''THRESHOLD_SWEEP_POINTS = [0.80, 0.90, 0.95, 0.98, 0.99]
CONFIDENCE_MARGIN_GRID = [(conf, margin) for conf in (0.95, 0.98, 0.99) for margin in (0.0, 0.10, 0.15)]
PER_SOURCE_DIAG_MIN_ROWS = 100
PER_SOURCE_DIAG_ACCURACY_GATE = 0.90
PER_SOURCE_DIAG_WRONG_TOOL_RECALL_GATE = 0.80


def threshold_sweep_diagnostics(scored: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Diagnostic-only sweep of valid false-block rates and per-label objection
    precision per confidence threshold. Never added to the blocking gate list; the
    0.90 gate in evaluate_promotion_gates stays the only blocking false-objection
    threshold."""
    valid_rows = scored[scored["true_label"] == "valid"]
    non_valid_labels = [label for label in LABELS if label != "valid"]
    thresholds_out: Dict[str, Any] = {}
    for threshold in THRESHOLD_SWEEP_POINTS:
        false_blocks = valid_rows[(valid_rows["pred_label"] != "valid") & (valid_rows["confidence"] >= threshold)]
        per_label_objections: Dict[str, Any] = {}
        for label in non_valid_labels:
            predicted = scored[(scored["pred_label"] == label) & (scored["confidence"] >= threshold)]
            per_label_objections[label] = {
                "predicted_rows": int(len(predicted)),
                "objection_precision": float((predicted["true_label"] == label).mean()) if len(predicted) else None,
            }
        thresholds_out[str(threshold)] = {
            "valid_false_block": {
                "false_blocks": int(len(false_blocks)),
                "valid_rows": int(len(valid_rows)),
                "rate": float(len(false_blocks) / len(valid_rows)) if len(valid_rows) else None,
            },
            "per_label_objections": per_label_objections,
        }
    return {"split": split_name, "thresholds": thresholds_out}


def confidence_margin_diagnostics(scored: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Diagnostic-only valid false-objection rates when an objection additionally
    requires a top1-top2 probability margin. Uses uncalibrated confidence, matching
    how the FO@0.90 gate measures."""
    valid_rows = scored[scored["true_label"] == "valid"]
    has_margin = "margin_top1_top2" in scored.columns
    grid: Dict[str, Any] = {}
    if has_margin:
        for conf, margin in CONFIDENCE_MARGIN_GRID:
            objections = valid_rows[
                (valid_rows["pred_label"] != "valid")
                & (valid_rows["confidence"] >= conf)
                & (valid_rows["margin_top1_top2"] >= margin)
            ]
            grid[f"conf_{conf:.2f}_margin_{margin:.2f}"] = {
                "false_objections": int(len(objections)),
                "valid_rows": int(len(valid_rows)),
                "rate": float(len(objections) / len(valid_rows)) if len(valid_rows) else None,
            }
    return {"split": split_name, "margin_column_present": bool(has_margin), "grid": grid}


def per_source_diagnostic_gates(scored: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Diagnostic-only per-source gates; never added to the blocking promotion gates.
    Sources under PER_SOURCE_DIAG_MIN_ROWS rows are reported as insufficient_support
    instead of being judged on noise-level metrics."""
    sources_out: Dict[str, Any] = {}
    for source_name, src_df in scored.groupby("source"):
        total = int(len(src_df))
        if total < PER_SOURCE_DIAG_MIN_ROWS:
            sources_out[str(source_name)] = {"status": "insufficient_support", "rows": total}
            continue
        accuracy = float((src_df["pred_label"] == src_df["true_label"]).mean())
        valid_rows_src = src_df[src_df["true_label"] == "valid"]
        fo_rate = None
        if len(valid_rows_src):
            fo_rows = valid_rows_src[(valid_rows_src["pred_label"] != "valid") & (valid_rows_src["confidence"] >= 0.90)]
            fo_rate = float(len(fo_rows) / len(valid_rows_src))
        wrong_tool_rows_src = src_df[src_df["true_label"] == "wrong_tool_semantic"]
        wrong_tool_recall = None
        if len(wrong_tool_rows_src):
            wrong_tool_recall = float((wrong_tool_rows_src["pred_label"] == "wrong_tool_semantic").mean())
        gates = {
            "accuracy": {
                "value": accuracy,
                "threshold": f">= {PER_SOURCE_DIAG_ACCURACY_GATE}",
                "passed": accuracy >= PER_SOURCE_DIAG_ACCURACY_GATE,
            },
            "valid_false_objection_at_0_90": {
                "value": fo_rate,
                "threshold": f"<= {VALID_FALSE_OBJECTION_90_GATE}",
                "passed": None if fo_rate is None else fo_rate <= VALID_FALSE_OBJECTION_90_GATE,
            },
            "wrong_tool_semantic_recall": {
                "value": wrong_tool_recall,
                "threshold": f">= {PER_SOURCE_DIAG_WRONG_TOOL_RECALL_GATE}",
                "passed": None if wrong_tool_recall is None else wrong_tool_recall >= PER_SOURCE_DIAG_WRONG_TOOL_RECALL_GATE,
            },
        }
        sources_out[str(source_name)] = {
            "status": "evaluated",
            "rows": total,
            "gates": gates,
            "passed": all(gate["passed"] for gate in gates.values() if gate["passed"] is not None),
        }
    return {
        "split": split_name,
        "note": "diagnostic_only; never added to blocking promotion gates",
        "min_rows": PER_SOURCE_DIAG_MIN_ROWS,
        "sources": sources_out,
    }


def derive_promotion_status(gate_report: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], bool]:
    """Single source of truth for artifact promotability, derived only from the
    promotion gate report. Status is never plain "promotable": ONNX parity, shadow
    replay, and advisory replay gates live outside this notebook."""
    blocked_reasons = [
        {"split": split, "gate": gate["name"], "value": gate["value"], "threshold": gate["threshold"]}
        for split in ("validation", "test")
        for gate in gate_report[split]["gates"]
        if not gate["passed"]
    ]
    artifact_promotable = bool(gate_report["validation"]["passed"] and gate_report["test"]["passed"])
    status = "promotable_pending_replay" if artifact_promotable else "blocked"
    return status, blocked_reasons, artifact_promotable'''

E4_NEW = E4_HELPERS + "\n\n\n" + E4_OLD


# ---------------------------------------------------------------------------
# E5: both-split diagnostics replace the test-only sweep loop (cell 29)
# ---------------------------------------------------------------------------
E5_OLD = r'''valid_false_block_rates = {}
for threshold in [0.80, 0.90, 0.95, 0.98, 0.99]:
    valid_true = test_scored[test_scored["true_label"] == "valid"]
    if len(valid_true):
        false_blocks = valid_true[(valid_true["pred_label"] != "valid") & (valid_true["confidence"] >= threshold)]
        rate = len(false_blocks) / len(valid_true)
        valid_false_block_rates[str(threshold)] = {"false_blocks": int(len(false_blocks)), "valid_rows": int(len(valid_true)), "rate": float(rate)}
        print(f"valid-call false block rate @ {threshold:.2f}: {len(false_blocks)}/{len(valid_true)} = {rate:.4f}")'''

E5_NEW = r'''threshold_sweep = {
    "validation": threshold_sweep_diagnostics(valid_scored, "validation"),
    "test": threshold_sweep_diagnostics(test_scored, "test"),
}
confidence_margin_report = {
    "validation": confidence_margin_diagnostics(valid_scored, "validation"),
    "test": confidence_margin_diagnostics(test_scored, "test"),
}
per_source_diagnostics = {
    "validation": per_source_diagnostic_gates(valid_scored, "validation"),
    "test": per_source_diagnostic_gates(test_scored, "test"),
}
# Backward-compat alias consumed by training_metrics.json / training_run_summary.json.
valid_false_block_rates = {
    threshold_key: dict(entry["valid_false_block"])
    for threshold_key, entry in threshold_sweep["test"]["thresholds"].items()
    if entry["valid_false_block"]["rate"] is not None
}
for threshold_key, block in valid_false_block_rates.items():
    print(f"valid-call false block rate @ {float(threshold_key):.2f}: {block['false_blocks']}/{block['valid_rows']} = {block['rate']:.4f}")'''


# ---------------------------------------------------------------------------
# E6: embed diagnostics + derived status in promotion_gate_report (cell 29)
# ---------------------------------------------------------------------------
E6_OLD = r'''    "validation": evaluate_promotion_gates(valid_scored, "validation"),
    "test": evaluate_promotion_gates(test_scored, "test"),
}'''

E6_NEW = r'''    "validation": evaluate_promotion_gates(valid_scored, "validation"),
    "test": evaluate_promotion_gates(test_scored, "test"),
    "threshold_sweep": threshold_sweep,
    "confidence_margin_diagnostics": confidence_margin_report,
    "per_source_diagnostics": per_source_diagnostics,
}
promotion_status, blocked_reasons, artifact_promotable = derive_promotion_status(promotion_gate_report)
promotion_gate_report["promotion_status"] = promotion_status
promotion_gate_report["blocked_reasons"] = blocked_reasons
promotion_gate_report["artifact_promotable"] = artifact_promotable'''


# ---------------------------------------------------------------------------
# E7: authoritative test_metrics + consistency assertion (cell 29)
# ---------------------------------------------------------------------------
E7_OLD = r'''(DATA_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))'''

E7_NEW = r'''# Authoritative promotability comes only from promotion_gate_report (derived above);
# eval_checkpoint_constrained_promotable is a checkpoint-selection signal, not artifact status.
test_metrics.pop("eval_constrained_promotable", None)
test_metrics["promotion_status"] = promotion_status
test_metrics["blocked_reasons"] = blocked_reasons
test_metrics["artifact_promotable"] = artifact_promotable


def assert_promotion_consistency(claim_label: str, claimed_promotable: Any) -> None:
    if bool(claimed_promotable) and not artifact_promotable:
        raise RuntimeError(
            f"Promotion consistency violation: {claim_label} claims promotable=True while "
            f"promotion_gate_report is blocked: {json.dumps(blocked_reasons)}"
        )


assert_promotion_consistency("test_metrics.artifact_promotable", test_metrics["artifact_promotable"])
if "eval_constrained_promotable" in test_metrics:
    raise RuntimeError("Ambiguous eval_constrained_promotable key must not be exported")
(DATA_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))'''


# ---------------------------------------------------------------------------
# E8: high-confidence mistake export + run-summary status fields (cell 29)
# ---------------------------------------------------------------------------
E8A_OLD = r'''print("High-confidence mistakes:")
display(mistakes.head(25))'''

E8A_NEW = r'''print("High-confidence mistakes:")
display(mistakes.head(25))

# Export wrong predictions at very high confidence across validation+test for manual
# audit (forge_argument_semantic showed >= 0.99 confidence on mispredicted rows).
HIGH_CONF_MISTAKE_THRESHOLD = 0.99
high_confidence_mistakes = pd.concat(
    [valid_scored.assign(eval_split="validation"), test_scored.assign(eval_split="test")],
    ignore_index=True,
)
high_confidence_mistakes = high_confidence_mistakes[
    (~high_confidence_mistakes["correct"].astype(bool))
    & (high_confidence_mistakes["confidence"] >= HIGH_CONF_MISTAKE_THRESHOLD)
][["eval_split", "source", "true_label", "pred_label", "confidence"]]
high_confidence_mistakes.to_json(DATA_DIR / "high_confidence_mistakes.jsonl", orient="records", lines=True, force_ascii=False)
print(f"High-confidence (>= {HIGH_CONF_MISTAKE_THRESHOLD}) mistakes exported: {len(high_confidence_mistakes)} rows")'''

E8B_OLD = r'''    training_run_summary["promotion_gate_report"] = promotion_gate_report'''

E8B_NEW = r'''    training_run_summary["promotion_gate_report"] = promotion_gate_report
    training_run_summary["promotion_status"] = promotion_status
    training_run_summary["blocked_reasons"] = blocked_reasons
    training_run_summary["artifact_promotable"] = artifact_promotable'''


def patch_e3_to_e8(cells):
    idx = find_cell_by_marker(cells, "def evaluate_promotion_gates(")
    assert idx is not None, "[E3-E8] promotion gates cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, E3_OLD, E3_NEW, "E3")
    src = replace_once(src, E4_OLD, E4_NEW, "E4")
    src = replace_once(src, E5_OLD, E5_NEW, "E5")
    src = replace_once(src, E6_OLD, E6_NEW, "E6")
    src = replace_once(src, E7_OLD, E7_NEW, "E7")
    src = replace_once(src, E8A_OLD, E8A_NEW, "E8a")
    src = replace_once(src, E8B_OLD, E8B_NEW, "E8b")
    set_cell_src(cells[idx], src)
    print(f"  [E3] margin_top1_top2 column installed in cell {idx}")
    print(f"  [E4] diagnostic helpers + derive_promotion_status installed in cell {idx}")
    print(f"  [E5] both-split threshold/margin/per-source diagnostics installed in cell {idx}")
    print(f"  [E6] promotion_gate_report status + diagnostics embedded in cell {idx}")
    print(f"  [E7] authoritative test_metrics + consistency assertion installed in cell {idx}")
    print(f"  [E8] high-confidence mistake export + run-summary status in cell {idx}")


# ---------------------------------------------------------------------------
# E9: status fields in thresholds files (cell 31)
# ---------------------------------------------------------------------------
E9_OLD = r'''calibration_report = {'''

E9_NEW = r'''if "promotion_gate_report" in globals():
    for _threshold_doc in (thresholds, candidate_thresholds):
        _threshold_doc["promotion_status"] = promotion_gate_report.get("promotion_status")
        _threshold_doc["blocked_reasons"] = promotion_gate_report.get("blocked_reasons")
        _threshold_doc["artifact_promotable"] = bool(promotion_gate_report.get("artifact_promotable", False))

calibration_report = {'''


def patch_e9(cells):
    idx = find_cell_by_marker(cells, "def choose_threshold_for_label(")
    assert idx is not None, "[E9] threshold calibration cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, E9_OLD, E9_NEW, "E9")
    set_cell_src(cells[idx], src)
    print(f"  [E9] thresholds/candidate_thresholds status fields installed in cell {idx}")


# ---------------------------------------------------------------------------
# E10: status fields + final assertion in artifact manifest (cell 38)
# ---------------------------------------------------------------------------
E10_OLD = r'''(ONNX_DIR / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2))'''

E10_NEW = r'''if "promotion_gate_report" in globals():
    artifact_manifest["promotion_status"] = promotion_gate_report.get("promotion_status")
    artifact_manifest["blocked_reasons"] = promotion_gate_report.get("blocked_reasons")
    artifact_manifest["artifact_promotable"] = bool(promotion_gate_report.get("artifact_promotable", False))
    artifact_manifest["high_confidence_mistakes_file"] = "high_confidence_mistakes.jsonl"
    if "assert_promotion_consistency" in globals():
        assert_promotion_consistency("artifact_manifest.artifact_promotable", artifact_manifest["artifact_promotable"])
        _embedded_summary = artifact_manifest.get("training_run_summary") or {}
        assert_promotion_consistency(
            "artifact_manifest.training_run_summary.artifact_promotable",
            _embedded_summary.get("artifact_promotable"),
        )
(ONNX_DIR / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2))'''


def patch_e10(cells):
    idx = find_cell_by_marker(cells, "Build artifact manifest")
    assert idx is not None, "[E10] artifact manifest cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, E10_OLD, E10_NEW, "E10")
    set_cell_src(cells[idx], src)
    print(f"  [E10] manifest status fields + consistency assertion installed in cell {idx}")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
# Observed validation metrics from the v5d-era failed run (epochs 1-3), kept to prove
# the tiered score ordering and epoch-3 argmax are unchanged by E1.
OBSERVED_EPOCHS = [
    # (valid_recall, wrong_tool_precision, wrong_tool_recall, present_f1, wrong_args_recall, fo_90)
    (0.9550970874, 0.8111587983, 0.8852459016, 0.5308444585, 0.0839694656, 0.0012135922),
    (0.9356796117, 0.9818181818, 0.8852459016, 0.7598799283, 0.9720101781, 0.0266990291),
    (0.9684466019, 0.9603960396, 0.9086651054, 0.9201119436, 0.9643765903, 0.0133495146),
]


def smoke_test_strict_boolean(cells):
    """Apply E1 to the live cell-28 source and exec the resulting helper."""
    from typing import Tuple
    idx = find_cell_by_marker(cells, "def checkpoint_selection_score(")
    assert idx is not None, "[smoke E1] checkpoint selection cell not found"
    patched = replace_once(cell_src(cells[idx]), E1_OLD, E1_NEW, "E1 smoke")
    start = patched.index("CHECKPOINT_VALID_RECALL_GATE = 0.94")
    end = patched.index("def compute_metrics(eval_pred):")
    ns = {"Tuple": Tuple}
    exec(patched[start:end], ns)
    score = ns["checkpoint_selection_score"]
    # The observed contradiction: test fo 0.01147 sits inside the 2.5x ceiling (tier 2
    # score) but above the 0.005 gate, so the boolean must now be False (was True).
    s_bug, p_bug = score(0.95, 0.989, 0.8049, 0.90, 0.90, 0.011472275334608031)
    assert s_bug >= 100.0, f"contradiction case must stay tier 2, got {s_bug}"
    assert p_bug is False, "strict boolean must deny fo above the 0.005 gate"
    # Strictly passing checkpoint stays promotable.
    s_good, p_good = score(0.95, 0.95, 0.92, 0.92, 0.95, 0.004)
    assert p_good is True and s_good >= 100.0
    # fo passes but wrong_tool precision fails: still tier 2, boolean False.
    _, p_wtp = score(0.95, 0.85, 0.90, 0.90, 0.90, 0.004)
    assert p_wtp is False, "strict boolean must deny wrong_tool precision below 0.90"
    # v5d replay: ordering and epoch-3 argmax unchanged, all booleans False.
    results = [score(*epoch) for epoch in OBSERVED_EPOCHS]
    scores = [s for s, _ in results]
    assert scores[0] < scores[1] < scores[2], f"scores must increase monotonically: {scores}"
    assert max(range(3), key=lambda i: scores[i]) == 2, "epoch 3 must still be selected"
    assert [p for _, p in results] == [False, False, False]
    print(f"  [smoke] E1 strict boolean: contradiction case denied at score {s_bug:.3f}, "
          f"v5d epoch ordering unchanged {['%.3f' % s for s in scores]}")


def _exec_e4_helpers(ns_extra):
    """Exec the E4 helper block with a controlled namespace."""
    from typing import Any, Dict, List, Tuple
    ns = {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple}
    ns.update(ns_extra)
    if "pd" not in ns:
        class _FakePD:  # annotations only need the attribute to resolve
            DataFrame = object
        ns["pd"] = _FakePD
    exec(E4_HELPERS, ns)
    return ns


def smoke_test_promotion_status():
    """derive_promotion_status + assert_promotion_consistency on synthetic reports."""
    ns = _exec_e4_helpers({})
    derive = ns["derive_promotion_status"]
    failed_report = {
        "validation": {"passed": True, "gates": [
            {"name": "valid_recall", "passed": True, "value": 0.97, "threshold": ">= 0.94"},
        ]},
        "test": {"passed": False, "gates": [
            {"name": "valid_recall", "passed": True, "value": 0.95, "threshold": ">= 0.94"},
            {"name": "valid_false_objection_at_0_90", "passed": False,
             "value": 0.011472275334608031, "threshold": "<= 0.005"},
        ]},
    }
    status, reasons, promotable = derive(failed_report)
    assert status == "blocked" and promotable is False
    assert reasons == [{
        "split": "test",
        "gate": "valid_false_objection_at_0_90",
        "value": 0.011472275334608031,
        "threshold": "<= 0.005",
    }], reasons
    passing_report = {
        "validation": {"passed": True, "gates": [{"name": "valid_recall", "passed": True, "value": 0.97, "threshold": ">= 0.94"}]},
        "test": {"passed": True, "gates": [{"name": "valid_recall", "passed": True, "value": 0.96, "threshold": ">= 0.94"}]},
    }
    status2, reasons2, promotable2 = derive(passing_report)
    assert (status2, reasons2, promotable2) == ("promotable_pending_replay", [], True)
    # The E7 assertion helper must raise on a contradictory promotable claim.
    from typing import Any
    assert_src = E7_NEW[E7_NEW.index("def assert_promotion_consistency"):E7_NEW.index("\n\n\nassert_promotion_consistency")]
    blocked_ns = {"json": json, "Any": Any, "artifact_promotable": False, "blocked_reasons": reasons}
    exec(assert_src, blocked_ns)
    check = blocked_ns["assert_promotion_consistency"]
    check("test_metrics.artifact_promotable", False)  # consistent: no raise
    try:
        check("artifact_manifest.artifact_promotable", True)
    except RuntimeError as exc:
        assert "Promotion consistency violation" in str(exc)
    else:
        raise AssertionError("contradictory promotable claim must raise")
    print("  [smoke] E4/E7 promotion status: blocked report yields one test reason; "
          "contradictory claim raises")


def smoke_test_sweep_and_margin():
    """threshold_sweep_diagnostics + confidence_margin_diagnostics + alias extraction."""
    try:
        import pandas as pd
    except ImportError:
        print("  [smoke] E4/E5 sweep/margin skipped (pandas not installed locally)")
        return
    labels = ["valid", "wrong_tool_semantic", "wrong_arguments_semantic",
              "tool_not_needed", "needs_clarification", "deterministic_invalid"]
    rows = []
    for _ in range(8):
        rows.append({"true_label": "valid", "pred_label": "valid", "confidence": 0.97, "margin_top1_top2": 0.9, "source": "s"})
    rows.append({"true_label": "valid", "pred_label": "wrong_tool_semantic", "confidence": 0.995, "margin_top1_top2": 0.5, "source": "s"})
    rows.append({"true_label": "valid", "pred_label": "wrong_arguments_semantic", "confidence": 0.96, "margin_top1_top2": 0.05, "source": "s"})
    for _ in range(3):
        rows.append({"true_label": "wrong_tool_semantic", "pred_label": "wrong_tool_semantic", "confidence": 0.99, "margin_top1_top2": 0.4, "source": "s"})
    rows.append({"true_label": "wrong_tool_semantic", "pred_label": "valid", "confidence": 0.6, "margin_top1_top2": 0.1, "source": "s"})
    scored = pd.DataFrame(rows)
    ns = _exec_e4_helpers({"pd": pd, "LABELS": labels, "VALID_FALSE_OBJECTION_90_GATE": 0.005})
    sweep = ns["threshold_sweep_diagnostics"](scored, "test")
    assert sweep["thresholds"]["0.99"]["valid_false_block"] == {"false_blocks": 1, "valid_rows": 10, "rate": 0.1}
    assert sweep["thresholds"]["0.95"]["valid_false_block"]["false_blocks"] == 2
    wts_095 = sweep["thresholds"]["0.95"]["per_label_objections"]["wrong_tool_semantic"]
    assert wts_095 == {"predicted_rows": 4, "objection_precision": 0.75}, wts_095
    margins = ns["confidence_margin_diagnostics"](scored, "test")
    assert margins["margin_column_present"] is True
    assert margins["grid"]["conf_0.95_margin_0.00"]["false_objections"] == 2
    assert margins["grid"]["conf_0.95_margin_0.15"]["false_objections"] == 1
    assert margins["grid"]["conf_0.99_margin_0.15"]["false_objections"] == 1
    alias = {
        threshold_key: dict(entry["valid_false_block"])
        for threshold_key, entry in sweep["thresholds"].items()
        if entry["valid_false_block"]["rate"] is not None
    }
    assert set(alias) == {"0.8", "0.9", "0.95", "0.98", "0.99"}
    assert alias["0.99"] == {"false_blocks": 1, "valid_rows": 10, "rate": 0.1}
    print("  [smoke] E4/E5 sweep/margin: rates, per-label precision, margin filtering, "
          "and back-compat alias verified")


def smoke_test_per_source_gates():
    """per_source_diagnostic_gates over passing, failing, and tiny sources."""
    try:
        import pandas as pd
    except ImportError:
        print("  [smoke] E4 per-source skipped (pandas not installed locally)")
        return
    labels = ["valid", "wrong_tool_semantic"]
    rows = []
    for _ in range(100):
        rows.append({"true_label": "valid", "pred_label": "valid", "confidence": 0.5, "source": "good"})
    for _ in range(20):
        rows.append({"true_label": "wrong_tool_semantic", "pred_label": "wrong_tool_semantic", "confidence": 0.5, "source": "good"})
    for _ in range(100):
        rows.append({"true_label": "valid", "pred_label": "valid", "confidence": 0.5, "source": "bad_wtr"})
    for _ in range(10):
        rows.append({"true_label": "wrong_tool_semantic", "pred_label": "wrong_tool_semantic", "confidence": 0.5, "source": "bad_wtr"})
    for _ in range(10):
        rows.append({"true_label": "wrong_tool_semantic", "pred_label": "valid", "confidence": 0.5, "source": "bad_wtr"})
    for _ in range(50):
        rows.append({"true_label": "valid", "pred_label": "valid", "confidence": 0.5, "source": "tiny"})
    scored = pd.DataFrame(rows)
    ns = _exec_e4_helpers({"pd": pd, "LABELS": labels, "VALID_FALSE_OBJECTION_90_GATE": 0.005})
    report = ns["per_source_diagnostic_gates"](scored, "test")
    assert report["sources"]["good"]["passed"] is True
    bad = report["sources"]["bad_wtr"]
    assert bad["passed"] is False
    assert bad["gates"]["wrong_tool_semantic_recall"]["value"] == 0.5
    assert bad["gates"]["accuracy"]["passed"] is True
    assert report["sources"]["tiny"] == {"status": "insufficient_support", "rows": 50}
    print("  [smoke] E4 per-source gates: pass/fail/insufficient_support verified")


# ---------------------------------------------------------------------------
def main():
    print(f"Loading notebook: {NB_PATH}")
    nb = load_nb()
    cells = nb["cells"]

    full_src = "\n".join(cell_src(c) for c in cells)
    for m in ("derive_promotion_status", "checkpoint_constrained_promotable",
              "per_source_diagnostic_gates", "margin_top1_top2",
              "threshold_sweep_diagnostics", "assert_promotion_consistency",
              "high_confidence_mistakes"):
        assert m not in full_src, f"v5e marker {m!r} already present; refusing to re-run"
    for m in ("checkpoint_selection_score", "QUANTIZED_PARITY_HARD_FAIL", "constrained_promotable"):
        assert m in full_src, f"prerequisite marker {m!r} missing; run patch_notebook_v5d first"

    print("\nRunning pre-apply smoke tests on patch constants...")
    smoke_test_strict_boolean(cells)
    smoke_test_promotion_status()
    smoke_test_sweep_and_margin()
    smoke_test_per_source_gates()

    print("\nApplying patches...")
    patch_e1_e2(cells)
    patch_e3_to_e8(cells)
    patch_e9(cells)
    patch_e10(cells)

    save_nb(nb)
    print(f"\nNotebook saved: {NB_PATH}")

    print("\nVerifying...")
    nb2 = load_nb()
    cells2 = nb2["cells"]
    full_src2 = "\n".join(cell_src(c) for c in cells2)
    checks = [
        ("strict_promotable = bool(", "strict checkpoint boolean"),
        ('"checkpoint_constrained_promotable": constrained_promotable,', "renamed metric key"),
        ('raw["margin_top1_top2"]', "margin column"),
        ("def threshold_sweep_diagnostics(", "threshold sweep helper"),
        ("def confidence_margin_diagnostics(", "confidence margin helper"),
        ("def per_source_diagnostic_gates(", "per-source diagnostic helper"),
        ("def derive_promotion_status(", "promotion status derivation"),
        ('promotion_gate_report["promotion_status"]', "report status embedding"),
        ("def assert_promotion_consistency(", "consistency assertion"),
        ('test_metrics["artifact_promotable"]', "authoritative test_metrics fields"),
        ('training_run_summary["artifact_promotable"]', "run summary status fields"),
        ('_threshold_doc["promotion_status"]', "thresholds status fields"),
        ('artifact_manifest["promotion_status"]', "manifest status fields"),
        ("high_confidence_mistakes.jsonl", "high-confidence mistake export"),
    ]
    all_ok = True
    for marker, label in checks:
        ok = marker in full_src2
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
        all_ok = all_ok and ok
    assert all_ok, "verification failed"
    assert '"constrained_promotable": constrained_promotable,' not in full_src2, "ambiguous metric key still present"
    assert "bool(competent and fo_ok)" not in full_src2, "old relaxed boolean still present"

    print("\nCompile-checking patched code cells...")
    for marker in ("def checkpoint_selection_score(", "def derive_promotion_status(",
                   "def choose_threshold_for_label(", "Build artifact manifest"):
        i = find_cell_by_marker(cells2, marker, cell_type="code")
        assert i is not None, f"patched cell with {marker!r} not found"
        try:
            compile(cell_src(cells2[i]), f"cell_{i}", "exec")
        except SyntaxError as exc:
            print(f"  [FAIL] cell {i}: {exc}")
            sys.exit(1)
        print(f"  [OK ] cell {i}")
    print("\nDone. All v5e patches applied and verified.")


if __name__ == "__main__":
    main()
