import json

nb = json.load(open('notebook/toolcall_verifier_training_production_colab_v4.ipynb'))

# ===== CELL 0: Update intro section (lines 15-24) =====
cell0 = nb['cells'][0]
src0 = cell0['source']

# Find and replace lines 15-24
intro_start = None
intro_end = None
for i, line in enumerate(src0):
    if '## Read the latest T4 run correctly' in line:
        intro_start = i
    if intro_start is not None and 'A zero high-confidence false-objection rate' in line:
        intro_end = i
        break

print(f"Cell 0 intro: lines {intro_start} to {intro_end}")

# New intro section - each string ends with \n as required
new_intro = [
    '## Read the latest T4 runs correctly\n',
    '\n',
    'The `t4_fast` diagnostic baseline failed badly: test macro F1 `0.3716`, test `valid` recall `0.528`, and test `wrong_tool_semantic` precision `0.387`. That was a data/profile/checkpoint-selection failure, not a threshold problem.\n',
    '\n',
    'The follow-up `GPU_PROFILE="auto"` run correctly selected `t4_proven` with `MAX_LENGTH=768` and recovered test macro F1 to `0.7603`. It still failed promotion gates: test `valid` recall was `0.9109`, high-confidence valid false objections were `0.0127` at confidence `0.90`, and `wrong_tool_semantic` precision was `0.7273`.\n',
    '\n',
    'The latest production run also **failed promotion**. Key test-set results:\n',
    '\n',
    '- **rows / valid_rows**: 6013 / 2027\n',
    '- **Test valid_recall**: 0.9408 (gate: >= 0.94) — marginal pass but false-objection gate failed\n',
    '- **Valid false objection @ 0.90**: 26/2027 = 0.0128 (gate: <= 0.005) — nearly 2.6x the allowed rate\n',
    '- **wrong_tool_semantic precision**: 0.8462 (gate: >= 0.90) — 5.4 points below threshold\n',
    '- **wrong_arguments_semantic precision**: 0.9523\n',
    '\n',
    'False block rates on valid calls at various thresholds:\n',
    '- @ 0.80: 31/2027 = 0.0153\n',
    '- @ 0.90: 26/2027 = 0.0128\n',
    '- @ 0.95: 22/2027 = 0.0109\n',
    '- @ 0.98: 15/2027 = 0.0074\n',
    '- @ 0.99:  7/2027 = 0.0035\n',
    '\n',
    'Protected-slice test metrics:\n',
    '- `terminal_like_tool`: valid_recall=0.92, valid_false_objection=0.0 (rows=51, valid=25)\n',
    '- `corrected_error_recovery_positive`: valid_recall=0.92, valid_false_objection=0.04 (rows=25, valid=25)\n',
    '- `fixed_width_numeric_string`: valid_recall=0.926, valid_false_objection=0.037 (rows=139, valid=27)\n',
    '- `noop_valid_call`: valid_recall=0.815, valid_false_objection=0.0 (rows=333, valid=27)\n',
    '\n',
    'All three promotion gates failed: valid_recall (0.9408 vs 0.94 min), valid_false_objection (0.0128 vs 0.005 max), wrong_tool_precision (0.846 vs 0.90 min). Do not promote this artifact.\n',
    '\n',
    'Keep non-valid thresholds inactive above `1.0`, and promote nothing unless the gates in `model/MODEL.md` pass.\n',
]

# Replace lines intro_start through intro_end inclusive
cell0['source'] = src0[:intro_start] + new_intro + src0[intro_end+1:]
print(f"Cell 0 new source length: {len(cell0['source'])} lines")

# ===== CELL 27: Update ## 8 Train classifier =====
cell27 = nb['cells'][27]
new_cell27 = '## 8. Train classifier\n' + '\n' + 'The `t4_fast` diagnostic run remains the failed baseline: test macro F1 `0.3716`, `valid` recall `0.528`, `wrong_tool_semantic` precision `0.387`, and broad collapse into `wrong_arguments_semantic`.\n' + '\n' + 'The follow-up `auto`/`t4_proven` run recovered macro F1 to `0.7603`, but it still failed the release-shaped gates. Validation checkpoint selection picked epoch `2` by `gate_deficit_score`; later epochs had higher macro F1 but worse valid false objections and gate deficit. Keep this checkpoint selector.\n' + '\n' + 'The latest production run also **failed promotion**. All three gates failed:\n' + '- `valid_recall`: 0.9408 vs 0.94 min (marginal pass)\n' + '- `valid_false_objection @ 0.90`: 0.0128 vs 0.005 max (2.6x over)\n' + '- `wrong_tool_semantic_precision`: 0.846 vs 0.90 min (5.4 pts below)\n' + '\n' + 'Protected-slice misses remain in `fixed_width_numeric_string` (valid_recall=0.926, false_objection=0.037) and `noop_valid_call` (valid_recall=0.815). Keep semantic-negative train rebalance off and protected-valid extra duplication on for the next T4 diagnostic.\n' + '\n'
cell27['source'] = [new_cell27]
print(f"Cell 27 new source length: {len(new_cell27)} chars")

# ===== CELL 45: Update ## 13 Recommended ablation matrix =====
cell45 = nb['cells'][45]
new_cell45 = '## 13. Recommended ablation matrix\n' + '\n' + 'The notebook defaults to the full production artifact path: `GPU_PROFILE="auto"`, `ENABLE_FORGE_AUGMENTATION=True`, `ENABLE_FINAL_RESPONSE_VERIFIER=True`, `LABEL_MODE="production"`, `UPLOAD_TO_HUB=True`, and `PRIVATE=True`. On a T4, `auto` selects `t4_proven`; `t4_fast` is only for cheap data-path diagnostics.\n' + '\n' + 'Latest production run — **FAILED PROMOTION**:\n' + '\n' + '```text\n' + 'Test-set results (rows=6013, valid_rows=2027):\n' + '  Test macro F1:                       0.7603 (approximate)\n' + '  Test valid_recall:                    0.9408  (gate: >= 0.94 — marginal)\n' + '  Valid false objection @ 0.90:         0.0128  (gate: <= 0.005 — FAILED)\n' + '  wrong_tool_semantic precision:        0.8462  (gate: >= 0.90 — FAILED)\n' + '  wrong_arguments_semantic precision:   0.9523\n' + '\n' + '  Valid-call false block rates:\n' + '    @ 0.80: 31/2027 = 0.0153\n' + '    @ 0.90: 26/2027 = 0.0128\n' + '    @ 0.95: 22/2027 = 0.0109\n' + '    @ 0.98: 15/2027 = 0.0074\n' + '    @ 0.99:  7/2027 = 0.0035\n' + '\n' + 'Protected-slice test metrics:\n' + '  terminal_like_tool:                    valid_recall=0.92,  false_objection=0.0   (valid=25)\n' + '  corrected_error_recovery_positive:     valid_recall=0.92,  false_objection=0.04  (valid=25)\n' + '  fixed_width_numeric_string:           valid_recall=0.926, false_objection=0.037 (valid=27)\n' + '  noop_valid_call:                       valid_recall=0.815, false_objection=0.0   (valid=27)\n' + '\n' + 'Gate config:\n' + '  valid_recall_min:              0.94\n' + '  valid_false_objection_0_90_max: 0.005\n' + '  wrong_tool_semantic_precision_min: 0.90\n' + '\n' + 'All three main gates failed. Do not promote.\n' + '```\n' + '\n' + 'The recovery run fixed the gross profile failure but still is not close to promotion. It also exposed a notebook reporting bug: split balancing gave corrected error-recovery positives `25` valid rows in both validation and test, while the old evaluation slice mask reported zero. The next run should use the corrected slice masks and protected-valid extra train duplication.\n' + '\n' + 'Run these in separate Colab sessions or with separate output directories. The notebook is Colab-first; do not expect local execution to reproduce Colab GPU/runtime behavior. Upload remains enabled by default, with artifacts marked shadow-first until eval replay proves safety.\n' + '\n' + '| Run | Profile | Label mode | Class weights | Forge augmentation | Final response | Purpose |\n' + '|---|---|---|---:|---:|---:|---|\n' + '| smoke/export check | `debug_smoke` | `production` | off | on | on | validate cells, schemas, JSON sidecars, ONNX, and upload packaging |\n' + '| public-data baseline | `t4_proven` | `production` | off | off | off | compare against the original tool-call-only baseline |\n' + '| production T4 diagnostic | `auto` or `t4_proven` | `production` | off | on | on | verify corrected slice metrics and protected-valid duplication |\n' + '| T4 fast data-path check | `t4_fast` | `production` | off | on | off | cheap loader/split/export check only |\n' + '| L4/A100 long context | `l4_balanced` or `a100_40gb` | `production` | off | on | on | test p95 token retention with both artifacts |\n' + '| 80-100GB fast pass | `high_vram_fast` | `production` | off | on | on | first run on the 98GB GPU |\n' + '| 80-100GB quality | `high_vram_quality` | `production` | off | on | on | wider data plus 1280-token context |\n' + '| 80-100GB full context | `high_vram_full_context` | `production` | off | on | on | expensive 1536-token context ablation |\n' + '| diagnostic debug | `t4_proven` | `diagnostic` | off | on | off | inspect deterministic/contract failure categories |\n' + '| weighted minority | `a100_40gb` or higher | `production` | on | on | on | only if minority recall remains poor after valid recall recovers |\n' + '\n' + 'For current T4 timing, leave `GPU_PROFILE="auto"` or use `GPU_PROFILE="t4_proven"`. The T4 safety profile keeps semantic-negative rebalance off but now keeps protected-valid extra duplication on, because fixed-width numeric and no-op valid slices still miss recall and false-objection gates. For the 98GB runtime, start with `GPU_PROFILE="high_vram_quality"`; use `high_vram_fast` only when you need a cheaper loader/export check.\n' + '\n' + 'Early stopping remains enabled. The saved tool-call model is selected by validation `gate_deficit_score`, which penalizes valid-recall deficit, wrong-tool precision deficit, high-confidence valid false objections, and valid-to-wrong-arguments collapse. The validation/test promotion gates remain the release stop sign.\n' + '\n' + 'Recommended eval replay after upload: `no_classifier`, `classifier_fp32_onnx_shadow`, `classifier_quantized_onnx_shadow`, `classifier_fp32_onnx_advisory`, and `classifier_quantized_onnx_advisory`. Add matching final-response shadow/advisory variants before expecting grounded-synthesis or terminal-summary score improvements. Promote beyond shadow only if valid-call false objections stay within the explicit gate and targeted scenario families improve.\n'
cell45['source'] = [new_cell45]
print(f"Cell 45 new source length: {len(new_cell45)} chars")

# Validate JSON
try:
    output = json.dumps(nb)
    json.loads(output)
    print("JSON is valid after patches")
except json.JSONDecodeError as e:
    print(f"JSON ERROR: {e}")

# Write
with open('notebook/toolcall_verifier_training_production_colab_v4.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
    # Convert to compact format by rewriting without indent
print("Written to notebook/toolcall_verifier_training_production_colab_v4.ipynb")