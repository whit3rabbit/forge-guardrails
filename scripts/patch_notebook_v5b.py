#!/usr/bin/env python3
"""
patch_notebook_v5b.py — second-round classifier training fixes for
toolcall_verifier_training_production_colab_v5.ipynb, applied in-place.

Run after patch_notebook_v5.py (one-shot; refuses to re-run).

Changes:
  [C0] v3-aware candidate payload extraction in cells 20/29 (bug fix: under the v3
       layout the extractors fed CANDIDATE_TOOL_SCHEMA text into json.loads, so all
       payload-derived protected-slice flags were silently False)
  [C1] schema-compatible wrong-tool generator replaces the random distractor in
       make_public_rows (cell 19), with an ambiguity filter for near-duplicate tools
  [C2] truncation hard guard (cell 25) + char-budget trim of OTHER_AVAILABLE_TOOLS
       in serialize_state_v3 (cell 9)
  [C3] scaled protected valid slices: 520 fixed-width + 520 error-recovery seeded
       scenarios with paired one-difference hard negatives (cell 14)
  [C4] needs_clarification support builder from underspecified public-request
       variants (cell 19)
  [C5] ablation markdown: seed x LR sweep matrix, corrected checkpoint-selection
       description (cell 45)
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
# C0: v3-aware candidate payload extraction (cells 20 and 29)
# ---------------------------------------------------------------------------
C0_OLD = r'''    raw = text.split(marker, 1)[1].strip()
    metadata_marker = "\n\nSCORING_METADATA:"
    if metadata_marker in raw:
        raw = raw.split(metadata_marker, 1)[0].strip()'''

C0_NEW = r'''    raw = text.split(marker, 1)[1].strip()
    # v3 layout: CANDIDATE_TOOL_SCHEMA follows the candidate call; strip it first or
    # json.loads receives trailing sections and every v3 row loses its payload flags.
    schema_marker = "\n\nCANDIDATE_TOOL_SCHEMA:"
    if schema_marker in raw:
        raw = raw.split(schema_marker, 1)[0].strip()
    metadata_marker = "\n\nSCORING_METADATA:"
    if metadata_marker in raw:
        raw = raw.split(metadata_marker, 1)[0].strip()'''


def patch_c0(cells):
    for fn_marker in ("def extract_candidate_payload_for_audit(", "def extract_candidate_payload("):
        idx = find_cell_by_marker(cells, fn_marker)
        assert idx is not None, f"[C0] cell with {fn_marker} not found"
        src = cell_src(cells[idx])
        src = replace_once(src, C0_OLD, C0_NEW, f"C0 {fn_marker}")
        set_cell_src(cells[idx], src)
        print(f"  [C0] v3-aware payload extraction patched in cell {idx} ({fn_marker.strip('(')})")


# ---------------------------------------------------------------------------
# C1: schema-compatible wrong-tool generator (cell 19)
# ---------------------------------------------------------------------------
C1_OLD_FN = r'''def wrong_tool_candidate(tools: List[Dict[str, Any]], gold_name: str, gold_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    others = [t for t in tools if t.get("name") != gold_name]
    if not others:
        return None
    t = random.choice(others)
    return {"name": t.get("name", "unknown_tool"), "arguments": gold_args or {}}'''

C1_NEW_FN = r'''# v5b: schema-compatible wrong-tool selection replaces the random distractor.
# Random distractors with verbatim gold args let the model shortcut on schema mismatch
# instead of intent, and near-duplicate tools on xLAM/ToolACE made many "wrong tool"
# labels noisy (several tools were semantically plausible).
WRONG_TOOL_AMBIGUITY_SKIP_THRESHOLD = 0.60  #@param {type:"number"}
WRONG_TOOL_MIN_COMPAT_SCORE = 0.15  #@param {type:"number"}
_WRONG_TOOL_RNG = random.Random(SEED + 9173)
_DESC_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "in", "on", "by", "with", "and", "or",
    "is", "are", "be", "this", "that", "it", "as", "at", "from", "into", "you", "your",
}


def _tool_props(tool: Dict[str, Any]) -> Dict[str, Any]:
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    return props if isinstance(props, dict) else {}


def _tool_required_set(tool: Dict[str, Any]) -> set:
    return set(required_args_for_tool(tool))


def _desc_tokens(text: Any) -> set:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text or "").lower())
    return {t for t in cleaned.split() if t not in _DESC_STOPWORDS}


def _desc_overlap(a: Any, b: Any) -> float:
    ta, tb = _desc_tokens(a), _desc_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _json_type_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _schema_type_compatible(value: Any, spec: Any) -> bool:
    if not isinstance(spec, dict):
        return True
    declared = spec.get("type")
    if not declared:
        return True
    declared = [str(d) for d in (declared if isinstance(declared, list) else [declared])]
    vt = _json_type_of(value)
    if vt == "integer" and "number" in declared:
        return True
    return vt in declared


def _fill_value_for_schema(spec: Any, rng: random.Random) -> Any:
    spec = spec if isinstance(spec, dict) else {}
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        return rng.choice(enum)
    if "default" in spec:
        return spec["default"]
    declared = spec.get("type")
    if isinstance(declared, list) and declared:
        declared = declared[0]
    if declared == "integer":
        return rng.randint(1, 9)
    if declared == "number":
        return float(rng.randint(1, 9))
    if declared == "boolean":
        return True
    if declared == "array":
        return []
    if declared == "object":
        return {}
    return "example"


def _ambiguity_score(gold_tool: Optional[Dict[str, Any]], cand_tool: Dict[str, Any]) -> float:
    """High score = the candidate tool is plausibly interchangeable with the gold tool,
    so labeling it wrong_tool_semantic would inject noise rather than signal."""
    if not gold_tool:
        return 0.0
    score = _desc_overlap(gold_tool.get("description"), cand_tool.get("description"))
    same_schema = (
        bool(_tool_props(cand_tool))
        and set(_tool_props(gold_tool)) == set(_tool_props(cand_tool))
        and _tool_required_set(gold_tool) == _tool_required_set(cand_tool)
    )
    if same_schema:
        score += 0.25
    return score


def _schema_compat_score(gold_args: Dict[str, Any], cand_tool: Dict[str, Any]) -> float:
    props = _tool_props(cand_tool)
    if not props:
        return 0.05
    carried = [k for k in gold_args if k in props]
    score = len(carried) / max(1, len(gold_args))
    if carried:
        typed = sum(1 for k in carried if _schema_type_compatible(gold_args[k], props.get(k)))
        score += 0.25 * (typed / len(carried))
    score -= 0.10 * len(_tool_required_set(cand_tool) - set(gold_args))
    return score


def _remap_args_onto_tool(gold_args: Dict[str, Any], cand_tool: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    props = _tool_props(cand_tool)
    if not props:
        return dict(gold_args)
    carried = {k: v for k, v in gold_args.items() if k in props}
    for req in sorted(_tool_required_set(cand_tool) - set(carried)):
        carried[req] = _fill_value_for_schema(props.get(req), rng)
    return carried


def schema_compatible_wrong_tool_candidate(
    tools: List[Dict[str, Any]], gold_name: str, gold_args: Dict[str, Any]
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Pick the most schema-compatible non-gold tool so the only mismatch is intent.

    Near-duplicate tools (high description overlap or identical schemas) are skipped.
    Name similarity is deliberately NOT filtered: get_/set_ and approve_/reject_
    pairs are exactly the hard negatives the classifier needs."""
    gold_tool = tool_by_name(tools, gold_name)
    gold_args = gold_args or {}
    scored = []
    for t in tools:
        name = t.get("name")
        if not name or name == gold_name:
            continue
        ambiguity = _ambiguity_score(gold_tool, t)
        if ambiguity >= WRONG_TOOL_AMBIGUITY_SKIP_THRESHOLD:
            continue
        compat = _schema_compat_score(gold_args, t)
        scored.append((compat, -ambiguity, str(name), t, ambiguity))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
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

C1_OLD_COUNTER = r'''    rows = []
    deterministic_counts = Counter()'''

C1_NEW_COUNTER = r'''    rows = []
    deterministic_counts = Counter()
    wrong_tool_counts: Dict[str, Counter] = {}'''

C1_OLD_CALLSITE = r'''            wrong = wrong_tool_candidate(tools, call["name"], call.get("arguments") or {})
            if wrong:
                rows.append(make_row(
                    source=source,
                    label="wrong_tool_semantic",
                    user_request=user_request,
                    tools=tools,
                    candidate=wrong,
                    rank_score=0.15,
                    metadata={"generator": "hard_negative", "negative_type": "wrong_tool", "gold_tool": call["name"]},
                    group_id=group_id,
                ))'''

C1_NEW_CALLSITE = r'''            wrong_pair = schema_compatible_wrong_tool_candidate(tools, call["name"], call.get("arguments") or {})
            if wrong_pair:
                wrong, wrong_info = wrong_pair
                wrong_tool_counts.setdefault(source, Counter())["kept"] += 1
                rows.append(make_row(
                    source=source,
                    label="wrong_tool_semantic",
                    user_request=user_request,
                    tools=tools,
                    candidate=wrong,
                    rank_score=0.15,
                    metadata={"generator": "hard_negative", "negative_type": "wrong_tool_schema_compatible", "gold_tool": call["name"], **wrong_info},
                    group_id=group_id,
                ))
            else:
                wrong_tool_counts.setdefault(source, Counter())["skipped"] += 1'''

C1_OLD_PRINT = r'''    print("Public deterministic negative rows kept:", dict(deterministic_counts))'''

C1_NEW_PRINT = r'''    print("Wrong-tool distractor selection by source (ambiguity-filtered):")
    for src_name in sorted(wrong_tool_counts):
        counts = wrong_tool_counts[src_name]
        total = counts["kept"] + counts["skipped"]
        skip_rate = counts["skipped"] / total if total else 0.0
        print(f"  {src_name}: kept={counts['kept']} skipped={counts['skipped']} skip_rate={skip_rate:.1%}")
    print("Public deterministic negative rows kept:", dict(deterministic_counts))'''


def patch_c1(cells):
    idx = find_cell_by_marker(cells, "def make_public_rows(")
    assert idx is not None, "[C1] make_public_rows cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, C1_OLD_FN, C1_NEW_FN, "C1 fn")
    src = replace_once(src, C1_OLD_COUNTER, C1_NEW_COUNTER, "C1 counter")
    src = replace_once(src, C1_OLD_CALLSITE, C1_NEW_CALLSITE, "C1 callsite")
    src = replace_once(src, C1_OLD_PRINT, C1_NEW_PRINT, "C1 print")
    set_cell_src(cells[idx], src)
    print(f"  [C1] schema-compatible wrong-tool generator installed in cell {idx}")


# ---------------------------------------------------------------------------
# C2a: char-budget trim in serialize_state_v3 (cell 9)
# ---------------------------------------------------------------------------
C2_V3_NEW = r'''SERIALIZE_V3_CHARS_PER_TOKEN = 3.5  # SentencePiece on JSON-ish text runs ~3-4 chars/token.


def serialize_state_v3(input_obj: Dict[str, Any]) -> str:
    """Candidate-first layout: candidate call and its schema appear before the tool list.
    OTHER_AVAILABLE_TOOLS (the lowest-value section) is char-budget trimmed so the
    high-value front sections plus SCORING_METADATA fit within MAX_LENGTH at
    ~SERIALIZE_V3_CHARS_PER_TOKEN chars/token. The cell 25 hard guard backstops
    an optimistic estimate.
    NOTE: v1 and v2 remain byte-stable; this is a new layout, not a replacement."""
    ws = input_obj["workflow_state"]
    metadata = input_obj.get("metadata") or {}
    candidate_schema = serialize_candidate_tool_schema(
        input_obj["available_tools"], input_obj["candidate_call"]
    )
    competing_sigs = serialize_competing_tool_signatures(
        input_obj["available_tools"], input_obj["candidate_call"]
    )
    front = f"""SCHEMA_VERSION:
{input_obj['schema_version']}

USER_REQUEST:
{input_obj['user_request']}

CANDIDATE_CALL:
{compact_json(input_obj['candidate_call'], 2400)}

CANDIDATE_TOOL_SCHEMA:
{candidate_schema}

WORKFLOW_STATE:
required_steps={ws.get('required_steps', [])}
completed_steps={ws.get('completed_steps', [])}
pending_steps={ws.get('pending_steps', [])}
terminal_tools={ws.get('terminal_tools', [])}
recent_errors={ws.get('recent_errors', [])}"""
    tail = f"""SCORING_METADATA:
scenario_family={_json_or_null(metadata.get('scenario_family'))}
requires_transform={_json_or_null(metadata.get('requires_transform'))}
requires_synthesis={_json_or_null(metadata.get('requires_synthesis'))}
requires_all_tool_facts={_json_or_null(metadata.get('requires_all_tool_facts'))}
must_acknowledge_missing_data={_json_or_null(metadata.get('must_acknowledge_missing_data'))}"""
    other_header = "\n\nOTHER_AVAILABLE_TOOLS:\n"
    char_budget = int(float(globals().get("MAX_LENGTH", 1024)) * SERIALIZE_V3_CHARS_PER_TOKEN)
    remaining = char_budget - len(front) - len(tail) - len(other_header) - 2
    sig_lines = [line for line in competing_sigs.split("\n") if line]
    kept_lines = []
    used = 0
    for line in sig_lines:
        cost = len(line) + 1
        if used + cost > remaining:
            break
        kept_lines.append(line)
        used += cost
    dropped = len(sig_lines) - len(kept_lines)
    if dropped > 0:
        kept_lines.append(f"...<+{dropped} more tools>")
    competing_block = "\n".join(kept_lines)
    return f"{front}{other_header}{competing_block}\n\n{tail}".strip()


'''


def patch_c2_serializer(cells):
    idx = find_cell_by_marker(cells, "def serialize_state_v3(")
    assert idx is not None, "[C2] serializer cell not found"
    src = cell_src(cells[idx])
    start = src.index("def serialize_state_v3(")
    end = src.index("def serialize_state_from_object")
    old_fn = src[start:end]
    assert 'OTHER_AVAILABLE_TOOLS:\n{competing_sigs}' in old_fn, "[C2] v3 body drifted from expected layout"
    assert "SERIALIZE_V3_CHARS_PER_TOKEN" not in src, "[C2] already applied"
    src = src[:start] + C2_V3_NEW + src[end:]
    set_cell_src(cells[idx], src)
    print(f"  [C2] char-budget serialize_state_v3 installed in cell {idx}")


# ---------------------------------------------------------------------------
# C2b: truncation hard guard (cell 25)
# ---------------------------------------------------------------------------
C2_GUARD_ANCHOR = 'print(json.dumps(tokenization_diagnostics, indent=2)[:6000])'

C2_GUARD_NEW = r'''

# v5b: truncation hard guard. Under the v3 candidate-first layout the candidate call
# must essentially never be truncated; treat anything above the threshold as a data
# layout bug, not a tolerable diagnostic.
CANDIDATE_TRUNCATION_HARD_FAIL_RATE = 0.005  #@param {type:"number"}
SCHEMA_MARKER = "CANDIDATE_TOOL_SCHEMA:\n"


def schema_marker_token_position(text: Any) -> Optional[int]:
    raw = str(text or "")
    if SCHEMA_MARKER not in raw:
        return None
    prefix = raw.split(SCHEMA_MARKER, 1)[0] + SCHEMA_MARKER
    return token_count(prefix)


if SERIALIZER_VERSION == "serialize_state_v3":
    schema_positions = token_diag_df["input_text"].map(schema_marker_token_position)
    schema_present = schema_positions.notna()
    schema_truncated = schema_present & schema_positions.map(
        lambda value: bool(pd.notna(value) and int(value) > int(MAX_LENGTH))
    )
    schema_truncated_rate = (
        float(schema_truncated.sum() / schema_present.sum()) if int(schema_present.sum()) else None
    )
    tokenization_diagnostics["schema_marker_truncated_rows"] = int(schema_truncated.sum())
    tokenization_diagnostics["schema_marker_truncated_rate"] = schema_truncated_rate
    tokenization_diagnostics["candidate_truncation_hard_fail_rate"] = float(CANDIDATE_TRUNCATION_HARD_FAIL_RATE)
    (DATA_DIR / "tokenization_diagnostics.json").write_text(
        json.dumps(tokenization_diagnostics, indent=2, ensure_ascii=False)
    )
    candidate_truncated_rate = float(tokenization_diagnostics.get("candidate_marker_truncated_rate") or 0.0)
    if candidate_truncated_rate > CANDIDATE_TRUNCATION_HARD_FAIL_RATE:
        raise RuntimeError(
            f"CANDIDATE_CALL marker truncated in {candidate_truncated_rate:.2%} of sampled rows "
            f"(hard limit {CANDIDATE_TRUNCATION_HARD_FAIL_RATE:.2%}, MAX_LENGTH={MAX_LENGTH}). "
            "Raise MAX_LENGTH or lower SERIALIZE_V3_CHARS_PER_TOKEN so the v3 char budget "
            "trims more of OTHER_AVAILABLE_TOOLS."
        )
    if schema_truncated_rate is not None and schema_truncated_rate > 0.02:
        print(
            f"WARNING: CANDIDATE_TOOL_SCHEMA truncated in {schema_truncated_rate:.2%} of sampled rows; "
            "the candidate schema is high-value context."
        )
    print(
        f"Truncation hard guard passed: candidate marker truncated rate "
        f"{candidate_truncated_rate:.3%} <= {CANDIDATE_TRUNCATION_HARD_FAIL_RATE:.3%}."
    )
else:
    print("Truncation hard guard skipped (serializer is not v3).")'''


def patch_c2_guard(cells):
    idx = find_cell_by_marker(cells, C2_GUARD_ANCHOR)
    assert idx is not None, "[C2] tokenization diagnostics cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, C2_GUARD_ANCHOR, C2_GUARD_ANCHOR + C2_GUARD_NEW, "C2 guard")
    set_cell_src(cells[idx], src)
    print(f"  [C2] truncation hard guard appended to cell {idx}")


# ---------------------------------------------------------------------------
# C3: scaled protected valid slices (cell 14)
# ---------------------------------------------------------------------------
C3_BUILDERS = r'''# ---------------------------------------------------------------------------
# v5b C3: Protected valid slices at scale (seeded generators).
# The hardcoded v5 lists above stay as-is; these add 500+ distinct valid rows per
# slice plus paired one-difference hard negatives so the slice gates measure real
# variation instead of duplicated rows.
# ---------------------------------------------------------------------------
FWNS_SCALED_SCENARIOS = 520  #@param {type:"integer"}
ER_SCALED_SCENARIOS = 520  #@param {type:"integer"}

_FWNS_DOMAINS = [
    ("order", "order", "order_id", "order ID"),
    ("invoice", "invoice", "invoice_id", "invoice ID"),
    ("shipment", "shipment", "shipment_id", "shipment ID"),
    ("ticket", "support ticket", "ticket_id", "ticket ID"),
    ("account", "account", "account_id", "account ID"),
    ("employee", "employee record", "employee_id", "employee ID"),
    ("batch", "transaction batch", "batch_id", "batch ID"),
    ("meter", "meter reading", "meter_id", "meter ID"),
    ("case", "case file", "case_id", "case ID"),
    ("device", "device", "device_id", "device ID"),
    ("bin", "warehouse bin", "bin_id", "bin ID"),
    ("policy", "insurance policy", "policy_id", "policy number"),
    ("loan", "loan application", "application_id", "application ID"),
    ("chart", "patient chart", "chart_id", "chart ID"),
    ("registration", "vehicle registration", "registration_id", "registration ID"),
]
_FWNS_VERBS = ["get", "lookup", "fetch", "retrieve", "load"]
_FWNS_PHRASES = [
    "Retrieve {np} {value}.",
    "Pull up {np} number {value} please.",
    "Get the details for {np} {value}.",
    "I need {np} {value}.",
    "Open {np} {value} for me.",
    "Show me {np} {value}.",
    "Look up {np} {value} in the system.",
    "Can you fetch {np} {value}?",
]
# Generalization extras: formats that are valid stringly-typed values but do NOT
# satisfy the zero-padded slice predicate; they broaden coverage without being
# counted on for slice support.
_FWNS_FORMAT_EXTRAS = [
    ("e164_phone", "customer phone", "phone_number", "E.164 phone number string with + prefix",
     lambda rng: "+1415555" + str(rng.randint(0, 9999)).zfill(4),
     lambda v: [("integer_instead_of_string", int(v[1:])), ("missing_plus_prefix", v[1:])]),
    ("iban", "bank account", "iban", "IBAN string including country prefix",
     lambda rng: "DE89" + str(rng.randint(0, 10 ** 16 - 1)).zfill(16),
     lambda v: [("missing_country_prefix", v[4:]), ("lowercase_prefix", v[:4].lower() + v[4:])]),
    ("sku_code", "product SKU", "sku", "SKU string in SKU-NNNNNN format",
     lambda rng: "SKU-" + str(rng.randint(1, 999999)).zfill(6),
     lambda v: [("bare_number_int", int(v.split("-", 1)[1])), ("missing_prefix", v.split("-", 1)[1])]),
    ("routing_number", "bank routing number", "routing_number", "9-digit routing number string",
     lambda rng: "0" + str(rng.randint(10 ** 7, 10 ** 8 - 1)),
     lambda v: [("integer_instead_of_string", int(v)), ("unpadded_string", str(int(v)))]),
]


def build_fixed_width_numeric_rows_scaled(n_scenarios: int = FWNS_SCALED_SCENARIOS) -> List[VerifierRow]:
    """Scaled fixed-width protected-valid coverage. Every core valid value is sampled
    below 10**(width-1) and zfilled, so it is guaranteed to start with '0' and fire
    has_fixed_width_numeric_string / the training protection flags."""
    rng = random.Random(SEED + 4242)
    rows: List[VerifierRow] = []
    fm = infer_scoring_metadata("fixed_width_numeric")

    def emit_scenario(i: int, user_request: str, tool_name: str, tool_desc: str,
                      param_name: str, param_desc: str, valid_value: Any,
                      negatives: List[Tuple[str, Any]]) -> None:
        tools = _fw_tools(tool_name, tool_desc, param_name, param_desc)
        required = [tool_name]
        terminal = ["respond"]
        group_id = stable_id("forge_fwns_v2", i, user_request, tool_name, param_name, valid_value)
        valid_meta = {
            "generator": "forge_fixed_width_numeric_v2",
            "scenario_family": "fixed_width_numeric",
            "source_kind": "synthetic_numeric_string",
            "corrected_positive": True,
            "valid_protection_fixed_width_numeric_string": True,
        }
        rows.append(make_row(
            "forge_fixed_width_numeric", "valid",
            user_request, tools, {"name": tool_name, "arguments": {param_name: valid_value}}, 1.0,
            valid_meta, required_steps=required, completed_steps=[], pending_steps=required,
            terminal_tools=terminal, group_id=group_id, scoring_metadata=fm,
        ))
        seen = {json.dumps(valid_value, default=str, sort_keys=True)}
        for neg_type, neg_val in negatives:
            nk = json.dumps(neg_val, default=str, sort_keys=True)
            if nk in seen:
                continue
            seen.add(nk)
            rows.append(make_row(
                "forge_fixed_width_numeric", "wrong_arguments_semantic",
                user_request, tools, {"name": tool_name, "arguments": {param_name: neg_val}}, 0.05,
                {"generator": "forge_fixed_width_numeric_v2", "scenario_family": "fixed_width_numeric",
                 "source_kind": "synthetic_numeric_string", "negative_type": f"fixed_width_{neg_type}",
                 "valid_counterpart": valid_value},
                required_steps=required, completed_steps=[], pending_steps=required,
                terminal_tools=terminal, group_id=group_id, scoring_metadata=fm,
            ))

    for i in range(int(n_scenarios)):
        noun, np_, param_name, param_noun = _FWNS_DOMAINS[i % len(_FWNS_DOMAINS)]
        verb = rng.choice(_FWNS_VERBS)
        width = rng.randint(3, 8)
        number = rng.randint(1, 10 ** (width - 1) - 1)
        valid_value = str(number).zfill(width)
        tool_name = f"{verb}_{noun}"
        tool_desc = f"{verb.capitalize()} a {np_}. The {param_noun} must be a zero-padded {width}-digit string."
        param_desc = f"{width}-digit zero-padded {param_noun} string."
        user_request = rng.choice(_FWNS_PHRASES).format(np=np_, value=valid_value)
        negatives = [("integer_instead_of_string", number), ("unpadded_string", str(number))]
        if rng.random() < 0.3:
            negatives.append(("over_padded", valid_value.zfill(width + 2)))
        emit_scenario(i, user_request, tool_name, tool_desc, param_name, param_desc, valid_value, negatives)
        if i % 5 == 4:
            fmt_name, fmt_np, fmt_param, fmt_param_desc, make_value, make_negs = (
                _FWNS_FORMAT_EXTRAS[(i // 5) % len(_FWNS_FORMAT_EXTRAS)]
            )
            extra_value = make_value(rng)
            extra_tool = f"verify_{fmt_param}"
            extra_desc = f"Verify a {fmt_np}. The value must be a {fmt_param_desc}."
            extra_request = rng.choice(_FWNS_PHRASES).format(np=fmt_np, value=extra_value)
            emit_scenario(10_000 + i, extra_request, extra_tool, extra_desc, fmt_param,
                          fmt_param_desc, extra_value, make_negs(extra_value))
    return rows


_ER_PREFIXES = [None, "ACC", "ORD", "REQ", "INV", "TKT"]
_ER_PHRASES = [
    "Retry: fetch {np} {value}. The previous call failed with an invalid {param} format.",
    "Fetch {np} {value} again; the last attempt passed {param} as a number and was rejected.",
    "The earlier {param} was malformed. Get {np} {value} with the corrected format.",
    "Previous request errored on {param}. Retrieve {np} {value} now.",
    "That failed because {param} was numeric. Please pull {np} {value} using the string form.",
]
_ER_DISTRACTORS = [
    ("search_records", "Search records by keyword query.",
     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
     lambda v: {"query": str(v)}),
    ("list_items", "List all available items.",
     {"type": "object", "properties": {}, "required": []},
     lambda v: {}),
    ("summarize", "Summarize previously fetched data.",
     {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
     lambda v: {"content": f"records for {v}"}),
    ("delete_record", "Delete a record by numeric ID.",
     {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
     lambda v: {"id": 7}),
]


def build_error_recovery_protected_rows_scaled(n_scenarios: int = ER_SCALED_SCENARIOS) -> List[VerifierRow]:
    """Scaled corrected-error-recovery protected-valid coverage: a prior call failed on
    argument format (recent_errors populated), the candidate is the corrected retry."""
    rng = random.Random(SEED + 7311)
    rows: List[VerifierRow] = []
    fm = infer_scoring_metadata("error_recovery")
    for i in range(int(n_scenarios)):
        noun, np_, param_name, param_noun = _FWNS_DOMAINS[i % len(_FWNS_DOMAINS)]
        verb = rng.choice(_FWNS_VERBS)
        prefix = rng.choice(_ER_PREFIXES)
        width = rng.randint(3, 6)
        number = rng.randint(1, 10 ** (width - 1) - 1)
        core = str(number).zfill(width)
        if prefix:
            valid_value = f"{prefix}-{core}"
            fmt_desc = f"{prefix}-" + "N" * width
            wrong_args = [
                ("integer_id", number),
                ("bare_number_str", core),
                ("wrong_prefix", f"{prefix.lower()}-{core}"),
            ]
        else:
            valid_value = core
            fmt_desc = f"zero-padded {width}-digit string"
            wrong_args = [("integer", number), ("unpadded_string", str(number))]
            if rng.random() < 0.3:
                wrong_args.append(("over_padded", core.zfill(width + 2)))
        tool_name = f"{verb}_{noun}"
        tool_desc = f"{verb.capitalize()} a {np_}. The {param_noun} must be a string in {fmt_desc} format."
        param_desc = f"{param_noun} string in {fmt_desc} format."
        user_request = rng.choice(_ER_PHRASES).format(np=np_, value=valid_value, param=param_name)
        recent_errors = [f"Error: {param_name} must be a string like '{valid_value}', got {number}."]
        d_name, d_desc, d_params, d_args = _ER_DISTRACTORS[i % len(_ER_DISTRACTORS)]
        tools = [
            {"name": tool_name, "description": tool_desc, "parameters": {
                "type": "object",
                "properties": {param_name: {"type": "string", "description": param_desc}},
                "required": [param_name]}},
            {"name": d_name, "description": d_desc, "parameters": d_params},
            {"name": "respond", "description": "Send final answer.", "parameters": {
                "type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
        ]
        required = [tool_name]
        terminal = ["respond"]
        group_id = stable_id("forge_er_v2", i, user_request, tool_name, param_name, valid_value)
        valid_meta = {
            "generator": "forge_error_recovery_protected_v2",
            "scenario_family": "error_recovery_scaled",
            "source_kind": "synthetic_error_recovery",
            "corrected_positive": True,
            "valid_protection_corrected_error_recovery": True,
        }
        rows.append(make_row(
            "forge_error_recovery_protected", "valid",
            user_request, tools, {"name": tool_name, "arguments": {param_name: valid_value}}, 1.0,
            valid_meta, required_steps=required, completed_steps=[], pending_steps=required,
            terminal_tools=terminal, recent_errors=recent_errors, group_id=group_id, scoring_metadata=fm,
        ))
        seen = {json.dumps(valid_value, default=str, sort_keys=True)}
        for neg_type, neg_val in wrong_args:
            nk = json.dumps(neg_val, default=str, sort_keys=True)
            if nk in seen:
                continue
            seen.add(nk)
            rows.append(make_row(
                "forge_error_recovery_protected", "wrong_arguments_semantic",
                user_request, tools, {"name": tool_name, "arguments": {param_name: neg_val}}, 0.05,
                {"generator": "forge_error_recovery_protected_v2", "scenario_family": "error_recovery_scaled",
                 "source_kind": "synthetic_error_recovery", "negative_type": f"er_{neg_type}",
                 "valid_counterpart": valid_value},
                required_steps=required, completed_steps=[], pending_steps=required,
                terminal_tools=terminal, recent_errors=recent_errors, group_id=group_id, scoring_metadata=fm,
            ))
        rows.append(make_row(
            "forge_error_recovery_protected", "wrong_tool_semantic",
            user_request, tools, {"name": d_name, "arguments": d_args(valid_value)}, 0.05,
            {"generator": "forge_error_recovery_protected_v2", "scenario_family": "error_recovery_scaled",
             "source_kind": "synthetic_error_recovery", "negative_type": "er_wrong_tool"},
            required_steps=required, completed_steps=[], pending_steps=required,
            terminal_tools=terminal, recent_errors=recent_errors, group_id=group_id, scoring_metadata=fm,
        ))
    return rows


'''

C3_OLD_SUM = r'''forge_rows = (
    build_forge_synthetic_rows()
    + build_argument_semantic_rows()
    + build_error_recovery_numeric_semantic_rows()
    + build_contrastive_wrong_tool_rows()
    + build_fixed_width_numeric_rows()
    + build_error_recovery_protected_rows()
)'''

C3_NEW_SUM = r'''forge_rows = (
    build_forge_synthetic_rows()
    + build_argument_semantic_rows()
    + build_error_recovery_numeric_semantic_rows()
    + build_contrastive_wrong_tool_rows()
    + build_fixed_width_numeric_rows()
    + build_error_recovery_protected_rows()
    + build_fixed_width_numeric_rows_scaled()
    + build_error_recovery_protected_rows_scaled()
)'''


def patch_c3(cells):
    idx = find_cell_by_marker(cells, C3_OLD_SUM)
    assert idx is not None, "[C3] forge_rows sum cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, C3_OLD_SUM, C3_BUILDERS + C3_NEW_SUM, "C3")
    set_cell_src(cells[idx], src)
    print(f"  [C3] scaled protected-slice builders installed in cell {idx}")


# ---------------------------------------------------------------------------
# C4: needs_clarification builder (cell 19)
# ---------------------------------------------------------------------------
C4_BUILDER = r'''# v5b C4: needs_clarification support from underspecified public-request variants.
# The gold call's key argument value is replaced with a vague reference, so calling
# the gold tool with the original concrete arguments is no longer justified.
NEEDS_CLARIFICATION_TARGET_ROWS = 1000  #@param {type:"integer"}

_NC_GENERIC_VAGUE = ["that one", "the usual one", "the one I mentioned", "the one from before"]
_NC_PARAM_VAGUE = {
    "id": "one of my records",
    "name": "that one",
    "city": "my city",
    "location": "around here",
    "date": "whenever it was",
    "time": "sometime soon",
    "count": "some of them",
    "number": "a few",
    "amount": "some amount",
    "query": "that thing",
    "symbol": "that stock",
    "email": "my usual address",
}


def _vague_replacement(param_name: str, rng: random.Random) -> str:
    key = str(param_name or "").lower()
    for token, phrase in _NC_PARAM_VAGUE.items():
        if token in key:
            return phrase
    return rng.choice(_NC_GENERIC_VAGUE)


def _droppable_argument(call: Dict[str, Any], user_request: str) -> Optional[Tuple[str, str]]:
    for k, v in (call.get("arguments") or {}).items():
        if isinstance(v, bool):
            continue
        if not isinstance(v, (str, int, float)):
            continue
        text = str(v)
        if len(text) >= 3 and text in user_request:
            return str(k), text
    return None


def build_needs_clarification_rows(
    examples: List[Dict[str, Any]], target_rows: int = NEEDS_CLARIFICATION_TARGET_ROWS
) -> List[VerifierRow]:
    """Underspecified variants of public requests. Shares the example group_id so each
    variant co-splits with the gold valid row emitted by make_public_rows, giving the
    model paired contrast between justified and unjustified concrete arguments.

    ~15% of the budget is spent on borderline cases distinguishing needs_clarification
    from valid-with-inferable-context and from tool_not_needed."""
    rng = random.Random(SEED + 5519)
    eligible = []
    for ex in examples:
        gold_calls = ex.get("gold_calls") or []
        if not gold_calls:
            continue
        call = gold_calls[0]
        if not call.get("name"):
            continue
        hit = _droppable_argument(call, ex.get("user_request") or "")
        if hit:
            eligible.append((ex, call, hit))
    if len(eligible) > int(target_rows):
        eligible = rng.sample(eligible, int(target_rows))
    rows: List[VerifierRow] = []
    nc_label_counts = Counter()
    for ex, call, (param_name, surface) in eligible:
        source = ex["source"]
        user_request = ex["user_request"]
        tools = ex["tools"]
        group_id = ex.get("group_id") or stable_id(source, user_request, tools, ex.get("gold_calls"))
        vague = _vague_replacement(param_name, rng)
        vague_request = user_request.replace(surface, vague, 1)
        roll = rng.random()
        if roll < 0.85:
            rows.append(make_row(
                source=source,
                label="needs_clarification",
                user_request=vague_request,
                tools=tools,
                candidate=call,
                rank_score=0.20,
                metadata={"generator": "underspecified_variant", "negative_type": "needs_clarification_entity_drop",
                          "dropped_param": param_name, "gold_tool": call["name"]},
                group_id=group_id,
            ))
            nc_label_counts["needs_clarification"] += 1
        elif roll < 0.90:
            # Borderline valid: vague phrasing, but the concrete value is restated, so
            # the gold call stays fully inferable. No negative_type, so the suspicious
            # valid hard-negative audit in the dataframe cell passes.
            inferable_request = f"{vague_request} To be specific, I mean {surface}."
            rows.append(make_row(
                source=source,
                label="valid",
                user_request=inferable_request,
                tools=tools,
                candidate=call,
                rank_score=0.90,
                metadata={"generator": "nc_borderline_inferable", "dropped_param": param_name,
                          "gold_tool": call["name"]},
                group_id=group_id,
            ))
            nc_label_counts["valid"] += 1
        elif roll < 0.95:
            # Borderline tool_not_needed: explanation-only request answerable without tools.
            explain_request = (
                f"Briefly explain what information you would need before you could look up {vague}."
            )
            rows.append(make_row(
                source=source,
                label="tool_not_needed",
                user_request=explain_request,
                tools=tools,
                candidate={"text_response": "I would need the specific identifier before any lookup."},
                rank_score=0.30,
                metadata={"generator": "nc_borderline_explanation", "negative_type": "text_instead_of_tool",
                          "gold_tool": call["name"]},
                group_id=group_id,
            ))
            nc_label_counts["tool_not_needed"] += 1
        else:
            # Borderline needs_clarification: a hedge that still names no concrete entity.
            hedged_request = f"{vague_request} Whichever makes sense."
            rows.append(make_row(
                source=source,
                label="needs_clarification",
                user_request=hedged_request,
                tools=tools,
                candidate=call,
                rank_score=0.20,
                metadata={"generator": "underspecified_variant", "negative_type": "needs_clarification_hedged",
                          "dropped_param": param_name, "gold_tool": call["name"]},
                group_id=group_id,
            ))
            nc_label_counts["needs_clarification"] += 1
    print("needs_clarification builder label counts:", dict(nc_label_counts))
    return rows


'''

C4_OLD_TAIL = r'''public_rows = make_public_rows(normalized_examples)
all_rows = public_rows + forge_rows + trace_rows
print("public rows:", len(public_rows))'''

C4_NEW_TAIL = r'''public_rows = make_public_rows(normalized_examples)
needs_clarification_rows = build_needs_clarification_rows(normalized_examples)
all_rows = public_rows + needs_clarification_rows + forge_rows + trace_rows
print("public rows:", len(public_rows))
print("needs_clarification rows:", len(needs_clarification_rows))'''


def patch_c4(cells):
    idx = find_cell_by_marker(cells, "def make_public_rows(")
    assert idx is not None, "[C4] make_public_rows cell not found"
    src = cell_src(cells[idx])
    anchor = "def make_public_rows("
    src = replace_once(src, anchor, C4_BUILDER + anchor, "C4 builder insert")
    src = replace_once(src, C4_OLD_TAIL, C4_NEW_TAIL, "C4 tail")
    set_cell_src(cells[idx], src)
    print(f"  [C4] needs_clarification builder installed in cell {idx}")


# ---------------------------------------------------------------------------
# C5: ablation markdown (cell 45)
# ---------------------------------------------------------------------------
C5_SWEEP_ANCHOR = "For current T4 timing, leave `GPU_PROFILE=\"auto\"`"

C5_SWEEP_SECTION = """### Seed x LR sweep

Run six sessions varying only the `SEED` and `LEARNING_RATE_OVERRIDE` form params (cell 2), keeping the profile fixed (`t4_proven` or `high_vram_quality`):

| Run | SEED | LEARNING_RATE_OVERRIDE |
|---|---:|---:|
| 1 | 42 | 1e-5 |
| 2 | 42 | 2e-5 |
| 3 | 1337 | 1e-5 |
| 4 | 1337 | 2e-5 |
| 5 | 2025 | 1e-5 |
| 6 | 2025 | 2e-5 |

Promote the best run under the constrained gate order: lowest valid false objection at 0.90 first, then highest valid recall, then wrong_tool_semantic precision, then wrong_tool_semantic recall. Note: the v5b schema-compatible wrong-tool generator changed the RNG stream, so row-for-row comparisons against pre-v5b runs are not meaningful even at SEED=42.

"""

C5_OLD_SELECTION = "Early stopping remains enabled. The saved tool-call model is selected by validation `gate_deficit_score`, which penalizes valid-recall deficit, wrong-tool precision deficit, high-confidence valid false objections, and valid-to-wrong-arguments collapse. The validation/test promotion gates remain the release stop sign."

C5_NEW_SELECTION = "Early stopping remains enabled. The saved tool-call model is selected by a constrained lexicographic rule: checkpoints whose valid false objection at 0.90 exceeds 2.5x the gate ceiling are discarded outright; among the survivors the selector maximizes valid recall, then wrong_tool_semantic precision, then wrong_tool_semantic recall, then macro F1. The validation/test promotion gates remain the release stop sign."


def patch_c5(cells):
    idx = find_cell_by_marker(cells, "## 13. Recommended ablation matrix", cell_type="markdown")
    assert idx is not None, "[C5] ablation markdown cell not found"
    src = cell_src(cells[idx])
    src = replace_once(src, C5_SWEEP_ANCHOR, C5_SWEEP_SECTION + C5_SWEEP_ANCHOR, "C5 sweep")
    src = replace_once(src, C5_OLD_SELECTION, C5_NEW_SELECTION, "C5 selection")
    set_cell_src(cells[idx], src)
    print(f"  [C5] ablation markdown updated in cell {idx}")


# ---------------------------------------------------------------------------
# Post-patch smoke tests (run against the patch constants, before saving)
# ---------------------------------------------------------------------------
def smoke_test_c1_selector():
    """Exec the C1 block with stubs; assert near-duplicate skip and arg remapping."""
    import random as _random

    ns = {
        "random": _random,
        "SEED": 42,
        "Any": object, "Dict": dict, "List": list, "Optional": object, "Tuple": tuple,
    }
    exec("from typing import Any, Dict, List, Optional, Tuple", ns)
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
    exec(C1_NEW_FN.replace("#@param {type:\"number\"}", ""), ns)
    select = ns["schema_compatible_wrong_tool_candidate"]

    get_user = {"name": "get_user", "description": "Get a user profile by user id.",
                "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}
    fetch_user = {"name": "fetch_user", "description": "Get a user profile by user id.",
                  "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}
    update_user = {"name": "update_user", "description": "Update fields on an existing account record.",
                   "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}, "email": {"type": "string"}}, "required": ["user_id"]}}

    # Near-duplicate only -> skipped entirely.
    result = select([get_user, fetch_user], "get_user", {"user_id": "u-1"})
    assert result is None, "near-duplicate distractor should be skipped"

    # Antonym-style pair with shared args -> selected, args remapped.
    result = select([get_user, fetch_user, update_user], "get_user", {"user_id": "u-1"})
    assert result is not None, "schema-compatible distractor should be selected"
    candidate, info = result
    assert candidate["name"] == "update_user", f"expected update_user, got {candidate['name']}"
    assert candidate["arguments"]["user_id"] == "u-1", "gold arg should carry over"
    assert info["distractor_tool"] == "update_user"
    # Deterministic across calls.
    again = select([get_user, fetch_user, update_user], "get_user", {"user_id": "u-1"})
    assert again[0]["name"] == "update_user"
    print("  [smoke] C1 selector: near-duplicate skipped, get/update remapped, deterministic")


def smoke_test_c3_builders():
    """Exec the C3 block with stubbed make_row machinery; check volume and invariants."""
    import json as _json
    import random as _random
    from types import SimpleNamespace

    captured = []

    def make_row(source, label, user_request, tools, candidate, rank_score,
                 metadata=None, **kw):
        row = SimpleNamespace(source=source, label=label, user_request=user_request,
                              tools=tools, candidate=candidate, rank_score=rank_score,
                              metadata=dict(metadata or {}), kw=kw)
        captured.append(row)
        return row

    ns = {
        "random": _random, "json": _json, "SEED": 42,
        "make_row": make_row,
        "stable_id": lambda *a: "sid_" + str(abs(hash(tuple(str(x) for x in a))) % 10 ** 10),
        "infer_scoring_metadata": lambda *a, **k: {},
        "_fw_tools": lambda tool_name, tool_desc, param_name, param_desc: [
            {"name": tool_name, "description": tool_desc, "parameters": {
                "type": "object", "properties": {param_name: {"type": "string", "description": param_desc}},
                "required": [param_name]}},
            {"name": "respond", "description": "Send final answer.", "parameters": {
                "type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
        ],
        "VerifierRow": SimpleNamespace,
    }
    exec("from typing import Any, Dict, List, Optional, Tuple", ns)
    exec(C3_BUILDERS.replace("#@param {type:\"integer\"}", ""), ns)

    fw_rows = ns["build_fixed_width_numeric_rows_scaled"]()
    fw_valid = [r for r in fw_rows if r.label == "valid"]
    fw_core_valid = [r for r in fw_valid if r.metadata.get("valid_protection_fixed_width_numeric_string")]
    assert len(fw_valid) >= 500, f"fixed-width valid rows: {len(fw_valid)} < 500"
    core_padded = [
        r for r in fw_core_valid
        if str(list(r.candidate["arguments"].values())[0]).isdigit()
    ]
    bad = [r for r in core_padded if not str(list(r.candidate["arguments"].values())[0]).startswith("0")]
    assert not bad, f"{len(bad)} all-digit fixed-width valid values lack a leading zero"
    assert len(core_padded) >= 500, f"zero-padded slice-support valid rows: {len(core_padded)} < 500"

    er_rows = ns["build_error_recovery_protected_rows_scaled"]()
    er_valid = [r for r in er_rows if r.label == "valid"]
    assert len(er_valid) >= 500, f"error-recovery valid rows: {len(er_valid)} < 500"
    missing_err = [r for r in er_valid if not r.kw.get("recent_errors")]
    assert not missing_err, f"{len(missing_err)} error-recovery valid rows lack recent_errors"
    er_wrong_tool = [r for r in er_rows if r.label == "wrong_tool_semantic"]
    assert len(er_wrong_tool) >= 500, f"error-recovery wrong_tool rows: {len(er_wrong_tool)} < 500"
    bad_family = [r for r in er_valid if "error_recovery" not in str(r.metadata.get("scenario_family"))]
    assert not bad_family, "error-recovery scenario_family must contain 'error_recovery'"
    print(
        f"  [smoke] C3 builders: fixed-width valid={len(fw_valid)} "
        f"(zero-padded slice support={len(core_padded)}), "
        f"error-recovery valid={len(er_valid)}, wrong_tool={len(er_wrong_tool)}"
    )


# ---------------------------------------------------------------------------
def main():
    print(f"Loading notebook: {NB_PATH}")
    nb = load_nb()
    cells = nb["cells"]
    print(f"Total cells: {len(cells)}")

    full_src = "\n".join(cell_src(c) for c in cells)
    # One-shot guard: refuse re-run.
    new_markers = [
        "schema_compatible_wrong_tool_candidate",
        "SERIALIZE_V3_CHARS_PER_TOKEN",
        "CANDIDATE_TRUNCATION_HARD_FAIL_RATE",
        "build_fixed_width_numeric_rows_scaled",
        "build_needs_clarification_rows",
    ]
    for m in new_markers:
        assert m not in full_src, f"v5b marker {m!r} already present; refusing to re-run"
    # Prerequisite guard: v5 must be applied.
    for m in ("serialize_state_v3", "FORGE_CONTRASTIVE_WRONG_TOOL_PAIRS", "constrained_promotable",
              "LEARNING_RATE_OVERRIDE = 0.0  #@param"):
        assert m in full_src, f"v5 prerequisite marker {m!r} missing; run patch_notebook_v5.py first"

    print("\nRunning pre-apply smoke tests on patch constants...")
    smoke_test_c1_selector()
    smoke_test_c3_builders()

    print("\nApplying patches...")
    patch_c0(cells)
    patch_c1(cells)
    patch_c2_serializer(cells)
    patch_c2_guard(cells)
    patch_c3(cells)
    patch_c4(cells)
    patch_c5(cells)

    save_nb(nb)
    print(f"\nNotebook saved: {NB_PATH}")

    print("\nVerifying patches...")
    nb2 = load_nb()
    cells2 = nb2["cells"]
    full_src2 = "\n".join(cell_src(c) for c in cells2)
    checks = [
        ('schema_marker = "\\n\\nCANDIDATE_TOOL_SCHEMA:"', "v3-aware payload extraction"),
        ("schema_compatible_wrong_tool_candidate", "schema-compatible wrong-tool selector"),
        ("wrong_tool_schema_compatible", "wrong-tool negative_type"),
        ("WRONG_TOOL_AMBIGUITY_SKIP_THRESHOLD", "ambiguity threshold param"),
        ("ambiguity_score", "ambiguity score metadata"),
        ("SERIALIZE_V3_CHARS_PER_TOKEN", "v3 char-budget constant"),
        ("...<+{dropped} more tools>", "competing-tools trim marker"),
        ("CANDIDATE_TRUNCATION_HARD_FAIL_RATE", "truncation hard-fail threshold"),
        ('SCHEMA_MARKER = "CANDIDATE_TOOL_SCHEMA:\\n"', "schema marker diagnostic"),
        ("build_fixed_width_numeric_rows_scaled", "scaled fixed-width builder"),
        ("build_error_recovery_protected_rows_scaled", "scaled error-recovery builder"),
        ("error_recovery_scaled", "scaled error-recovery scenario family"),
        ("build_needs_clarification_rows", "needs_clarification builder"),
        ("needs_clarification_entity_drop", "needs_clarification negative_type"),
        ("nc_borderline_inferable", "borderline inferable-valid generator tag"),
        ("Seed x LR sweep", "sweep matrix markdown"),
        ("constrained lexicographic rule", "checkpoint selection description"),
    ]
    all_ok = True
    for marker, label in checks:
        ok = marker in full_src2
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {label}")
        all_ok = all_ok and ok
    assert all_ok, "verification failed"
    # C0 must be present in BOTH extractor cells.
    extractor_hits = sum(
        1 for c in cells2 if 'schema_marker = "\\n\\nCANDIDATE_TOOL_SCHEMA:"' in cell_src(c)
    )
    assert extractor_hits == 2, f"expected v3-aware extraction in 2 cells, found {extractor_hits}"

    # Cells 1/4 contain IPython magics that plain compile() cannot parse, so only the
    # cells this patch modifies are checked; none of them contain magics.
    print("\nCompile-checking patched code cells...")
    patched_markers = [
        "def serialize_state_v3(",
        "build_fixed_width_numeric_rows_scaled",
        "def make_public_rows(",
        "def extract_candidate_payload_for_audit(",
        "CANDIDATE_TRUNCATION_HARD_FAIL_RATE",
        "def extract_candidate_payload(",
    ]
    for marker in patched_markers:
        i = find_cell_by_marker(cells2, marker, cell_type="code")
        assert i is not None, f"patched cell with {marker!r} not found"
        try:
            compile(cell_src(cells2[i]), f"cell_{i}", "exec")
        except SyntaxError as exc:
            print(f"  [FAIL] cell {i} ({marker}): {exc}")
            sys.exit(1)
        print(f"  [OK ] cell {i} ({marker})")
    print("\nDone. All v5b patches applied and verified.")


if __name__ == "__main__":
    main()
