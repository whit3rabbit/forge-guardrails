#!/usr/bin/env python3
"""Generate OpenToken tool-output filter fixtures from the upstream source.

This script checks out the exact OpenToken commit used by Forge's parity
fixtures, invokes the TypeScript filter functions through tsx, applies local
Forge safety overrides for known false summaries, and rewrites the checked-in
JSON fixture with `expected_output` values.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


REPO_URL = "https://github.com/MrGray17/opentoken"
COMMIT = "5998ffab786d12a0f1f635604a3f6eff8b359967"
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/parity/fixtures/opentoken_tool_output_filters.json"


CASES = [
    {
        "name": "git_diff",
        "tool": "bash",
        "opentoken_filter": "git",
        "args": {"command": "git diff"},
        "input": "diff --git a/src/lib.rs b/src/lib.rs\nindex 111..222 100644\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n context line\n-old value\n+new value\n",
    },
    {
        "name": "git_status_porcelain",
        "tool": "bash",
        "opentoken_filter": "git",
        "args": {"command": "git status --short"},
        "input": " M src/lib.rs\nA  src/main.rs\n?? README.md\n",
    },
    {
        "name": "git_status_conflict_and_rename",
        "tool": "bash",
        "opentoken_filter": "git",
        "args": {"command": "git status --short"},
        "input": "UU src/conflict.rs\nR  src/old.rs -> src/new.rs\n?? notes.txt\n",
    },
    {
        "name": "git_diff_rename",
        "tool": "bash",
        "opentoken_filter": "git",
        "args": {"command": "git diff --find-renames"},
        "input": "diff --git a/src/old.rs b/src/new.rs\nsimilarity index 100%\nrename from src/old.rs\nrename to src/new.rs\n",
    },
    {
        "name": "git_log_merge",
        "tool": "bash",
        "opentoken_filter": "git",
        "args": {"command": "git log"},
        "input": "commit abcdef1234567890\nAuthor: Dev <dev@example.test>\nDate: Thu May 28 12:00:00 2026 -0500\n\n    initial commit\ncommit 123456abcdef7890\nMerge: 1111111 2222222\nAuthor: Dev <dev@example.test>\nDate: Thu May 28 13:00:00 2026 -0500\n\n    merge branch\n",
    },
    {
        "name": "cargo_build",
        "tool": "bash",
        "opentoken_filter": "cargo",
        "args": {"command": "cargo build"},
        "input_parts": [
            {"repeat": 30, "text": "Compiling noisy_crate v0.1.0\n"},
            {
                "text": "error[E0425]: cannot find value `missing` in this scope\n --> src/lib.rs:7:5\n  |\n7 |     missing\n  |     ^^^^^^^ not found in this scope\n\n"
            },
        ],
    },
    {
        "name": "cargo_build_multi_diagnostic",
        "tool": "bash",
        "opentoken_filter": "cargo",
        "args": {"command": "cargo clippy"},
        "input_parts": [
            {"repeat": 30, "text": "Checking noisy_crate v0.1.0\n"},
            {
                "text": "warning[clippy::needless_return]: unneeded return statement\n --> src/lib.rs:3:5\n  |\n3 |     return value;\n  |     ^^^^^^^^^^^^\n\nerror[E0308]: mismatched types\n --> src/lib.rs:8:5\n  |\n8 |     \"wrong\"\n  |     ^^^^^^^ expected i32\n\nerror: could not compile `demo` due to previous error\n"
            },
        ],
    },
    {
        "name": "cargo_test_failure",
        "tool": "bash",
        "opentoken_filter": "cargo",
        "args": {"command": "cargo test"},
        "input": "running 2 tests\ntest alpha ... ok\ntest beta ... FAILED\n\nfailures:\n    beta\n\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; finished in 0.01s\n",
    },
    {
        "name": "cargo_json_diagnostic",
        "tool": "bash",
        "opentoken_filter": "cargo",
        "args": {"command": "cargo check --message-format=json"},
        "input": '{"reason":"compiler-artifact","package_id":"demo 0.1.0","target":{"name":"demo"}}\n'
        '{"reason":"compiler-message","message":{"level":"error","message":"cannot find value `missing` in this scope","code":{"code":"E0425"},"spans":[{"file_name":"src/lib.rs","line_start":7,"column_start":5,"line_end":7,"column_end":12,"text":[{"text":"    missing","highlight_start":5,"highlight_end":12}]}],"rendered":"error[E0425]: cannot find value `missing` in this scope\\n --> src/lib.rs:7:5\\n  |\\n7 |     missing\\n  |     ^^^^^^^ not found in this scope\\n"}}}\n'
        '{"reason":"build-finished","success":false}\n',
    },
    {
        "name": "rustc_diagnostic",
        "tool": "bash",
        "opentoken_filter": "cargo",
        "args": {"command": "rustc src/lib.rs"},
        "input": "error[E0308]: mismatched types\n --> src/lib.rs:2:5\n  |\n2 |     \"wrong\"\n  |     ^^^^^^^ expected i32, found &str\n\nerror: aborting due to previous error\n",
    },
    {
        "name": "npm_install",
        "tool": "bash",
        "opentoken_filter": "npm",
        "args": {"command": "npm install"},
        "input": "added 312 packages, and audited 313 packages in 6s\nnpm warn deprecated old-lib@1.0.0: unsupported\nfound 0 vulnerabilities\n",
    },
    {
        "name": "npm_lint",
        "tool": "bash",
        "opentoken_filter": "npm",
        "args": {"command": "npm run lint"},
        "input": "src/a.ts\nsrc/a.ts:1:1 error no-console\nsrc/a.ts:2:1 Warning missing semicolon\n2 problems\n",
    },
    {
        "name": "pnpm_test_failure",
        "tool": "bash",
        "opentoken_filter": "npm",
        "args": {"command": "pnpm test"},
        "input": "FAIL src/foo.test.ts\nError: boom\n    at Object.<anonymous> (src/foo.test.ts:1:1)\n\nTest Suites: 1 failed, 1 total\nTests: 1 failed, 1 total\n",
    },
    {
        "name": "yarn_test_failure",
        "tool": "bash",
        "opentoken_filter": "npm",
        "args": {"command": "yarn test"},
        "input": "yarn run v1.22.22\n$ vitest run\nFAIL src/foo.test.ts\nError: expected true to be false\n\nTest Files 1 failed (1)\nTests 1 failed (1)\nerror Command failed with exit code 1.\n",
    },
    {
        "name": "bun_test_failure",
        "tool": "bash",
        "opentoken_filter": "npm",
        "args": {"command": "bun test"},
        "input": "bun test v1.1.0\nsrc/foo.test.ts:\n(fail) adds numbers [0.12ms]\n  expect(received).toBe(expected)\n\n1 fail\n2 pass\nerror: script \"test\" exited with code 1\n",
    },
    {
        "name": "docker_build",
        "tool": "bash",
        "opentoken_filter": "docker",
        "args": {"command": "docker build ."},
        "input": "#1 [internal] load build definition\n#2 [1/4] FROM alpine\nDownloading layer\nExtracting layer\nSuccessfully built abcdef123456\n",
    },
    {
        "name": "pip_install",
        "tool": "bash",
        "opentoken_filter": "pip",
        "args": {"command": "pip install -r requirements.txt"},
        "input": "Collecting flask\n  Downloading flask.whl\nRequirement already satisfied: click in .venv/lib/python3.12/site-packages\nRequirement already satisfied: itsdangerous in .venv/lib/python3.12/site-packages\nSuccessfully installed flask-3.0.0\n",
    },
    {
        "name": "make_progress",
        "tool": "bash",
        "opentoken_filter": "make",
        "args": {"command": "make"},
        "input": "[ 10%] Building C object a.o\nwarning: generated header is stale\n[ 50%] Building C object b.o\n[100%] Built target app\n",
    },
    {
        "name": "find_grouping",
        "tool": "bash",
        "opentoken_filter": "fs",
        "args": {"command": "find . -type f"},
        "input_parts": [
            {"repeat": 110, "text": "src/generated/file.rs\n"},
            {"repeat": 20, "text": "target/debug/noise.o\n"},
        ],
    },
    {
        "name": "tree_summary",
        "tool": "bash",
        "opentoken_filter": "fs",
        "args": {"command": "tree"},
        "input": ".\n├── src\n│   ├── lib.rs\n│   └── main.rs\n└── target\n    └── debug\n"
        + "".join(f"        ├── artifact_{idx}.o\n" for idx in range(20)),
    },
    {
        "name": "generic_stack_trace",
        "tool": "bash",
        "opentoken_filter": "generic",
        "args": {"command": "custom-runner"},
        "input": "Error: boom\n    at first (src/a.js:1:1)\n    at second (src/a.js:2:1)\n    at third (src/a.js:3:1)\n    at fourth (src/a.js:4:1)\n    at fifth (src/a.js:5:1)\n    at sixth (src/a.js:6:1)\n    at seventh (src/a.js:7:1)\n",
    },
    {
        "name": "pytest_failure_block",
        "tool": "bash",
        "opentoken_filter": "test",
        "args": {"command": "pytest"},
        "input": "___ test_adds_numbers ___\n\n    assert add(1, 1) == 3\nE   assert 2 == 3\n\nFAILED tests/test_math.py::test_adds_numbers - AssertionError\n1 failed, 2 passed in 0.03s\n",
    },
    {
        "name": "pytest_param_failure",
        "tool": "bash",
        "opentoken_filter": "test",
        "args": {"command": "pytest -q"},
        "input": "___ test_adds_numbers[1-2-4] ___\n\nleft = 1, right = 2, expected = 4\n\n    assert add(left, right) == expected\nE   assert 3 == 4\n\nFAILED tests/test_math.py::test_adds_numbers[1-2-4] - AssertionError\n1 failed, 8 passed in 0.04s\n",
    },
    {
        "name": "read_source_outline",
        "tool": "read_file",
        "opentoken_filter": "read",
        "args": {"path": "src/example.rs"},
        "input_parts": [
            {"text": "use crate::thing;\n"},
            {"repeat": 90, "text": "let noise = 1;\n"},
            {"text": "pub struct Widget {\n    id: String,\n}\nfn run_widget() {}\n"},
        ],
    },
    {
        "name": "grep_rg_json",
        "tool": "grep",
        "opentoken_filter": "grep",
        "args": {"query": "Widget"},
        "input": '{"type":"match","data":{"path":{"text":"src/lib.rs"},"line_number":10,"lines":{"text":"pub struct Widget;\\n"}}}\n{"type":"match","data":{"path":{"text":"src/lib.rs"},"line_number":20,"lines":{"text":"impl Widget {}\\n"}}}\n{"type":"match","data":{"path":{"text":"build/generated/noise.rs"},"line_number":1,"lines":{"text":"noise\\n"}}}\n',
    },
    {
        "name": "grep_rg_json_context_events",
        "tool": "grep",
        "opentoken_filter": "grep",
        "args": {"query": "Widget"},
        "input": '{"type":"context","data":{"path":{"text":"src/lib.rs"},"line_number":9,"lines":{"text":"#[derive(Debug)]\\n"}}}\n{"type":"match","data":{"path":{"text":"src/lib.rs"},"line_number":10,"lines":{"text":"pub struct Widget;\\n"}}}\n{"type":"context","data":{"path":{"text":"src/lib.rs"},"line_number":11,"lines":{"text":"impl Widget {\\n"}}}\n{"type":"match","data":{"path":{"text":"src/ui.rs"},"line_number":42,"submatches":[{"match":{"text":"Widget"},"start":7,"end":13}]}}\n{"type":"summary","data":{"elapsed_total":{"human":"0.01s"}}}\n',
    },
    {
        "name": "grep_vimgrep",
        "tool": "grep",
        "opentoken_filter": "grep",
        "args": {"query": "fn"},
        "input": "src/lib.rs:10:5:fn main() {}\nsrc/lib.rs:20:1:fn helper() {}\nbuild/generated.rs:1:1:fn noise() {}\n",
    },
    {
        "name": "grep_per_file_cap",
        "tool": "grep",
        "opentoken_filter": "grep",
        "args": {"query": "match"},
        "input": "".join(f"src/lib.rs:{idx}:match {idx}\n" for idx in range(1, 13)),
    },
    {
        "name": "glob_grouping",
        "tool": "glob",
        "opentoken_filter": "glob",
        "input": "".join(f"src/generated/item_{idx}.rs\n" for idx in range(105))
        + "".join(f"node_modules/pkg_{idx}/index.js\n" for idx in range(10)),
    },
    {
        "name": "glob_hidden_noisy_paths",
        "tool": "glob",
        "opentoken_filter": "glob",
        "input": "src/lib.rs\n.git/config\n.cache/tool/tmp.json\n.env\n.DS_Store\ntarget/debug/build.log\nnode_modules/pkg/index.js\nsrc/.hidden.rs\n",
    },
]


DRIVER = """
import fs from "node:fs";
import {{ filterGitOutput }} from "{src}/families/git.ts";
import {{ filterCargoOutput }} from "{src}/families/cargo.ts";
import {{ filterNpmOutput }} from "{src}/families/npm.ts";
import {{ filterDockerOutput }} from "{src}/families/docker.ts";
import {{ filterPipOutput }} from "{src}/families/pip.ts";
import {{ filterMakeOutput }} from "{src}/families/make.ts";
import {{ filterFsOutput }} from "{src}/families/fs.ts";
import {{ filterGeneric }} from "{src}/families/generic.ts";
import {{ filterTestOutput }} from "{src}/families/test.ts";
import {{ filterRead }} from "{src}/filters/read.ts";
import {{ filterGrep }} from "{src}/filters/grep.ts";
import {{ filterGlob }} from "{src}/filters/glob.ts";

const cases = JSON.parse(fs.readFileSync(0, "utf8"));

function expandInput(testCase) {{
  if (typeof testCase.input === "string") return testCase.input;
  return testCase.input_parts.map((part) => part.text.repeat(part.repeat ?? 1)).join("");
}}

function expectedOutput(testCase) {{
  const input = expandInput(testCase);
  const command = testCase.args?.command ?? "";
  switch (testCase.opentoken_filter) {{
    case "git":
      return filterGitOutput(command, input);
    case "cargo":
      return filterCargoOutput(command, input);
    case "npm":
      return filterNpmOutput(command, input);
    case "docker":
      return filterDockerOutput(command, input);
    case "pip":
      return filterPipOutput(command, input);
    case "make":
      return filterMakeOutput(command, input);
    case "fs":
      return filterFsOutput(command, input);
    case "generic":
      return filterGeneric(input);
    case "test":
      return filterTestOutput(command, input);
    case "read":
      return filterRead(testCase.args?.path ?? testCase.args?.file ?? "", input);
    case "grep":
      return filterGrep(input);
    case "glob":
      return filterGlob(input);
    default:
      throw new Error(`unknown filter ${{testCase.opentoken_filter}}`);
  }}
}}

console.log(JSON.stringify(cases.map((testCase) => ({{
  ...testCase,
  expected_output: expectedOutput(testCase),
}})), null, 2));
"""


def run(command: list[str], cwd: Path | None = None, input_text: str | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def expand_input(case: dict) -> str:
    if "input" in case:
        return case["input"]
    return "".join(
        part["text"] * int(part.get("repeat", 1)) for part in case["input_parts"]
    )


FORGE_SAFETY_PRESERVE_INPUTS = {"cargo_json_diagnostic"}

FORGE_SAFETY_EXPECTED_OUTPUTS = {
    "grep_rg_json_context_events": "2 files, 2 matches:\n\n"
    "src/lib.rs:\n"
    "  10: pub struct Widget;\n\n"
    "src/ui.rs:\n"
    "  42: Widget",
}


def apply_forge_safety_overrides(cases: list[dict]) -> None:
    for case in cases:
        if case["name"] in FORGE_SAFETY_PRESERVE_INPUTS:
            case["expected_output"] = expand_input(case)
            continue
        override = FORGE_SAFETY_EXPECTED_OUTPUTS.get(case["name"])
        if override is not None:
            case["expected_output"] = override


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="forge-opentoken-") as temp:
        temp_path = Path(temp)
        checkout = temp_path / "opentoken"
        run(["git", "init", str(checkout)])
        run(["git", "remote", "add", "origin", REPO_URL], cwd=checkout)
        run(["git", "fetch", "--depth", "1", "origin", COMMIT], cwd=checkout)
        run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=checkout)

        src = (checkout / "packages/core/src").as_posix()
        driver = temp_path / "driver.ts"
        driver.write_text(textwrap.dedent(DRIVER).format(src=src), encoding="utf-8")

        generated = run(
            ["npm", "exec", "--yes", "--package", "tsx", "--", "tsx", str(driver)],
            input_text=json.dumps(CASES),
        )
        cases = json.loads(generated)
        apply_forge_safety_overrides(cases)

    fixture = {
        "source": "MrGray17/opentoken",
        "commit": COMMIT,
        "generator": "tests/parity/generate_opentoken_tool_output_fixtures.py",
        "cases": cases,
    }

    for case in fixture["cases"]:
        case["input_len"] = len(expand_input(case))

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
