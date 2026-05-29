#!/usr/bin/env python3
"""Generate OpenToken tool-output filter fixtures from the upstream source.

This script checks out the exact OpenToken commit used by Forge's parity
fixtures, invokes the TypeScript filter functions through tsx, and rewrites the
checked-in JSON fixture with upstream `expected_output` values.
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
