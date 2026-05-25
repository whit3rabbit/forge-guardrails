from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "eval_openai_proxy.py"
COMPARE_PATH = ROOT / "scripts" / "compare_published_eval.py"

spec = importlib.util.spec_from_file_location("eval_openai_proxy", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
proxy_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = proxy_eval
spec.loader.exec_module(proxy_eval)

compare_spec = importlib.util.spec_from_file_location(
    "compare_published_eval", COMPARE_PATH
)
if compare_spec is None or compare_spec.loader is None:
    raise RuntimeError(f"cannot load {COMPARE_PATH}")
compare_eval = importlib.util.module_from_spec(compare_spec)
sys.modules[compare_spec.name] = compare_eval
compare_spec.loader.exec_module(compare_eval)


def scenario_by_name(name: str) -> Any:
    return next(scenario for scenario in proxy_eval.ALL_SCENARIOS if scenario.name == name)


class FakeSseResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    async def aiter_lines(self) -> Any:
        for line in self.lines:
            yield line


class FakeProxyClient:
    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.tool_names_by_call: list[list[str]] = []
        self.messages_by_call: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
        stream: bool = False,
        required_steps: list[str] | None = None,
        terminal_tools: list[str] | None = None,
    ) -> Any:
        self.tool_names_by_call.append([tool.name for tool in tools or []])
        self.messages_by_call.append(list(messages))
        if not self.turns:
            raise AssertionError("unexpected extra proxy call")
        return self.turns.pop(0)


class EvalOpenAIProxyTests(unittest.IsolatedAsyncioTestCase):
    def test_recommended_sampling_merges_and_overrides(self) -> None:
        client = proxy_eval.OpenAIProxyClient(
            "http://127.0.0.1:8081/v1",
            "Ministral-3-8B-Instruct-2512-Q8_0",
        )

        body = client._body([], None, None, stream=False)
        self.assertEqual(body["temperature"], 0.05)

        overridden = client._body([], None, {"temperature": 0.2}, stream=False)
        self.assertEqual(overridden["temperature"], 0.2)

        disabled = proxy_eval.OpenAIProxyClient(
            "http://127.0.0.1:8081/v1",
            "Ministral-3-8B-Instruct-2512-Q8_0",
            recommended_sampling=False,
        )
        self.assertNotIn("temperature", disabled._body([], None, None, stream=False))

    def test_nonstream_parser_preserves_tool_ids_and_arguments(self) -> None:
        turn = proxy_eval._parse_openai_response({
            "choices": [{
                "message": {
                    "content": "Use lookup.",
                    "tool_calls": [{
                        "id": "call_lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup_user",
                            "arguments": "{\"name\":\"Alice\"}",
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        })

        self.assertEqual(turn.kind, "tool_call")
        self.assertEqual(turn.input_tokens, 5)
        self.assertEqual(turn.output_tokens, 7)
        self.assertEqual(turn.tool_calls[0].id, "call_lookup")
        self.assertEqual(turn.tool_calls[0].name, "lookup_user")
        self.assertEqual(turn.tool_calls[0].args, {"name": "Alice"})
        self.assertEqual(turn.tool_calls[0].arguments_json, "{\"name\":\"Alice\"}")
        self.assertEqual(turn.tool_calls[0].reasoning, "Use lookup.")

    async def test_stream_parser_preserves_tool_ids_and_arguments(self) -> None:
        def event(payload: dict[str, Any]) -> str:
            return "data: " + json.dumps(payload, separators=(",", ":"))

        response = FakeSseResponse([
            event({"choices": [{"delta": {"content": "Use lookup."}}]}),
            event({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_lookup",
                            "function": {
                                "name": "lookup_user",
                                "arguments": "{\"name\"",
                            },
                        }],
                    },
                }],
            }),
            event({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": ":\"Alice\"}"},
                        }],
                    },
                }],
            }),
            event({
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }),
            "data: [DONE]",
        ])

        turn = await proxy_eval._parse_openai_sse(response)

        self.assertEqual(turn.kind, "tool_call")
        self.assertEqual(turn.input_tokens, 3)
        self.assertEqual(turn.output_tokens, 4)
        self.assertEqual(turn.tool_calls[0].id, "call_lookup")
        self.assertEqual(turn.tool_calls[0].name, "lookup_user")
        self.assertEqual(turn.tool_calls[0].args, {"name": "Alice"})
        self.assertEqual(turn.tool_calls[0].arguments_json, "{\"name\":\"Alice\"}")
        self.assertEqual(turn.tool_calls[0].reasoning, "Use lookup.")

    async def test_tool_calls_then_text_complete_tool_selection(self) -> None:
        client = FakeProxyClient([
            proxy_eval.ProxyTurn(
                "tool_call",
                tool_calls=[
                    proxy_eval.ProxyToolCall(
                        "call_lookup",
                        "lookup_user",
                        {"name": "Alice"},
                        "{\"name\":\"Alice\"}",
                    ),
                ],
            ),
            proxy_eval.ProxyTurn(
                "tool_call",
                tool_calls=[
                    proxy_eval.ProxyToolCall(
                        "call_permissions",
                        "get_permissions",
                        {"user_id": "U-1001"},
                        "{\"user_id\":\"U-1001\"}",
                    ),
                ],
            ),
            proxy_eval.ProxyTurn(
                "text",
                content="Alice has read, write, and admin permissions.",
            ),
        ])

        result = await proxy_eval.run_proxy_scenario(
            client,
            scenario_by_name("tool_selection"),
            stream=True,
            budget_tokens=8192,
            ablation=proxy_eval.ABLATION_PRESETS["reforged"],
        )

        self.assertTrue(result.completeness)
        self.assertTrue(result.accuracy)
        self.assertEqual(result.proxy_terminal_source, "text")
        self.assertEqual(result.tool_sequence, ["lookup_user", "get_permissions"])
        self.assertIn("admin", result.terminal_args["answer"])
        self.assertTrue(
            all("respond" not in names for names in client.tool_names_by_call)
        )
        self.assertEqual(
            client.messages_by_call[1][-1]["tool_call_id"],
            "call_lookup",
        )
        self.assertEqual(
            client.messages_by_call[1][-1]["_forge"]["tool_status"],
            "ok",
        )
        self.assertEqual(
            client.messages_by_call[2][-1]["tool_call_id"],
            "call_permissions",
        )
        self.assertEqual(
            client.messages_by_call[2][-1]["_forge"]["tool_status"],
            "ok",
        )

    async def test_tool_execution_error_marks_tool_status_error(self) -> None:
        client = FakeProxyClient([
            proxy_eval.ProxyTurn(
                "tool_call",
                tool_calls=[
                    proxy_eval.ProxyToolCall(
                        "call_fetch",
                        "fetch",
                        {"count": "10"},
                        "{\"count\":\"10\"}",
                    ),
                ],
            ),
            proxy_eval.ProxyTurn("text", content="Fetched 10 records."),
        ])

        result = await proxy_eval.run_proxy_scenario(
            client,
            scenario_by_name("error_recovery"),
            stream=False,
            budget_tokens=8192,
            ablation=proxy_eval.ABLATION_PRESETS["reforged"],
        )

        self.assertTrue(result.completeness)
        self.assertEqual(result.tool_errors, 1)
        self.assertEqual(
            client.messages_by_call[1][-1]["_forge"]["tool_status"],
            "error",
        )

    async def test_premature_text_is_not_client_side_nudged(self) -> None:
        client = FakeProxyClient([
            proxy_eval.ProxyTurn("text", content="Alice is an engineer."),
        ])

        result = await proxy_eval.run_proxy_scenario(
            client,
            scenario_by_name("tool_selection"),
            stream=False,
            budget_tokens=8192,
            ablation=proxy_eval.ABLATION_PRESETS["reforged"],
        )

        self.assertTrue(result.completeness)
        self.assertFalse(result.accuracy)
        self.assertEqual(result.step_nudges, 0)
        self.assertEqual(result.iterations_used, 1)
        self.assertEqual(len(client.messages_by_call), 1)

    def test_result_row_labels_proxy_backend(self) -> None:
        scenario = scenario_by_name("tool_selection")
        result = proxy_eval.ProxyRunResult(
            scenario_name="tool_selection",
            completeness=True,
            iterations_used=3,
            terminal_args={"answer": "Alice has read, write, and admin permissions."},
            accuracy=True,
            final_text="Alice has read, write, and admin permissions.",
            proxy_terminal_source="text",
        )

        row = proxy_eval._result_row(
            result,
            scenario,
            1,
            "Ministral-3-8B-Instruct-2512-Q8_0",
            True,
            "reforged",
            8192,
            "llamaserver",
            "proxy",
            "openai-proxy",
        )

        self.assertEqual(row["backend"], "llamaserver")
        self.assertEqual(row["mode"], "proxy")
        self.assertEqual(row["eval_target_backend"], "openai-proxy")
        self.assertEqual(row["proxy_terminal_source"], "text")

    def test_result_row_can_include_proxy_backend_mode(self) -> None:
        scenario = scenario_by_name("tool_selection")
        result = proxy_eval.ProxyRunResult(
            scenario_name="tool_selection",
            completeness=True,
            iterations_used=3,
            terminal_args={"answer": "Alice has read, write, and admin permissions."},
            accuracy=True,
            final_text="Alice has read, write, and admin permissions.",
            proxy_terminal_source="text",
        )

        row = proxy_eval._result_row(
            result,
            scenario,
            1,
            "Ministral-3-8B-Instruct-2512-Q8_0",
            True,
            "reforged",
            8192,
            "llamaserver",
            "proxy",
            "openai-proxy",
            "native",
        )

        self.assertEqual(row["proxy_backend_mode"], "native")

    def test_proxy_terminal_tools_omit_respond_when_real_terminal_exists(self) -> None:
        scenario = scenario_by_name("error_recovery")

        self.assertEqual(
            proxy_eval._proxy_terminal_tools(scenario.workflow),
            ["summarize"],
        )

    def test_proxy_terminal_tools_keep_respond_without_real_terminal(self) -> None:
        class WorkflowStub:
            terminal_tools = {"respond"}

        self.assertEqual(
            proxy_eval._proxy_terminal_tools(WorkflowStub()),
            ["respond"],
        )

    def test_terminal_redaction_has_specific_failure_classification(self) -> None:
        result = proxy_eval.ProxyRunResult(
            scenario_name="error_recovery",
            completeness=True,
            iterations_used=3,
            terminal_args={"content": "[REDACTED]"},
            accuracy=False,
            final_text="[REDACTED]",
            proxy_terminal_source="tool_call",
        )

        self.assertEqual(
            proxy_eval._proxy_failure_classification(result),
            "terminal_redacted",
        )

    def test_published_compare_skips_proxy_rows_for_native_baseline(self) -> None:
        model = "Ministral-3-8B-Instruct-2512-Q8_0"
        row = {
            "scenario": "basic_2step",
            "model": model,
            "ablation": "reforged",
            "backend": "llamaserver",
            "mode": "proxy",
            "eval_target_backend": "openai-proxy",
            "success": True,
            "completeness": True,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            old_argv = sys.argv
            sys.argv = [
                "compare_published_eval.py",
                handle.name,
                "--model",
                model,
                "--backend-mode",
                "LS/N",
                "--local-model",
                model,
            ]
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    status = compare_eval.main()
            finally:
                sys.argv = old_argv

        self.assertEqual(status, 0)
        self.assertIn("Published comparison skipped", stdout.getvalue())
        self.assertIn("llamaserver/proxy", stdout.getvalue())

    def test_published_compare_allows_proxy_rows_for_prompt_baseline(self) -> None:
        model = "Ministral-3-8B-Instruct-2512-Q8_0"
        published_text = f"""
Model/Backend Scr Acc Cmp Eff Wst Spd N b2s
{model} LS/P [reforged] 100.0% 100.0% 100.0% 100% 0.0 1.0s 1 100
rel=relevance_detection, b2s=basic_2step
"""
        row = {
            "scenario": "basic_2step",
            "model": model,
            "ablation": "reforged",
            "backend": "llamaserver",
            "mode": "proxy",
            "eval_target_backend": "openai-proxy",
            "success": True,
            "completeness": True,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".md") as published:
            published.write(published_text)
            published.flush()
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                old_argv = sys.argv
                sys.argv = [
                    "compare_published_eval.py",
                    handle.name,
                    "--published",
                    published.name,
                    "--model",
                    model,
                    "--local-model",
                    model,
                ]
                try:
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        status = compare_eval.main()
                finally:
                    sys.argv = old_argv

        self.assertEqual(status, 0)
        self.assertIn("Published comparison passed", stdout.getvalue())
        self.assertNotIn("Published comparison skipped", stdout.getvalue())

    def test_published_compare_allows_native_proxy_rows_for_native_baseline(self) -> None:
        model = "Ministral-3-8B-Instruct-2512-Q8_0"
        published_text = f"""
Model/Backend Scr Acc Cmp Eff Wst Spd N b2s
{model} LS/N [reforged] 100.0% 100.0% 100.0% 100% 0.0 1.0s 1 100
rel=relevance_detection, b2s=basic_2step
"""
        row = {
            "scenario": "basic_2step",
            "model": model,
            "ablation": "reforged",
            "backend": "llamaserver",
            "mode": "proxy",
            "eval_target_backend": "openai-proxy",
            "proxy_backend_mode": "native",
            "success": True,
            "completeness": True,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".md") as published:
            published.write(published_text)
            published.flush()
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                old_argv = sys.argv
                sys.argv = [
                    "compare_published_eval.py",
                    handle.name,
                    "--published",
                    published.name,
                    "--model",
                    model,
                    "--backend-mode",
                    "LS/N",
                    "--local-model",
                    model,
                ]
                try:
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        status = compare_eval.main()
                finally:
                    sys.argv = old_argv

        self.assertEqual(status, 0)
        self.assertIn("Published comparison passed", stdout.getvalue())
        self.assertNotIn("Published comparison skipped", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
