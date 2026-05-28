from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "proxy_resource_baseline.py"

spec = importlib.util.spec_from_file_location("proxy_resource_baseline", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
resource = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resource
spec.loader.exec_module(resource)


class ResourceBaselineTests(unittest.TestCase):
    def test_classify_roles(self) -> None:
        root_pid = 100

        self.assertEqual(
            resource.classify_role(
                resource.ProcessInfo(100, 1, 0.0, 100, "bash scripts/start_llamaserver_proxy.sh"),
                root_pid,
            ),
            "wrapper",
        )
        self.assertEqual(
            resource.classify_role(
                resource.ProcessInfo(101, 100, 1.0, 200, "/tmp/forge-guardrails-proxy --port 8081"),
                root_pid,
            ),
            "proxy",
        )
        self.assertEqual(
            resource.classify_role(
                resource.ProcessInfo(102, 101, 50.0, 300, "/opt/homebrew/bin/llama-server -m model.gguf"),
                root_pid,
            ),
            "backend",
        )
        self.assertEqual(
            resource.classify_role(
                resource.ProcessInfo(103, 101, 0.1, 50, "sleep 10"),
                root_pid,
            ),
            "other",
        )

    def test_descendant_fallback_and_sample_aggregation(self) -> None:
        rows = [
            resource.ProcessInfo(100, 1, 0.1, 1000, "bash scripts/start_llamaserver_proxy.sh"),
            resource.ProcessInfo(101, 100, 12.5, 2048, "forge-guardrails-proxy --port 8081"),
            resource.ProcessInfo(102, 101, 150.0, 1048576, "llama-server -m model.gguf"),
        ]

        self.assertEqual(resource.descendants_from_rows(100, rows), {101, 102})
        sample = resource.build_sample(
            label="budget_8192",
            root_pid=100,
            rows=rows,
            started_at=1000.0,
            now=1002.0,
        )

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["roles"]["proxy"]["process_count"], 1)
        self.assertEqual(sample["roles"]["proxy"]["cpu_percent"], 12.5)
        self.assertEqual(sample["roles"]["backend"]["rss_kib"], 1048576)
        self.assertEqual(sample["combined"]["process_count"], 3)
        self.assertEqual(sample["combined"]["rss_kib"], 1051624)

    def test_percentile_and_series_stats(self) -> None:
        self.assertEqual(resource.percentile([1.0, 2.0, 100.0], 95), 100.0)
        self.assertEqual(resource.percentile([1.0, 2.0, 100.0], 50), 2.0)

        stats = resource.series_stats([1.0, 2.0, 100.0], [1024, 2048, 4096])
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(stats["cpu_percent"]["mean"], 34.33)
        self.assertEqual(stats["cpu_percent"]["p95"], 100.0)
        self.assertEqual(stats["rss_mib"]["first"], 1.0)
        self.assertEqual(stats["rss_mib"]["last"], 4.0)
        self.assertEqual(stats["rss_mib"]["delta"], 3.0)

    def test_summary_and_report_format_multiple_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_samples = [
                self._sample("budget_8192", 1000.0, 0.0, proxy_cpu=10.0, backend_cpu=100.0),
                self._sample("budget_8192", 1001.0, 1.0, proxy_cpu=30.0, backend_cpu=120.0),
            ]
            second_samples = [
                self._sample("compaction_chain_p1", 2000.0, 0.0, proxy_cpu=20.0, backend_cpu=200.0)
            ]

            for label, samples in {
                "budget_8192": first_samples,
                "compaction_chain_p1": second_samples,
            }.items():
                sample_path = tmp_path / f"resource_samples_{label}.jsonl"
                with sample_path.open("w") as handle:
                    for sample in samples:
                        handle.write(json.dumps(sample) + "\n")
                summary = resource.summarize_samples(label, sample_path, samples)
                (tmp_path / f"resource_summary_{label}.json").write_text(
                    json.dumps(summary)
                )

            summaries = resource.load_summaries(tmp_path)
            report = resource.format_report(summaries)

            self.assertIn("Label: budget_8192", report)
            self.assertIn("Label: compaction_chain_p1", report)
            self.assertLess(
                report.index("Proxy (headline)"),
                report.index("Backend"),
            )
            self.assertIn("Combined tree", report)

    def test_report_command_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample_path = tmp_path / "resource_samples_budget_8192.jsonl"
            samples = [self._sample("budget_8192", 1000.0, 0.0)]
            summary = resource.summarize_samples("budget_8192", sample_path, samples)
            (tmp_path / "resource_summary_budget_8192.json").write_text(
                json.dumps(summary)
            )
            report_path = tmp_path / "resource_baseline_report.txt"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = resource.main([
                    "report",
                    "--output-dir",
                    str(tmp_path),
                    "--report",
                    str(report_path),
                ])

            self.assertEqual(status, 0)
            self.assertIn("Proxy Resource Baseline", report_path.read_text())

    def _sample(
        self,
        label: str,
        timestamp: float,
        elapsed_s: float,
        *,
        proxy_cpu: float = 10.0,
        backend_cpu: float = 100.0,
    ) -> dict[str, object]:
        roles = {
            "proxy": {
                "process_count": 1,
                "cpu_percent": proxy_cpu,
                "rss_kib": 2048,
            },
            "backend": {
                "process_count": 1,
                "cpu_percent": backend_cpu,
                "rss_kib": 1048576,
            },
            "wrapper": {
                "process_count": 1,
                "cpu_percent": 0.1,
                "rss_kib": 1024,
            },
            "other": {
                "process_count": 0,
                "cpu_percent": 0.0,
                "rss_kib": 0,
            },
        }
        return {
            "label": label,
            "timestamp": timestamp,
            "elapsed_s": elapsed_s,
            "root_pid": 100,
            "roles": roles,
            "combined": {
                "process_count": 3,
                "cpu_percent": proxy_cpu + backend_cpu + 0.1,
                "rss_kib": 1051648,
            },
            "processes": [],
        }


if __name__ == "__main__":
    unittest.main()
