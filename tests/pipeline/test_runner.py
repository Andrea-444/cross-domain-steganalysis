from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.configuration import PipelineConfiguration
from src.pipeline.context import PipelineContext
from src.pipeline.runner import PipelineRunner
from src.pipeline.steps.base import StageResult


class RecordingStage:
    def __init__(self, name: str, calls: list[str], assert_local_state: bool = False):
        self.name = name
        self.calls = calls
        self.assert_local_state = assert_local_state

    def execute(self, context: PipelineContext) -> StageResult:
        if self.assert_local_state:
            self.assert_artifacts_exist(context)
        self.calls.append(self.name)
        return StageResult(outputs={"stage": self.name})

    @staticmethod
    def assert_artifacts_exist(context: PipelineContext) -> None:
        if not (context.run_directory / "pipeline_config.json").exists():
            raise AssertionError("Configuration was not persisted before tracking.")
        if not context.state_path.exists():
            raise AssertionError("State was not persisted before tracking.")


class FailingStage:
    name = "failing"

    def execute(self, context: PipelineContext) -> StageResult:
        raise RuntimeError("expected failure")


class PipelineRunnerTests(unittest.TestCase):
    def configuration(self, output: Path) -> PipelineConfiguration:
        return PipelineConfiguration.from_mapping(
            {
                "run": {"name": "test", "phase": "smoke", "output": str(output)},
                "dataset": {"clean_manifest": "clean.jsonl"},
                "steganography": {"stego_manifest": "stego.jsonl"},
                "detector": {"enabled": False},
                "tracking": {"enabled": False, "mode": "disabled"},
            }
        )

    def test_runs_stages_in_order_and_persists_before_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls: list[str] = []
            stages = [
                RecordingStage("dataset", calls),
                RecordingStage("embedding", calls),
                RecordingStage("tracking", calls, assert_local_state=True),
            ]
            run_directory = PipelineRunner(
                self.configuration(root),
                root / "input.json",
                stages=stages,
            ).run()

            self.assertEqual(calls, ["dataset", "embedding", "tracking"])
            state = json.loads((run_directory / "pipeline_state.json").read_text())
            self.assertEqual(
                [stage["status"] for stage in state["stages"]],
                ["completed", "completed", "completed"],
            )

    def test_persists_failed_stage_before_reraising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = PipelineRunner(
                self.configuration(root),
                root / "input.json",
                stages=[FailingStage()],
            )
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                runner.run()
            run_directory = next(root.iterdir())
            state = json.loads((run_directory / "pipeline_state.json").read_text())
            self.assertEqual(state["stages"][0]["status"], "failed")
            self.assertEqual(state["stages"][0]["error"], "expected failure")


if __name__ == "__main__":
    unittest.main()
