from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.datasets.manifest import write_manifest
from src.pipeline.configuration import PipelineConfiguration
from src.pipeline.context import PipelineContext
from src.pipeline.steps import DatasetStage, DetectorStage, EmbeddingStage, TrackingStage


def record(image_id: str, domain: str, stego: bool = False) -> dict[str, object]:
    item: dict[str, object] = {
        "image_id": image_id,
        "domain": domain,
        "clean_path": f"/{image_id}.png",
        "split": "test",
        "source": "fixture",
    }
    if stego:
        item.update(
            {
                "stego_path": f"/{image_id}_stego.png",
                "algorithm": "suniward",
                "payload_bpp": 0.2,
                "embedding_seed": 1,
            }
        )
    return item


class PipelineStageTests(unittest.TestCase):
    def test_existing_manifests_and_disabled_terminal_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = write_manifest(
                root / "clean.jsonl",
                [record("real-1", "real"), record("ai-1", "ai")],
            )
            stego = write_manifest(
                root / "stego.jsonl",
                [record("real-1", "real", True), record("ai-1", "ai", True)],
            )
            configuration = PipelineConfiguration.from_mapping(
                {
                    "run": {"name": "test", "phase": "smoke", "output": str(root)},
                    "dataset": {"clean_manifest": str(clean)},
                    "steganography": {"stego_manifest": str(stego)},
                    "detector": {"enabled": False},
                    "tracking": {"enabled": False, "mode": "disabled"},
                }
            )
            context = PipelineContext(configuration, root / "run", root / "config.json")
            context.run_directory.mkdir()

            dataset_result = DatasetStage().execute(context)
            embedding_result = EmbeddingStage().execute(context)
            detector_result = DetectorStage().execute(context)
            tracking_result = TrackingStage().execute(context)

            self.assertEqual(dataset_result.outputs["record_count"], 2)
            self.assertEqual(embedding_result.outputs["record_count"], 2)
            self.assertEqual(detector_result.outputs["status"], "skipped")
            self.assertEqual(tracking_result.outputs["status"], "disabled")

    def test_tracking_failure_does_not_invalidate_local_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = PipelineConfiguration.from_mapping(
                {
                    "run": {"name": "test", "phase": "smoke", "output": str(root)},
                    "dataset": {"clean_manifest": "clean.jsonl"},
                    "steganography": {"stego_manifest": "stego.jsonl"},
                    "detector": {"enabled": False},
                    "tracking": {"enabled": True, "mode": "offline"},
                }
            )
            context = PipelineContext(configuration, root / "run", root / "config.json")
            context.run_directory.mkdir()
            with patch(
                "src.pipeline.steps.tracking.log_pipeline_run",
                side_effect=RuntimeError("tracking unavailable"),
            ):
                result = TrackingStage().execute(context)
            self.assertEqual(result.outputs["status"], "failed")
            self.assertEqual(result.outputs["error"], "tracking unavailable")


if __name__ == "__main__":
    unittest.main()
