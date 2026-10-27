from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.datasets.manifest import write_manifest
from src.steganalysis.evaluate import evaluate_checkpoints
from src.steganalysis.srnet import SRNET_REFERENCE_COMMIT


class FakeModel:
    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def to(self, device):
        return self


class CheckpointEvaluationTests(unittest.TestCase):
    def manifest(self, root: Path) -> Path:
        path = root / "sd3_manifest.jsonl"
        rows = []
        for index in range(2):
            rows.append(
                {
                    "image_id": f"sd3_{index}",
                    "domain": "ai",
                    "split": "test",
                    "clean_path": f"/clean/{index}.png",
                    "stego_path": f"/stego/{index}.png",
                    "source": "genimagepp-sd3",
                    "algorithm": "suniward",
                    "payload_bpp": 0.4,
                    "embedding_seed": index + 1,
                }
            )
        write_manifest(path, rows)
        return path

    def checkpoint(self, root: Path, folder: str, seed: int) -> Path:
        run = root / folder
        checkpoint = run / "checkpoints" / f"seed{seed}_mixed.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            {
                "state_dict": {"weight": torch.tensor([float(seed)])},
                "seed": seed,
                "train_domain": "mixed",
                "selected_epoch": 290,
                "selection_criterion": "val_loss",
                "val_loss": 0.6,
                "detector_commit": SRNET_REFERENCE_COMMIT,
            },
            checkpoint,
        )
        (run / "config.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": "source-manifest",
                    "epochs": 300,
                    "batch_size": 32,
                    "learning_rate": 1e-4,
                    "train_pairs": None,
                    "mixed_pairs_per_domain": None,
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def test_evaluates_and_aggregates_checkpoints_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            checkpoints = [
                self.checkpoint(root, "srnet_a", 1),
                self.checkpoint(root, "srnet_b", 2),
            ]
            prediction = (
                [0, 1, 0, 1],
                [0, 1, 1, 1],
                [0.1, 0.8, 0.7, 0.9],
                ["a", "a", "b", "b"],
            )
            with patch(
                "src.steganalysis.evaluate.initialize_srnet", return_value=FakeModel()
            ), patch(
                "src.steganalysis.evaluate.predict_srnet", return_value=prediction
            ), patch(
                "src.steganalysis.evaluate.accelerator_state", return_value={}
            ):
                run = evaluate_checkpoints(
                    manifest=manifest,
                    checkpoints=checkpoints,
                    test_domain="ai",
                    test_set_name="genimagepp-sd3",
                    batch_size=4,
                    output=root / "results",
                    reference_root=root / "reference",
                    device_name="cpu",
                )

            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["evaluations"]), 2)
            self.assertEqual(len(summary["conditions"]), 1)
            self.assertEqual(summary["conditions"][0]["seeds"], [1, 2])
            self.assertEqual(summary["sample_count"], 4)
            self.assertEqual(len(list((run / "predictions").glob("*.jsonl"))), 2)
            self.assertTrue((run / "dataset_manifest.jsonl").exists())

    def test_rejects_a_checkpoint_from_another_detector_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            checkpoint = self.checkpoint(root, "srnet_a", 1)
            payload = torch.load(checkpoint, weights_only=True)
            payload["detector_commit"] = "wrong"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "detector commit"):
                evaluate_checkpoints(
                    manifest=manifest,
                    checkpoints=[checkpoint],
                    test_domain="ai",
                    test_set_name="sd3",
                    batch_size=4,
                    output=root / "results",
                    reference_root=root / "reference",
                    device_name="cpu",
                )


if __name__ == "__main__":
    unittest.main()
