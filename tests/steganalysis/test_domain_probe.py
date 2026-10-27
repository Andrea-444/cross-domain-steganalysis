from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.steganalysis.domain_probe import DomainDataset, run_domain_probe


def build_cohort(directory: Path, per_split: int = 4) -> Path:
    records = []
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for domain in ("real", "ai"):
            for index in range(per_split):
                path = directory / split / domain / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    rng.integers(0, 255, (256, 256), dtype=np.uint8), mode="L"
                ).save(path)
                records.append(
                    {
                        "image_id": f"{split}_{domain}_{index}",
                        "domain": domain,
                        "split": split,
                        "clean_path": str(path),
                        "class_id": "n0000000",
                    }
                )
    manifest = directory / "clean.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return manifest


class DomainDatasetTests(unittest.TestCase):
    def test_labels_by_domain_not_by_stego(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = build_cohort(Path(raw))
            records = [json.loads(l) for l in open(manifest)]
            dataset = DomainDataset(records, split="train")
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
            self.assertEqual(sorted(labels), [0] * 4 + [1] * 4)

    def test_uses_clean_covers_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = build_cohort(Path(raw))
            records = [json.loads(l) for l in open(manifest)]
            dataset = DomainDataset(records, split="train")
            # one entry per cover, not two as the clean/stego pair dataset yields
            self.assertEqual(len(dataset), 8)

    def test_rejects_empty_split(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = build_cohort(Path(raw))
            records = [json.loads(l) for l in open(manifest)]
            with self.assertRaises(ValueError):
                DomainDataset(records, split="nonexistent")


class ProbeOutputTests(unittest.TestCase):
    def fake_training(self, **kwargs):
        from src.steganalysis.domain_probe import ProbeOutcome

        return ProbeOutcome(
            model=object(),
            device=torch.device("cpu"),
            history=[{"epoch": 1, "train_loss": 0.7, "val_loss": 0.69,
                      "val_balanced_accuracy": 0.5}],
            best_epoch=1,
            best_val_loss=0.69,
        )

    def fake_predict(self, **kwargs):
        return [0, 0, 1, 1], [0, 1, 1, 1], [0.2, 0.7, 0.8, 0.9], ["a", "b", "c", "d"]

    def test_writes_summary_with_per_seed_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = build_cohort(Path(raw))
            with patch(
                "src.steganalysis.domain_probe.train_domain_model",
                side_effect=self.fake_training,
            ), patch(
                "src.steganalysis.domain_probe.predict_domain",
                side_effect=self.fake_predict,
            ):
                out = run_domain_probe(
                    manifest=manifest, seeds=[1, 2], epochs=1, batch_size=2,
                    learning_rate=1e-3, output=Path(raw) / "out",
                    reference_root="external/srnet-reference", device="cpu",
                )
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual([r["seed"] for r in summary["runs"]], [1, 2])
            self.assertIn("balanced_accuracy_mean", summary["aggregate"])
            self.assertIn("roc_auc_mean", summary["aggregate"])
            self.assertIn("provenance", summary)
            self.assertTrue((out / "predictions" / "seed1.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
