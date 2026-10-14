from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.datasets.manifest import write_manifest
from src.evaluation.metrics import calculate_metrics
from src.steganalysis.experiments import (
    embedding_condition,
    required_train_domains,
    run_srnet_protocol,
    select_directions,
)
from src.steganalysis.srnet import TrainingOutcome, resolve_device


class DeviceSelectionTests(unittest.TestCase):
    def test_prefers_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device(None).type, "cuda")

    def test_prefers_mps_over_cpu_when_cuda_absent(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            pass
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            self.assertEqual(resolve_device(None).type, "mps")

    def test_falls_back_to_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            self.assertEqual(resolve_device(None).type, "cpu")

    def test_explicit_override_wins(self) -> None:
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            self.assertEqual(resolve_device("cpu").type, "cpu")


class ProbabilityMetricTests(unittest.TestCase):
    def test_probability_metrics_present(self) -> None:
        labels = [0, 0, 1, 1]
        predictions = [0, 1, 0, 1]
        probabilities = [0.1, 0.6, 0.4, 0.9]
        metrics = calculate_metrics(labels, predictions, probabilities)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIsInstance(metrics["roc_auc"], float)

    def test_metrics_without_probabilities_omit_auc(self) -> None:
        metrics = calculate_metrics([0, 1], [0, 1])
        self.assertNotIn("roc_auc", metrics)
        self.assertIn("accuracy", metrics)

    def test_single_class_labels_degrade_safely(self) -> None:
        metrics = calculate_metrics([1, 1], [1, 1], [0.8, 0.9])
        self.assertIsNone(metrics["roc_auc"])


if __name__ == "__main__":
    unittest.main()


class EmbeddingConditionTests(unittest.TestCase):
    def test_uniform_values_collapse_to_scalar(self) -> None:
        records = [{"algorithm": "suniward", "payload_bpp": 0.4, "embedding_seed": 42}] * 3
        condition = embedding_condition(records)
        self.assertEqual(condition["algorithm"], "suniward")
        self.assertEqual(condition["payload_bpp"], 0.4)
        self.assertEqual(condition["embedding_seed"], 42)

    def test_per_image_seeds_summarised_not_listed(self) -> None:
        records = [
            {"algorithm": "suniward", "payload_bpp": 0.4, "embedding_seed": seed}
            for seed in range(50)
        ]
        condition = embedding_condition(records)
        self.assertEqual(condition["embedding_seed"], {"per_image": True, "distinct": 50})

    def test_mixed_algorithms_are_reported(self) -> None:
        records = [
            {"algorithm": "suniward", "payload_bpp": 0.4},
            {"algorithm": "mipod", "payload_bpp": 0.4},
        ]
        self.assertEqual(embedding_condition(records)["algorithm"], ["mipod", "suniward"])


class ProtocolHarness:
    """Fake training and prediction, so the tests exercise the protocol only."""

    def build_manifest(self, directory: Path) -> Path:
        records = []
        for domain in ("real", "ai"):
            for index in range(4):
                split = ("train", "val", "test", "test")[index]
                records.append(
                    {
                        "image_id": f"{domain}_{index}",
                        "domain": domain,
                        "split": split,
                        "clean_path": f"/{domain}_{index}.png",
                        "stego_path": f"/{domain}_{index}_stego.png",
                        "algorithm": "suniward",
                        "payload_bpp": 0.4,
                        "embedding_seed": 42,
                        "source": "fixture",
                    }
                )
        path = directory / "manifest.jsonl"
        write_manifest(path, records)
        return path

    def fake_training(self, **kwargs):
        history = [
            {
                "epoch": 1,
                "train_loss": 0.7,
                "train_accuracy": 0.5,
                "val_loss": 0.6,
                "val_accuracy": 0.55,
                "val_balanced_accuracy": 0.55,
                "learning_rate": 1e-3,
                "duration_seconds": 0.01,
            }
        ]
        return TrainingOutcome(
            model=object(),
            device=torch.device("cpu"),
            history=history,
            best_epoch=1,
            best_val_loss=0.6,
            best_state={"weight": torch.zeros(1)},
        )

    def fake_prediction(self, **kwargs):
        domain = kwargs["test_domain"]
        image_ids = [f"{domain}_2", f"{domain}_2", f"{domain}_3", f"{domain}_3"]
        return [0, 1, 0, 1], [0, 1, 1, 1], [0.2, 0.8, 0.6, 0.9], image_ids

    def run_protocol(
        self,
        directory: Path,
        directions: list[str] | None = None,
    ) -> Path:
        manifest = self.build_manifest(directory)
        with patch(
            "src.steganalysis.experiments.train_srnet", side_effect=self.fake_training
        ) as training, patch(
            "src.steganalysis.experiments.predict_srnet", side_effect=self.fake_prediction
        ):
            self.training_calls = training
            return run_srnet_protocol(
                manifest=manifest,
                seeds=[1, 2],
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                output=directory / "results",
                reference_root="external/srnet-reference",
                device="cpu",
                directions=directions,
            )



class RunOutputTests(ProtocolHarness, unittest.TestCase):
    """The protocol run must persist everything needed to interpret it later."""

    def test_predictions_written_per_direction_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(Path(raw))
            files = sorted(p.name for p in (run_dir / "predictions").glob("*.jsonl"))
            self.assertEqual(len(files), 12)  # 6 directions x 2 seeds
            self.assertIn("seed1_real_to_ai.jsonl", files)
            self.assertIn("seed1_mixed_to_ai.jsonl", files)
            self.assertIn("seed1_mixed_to_real.jsonl", files)
            rows = [
                json.loads(line)
                for line in (run_dir / "predictions" / "seed1_real_to_ai.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 4)
            for key in ("image_id", "domain", "label", "prediction", "probability", "split"):
                self.assertIn(key, rows[0])

    def test_epoch_history_and_checkpoints_written(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(Path(raw))
            history = json.loads((run_dir / "history" / "seed1_real.json").read_text())
            self.assertEqual(history["epochs"][0]["epoch"], 1)
            self.assertIn("val_accuracy", history["epochs"][0])
            self.assertEqual(history["selection"]["criterion"], "val_loss")
            self.assertEqual(history["selection"]["epoch"], 1)
            self.assertTrue((run_dir / "checkpoints" / "seed1_real.pt").exists())

    def test_per_seed_metrics_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(Path(raw))
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual([run["seed"] for run in summary["runs"]], [1, 2])
            self.assertIn("roc_auc", summary["runs"][0]["directions"]["ai_to_ai"])
            self.assertIn("ai_to_ai", summary["summary"])

    def test_summary_aggregates_probability_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(Path(raw))
            summary = json.loads((run_dir / "summary.json").read_text())
            direction = summary["summary"]["ai_to_ai"]
            self.assertIn("roc_auc_mean", direction)
            self.assertIn("pr_auc_mean", direction)

    def test_provenance_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(Path(raw))
            summary = json.loads((run_dir / "summary.json").read_text())
            provenance = summary["provenance"]
            for key in (
                "manifest_sha256",
                "dataset_counts",
                "embedding",
                "detector_commit",
                "repository_commit",
                "repository_dirty",
                "device",
                "platform",
                "dependencies",
                "started_at",
                "finished_at",
                "duration_seconds",
            ):
                self.assertIn(key, provenance)
            self.assertEqual(provenance["embedding"]["algorithm"], "suniward")
            self.assertEqual(provenance["dataset_counts"]["real"]["test"], 2)


class DirectionSelectionTests(unittest.TestCase):
    def test_default_is_every_direction_in_priority_order(self) -> None:
        self.assertEqual(
            select_directions(None),
            [
                "real_to_real",
                "ai_to_ai",
                "mixed_to_real",
                "mixed_to_ai",
                "real_to_ai",
                "ai_to_real",
            ],
        )

    def test_requested_order_does_not_change_protocol_order(self) -> None:
        self.assertEqual(
            select_directions(["real_to_ai", "real_to_real"]),
            ["real_to_real", "real_to_ai"],
        )

    def test_duplicate_request_runs_once(self) -> None:
        self.assertEqual(select_directions(["ai_to_ai", "ai_to_ai"]), ["ai_to_ai"])

    def test_unknown_direction_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as error:
            select_directions(["real_to_real", "real_to_sky"])
        self.assertIn("real_to_sky", str(error.exception))

    def test_train_domains_cover_the_selected_directions_only(self) -> None:
        self.assertEqual(required_train_domains(["real_to_real", "real_to_ai"]), ["real"])
        self.assertEqual(
            required_train_domains(["mixed_to_ai", "ai_to_real"]), ["ai", "mixed"]
        )

    def test_train_domains_are_not_repeated(self) -> None:
        self.assertEqual(required_train_domains(list(select_directions(None))),
                         ["real", "ai", "mixed"])


class StagedRunTests(ProtocolHarness, unittest.TestCase):
    def test_stage_trains_only_the_needed_domain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(
                Path(raw), directions=["real_to_real", "real_to_ai"]
            )
            trained = {call.kwargs["train_domain"] for call in self.training_calls.call_args_list}
            self.assertEqual(trained, {"real"})
            checkpoints = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
            self.assertEqual(checkpoints, ["seed1_real.pt", "seed2_real.pt"])

    def test_stage_evaluates_only_the_selected_directions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(
                Path(raw), directions=["real_to_real", "real_to_ai"]
            )
            files = sorted(p.name for p in (run_dir / "predictions").glob("*.jsonl"))
            self.assertEqual(
                files,
                [
                    "seed1_real_to_ai.jsonl",
                    "seed1_real_to_real.jsonl",
                    "seed2_real_to_ai.jsonl",
                    "seed2_real_to_real.jsonl",
                ],
            )

    def test_stage_summary_holds_the_selected_directions_and_their_gap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(
                Path(raw), directions=["real_to_real", "real_to_ai"]
            )
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(
                sorted(key for key in summary["summary"] if key != "generalization_gaps"),
                ["real_to_ai", "real_to_real"],
            )
            gaps = summary["summary"]["generalization_gaps"]
            self.assertIn("real_to_real_minus_real_to_ai", gaps)
            self.assertNotIn("ai_to_ai_minus_ai_to_real", gaps)

    def test_stage_config_records_the_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = self.run_protocol(Path(raw), directions=["mixed_to_ai"])
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["directions"], ["mixed_to_ai"])
            self.assertEqual(config["train_domains"], ["mixed"])


class TrackedRunIdentityTests(ProtocolHarness, unittest.TestCase):
    """The name W&B receives must come from the run, not from a flag."""

    def write_chain(self, directory: Path) -> None:
        """A manifest with the embedding and preprocessing behind it."""
        prepared = directory / "cohort_clean.jsonl"
        prepared.write_text("", encoding="utf-8")
        (directory / "cohort_clean.config.json").write_text(
            json.dumps({"name": "cohort", "preprocess_mode": "crop", "size": 256}),
            encoding="utf-8",
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "algorithm": "suniward",
                    "payload_bpp": 0.4,
                    "record_count": 8,
                    "input_manifest": str(prepared),
                }
            ),
            encoding="utf-8",
        )

    def start_tracked(self, directory: Path, **kwargs) -> dict:
        self.write_chain(directory)
        manifest = self.build_manifest(directory)
        captured: dict = {}

        def capture(**call):
            captured.update(call)
            return None  # no tracker, so the run continues untracked

        with patch(
            "src.steganalysis.experiments.train_srnet", side_effect=self.fake_training
        ), patch(
            "src.steganalysis.experiments.predict_srnet", side_effect=self.fake_prediction
        ), patch(
            "src.steganalysis.experiments.start_experiment_run", side_effect=capture
        ):
            run_srnet_protocol(
                manifest=manifest,
                seeds=kwargs.get("seeds", [1]),
                epochs=1,
                batch_size=2,
                learning_rate=kwargs.get("learning_rate", 1e-4),
                output=directory / "results",
                reference_root="external/srnet-reference",
                device="cpu",
                directions=kwargs.get("directions"),
                wandb_mode="online",
                wandb_tags=kwargs.get("wandb_tags"),
            )
        return captured

    def test_name_states_role_domain_condition_rate_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            captured = self.start_tracked(
                Path(raw), seeds=[1], directions=["real_to_real", "real_to_ai"]
            )
        self.assertEqual(
            captured["name"], "pilot-real-suniward040-crop-lr1e-4-s1"
        )

    def test_three_seeds_on_a_subset_is_a_stage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            captured = self.start_tracked(
                Path(raw), seeds=[1, 2], directions=["ai_to_ai", "ai_to_real"]
            )
        self.assertTrue(captured["name"].startswith("stage-ai-suniward040-crop-"))
        self.assertEqual(captured["job_type"], "stage")

    def test_group_holds_the_condition_and_the_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            captured = self.start_tracked(Path(raw), directions=["real_to_real"])
        self.assertTrue(captured["group"].startswith("suniward040-crop-"))

    def test_tags_carry_every_facet_and_keep_the_user_tags(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            captured = self.start_tracked(
                Path(raw), directions=["real_to_real"], wandb_tags=["rerun"]
            )
        for expected in ("role:pilot", "detector:srnet", "algorithm:suniward",
                         "preprocess:crop", "lr:1e-4", "domain:real",
                         "direction:real_to_real", "rerun"):
            self.assertIn(expected, captured["tags"])

    def test_the_run_config_records_what_was_sent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.start_tracked(directory, directions=["real_to_real"])
            runs = sorted((directory / "results").glob("srnet_*"))
            config = json.loads((runs[-1] / "config.json").read_text())
        self.assertEqual(config["role"], "pilot")
        self.assertEqual(config["condition"]["preprocess_mode"], "crop")
        self.assertEqual(
            config["tracking"]["run_name"], "pilot-real-suniward040-crop-lr1e-4-s1"
        )
