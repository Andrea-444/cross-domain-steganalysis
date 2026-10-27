from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from src.datasets.class_mapping import (
    _numeric_filename_base,
    _numeric_prompt_mapping,
    build_genimagepp_class_maps,
)
from src.datasets.manifest import read_manifest, write_manifest
from src.datasets.preparation import prepare_extracted_selection


class ClassMappingTests(unittest.TestCase):
    def test_numeric_prompt_indices_map_six_images_per_class(self) -> None:
        prompts = [f"prompt {index}" for index in range(12)]
        self.assertEqual(_numeric_prompt_mapping("0", prompts, 2), (0, 0))
        self.assertEqual(_numeric_prompt_mapping("5", prompts, 2), (5, 0))
        self.assertEqual(_numeric_prompt_mapping("6", prompts, 2), (6, 1))
        self.assertEqual(_numeric_prompt_mapping("11", prompts, 2), (11, 1))

    def test_one_based_filenames_map_to_zero_based_prompts(self) -> None:
        prompts = [f"prompt {index}" for index in range(12)]
        self.assertEqual(
            _numeric_prompt_mapping("1", prompts, 2, filename_base=1), (0, 0)
        )
        self.assertEqual(
            _numeric_prompt_mapping("6", prompts, 2, filename_base=1), (5, 0)
        )
        self.assertEqual(
            _numeric_prompt_mapping("7", prompts, 2, filename_base=1), (6, 1)
        )
        self.assertEqual(
            _numeric_prompt_mapping("12", prompts, 2, filename_base=1), (11, 1)
        )

    def test_numeric_filename_base_is_detected_from_the_complete_set(self) -> None:
        self.assertEqual(_numeric_filename_base([str(i) for i in range(12)], 12), 0)
        self.assertEqual(
            _numeric_filename_base([str(i) for i in range(1, 13)], 12), 1
        )

    def test_numeric_filename_set_must_be_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly once"):
            _numeric_filename_base(["1", "2", "4"], 3)

    def test_numeric_filenames_require_the_official_prompt_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "--indexed-prompts"):
            _numeric_prompt_mapping("0", None, 2)

    def test_numeric_prompt_list_must_have_six_entries_per_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 12 indexed prompts"):
            _numeric_prompt_mapping("0", ["too short"], 2)

    def test_numeric_prompt_index_must_be_in_range(self) -> None:
        prompts = [f"prompt {index}" for index in range(12)]
        with self.assertRaisesRegex(ValueError, "outside"):
            _numeric_prompt_mapping("12", prompts, 2)

    def test_rejects_non_thousand_class_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "real").mkdir()
            archive = root / "ai.zip"
            with zipfile.ZipFile(archive, "w"):
                pass
            labels = root / "labels.json"
            labels.write_text(json.dumps({"labels": ["one"]}), encoding="utf-8")
            simple = root / "simple.json"
            simple.write_text(json.dumps(["one"]), encoding="utf-8")
            index = root / "index.json"
            index.write_text(json.dumps({"0": ["n1", "one"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "1,000"):
                build_genimagepp_class_maps(
                    root / "real", archive, labels, simple, index, root / "maps"
                )


class PrepareSelectionTests(unittest.TestCase):
    def extraction_fixture(self, root: Path) -> Path:
        records = []
        for domain, value, candidate_id in (
            ("real", 20, "a" * 24),
            ("ai", 40, "b" * 24),
        ):
            source = root / f"{domain}.jpg"
            Image.new("RGB", (20, 18), (value, value, value)).save(source)
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            records.append(
                {
                    "candidate_id": candidate_id,
                    "dataset_id": f"{domain}-fixture",
                    "dataset_version": "v1",
                    "domain": domain,
                    "source_kind": "directory",
                    "source_path": str(root),
                    "source_member": source.name,
                    "class_id": "n00000001",
                    "extension": ".jpg",
                    "selection_seed": 42,
                    "split": "test",
                    "cohort": "full_addition",
                    "local_path": str(source),
                    "sha256": checksum,
                }
            )
        extraction = root / "extraction.jsonl"
        write_manifest(extraction, records)
        return extraction

    def test_preserves_domain_class_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self.extraction_fixture(root)
            manifest = prepare_extracted_selection(
                extraction, "fixture", root / "processed", size=16
            )
            clean_records = read_manifest(manifest)
            self.assertEqual(len(clean_records), 2)
            self.assertEqual({record["class_id"] for record in clean_records}, {"n00000001"})
            self.assertEqual({record["split"] for record in clean_records}, {"test"})
            for record in clean_records:
                with Image.open(record["clean_path"]) as image:
                    self.assertEqual(image.mode, "L")
                    self.assertEqual(image.size, (16, 16))

    def test_can_prepare_only_the_external_evaluation_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = self.extraction_fixture(root)
            manifest = prepare_extracted_selection(
                extraction,
                "ai-only",
                root / "processed",
                size=16,
                domains=("ai",),
            )
            clean_records = read_manifest(manifest)
            self.assertEqual(len(clean_records), 1)
            self.assertEqual(clean_records[0]["domain"], "ai")
            config = json.loads(
                manifest.with_suffix(".config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["domains"], ["ai"])


if __name__ == "__main__":
    unittest.main()
