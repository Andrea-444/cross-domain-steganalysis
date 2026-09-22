from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from src.datasets.class_mapping import build_genimagepp_class_maps
from src.datasets.manifest import read_manifest, write_manifest
from src.datasets.preparation import prepare_extracted_selection


class ClassMappingTests(unittest.TestCase):
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
    def test_preserves_domain_class_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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


if __name__ == "__main__":
    unittest.main()
