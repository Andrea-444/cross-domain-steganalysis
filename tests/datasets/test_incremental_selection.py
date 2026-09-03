from __future__ import annotations

import csv
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from src.datasets.extractor import extract_selection_stage
from src.datasets.index import build_dataset_index
from src.datasets.manifest import read_manifest
from src.datasets.selection import create_selection_manifest


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


class IncrementalDatasetSelectionTests(unittest.TestCase):
    def build_sources(self, root: Path) -> tuple[Path, Path]:
        real = root / "real"
        ai_source = root / "ai-source"
        for class_index, class_id in enumerate(("class-a", "class-b", "class-c")):
            for image_index in range(8):
                write_image(
                    real / class_id / f"real-{image_index}.jpg",
                    10 + class_index * 20 + image_index,
                )
                write_image(
                    ai_source / class_id / f"ai-{image_index}.png",
                    100 + class_index * 20 + image_index,
                )
        ai_archive = root / "ai.zip"
        with zipfile.ZipFile(ai_archive, "w") as archive:
            for path in sorted(ai_source.rglob("*.png")):
                archive.write(path, path.relative_to(ai_source).as_posix())
        return real, ai_archive

    def create_selection(self, root: Path) -> Path:
        real, ai_archive = self.build_sources(root)
        real_index = build_dataset_index(
            source=real,
            dataset_id="natural-fixture",
            dataset_version="v1",
            domain="real",
            output=root / "real-index.jsonl",
        )
        ai_index = build_dataset_index(
            source=ai_archive,
            dataset_id="ai-fixture",
            dataset_version="v1",
            domain="ai",
            output=root / "ai-index.jsonl",
        )
        return create_selection_manifest(
            real_index=real_index,
            ai_index=ai_index,
            output=root / "selection.jsonl",
            count_per_domain=18,
            audit_count=3,
            pilot_total=6,
            seed=17,
        )

    def test_selection_is_matched_deterministic_and_test_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = self.create_selection(root)
            records = read_manifest(selection)

            self.assertEqual(Counter(record["domain"] for record in records), {"real": 18, "ai": 18})
            for domain in ("real", "ai"):
                domain_records = [record for record in records if record["domain"] == domain]
                self.assertEqual(
                    Counter(record["class_id"] for record in domain_records),
                    {"class-a": 6, "class-b": 6, "class-c": 6},
                )
                self.assertEqual(
                    Counter(record["split"] for record in domain_records),
                    {"train": 12, "val": 3, "test": 3},
                )
                self.assertEqual(
                    sum(record["cohort"] == "audit" for record in domain_records),
                    3,
                )
                self.assertEqual(
                    sum(
                        record["cohort"] in {"audit", "pilot_addition"}
                        for record in domain_records
                    ),
                    6,
                )
            self.assertTrue(
                all(
                    record["cohort"] == "full_addition"
                    for record in records
                    if record["split"] == "test"
                )
            )

            second = create_selection_manifest(
                real_index=root / "real-index.jsonl",
                ai_index=root / "ai-index.jsonl",
                output=root / "selection-second.jsonl",
                count_per_domain=18,
                audit_count=3,
                pilot_total=6,
                seed=17,
            )
            comparable_fields = (
                "candidate_id",
                "domain",
                "class_id",
                "split",
                "cohort",
                "domain_selection_rank",
            )
            self.assertEqual(
                [tuple(record[field] for field in comparable_fields) for record in records],
                [
                    tuple(record[field] for field in comparable_fields)
                    for record in read_manifest(second)
                ],
            )

    def test_incremental_stages_extract_only_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = self.create_selection(root)
            output = root / "selected"

            audit = extract_selection_stage(selection, output, "audit")
            pilot = extract_selection_stage(selection, output, "pilot")
            full = extract_selection_stage(selection, output, "full")
            repeated = extract_selection_stage(selection, output, "full")

            self.assertEqual(
                (audit["requested_count"], audit["extracted_count"], audit["reused_count"]),
                (6, 6, 0),
            )
            self.assertEqual(
                (pilot["requested_count"], pilot["extracted_count"], pilot["reused_count"]),
                (12, 6, 6),
            )
            self.assertEqual(
                (full["requested_count"], full["extracted_count"], full["reused_count"]),
                (36, 24, 12),
            )
            self.assertEqual(
                (repeated["requested_count"], repeated["extracted_count"], repeated["reused_count"]),
                (36, 0, 36),
            )
            self.assertEqual(len(read_manifest(output / "extraction_manifest.jsonl")), 36)
            self.assertEqual(
                len([path for path in output.rglob("*") if path.suffix in {".jpg", ".png"}]),
                36,
            )

    def test_modified_extracted_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = self.create_selection(root)
            output = root / "selected"
            extract_selection_stage(selection, output, "audit")
            extracted = next(path for path in output.rglob("*.jpg"))
            extracted.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                extract_selection_stage(selection, output, "audit")

    def test_flat_archive_uses_explicit_class_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            write_image(images / "one.jpg", 10)
            write_image(images / "two.jpg", 20)
            archive_path = root / "flat.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(images / "one.jpg", "one.jpg")
                archive.write(images / "two.jpg", "two.jpg")
            class_map = root / "classes.csv"
            with class_map.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "synset"])
                writer.writeheader()
                writer.writerow({"filename": "one.jpg", "synset": "n0001"})
                writer.writerow({"filename": "two.jpg", "synset": "n0002"})

            index = build_dataset_index(
                source=archive_path,
                dataset_id="flat",
                dataset_version="v1",
                domain="real",
                output=root / "flat-index.jsonl",
                class_map_path=class_map,
                class_map_member_column="filename",
                class_map_class_column="synset",
                infer_class_from_parent=False,
            )
            self.assertEqual(
                {record["class_id"] for record in read_manifest(index)},
                {"n0001", "n0002"},
            )

    def test_index_rejects_archive_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            write_image(image, 10)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(image, "../class-a/image.png")
            with self.assertRaisesRegex(ValueError, "Unsafe dataset member"):
                build_dataset_index(
                    archive_path,
                    "unsafe",
                    "v1",
                    "ai",
                    root / "unsafe-index.jsonl",
                )

    @unittest.skipUnless(
        shutil.which("zstd") and shutil.which("tar"),
        "TAR.ZST tools are not installed",
    )
    def test_tar_zstd_index_and_incremental_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            ai = root / "ai"
            for index in range(3):
                write_image(real / "class-a" / f"real-{index}.jpg", 10 + index)
                write_image(ai / "class-a" / f"ai-{index}.png", 100 + index)
            tar_path = root / "ai.tar"
            with tarfile.open(tar_path, "w") as archive:
                archive.add(ai / "class-a", arcname="class-a")
            compressed = root / "ai.tar.zst"
            subprocess.run(
                ["zstd", "-q", "-f", str(tar_path), "-o", str(compressed)],
                check=True,
            )
            real_index = build_dataset_index(
                real,
                "real-zstd-fixture",
                "v1",
                "real",
                root / "real-zstd-index.jsonl",
            )
            ai_index = build_dataset_index(
                compressed,
                "ai-zstd-fixture",
                "v1",
                "ai",
                root / "ai-zstd-index.jsonl",
            )
            selection = create_selection_manifest(
                real_index,
                ai_index,
                root / "zstd-selection.jsonl",
                count_per_domain=2,
                audit_count=1,
                pilot_total=1,
                seed=7,
            )
            result = extract_selection_stage(selection, root / "zstd-selected", "full")
            self.assertEqual(result["extracted_count"], 4)


if __name__ == "__main__":
    unittest.main()
