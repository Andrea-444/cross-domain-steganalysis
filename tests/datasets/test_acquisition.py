from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.datasets.acquisition import fetch_repo_file, stream_images


JPEG_BYTES = b"\xff\xd8\xff\xe0FAKEJPEGBODY\xff\xd9"
PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKEPNGBODY"


def row(label: int, payload: bytes, name: str) -> dict[str, object]:
    return {"image": {"bytes": payload, "path": name}, "label": label}


class FakeFeature:
    names = ["n01440764", "n01443537", "n01484850"]


class FakeDataset:
    """Stands in for a streaming datasets.IterableDataset."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.features = {"label": FakeFeature()}

    def __iter__(self):
        return iter(self.rows)

    def cast_column(self, *args, **kwargs):
        return self


class StreamImagesTests(unittest.TestCase):
    def build(self, per_class: int = 4) -> FakeDataset:
        rows = []
        for label in range(3):
            for index in range(per_class):
                rows.append(row(label, JPEG_BYTES, f"{label}_{index}.JPEG"))
        return FakeDataset(rows)

    def test_writes_original_bytes_unmodified(self) -> None:
        """Re-encoding would add a JPEG generation and corrupt the residuals."""
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=self.build()
        ):
            out = stream_images("repo", "validation", Path(raw) / "img", limit_per_class=2)
            files = sorted(out.rglob("*.JPEG"))
            self.assertTrue(files)
            self.assertEqual(files[0].read_bytes(), JPEG_BYTES)

    def test_respects_limit_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=self.build()
        ):
            out = stream_images("repo", "validation", Path(raw) / "img", limit_per_class=2)
            for synset in FakeFeature.names:
                self.assertEqual(len(list((out / synset).glob("*"))), 2)

    def test_class_directories_named_by_synset(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=self.build()
        ):
            out = stream_images("repo", "validation", Path(raw) / "img", limit_per_class=1)
            self.assertEqual(
                sorted(p.name for p in out.iterdir() if p.is_dir()), FakeFeature.names
            )

    def test_class_allowlist_filters(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=self.build()
        ):
            out = stream_images(
                "repo", "validation", Path(raw) / "img",
                limit_per_class=2, classes=["n01443537"],
            )
            self.assertEqual([p.name for p in out.iterdir() if p.is_dir()], ["n01443537"])

    def test_stops_early_once_every_target_class_is_full(self) -> None:
        dataset = self.build(per_class=50)
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=dataset
        ):
            out = stream_images("repo", "validation", Path(raw) / "img", limit_per_class=1)
            self.assertEqual(len(list(out.rglob("*.JPEG"))), 3)

    def test_max_classes_limits_breadth(self) -> None:
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=self.build()
        ):
            out = stream_images(
                "repo", "validation", Path(raw) / "img",
                limit_per_class=2, max_classes=2,
            )
            self.assertEqual(len([p for p in out.iterdir() if p.is_dir()]), 2)

    def test_extension_follows_source_payload(self) -> None:
        dataset = FakeDataset([row(0, PNG_BYTES, "a.png"), row(1, JPEG_BYTES, "b.JPEG")])
        with tempfile.TemporaryDirectory() as raw, patch(
            "src.datasets.acquisition.load_dataset", return_value=dataset
        ):
            out = stream_images("repo", "validation", Path(raw) / "img", limit_per_class=1)
            self.assertTrue(list(out.rglob("*.png")))
            self.assertTrue(list(out.rglob("*.JPEG")))


class FetchRepoFileTests(unittest.TestCase):
    def test_downloads_into_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "cached.tar.zst"
            source.write_bytes(b"archive")
            with patch(
                "src.datasets.acquisition.hf_hub_download", return_value=str(source)
            ) as download:
                result = fetch_repo_file(
                    "Lunahera/genimagepp", "archives/flux.tar.zst", Path(raw) / "out"
                )
            self.assertTrue(result.exists())
            self.assertEqual(result.name, "flux.tar.zst")
            self.assertEqual(download.call_args.kwargs["repo_type"], "dataset")

    def test_existing_file_is_not_redownloaded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "out"
            out.mkdir()
            (out / "flux.tar.zst").write_bytes(b"already here")
            with patch("src.datasets.acquisition.hf_hub_download") as download:
                result = fetch_repo_file(
                    "Lunahera/genimagepp", "archives/flux.tar.zst", out
                )
            download.assert_not_called()
            self.assertEqual(result.read_bytes(), b"already here")


if __name__ == "__main__":
    unittest.main()
