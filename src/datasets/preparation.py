"""Normalize an extracted, split-locked selection into clean cover images."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from src.datasets.extractor import file_sha256
from src.logging_config import ProgressReporter, get_logger
from src.datasets.manifest import read_manifest, validate_clean_manifest, write_manifest
from src.datasets.selection import validate_selection_manifest


LOGGER = get_logger(__name__)


class UndersizedImageError(ValueError):
    """Raised when crop-only preprocessing cannot fill the target size."""


def normalize_image(image: Image.Image, size: int, mode: str = "fit") -> Image.Image:
    """Convert to grayscale and reduce to `size` x `size`.

    ``fit`` crops to aspect then resamples, so any source resolution works.
    ``crop`` takes a centred native crop and never resamples: resampling
    attenuates the high-frequency residuals steganalysis depends on, and
    domains whose sources differ in resolution would be attenuated by
    different amounts, which is itself a confound.
    """
    if mode not in {"fit", "crop"}:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    image = ImageOps.exif_transpose(image).convert("L")
    if mode == "fit":
        return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    width, height = image.size
    if min(width, height) < size:
        raise UndersizedImageError(
            f"Image is {width}x{height}, smaller than the {size}x{size} crop."
        )
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare_extracted_selection(
    extraction_manifest: str | Path,
    name: str,
    output: str | Path,
    size: int = 256,
    mode: str = "fit",
    skip_undersized: bool = False,
) -> Path:
    if not name.strip():
        raise ValueError("Dataset name must not be empty.")
    if size <= 0:
        raise ValueError("Image size must be positive.")
    extraction_manifest = Path(extraction_manifest).resolve()
    extraction_records = read_manifest(extraction_manifest)
    validate_selection_manifest(extraction_records)
    output = Path(output)
    dataset_root = output / name
    manifest_path = output / "manifests" / f"{name}_clean.jsonl"
    config_path = output / "manifests" / f"{name}_clean.config.json"
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError(
            f"Clean manifest or configuration already exists: {manifest_path}"
        )

    clean_records: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    LOGGER.info(
        "Preparing %d covers with mode=%s at size %d.",
        len(extraction_records), mode, size,
    )
    progress = ProgressReporter(LOGGER, total=len(extraction_records), label="prepare")
    for record in extraction_records:
        source = Path(str(record.get("local_path", "")))
        if not source.is_file():
            raise ValueError(f"Extracted source image is missing: {source}")
        checksum = file_sha256(source)
        if checksum != record.get("sha256"):
            raise ValueError(f"Extracted source image changed: {source}")
        domain = str(record["domain"])
        image_id = str(record["candidate_id"])
        destination = dataset_root / domain / f"{image_id}.png"
        try:
            _normalize_incrementally(source, destination, size, mode=mode)
        except UndersizedImageError as error:
            if not skip_undersized:
                raise
            # Recorded rather than silently dropped: excluding covers changes
            # the class balance the selection deliberately established.
            skipped.append(
                {"image_id": image_id, "domain": domain, "reason": str(error)}
            )
            progress.advance()
            continue
        with Image.open(destination) as image:
            if image.mode != "L" or image.size != (size, size):
                raise ValueError(f"Invalid processed image: {destination}")
        progress.advance()
        clean_records.append(
            {
                "image_id": image_id,
                "candidate_id": image_id,
                "domain": domain,
                "class_id": str(record["class_id"]),
                "source": str(record["dataset_id"]),
                "dataset_version": str(record["dataset_version"]),
                "source_path": str(source.resolve()),
                "source_member": str(record["source_member"]),
                "source_sha256": checksum,
                "clean_path": str(destination.resolve()),
                "split": str(record["split"]),
                "selection_seed": int(record["selection_seed"]),
                "width": size,
                "height": size,
                "color_mode": "L",
            }
        )

    progress.finish()
    if skipped:
        LOGGER.warning(
            "Skipped %d undersized cover(s); see the run configuration.", len(skipped)
        )
    clean_records.sort(key=lambda record: str(record["image_id"]))
    validate_clean_manifest(clean_records)
    write_manifest(manifest_path, clean_records)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "extraction_manifest": str(extraction_manifest),
                "extraction_manifest_sha256": file_sha256(extraction_manifest),
                "size": size,
                "preprocess_mode": mode,
                "color_mode": "L",
                "image_format": "PNG",
                "record_count": len(clean_records),
                "skipped_undersized": skipped,
                "clean_manifest_sha256": file_sha256(manifest_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _normalize_incrementally(
    source: Path, destination: Path, size: int, mode: str = "fit"
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with Image.open(source) as image:
            image = normalize_image(image, size=size, mode=mode)
            image.save(temporary, format="PNG", optimize=False)
        if destination.exists():
            if _sha256(destination) != _sha256(temporary):
                raise FileExistsError(
                    f"Processed destination exists with different content: {destination}"
                )
            return
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
