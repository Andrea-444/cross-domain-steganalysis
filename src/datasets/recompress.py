"""Equalize compression history across cover domains.

The natural covers in this project come from JPEG sources and the AI covers
from lossless PNG. Crop-only preprocessing removes the resampling mismatch but
leaves that difference untouched, and a steganalyzer reads exactly the
high-frequency band JPEG quantization disturbs.

JPEG history cannot be removed from an image that already has it, so the only
direction available is to add matching history to the other domain. This
module re-encodes chosen domains through JPEG at a given quality and decodes
back, leaving the stored file lossless PNG so the rest of the pipeline is
unchanged while the pixel content now carries the quantization artifacts.

Applying this to a domain that is already JPEG-sourced produces double
compression, which has its own detectable signature. Both variants are worth
measuring rather than assuming.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

from PIL import Image

from src.datasets.manifest import read_manifest, write_manifest
from src.logging_config import get_logger


LOGGER = get_logger(__name__)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    """Encode through JPEG at `quality` and decode back to a pixel array."""
    buffer = io.BytesIO()
    image.convert("L").save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("L").copy()


def recompress_cohort(
    manifest: str | Path,
    name: str,
    output: str | Path,
    quality: int,
    domains: list[str],
) -> Path:
    """Write a cohort variant in which `domains` carry JPEG history."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")
    records = read_manifest(manifest)
    present = {str(record["domain"]) for record in records}
    unknown = sorted(set(domains) - present)
    if unknown:
        raise ValueError(f"Domains not present in manifest: {unknown}")

    output = Path(output)
    dataset_root = output / name
    manifest_path = output / "manifests" / f"{name}_clean.jsonl"
    config_path = output / "manifests" / f"{name}_clean.config.json"
    if manifest_path.exists():
        raise FileExistsError(f"Variant manifest already exists: {manifest_path}")

    targets = set(domains)
    new_records: list[dict[str, object]] = []
    for record in records:
        domain = str(record["domain"])
        source = Path(str(record["clean_path"]))
        destination = dataset_root / domain / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if domain in targets:
            with Image.open(source) as image:
                jpeg_roundtrip(image, quality).save(
                    destination, format="PNG", optimize=False
                )
            applied: int | None = quality
        else:
            # Copied byte-for-byte so the untouched domain is provably unchanged.
            shutil.copy2(source, destination)
            applied = None
        updated = dict(record)
        updated["clean_path"] = str(destination.resolve())
        updated["jpeg_quality_applied"] = applied
        updated["source_clean_path"] = str(source.resolve())
        new_records.append(updated)

    write_manifest(manifest_path, new_records)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "source_manifest": str(Path(manifest).resolve()),
                "quality": quality,
                "domains": list(domains),
                "record_count": len(new_records),
                "note": (
                    "JPEG-quantized content stored losslessly as PNG. Domains "
                    "not listed are copied unchanged."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def add_arguments(subparsers) -> None:
    parser = subparsers.add_parser(
        "recompress-cohort",
        help="Add JPEG history to chosen domains to equalize compression.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality", type=int, required=True)
    parser.add_argument("--domains", nargs="+", required=True)


def run(args: argparse.Namespace) -> Path:
    path = recompress_cohort(
        manifest=args.manifest,
        name=args.name,
        output=args.output,
        quality=args.quality,
        domains=args.domains,
    )
    LOGGER.info(f"Manifest: {path.resolve()}")
    return path
