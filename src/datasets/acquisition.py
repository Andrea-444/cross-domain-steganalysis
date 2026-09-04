"""Acquisition of native source images from Hugging Face dataset repositories.

Two access shapes are needed, because the two cover domains are published
differently:

* GenImage++ ships one compressed archive per generator, so only the archive
  for the chosen generator is downloaded and later indexed in place.
* ImageNet-1k ships parquet shards behind a licence gate, so images are
  streamed row by row and only the bytes actually consumed are transferred.

Both paths write the **original encoded bytes**. Decoding an image and
re-saving it would add a second JPEG generation, and the resulting compression
artifacts sit in exactly the high-frequency band a steganalyzer reads.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import Image as DatasetsImage
from datasets import load_dataset
from huggingface_hub import hf_hub_download


DEFAULT_EXTENSION = ".jpg"


def fetch_repo_file(
    repo_id: str,
    filename: str,
    output: str | Path,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download one file from a Hugging Face dataset repository.

    The archive is kept immutable: an existing destination is reused rather
    than re-downloaded, so repeated runs cost nothing and cannot replace a file
    that other artifacts already reference by checksum.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / Path(filename).name
    if destination.exists():
        return destination
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    shutil.copy2(cached, destination)
    return destination


def stream_images(
    repo_id: str,
    split: str,
    output: str | Path,
    limit_per_class: int,
    classes: list[str] | None = None,
    max_classes: int | None = None,
    revision: str | None = None,
    token: str | None = None,
    class_column: str = "label",
    image_column: str = "image",
) -> Path:
    """Stream an image dataset into `output/<class>/<file>` without decoding.

    Streaming avoids materializing the full split: for ImageNet-1k validation,
    a class-balanced sample of a few thousand images transfers a small fraction
    of the 50,000-image split.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        repo_id, split=split, streaming=True, revision=revision, token=token
    )
    # decode=False keeps the source bytes exactly as published.
    dataset = dataset.cast_column(image_column, DatasetsImage(decode=False))
    names = _class_names(dataset, class_column)
    allowed = set(classes) if classes else None

    counts: dict[str, int] = defaultdict(int)
    for row in dataset:
        class_id = _class_id(row[class_column], names)
        if allowed is not None and class_id not in allowed:
            continue
        if counts[class_id] >= limit_per_class:
            if _is_complete(counts, allowed, max_classes, limit_per_class):
                break
            continue
        if (
            max_classes is not None
            and class_id not in counts
            and len(counts) >= max_classes
        ):
            continue
        payload = row[image_column]
        data = payload["bytes"] if isinstance(payload, dict) else payload
        if not data:
            continue
        extension = _extension(payload)
        destination = output / class_id / f"{class_id}_{counts[class_id]:05d}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        counts[class_id] += 1
        if _is_complete(counts, allowed, max_classes, limit_per_class):
            break
    return output


def _class_names(dataset, class_column: str) -> list[str] | None:
    feature = getattr(dataset, "features", None)
    if not feature or class_column not in feature:
        return None
    return getattr(feature[class_column], "names", None)


def _class_id(raw: object, names: list[str] | None) -> str:
    if isinstance(raw, int) and names and 0 <= raw < len(names):
        return str(names[raw])
    return str(raw)


def _extension(payload: object) -> str:
    if isinstance(payload, dict) and payload.get("path"):
        suffix = Path(str(payload["path"])).suffix
        if suffix:
            return suffix
    return DEFAULT_EXTENSION


def _is_complete(
    counts: dict[str, int],
    allowed: set[str] | None,
    max_classes: int | None,
    limit_per_class: int,
) -> bool:
    """True once every class we intend to fill has reached its quota."""
    target = len(allowed) if allowed is not None else max_classes
    if target is None:
        return False
    if len(counts) < target:
        return False
    return all(value >= limit_per_class for value in counts.values())


def add_arguments(subparsers) -> None:
    archive = subparsers.add_parser(
        "fetch-archive", help="Download one archive from a Hugging Face dataset repo."
    )
    archive.add_argument("--repo", required=True)
    archive.add_argument("--file", required=True, dest="filename")
    archive.add_argument("--revision", default=None)
    archive.add_argument("--output", required=True)

    stream = subparsers.add_parser(
        "fetch-images",
        help="Stream images from a Hugging Face dataset into class directories.",
    )
    stream.add_argument("--repo", required=True)
    stream.add_argument("--split", default="validation")
    stream.add_argument("--limit-per-class", type=int, required=True)
    stream.add_argument("--max-classes", type=int, default=None)
    stream.add_argument("--classes", nargs="*", default=None)
    stream.add_argument("--revision", default=None)
    stream.add_argument("--output", required=True)


def run(args: argparse.Namespace) -> Path:
    if args.command == "fetch-archive":
        path = fetch_repo_file(
            repo_id=args.repo,
            filename=args.filename,
            output=args.output,
            revision=args.revision,
        )
        print(f"Archive: {path.resolve()}")
        return path
    path = stream_images(
        repo_id=args.repo,
        split=args.split,
        output=args.output,
        limit_per_class=args.limit_per_class,
        classes=args.classes,
        max_classes=args.max_classes,
        revision=args.revision,
    )
    total = sum(1 for item in path.rglob("*") if item.is_file())
    classes = sum(1 for item in path.iterdir() if item.is_dir())
    print(f"Streamed {total} images across {classes} classes into {path.resolve()}")
    return path
