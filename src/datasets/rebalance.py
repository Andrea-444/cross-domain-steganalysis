"""Restore per-class domain matching after undersized covers are skipped.

Crop-only preprocessing cannot fill the target size from a smaller source, and
the loss is one-sided: natural covers vary in resolution while the AI covers
are uniformly large. A cohort can therefore come out of preparation with fewer
natural than AI covers in some classes.

That matters twice over. The accepted protocol matches class counts between
domains, and unequal domain totals would also give the two detectors different
amounts of training data, so a `real_to_real` against `ai_to_ai` difference
could reflect training-set size rather than domain difficulty.

Matching is applied per class **and** per split, because splits are locked at
selection time and a cover must never move between them. This is a manifest
operation: no image is read, written or modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.datasets.manifest import read_manifest, write_manifest


DOMAINS = ("real", "ai")


def rebalance_cohort(
    manifest: str | Path,
    name: str,
    output: str | Path,
) -> Path:
    """Write a variant in which every class and split holds equal domain counts."""
    records = read_manifest(manifest)
    output = Path(output)
    manifest_path = output / "manifests" / f"{name}_clean.jsonl"
    config_path = output / "manifests" / f"{name}_clean.config.json"
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError(f"Rebalanced manifest already exists: {manifest_path}")

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (str(record["class_id"]), str(record["split"]), str(record["domain"]))
        grouped[key].append(record)
    # Sorting by image_id makes the choice of which surplus cover to drop
    # deterministic, so the same input always yields the same cohort.
    for bucket in grouped.values():
        bucket.sort(key=lambda record: str(record["image_id"]))

    kept: list[dict[str, object]] = []
    dropped: list[dict[str, str]] = []
    class_splits = sorted(
        {(class_id, split) for class_id, split, _ in grouped}
    )
    for class_id, split in class_splits:
        counts = {
            domain: len(grouped.get((class_id, split, domain), []))
            for domain in DOMAINS
        }
        target = min(counts.values())
        for domain in DOMAINS:
            bucket = grouped.get((class_id, split, domain), [])
            kept.extend(bucket[:target])
            for record in bucket[target:]:
                dropped.append(
                    {
                        "image_id": str(record["image_id"]),
                        "domain": domain,
                        "class_id": class_id,
                        "split": split,
                    }
                )

    kept.sort(key=lambda record: str(record["image_id"]))
    write_manifest(manifest_path, kept)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "source_manifest": str(Path(manifest).resolve()),
                "strategy": "min-count-per-class-and-split",
                "source_record_count": len(records),
                "record_count": len(kept),
                "dropped_count": len(dropped),
                "dropped_by_domain": dict(
                    Counter(entry["domain"] for entry in dropped)
                ),
                "dropped": dropped,
                "note": (
                    "Manifest-only operation. Image files are untouched and "
                    "clean_path values are unchanged."
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
        "rebalance-cohort",
        help="Equalize domain counts per class and split after skipped covers.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)


def run(args: argparse.Namespace) -> Path:
    path = rebalance_cohort(
        manifest=args.manifest, name=args.name, output=args.output
    )
    config = json.loads(
        (path.parent / f"{path.stem}.config.json").read_text(encoding="utf-8")
    )
    print(f"Kept {config['record_count']} of {config['source_record_count']} covers.")
    print(f"Dropped {config['dropped_count']}: {config['dropped_by_domain']}")
    print(f"Manifest: {path.resolve()}")
    return path
