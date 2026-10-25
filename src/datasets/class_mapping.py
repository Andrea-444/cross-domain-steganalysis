"""Exact ImageNet synset mapping for ImageNet and GenImage++ candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from src.datasets.index import IMAGE_EXTENSIONS, list_source_members
from src.datasets.manifest import write_manifest


def build_genimagepp_class_maps(
    real_source: str | Path,
    ai_source: str | Path,
    imagenet_labels: str | Path,
    simple_labels: str | Path,
    class_index: str | Path,
    output: str | Path,
) -> tuple[Path, Path]:
    """Join both domains to synsets by their official ordered class metadata."""
    real_source = Path(real_source).resolve()
    ai_source = Path(ai_source).resolve()
    imagenet_labels = Path(imagenet_labels).resolve()
    simple_labels = Path(simple_labels).resolve()
    class_index = Path(class_index).resolve()
    output = Path(output)
    real_output = output / "imagenet-real-class-map.jsonl"
    ai_output = output / "genimagepp-ai-class-map.jsonl"
    config_output = output / "mapping_config.json"
    existing = [path for path in (real_output, ai_output, config_output) if path.exists()]
    if existing:
        raise FileExistsError(
            "Class-map output already exists; validate it instead of overwriting: "
            + ", ".join(str(path) for path in existing)
        )

    hf_labels = _load_hf_labels(imagenet_labels)
    simple = _load_string_list(simple_labels, "simple labels")
    raw_class_index = json.loads(class_index.read_text(encoding="utf-8"))
    if len(hf_labels) != 1000 or len(simple) != 1000 or len(raw_class_index) != 1000:
        raise ValueError(
            "Expected exactly 1,000 entries in every ImageNet metadata source."
        )
    if len(set(hf_labels)) != 1000 or len(set(simple)) != 1000:
        raise ValueError("ImageNet label metadata contains duplicate class names.")

    synsets: list[str] = []
    for index in range(1000):
        entry = raw_class_index.get(str(index))
        if not isinstance(entry, list) or len(entry) < 2:
            raise ValueError(f"Invalid ImageNet class-index entry {index}: {entry!r}")
        synsets.append(str(entry[0]))
    if len(set(synsets)) != 1000:
        raise ValueError("ImageNet class-index metadata contains duplicate synsets.")

    hf_to_index = {label: index for index, label in enumerate(hf_labels)}
    simple_to_index = {label: index for index, label in enumerate(simple)}
    real_kind, real_members = list_source_members(real_source)
    ai_kind, ai_members = list_source_members(ai_source)
    if real_kind != "directory":
        raise ValueError("ImageNet real source must be a directory.")
    if ai_kind not in {"tar-zstd", "tar", "zip"}:
        raise ValueError("GenImage++ AI source must be an archive.")

    real_records: list[dict[str, object]] = []
    for member, _ in real_members:
        source_label = PurePosixPath(member).parent.name
        if source_label not in hf_to_index:
            raise ValueError(f"Unknown ImageNet source label: {source_label!r}")
        label_index = hf_to_index[source_label]
        real_records.append(
            {
                "source_member": member,
                "class_id": synsets[label_index],
                "label_index": label_index,
                "source_label": source_label,
            }
        )

    ai_records: list[dict[str, object]] = []
    for member, _ in ai_members:
        path = PurePosixPath(member)
        if path.suffix.lower() not in IMAGE_EXTENSIONS or "1_fake" not in path.parts:
            continue
        stem = path.stem
        source_label, separator, image_number = stem.rpartition("_")
        if not separator or not image_number.isdigit():
            raise ValueError(f"Invalid GenImage++ filename: {member}")
        if source_label not in simple_to_index:
            raise ValueError(f"Unknown GenImage++ class label: {source_label!r}")
        label_index = simple_to_index[source_label]
        ai_records.append(
            {
                "source_member": member,
                "class_id": synsets[label_index],
                "label_index": label_index,
                "source_label": source_label,
            }
        )

    _validate_maps(real_records, ai_records)
    real_records.sort(key=lambda record: str(record["source_member"]))
    ai_records.sort(key=lambda record: str(record["source_member"]))
    write_manifest(real_output, real_records)
    write_manifest(ai_output, ai_records)
    config_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mapping_method": "ordered ImageNet class index; exact label equality",
                "fuzzy_matching": False,
                "real_source": str(real_source),
                "ai_source": str(ai_source),
                "imagenet_labels": str(imagenet_labels),
                "imagenet_labels_sha256": _sha256(imagenet_labels),
                "simple_labels": str(simple_labels),
                "simple_labels_sha256": _sha256(simple_labels),
                "class_index": str(class_index),
                "class_index_sha256": _sha256(class_index),
                "records_per_domain": 6000,
                "class_count": 1000,
                "records_per_class": 6,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return real_output, ai_output


def _load_hf_labels(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("labels")
    return _validate_string_list(value, "Hugging Face labels")


def _load_string_list(path: Path, description: str) -> list[str]:
    return _validate_string_list(
        json.loads(path.read_text(encoding="utf-8")), description
    )


def _validate_string_list(value: object, description: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{description} must be a JSON list of strings.")
    return [str(item) for item in value]


def _validate_maps(
    real_records: list[dict[str, object]], ai_records: list[dict[str, object]]
) -> None:
    if len(real_records) != 6000 or len(ai_records) != 6000:
        raise ValueError(
            f"Expected 6,000 records per domain; found real={len(real_records)}, "
            f"ai={len(ai_records)}."
        )
    distributions = {
        "real": Counter(str(record["class_id"]) for record in real_records),
        "ai": Counter(str(record["class_id"]) for record in ai_records),
    }
    for domain, counts in distributions.items():
        if len(counts) != 1000 or set(counts.values()) != {6}:
            raise ValueError(
                f"Expected 1,000 {domain} classes with six records each."
            )
    if distributions["real"] != distributions["ai"]:
        raise ValueError("Natural and AI class distributions do not match.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
