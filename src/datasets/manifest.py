"""Manifest helpers shared by dataset, embedding, and experiment stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ManifestRecord = dict[str, Any]


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    path = Path(path)
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def stable_id(*parts: str, length: int = 16) -> str:
    value = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:length]


def deterministic_seed(base_seed: int, *parts: str) -> int:
    value = stable_id(str(base_seed), *parts, length=16)
    return int(value, 16) % (2**31 - 1)


def validate_clean_manifest(records: list[ManifestRecord]) -> None:
    required = {"image_id", "domain", "clean_path", "split", "source"}
    seen_ids: set[str] = set()
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Manifest record is missing fields {sorted(missing)}: {record}")
        if record["domain"] not in {"real", "ai"}:
            raise ValueError(f"Invalid domain: {record['domain']}")
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {record['split']}")
        image_id = str(record["image_id"])
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        seen_ids.add(image_id)


def validate_stego_manifest(records: list[ManifestRecord]) -> None:
    validate_clean_manifest(records)
    required = {"stego_path", "algorithm", "payload_bpp", "embedding_seed"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Stego manifest record is missing fields {sorted(missing)}: {record}")
