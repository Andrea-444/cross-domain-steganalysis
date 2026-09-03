"""Build immutable indexes of image candidates without normalizing them."""

from __future__ import annotations

import csv
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from src.datasets.manifest import stable_id, write_manifest


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_dataset_index(
    source: str | Path,
    dataset_id: str,
    dataset_version: str,
    domain: str,
    output: str | Path,
    class_map_path: str | Path | None = None,
    class_map_member_column: str = "source_member",
    class_map_class_column: str = "class_id",
    infer_class_from_parent: bool = True,
    allow_unclassified: bool = False,
) -> Path:
    """Index image members in a directory or archive using stable source IDs."""
    if domain not in {"real", "ai"}:
        raise ValueError("domain must be real or ai.")
    if not dataset_id.strip() or not dataset_version.strip():
        raise ValueError("dataset_id and dataset_version must not be empty.")

    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f"Dataset source does not exist: {source}")
    output = Path(output)
    config_path = output.with_suffix(".config.json")
    if output.exists() or config_path.exists():
        raise FileExistsError(
            f"Dataset index or companion configuration already exists: {output}"
        )

    class_map = load_class_map(
        class_map_path,
        member_column=class_map_member_column,
        class_column=class_map_class_column,
    )
    source_kind, members = list_source_members(source)
    records: list[dict[str, object]] = []
    missing_classes: list[str] = []
    seen_members: set[str] = set()

    for member, size_bytes in members:
        member = normalize_member(member)
        if member in seen_members:
            raise ValueError(f"Duplicate source member in dataset source: {member}")
        seen_members.add(member)
        class_id = resolve_class_id(
            member,
            class_map=class_map,
            infer_from_parent=infer_class_from_parent,
        )
        if class_id is None:
            missing_classes.append(member)
            if not allow_unclassified:
                continue
            class_id = "unclassified"
        records.append(
            {
                "candidate_id": stable_id(
                    "dataset-candidate",
                    dataset_id,
                    dataset_version,
                    domain,
                    member,
                    length=24,
                ),
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "domain": domain,
                "source_kind": source_kind,
                "source_path": str(source),
                "source_member": member,
                "class_id": class_id,
                "extension": PurePosixPath(member).suffix.lower(),
                "size_bytes": size_bytes,
            }
        )

    if missing_classes and not allow_unclassified:
        preview = ", ".join(missing_classes[:3])
        raise ValueError(
            f"Could not determine a class for {len(missing_classes)} image members "
            f"(examples: {preview}). Provide --class-map or use class directories."
        )
    if not records:
        raise ValueError(f"No indexed images found in {source}.")

    records.sort(key=lambda item: str(item["candidate_id"]))
    write_manifest(output, records)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "domain": domain,
                "source_kind": source_kind,
                "source_path": str(source),
                "class_map_path": (
                    str(Path(class_map_path).resolve()) if class_map_path else None
                ),
                "infer_class_from_parent": infer_class_from_parent,
                "allow_unclassified": allow_unclassified,
                "record_count": len(records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def list_source_members(source: Path) -> tuple[str, list[tuple[str, int | None]]]:
    if source.is_dir():
        members = [
            (path.relative_to(source).as_posix(), path.stat().st_size)
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return "directory", members
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = [
                (info.filename, info.file_size)
                for info in archive.infolist()
                if not info.is_dir()
                and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTENSIONS
            ]
        return "zip", members
    if is_zstd_tar(source):
        members = [
            (member, None)
            for member in list_zstd_tar_members(source)
            if PurePosixPath(member).suffix.lower() in IMAGE_EXTENSIONS
        ]
        return "tar-zstd", members
    if tarfile.is_tarfile(source):
        with tarfile.open(source, mode="r:*") as archive:
            members = [
                (member.name, member.size)
                for member in archive.getmembers()
                if member.isfile()
                and PurePosixPath(member.name).suffix.lower() in IMAGE_EXTENSIONS
            ]
        return "tar", members
    raise ValueError(
        "Dataset source must be a directory, ZIP, TAR, or TAR.ZST archive."
    )


def load_class_map(
    path: str | Path | None,
    member_column: str,
    class_column: str,
) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Class map does not exist: {path}")

    suffix = path.suffix.lower()
    rows: Iterable[dict[str, Any]]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return _normalize_class_map(value)
        if not isinstance(value, list):
            raise ValueError("JSON class map must be an object or a list of objects.")
        rows = value
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Class map must be JSON, JSONL, or CSV.")

    mapping: dict[str, str] = {}
    for row in rows:
        if member_column not in row or class_column not in row:
            raise ValueError(
                f"Class map rows require {member_column!r} and {class_column!r}."
            )
        member = normalize_member(str(row[member_column]))
        class_id = str(row[class_column]).strip()
        if not member or not class_id:
            raise ValueError("Class map member and class values must not be empty.")
        _add_class_mapping(mapping, member, class_id)
    return mapping


def resolve_class_id(
    member: str,
    class_map: dict[str, str],
    infer_from_parent: bool,
) -> str | None:
    normalized = normalize_member(member)
    for key in (normalized, PurePosixPath(normalized).name):
        if key in class_map:
            return class_map[key]
    if infer_from_parent:
        parent = PurePosixPath(normalized).parent.name.strip()
        if parent and parent != ".":
            return parent
    return None


def normalize_member(member: str) -> str:
    value = member.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("-")
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"Unsafe dataset member path: {member}")
    return path.as_posix()


def is_zstd_tar(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.zst") or name.endswith(".tzst")


def list_zstd_tar_members(path: Path) -> list[str]:
    direct = subprocess.run(
        ["tar", "-tf", str(path)],
        capture_output=True,
        text=True,
    )
    if direct.returncode == 0:
        return [line for line in direct.stdout.splitlines() if line.strip()]

    decompressor = subprocess.Popen(["unzstd", "-c", str(path)], stdout=subprocess.PIPE)
    try:
        listing = subprocess.run(
            ["tar", "-tf", "-"],
            stdin=decompressor.stdout,
            capture_output=True,
            text=True,
        )
    finally:
        if decompressor.stdout is not None:
            decompressor.stdout.close()
    decompressor_return_code = decompressor.wait()
    if listing.returncode != 0 or decompressor_return_code != 0:
        raise RuntimeError(
            "Could not list TAR.ZST archive. Install working `tar` and `unzstd` tools."
        )
    return [line for line in listing.stdout.splitlines() if line.strip()]


def _normalize_class_map(value: dict[object, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_member, raw_class in value.items():
        member = normalize_member(str(raw_member))
        class_id = str(raw_class).strip()
        if not class_id:
            raise ValueError("Class map values must not be empty.")
        _add_class_mapping(mapping, member, class_id)
    return mapping


def _add_class_mapping(mapping: dict[str, str], member: str, class_id: str) -> None:
    previous = mapping.get(member)
    if previous is not None and previous != class_id:
        raise ValueError(f"Conflicting class mapping for {member}: {previous}, {class_id}")
    mapping[member] = class_id
