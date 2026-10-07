"""Incrementally materialize selected native images without duplicate extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable

from src.datasets.index import normalize_member
from src.datasets.manifest import read_manifest
from src.datasets.selection import validate_selection_manifest
from src.logging_config import ProgressReporter, get_logger


STAGE_COHORTS = {
    "audit": {"audit"},
    "pilot": {"audit", "pilot_addition"},
    "full": {"audit", "pilot_addition", "full_addition"},
}


LOGGER = get_logger(__name__)


def extract_selection_stage(
    selection_manifest: str | Path,
    output: str | Path,
    stage: str,
    extraction_manifest: str | Path | None = None,
    materialization: str = "copy",
) -> dict[str, object]:
    """Materialize one cumulative selection stage and record content checksums."""
    if stage not in STAGE_COHORTS:
        raise ValueError(f"stage must be one of {sorted(STAGE_COHORTS)}.")
    if materialization not in {"copy", "hardlink", "symlink"}:
        raise ValueError("materialization must be copy, hardlink, or symlink.")

    selection_manifest = Path(selection_manifest)
    records = read_manifest(selection_manifest)
    validate_selection_manifest(records)
    selected = [
        record for record in records if record["cohort"] in STAGE_COHORTS[stage]
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = (
        Path(extraction_manifest)
        if extraction_manifest is not None
        else output / "extraction_manifest.jsonl"
    )
    state = load_extraction_state(state_path)

    pending_by_source: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    reused = 0
    for record in selected:
        destination = destination_for_record(output, record)
        candidate_id = str(record["candidate_id"])
        previous = state.get(candidate_id)
        if previous is not None:
            verify_existing_extraction(destination, previous, record)
            reused += 1
            continue
        pending_by_source[
            (str(record["source_kind"]), str(record["source_path"]))
        ].append(record)

    extracted = 0
    adopted = 0
    for (source_kind, source_path), source_records in pending_by_source.items():
        source = Path(source_path)
        if not source.exists():
            raise ValueError(f"Indexed dataset source is missing: {source}")
        if source_kind == "directory":
            for record in source_records:
                outcome = materialize_directory_record(
                    source,
                    record,
                    output,
                    state_path,
                    materialization,
                )
                extracted += outcome == "extracted"
                adopted += outcome == "adopted"
        elif source_kind == "zip":
            if materialization != "copy":
                raise ValueError("Archive members support only copy materialization.")
            with zipfile.ZipFile(source) as archive:
                for record in source_records:
                    member = normalize_member(str(record["source_member"]))
                    with archive.open(member) as handle:
                        outcome = materialize_stream_record(
                            handle,
                            record,
                            output,
                            state_path,
                        )
                    extracted += outcome == "extracted"
                    adopted += outcome == "adopted"
        elif source_kind == "tar":
            if materialization != "copy":
                raise ValueError("Archive members support only copy materialization.")
            with tarfile.open(source, mode="r:*") as archive:
                for record in source_records:
                    member_name = normalize_member(str(record["source_member"]))
                    member = archive.getmember(member_name)
                    if not member.isfile():
                        raise ValueError(f"Selected TAR member is not a file: {member_name}")
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise RuntimeError(f"Could not read TAR member: {member_name}")
                    with handle:
                        outcome = materialize_stream_record(
                            handle,
                            record,
                            output,
                            state_path,
                        )
                    extracted += outcome == "extracted"
                    adopted += outcome == "adopted"
        elif source_kind == "tar-zstd":
            if materialization != "copy":
                raise ValueError("Archive members support only copy materialization.")
            outcomes = materialize_zstd_tar_records(
                source,
                source_records,
                output,
                state_path,
            )
            extracted += outcomes.count("extracted")
            adopted += outcomes.count("adopted")
        else:
            raise ValueError(f"Unsupported indexed source kind: {source_kind}")

    return {
        "stage": stage,
        "requested_count": len(selected),
        "extracted_count": extracted,
        "adopted_count": adopted,
        "reused_count": reused,
        "output": str(output.resolve()),
        "extraction_manifest": str(state_path.resolve()),
    }


def materialize_directory_record(
    source_root: Path,
    record: dict[str, object],
    output: Path,
    state_path: Path,
    materialization: str,
) -> str:
    member = normalize_member(str(record["source_member"]))
    source = source_root / member
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Selected source member is not a regular file: {source}")
    destination = destination_for_record(output, record)
    source_hash = file_sha256(source)
    if destination.exists():
        if file_sha256(destination) != source_hash:
            raise FileExistsError(
                f"Destination exists with different content: {destination}"
            )
        append_extraction_record(state_path, record, destination, source_hash)
        return "adopted"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_destination(destination)
    try:
        if materialization == "copy":
            shutil.copy2(source, temporary)
        elif materialization == "hardlink":
            os.link(source, temporary)
        else:
            temporary.symlink_to(source)
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    append_extraction_record(state_path, record, destination, source_hash)
    return "extracted"


def materialize_stream_record(
    handle: BinaryIO,
    record: dict[str, object],
    output: Path,
    state_path: Path,
) -> str:
    destination = destination_for_record(output, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_destination(destination)
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as target:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
        source_hash = digest.hexdigest()
        if destination.exists():
            if file_sha256(destination) != source_hash:
                raise FileExistsError(
                    f"Destination exists with different content: {destination}"
                )
            temporary.unlink()
            outcome = "adopted"
        else:
            os.replace(temporary, destination)
            outcome = "extracted"
    finally:
        if temporary.exists():
            temporary.unlink()
    append_extraction_record(state_path, record, destination, source_hash)
    return outcome


def materialize_zstd_tar_records(
    source: Path,
    records: list[dict[str, object]],
    output: Path,
    state_path: Path,
) -> list[str]:
    members = [normalize_member(str(record["source_member"])) for record in records]
    with tempfile.TemporaryDirectory(prefix="dataset-extract-") as temporary_directory:
        staging = Path(temporary_directory)
        member_list = staging / "members.txt"
        member_list.write_text("".join(f"{member}\n" for member in members), encoding="utf-8")
        extracted_root = staging / "files"
        extracted_root.mkdir()
        extract_zstd_tar_members(source, member_list, extracted_root)
        outcomes: list[str] = []
        for record, member in zip(records, members):
            extracted_source = extracted_root / member
            if extracted_source.is_symlink() or not extracted_source.is_file():
                raise ValueError(
                    f"Selected TAR.ZST member is not a regular file: {member}"
                )
            resolved = extracted_source.resolve()
            if not resolved.is_relative_to(extracted_root.resolve()):
                raise ValueError(f"Archive member escaped extraction root: {member}")
            destination = destination_for_record(output, record)
            source_hash = file_sha256(extracted_source)
            if destination.exists():
                if file_sha256(destination) != source_hash:
                    raise FileExistsError(
                        f"Destination exists with different content: {destination}"
                    )
                outcome = "adopted"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_destination_path = temporary_destination(destination)
                try:
                    shutil.copy2(extracted_source, temporary_destination_path)
                    os.replace(temporary_destination_path, destination)
                finally:
                    if temporary_destination_path.exists():
                        temporary_destination_path.unlink()
                outcome = "extracted"
            append_extraction_record(state_path, record, destination, source_hash)
            outcomes.append(outcome)
        return outcomes


def extract_zstd_tar_members(source: Path, member_list: Path, output: Path) -> None:
    direct = subprocess.run(
        ["tar", "-xf", str(source), "-C", str(output), "-T", str(member_list)],
        capture_output=True,
        text=True,
    )
    if direct.returncode == 0:
        return

    decompressor = subprocess.Popen(["unzstd", "-c", str(source)], stdout=subprocess.PIPE)
    try:
        extraction = subprocess.run(
            ["tar", "-xf", "-", "-C", str(output), "-T", str(member_list)],
            stdin=decompressor.stdout,
            capture_output=True,
            text=True,
        )
    finally:
        if decompressor.stdout is not None:
            decompressor.stdout.close()
    decompressor_return_code = decompressor.wait()
    if extraction.returncode != 0 or decompressor_return_code != 0:
        raise RuntimeError(
            "Could not extract TAR.ZST members. Install working `tar` and "
            "`unzstd` tools and verify the archive."
        )


def destination_for_record(output: Path, record: dict[str, object]) -> Path:
    domain = str(record["domain"])
    if domain not in {"real", "ai"}:
        raise ValueError(f"Invalid selected domain: {domain}")
    class_component = safe_component(str(record["class_id"]))
    extension = str(record.get("extension") or "").lower()
    if not re.fullmatch(r"\.[a-z0-9]+", extension):
        raise ValueError(f"Unsafe selected image extension: {extension}")
    candidate_id = str(record["candidate_id"])
    if not re.fullmatch(r"[a-f0-9]{16,64}", candidate_id):
        raise ValueError(f"Invalid candidate_id: {candidate_id}")
    return output / domain / class_component / f"{candidate_id}{extension}"


def load_extraction_state(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    records = read_manifest(path)
    state: dict[str, dict[str, object]] = {}
    for record in records:
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"Extraction state record lacks candidate_id: {record}")
        if candidate_id in state:
            raise ValueError(f"Duplicate candidate in extraction state: {candidate_id}")
        state[candidate_id] = record
    return state


def verify_existing_extraction(
    destination: Path,
    previous: dict[str, object],
    selection_record: dict[str, object],
) -> None:
    if str(previous.get("source_member")) != str(selection_record["source_member"]):
        raise ValueError(
            f"Extraction state source mismatch for {selection_record['candidate_id']}."
        )
    if Path(str(previous.get("local_path"))).resolve() != destination.resolve():
        raise ValueError(
            f"Extraction state destination mismatch for {selection_record['candidate_id']}."
        )
    if not destination.exists():
        raise ValueError(f"Previously extracted image is missing: {destination}")
    checksum = file_sha256(destination)
    if checksum != previous.get("sha256"):
        raise ValueError(f"Previously extracted image changed: {destination}")


def append_extraction_record(
    path: Path,
    selection_record: dict[str, object],
    destination: Path,
    checksum: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **selection_record,
        "local_path": str(destination.resolve()),
        "sha256": checksum,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def temporary_destination(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not component:
        raise ValueError(f"Unsafe empty path component derived from {value!r}.")
    return component


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
