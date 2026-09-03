"""Deterministic, class-matched selection with fixed splits and cohorts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.datasets.manifest import read_manifest, stable_id, write_manifest


SPLITS = ("train", "val", "test")
COHORTS = ("audit", "pilot_addition", "full_addition")


def create_selection_manifest(
    real_index: str | Path,
    ai_index: str | Path,
    output: str | Path,
    count_per_domain: int,
    audit_count: int,
    pilot_total: int,
    seed: int,
    train_ratio: float = 2 / 3,
    val_ratio: float = 1 / 6,
) -> Path:
    """Create an immutable selection manifest shared by incremental stages."""
    validate_selection_parameters(
        count_per_domain=count_per_domain,
        audit_count=audit_count,
        pilot_total=pilot_total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    output = Path(output)
    config_path = output.with_suffix(".config.json")
    if output.exists() or config_path.exists():
        raise FileExistsError(
            f"Selection manifest or companion configuration already exists: {output}"
        )

    index_paths = {"real": Path(real_index), "ai": Path(ai_index)}
    indexes = {domain: read_manifest(path) for domain, path in index_paths.items()}
    for domain, records in indexes.items():
        validate_dataset_index(records, expected_domain=domain)

    grouped = {
        domain: group_by_class(records) for domain, records in indexes.items()
    }
    common_classes = sorted(set(grouped["real"]) & set(grouped["ai"]))
    if not common_classes:
        raise ValueError("Natural and AI indexes do not share any class IDs.")
    capacities = {
        class_id: min(
            len(grouped["real"][class_id]),
            len(grouped["ai"][class_id]),
        )
        for class_id in common_classes
    }
    allocation = balanced_allocation(capacities, count_per_domain, seed, "selection")

    selected_by_domain_class: dict[str, dict[str, list[dict[str, object]]]] = {
        "real": {},
        "ai": {},
    }
    split_counts_by_class: dict[str, dict[str, int]] = {}
    for class_id in common_classes:
        class_count = allocation[class_id]
        if class_count == 0:
            continue
        split_counts_by_class[class_id] = proportional_counts(
            class_count,
            {
                "train": train_ratio,
                "val": val_ratio,
                "test": 1 - train_ratio - val_ratio,
            },
        )
        for domain in ("real", "ai"):
            ranked = sorted(
                grouped[domain][class_id],
                key=lambda record: stable_id(
                    "candidate-rank",
                    str(seed),
                    domain,
                    class_id,
                    str(record["candidate_id"]),
                    length=64,
                ),
            )
            selected_by_domain_class[domain][class_id] = ranked[:class_count]

    development_capacities = {
        class_id: counts["train"] + counts["val"]
        for class_id, counts in split_counts_by_class.items()
    }
    if sum(development_capacities.values()) < pilot_total:
        raise ValueError(
            "The requested pilot does not fit inside the train/validation pool."
        )
    audit_allocation = balanced_allocation(
        development_capacities,
        audit_count,
        seed,
        "audit",
    )
    pilot_allocation = balanced_allocation(
        development_capacities,
        pilot_total,
        seed,
        "audit",
    )

    selection_records: list[dict[str, object]] = []
    for domain in ("real", "ai"):
        domain_rank = 0
        for class_id in deterministic_class_order(
            split_counts_by_class,
            seed,
            "selection",
        ):
            candidates = selected_by_domain_class[domain][class_id]
            split_counts = split_counts_by_class[class_id]
            ranked_for_split = sorted(
                candidates,
                key=lambda record: stable_id(
                    "split-rank",
                    str(seed),
                    domain,
                    class_id,
                    str(record["candidate_id"]),
                    length=64,
                ),
            )
            assigned = assign_splits(ranked_for_split, split_counts)
            development = sorted(
                [item for item in assigned if item[1] != "test"],
                key=lambda item: stable_id(
                    "cohort-rank",
                    str(seed),
                    domain,
                    class_id,
                    str(item[0]["candidate_id"]),
                    length=64,
                ),
            )
            cohorts_by_candidate: dict[str, str] = {}
            for position, (record, _) in enumerate(development):
                if position < audit_allocation[class_id]:
                    cohort = "audit"
                elif position < pilot_allocation[class_id]:
                    cohort = "pilot_addition"
                else:
                    cohort = "full_addition"
                cohorts_by_candidate[str(record["candidate_id"])] = cohort

            for class_rank, (record, split) in enumerate(assigned):
                candidate_id = str(record["candidate_id"])
                cohort = (
                    "full_addition"
                    if split == "test"
                    else cohorts_by_candidate[candidate_id]
                )
                selection_records.append(
                    {
                        **record,
                        "selection_id": stable_id(
                            "selection-record",
                            str(seed),
                            candidate_id,
                            length=24,
                        ),
                        "selection_seed": seed,
                        "domain_selection_rank": domain_rank,
                        "class_selection_rank": class_rank,
                        "split": split,
                        "cohort": cohort,
                    }
                )
                domain_rank += 1

    validate_selection_manifest(
        selection_records,
        expected_count_per_domain=count_per_domain,
        expected_audit_count=audit_count,
        expected_pilot_total=pilot_total,
    )
    selection_records.sort(
        key=lambda record: (
            str(record["domain"]),
            int(record["domain_selection_rank"]),
        )
    )
    write_manifest(output, selection_records)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "real_index": str(index_paths["real"].resolve()),
                "real_index_sha256": file_sha256(index_paths["real"]),
                "ai_index": str(index_paths["ai"].resolve()),
                "ai_index_sha256": file_sha256(index_paths["ai"]),
                "count_per_domain": count_per_domain,
                "audit_count_per_domain": audit_count,
                "pilot_total_per_domain": pilot_total,
                "seed": seed,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": 1 - train_ratio - val_ratio,
                "class_count": len(split_counts_by_class),
                "selection_manifest_sha256": file_sha256(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def validate_dataset_index(
    records: list[dict[str, object]],
    expected_domain: str,
) -> None:
    required = {
        "candidate_id",
        "dataset_id",
        "dataset_version",
        "domain",
        "source_kind",
        "source_path",
        "source_member",
        "class_id",
        "extension",
    }
    seen: set[str] = set()
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Dataset index record is missing {sorted(missing)}.")
        if record["domain"] != expected_domain:
            raise ValueError(
                f"Expected {expected_domain} index but found domain={record['domain']}."
            )
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id in dataset index: {candidate_id}")
        seen.add(candidate_id)


def validate_selection_manifest(
    records: list[dict[str, object]],
    expected_count_per_domain: int | None = None,
    expected_audit_count: int | None = None,
    expected_pilot_total: int | None = None,
) -> None:
    required = {
        "candidate_id",
        "domain",
        "class_id",
        "source_kind",
        "source_path",
        "source_member",
        "split",
        "cohort",
        "selection_seed",
    }
    seen: set[str] = set()
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Selection record is missing {sorted(missing)}.")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate selected: {candidate_id}")
        seen.add(candidate_id)
        if record["domain"] not in {"real", "ai"}:
            raise ValueError(f"Invalid selected domain: {record['domain']}")
        if record["split"] not in SPLITS:
            raise ValueError(f"Invalid selected split: {record['split']}")
        if record["cohort"] not in COHORTS:
            raise ValueError(f"Invalid selected cohort: {record['cohort']}")
        if record["split"] == "test" and record["cohort"] != "full_addition":
            raise ValueError("Final test images cannot appear in audit or pilot cohorts.")

    counts = Counter(str(record["domain"]) for record in records)
    if counts["real"] != counts["ai"]:
        raise ValueError(f"Domain selection is unbalanced: {dict(counts)}")
    if expected_count_per_domain is not None and any(
        counts[domain] != expected_count_per_domain for domain in ("real", "ai")
    ):
        raise ValueError(f"Unexpected selected domain counts: {dict(counts)}")

    class_counts = {
        domain: Counter(
            str(record["class_id"])
            for record in records
            if record["domain"] == domain
        )
        for domain in ("real", "ai")
    }
    if class_counts["real"] != class_counts["ai"]:
        raise ValueError("Natural and AI selections are not class-count matched.")

    if expected_audit_count is not None:
        for domain in ("real", "ai"):
            audit_count = sum(
                record["domain"] == domain and record["cohort"] == "audit"
                for record in records
            )
            if audit_count != expected_audit_count:
                raise ValueError(f"Unexpected {domain} audit count: {audit_count}")
    if expected_pilot_total is not None:
        for domain in ("real", "ai"):
            pilot_count = sum(
                record["domain"] == domain
                and record["cohort"] in {"audit", "pilot_addition"}
                for record in records
            )
            if pilot_count != expected_pilot_total:
                raise ValueError(f"Unexpected {domain} pilot count: {pilot_count}")


def balanced_allocation(
    capacities: dict[str, int],
    total: int,
    seed: int,
    namespace: str,
) -> dict[str, int]:
    if total < 0:
        raise ValueError("Allocation total must not be negative.")
    if sum(capacities.values()) < total:
        raise ValueError(
            f"Requested {total} images but matched class capacity is "
            f"{sum(capacities.values())}."
        )
    allocation = {class_id: 0 for class_id in capacities}
    order = deterministic_class_order(capacities, seed, namespace)
    assigned = 0
    while assigned < total:
        progressed = False
        for class_id in order:
            if allocation[class_id] >= capacities[class_id]:
                continue
            allocation[class_id] += 1
            assigned += 1
            progressed = True
            if assigned == total:
                break
        if not progressed:
            raise RuntimeError("Balanced allocation stopped before reaching its target.")
    return allocation


def proportional_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    priority = sorted(
        ratios,
        key=lambda name: (-(raw[name] - counts[name]), list(ratios).index(name)),
    )
    for name in priority[:remaining]:
        counts[name] += 1
    return counts


def deterministic_class_order(
    classes: Iterable[str] | dict[str, object],
    seed: int,
    namespace: str,
) -> list[str]:
    return sorted(
        classes,
        key=lambda class_id: stable_id(
            namespace,
            str(seed),
            str(class_id),
            length=64,
        ),
    )


def assign_splits(
    records: list[dict[str, object]],
    split_counts: dict[str, int],
) -> list[tuple[dict[str, object], str]]:
    assigned: list[tuple[dict[str, object], str]] = []
    offset = 0
    for split in SPLITS:
        count = split_counts[split]
        assigned.extend((record, split) for record in records[offset : offset + count])
        offset += count
    if offset != len(records):
        raise RuntimeError("Split allocation did not consume every selected record.")
    return assigned


def group_by_class(
    records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["class_id"])].append(record)
    return dict(grouped)


def validate_selection_parameters(
    count_per_domain: int,
    audit_count: int,
    pilot_total: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    if count_per_domain <= 0:
        raise ValueError("count_per_domain must be greater than zero.")
    if not 0 <= audit_count <= pilot_total <= count_per_domain:
        raise ValueError(
            "Require 0 <= audit_count <= pilot_total <= count_per_domain."
        )
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must leave positive train and test portions.")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
