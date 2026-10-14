"""Names, tags and groups for tracked runs, derived from the run itself.

A hand-typed run name can disagree with what actually ran. Everything here
comes from the manifest and the command arguments, so a name in W&B is
evidence rather than a label. The manifest chain supplies the embedding
algorithm, the payload and the preprocessing mode, which no command argument
states.

Format:

    <role>-<domains>-<algorithm><payload>-<preprocess>-lr<rate>-s<seeds>

For example `pilot-real-suniward040-crop-lr1e-4-s1`, or
`stage-real+ai+mixed-suniward040-crop-lr1e-4-s1-2-3`.
"""
from __future__ import annotations

from pathlib import Path

from src.logging_config import get_logger


LOGGER = get_logger(__name__)

UNKNOWN = "unknown"


def manifest_condition(manifest: str | Path) -> dict[str, object]:
    """Read the embedding and preprocessing behind a stego manifest.

    The embedding run writes `config.json` beside its manifest, and that config
    names the clean manifest it consumed. The preprocessing mode is two steps
    further back, so this follows the chain rather than trusting a file name.
    """
    condition: dict[str, object] = {
        "algorithm": None,
        "payload_bpp": None,
        "preprocess_mode": None,
    }
    run_config = Path(manifest).parent / "config.json"
    if not run_config.exists():
        LOGGER.warning(
            "No config.json beside %s, so the run name cannot state the "
            "embedding condition.", manifest
        )
        return condition
    try:
        from src.steganalysis.diagnostics import check_provenance

        walked = check_provenance(run_config)
    except Exception as error:
        LOGGER.warning("Could not read the manifest chain: %s", error)
        return condition
    embedding = walked.get("embedding") or {}
    condition["algorithm"] = embedding.get("algorithm")
    condition["payload_bpp"] = embedding.get("payload_bpp")
    condition["preprocess_mode"] = walked.get("preprocess_mode")
    return condition


def format_payload(payload: object) -> str:
    """0.4 becomes 040, so names sort and never hold a decimal point.

    Three digits are bits per pixel times one hundred, so 0.1 gives 010 and
    1.0 gives 100. The project uses 0.1, 0.2 and 0.4.
    """
    if payload is None:
        return UNKNOWN
    return f"{int(round(float(payload) * 100)):03d}"


def format_learning_rate(rate: float) -> str:
    """1e-4 stays 1e-4 rather than becoming 0.0001."""
    text = f"{rate:.0e}"
    mantissa, exponent = text.split("e")
    return f"{mantissa}e{int(exponent)}"


def format_seeds(seeds: list[int]) -> str:
    return "-".join(str(seed) for seed in seeds)


def run_role(seeds: list[int], direction_count: int, total_directions: int) -> str:
    """One seed is a pilot. The protocol needs repeats before a result."""
    if len(seeds) < 2:
        return "pilot"
    if direction_count >= total_directions:
        return "full"
    return "stage"


def run_name(
    *,
    role: str,
    train_domains: list[str],
    condition: dict[str, object],
    learning_rate: float,
    seeds: list[int],
    stamp: str | None = None,
) -> str:
    """Build the run name, with the results-folder stamp at the end.

    The stamp does two jobs. It separates two runs that used identical
    settings, which the rest of the name cannot do. It also names the local
    folder: a run called `...-20261014_153201` wrote
    `results/srnet_20261014_153201/`. A plain date does neither, because two
    runs on one day collide.
    """
    algorithm = condition.get("algorithm") or UNKNOWN
    preprocess = condition.get("preprocess_mode") or UNKNOWN
    parts = [
        role,
        "+".join(train_domains),
        f"{algorithm}{format_payload(condition.get('payload_bpp'))}",
        str(preprocess),
        f"lr{format_learning_rate(learning_rate)}",
        f"s{format_seeds(seeds)}",
    ]
    if stamp:
        parts.append(stamp)
    return "-".join(parts)


def results_stamp(run_directory: str | Path) -> str:
    """The timestamp part of a results folder, `srnet_20261014_153201`."""
    name = Path(run_directory).name
    _, _, stamp = name.partition("_")
    return stamp or name


def run_tags(
    *,
    role: str,
    detector: str,
    train_domains: list[str],
    directions: list[str],
    condition: dict[str, object],
    learning_rate: float,
    seeds: list[int],
    extra: list[str] | None = None,
) -> list[str]:
    """Facets for filtering. Each tag is one fact, as `key:value`."""
    tags = [
        f"role:{role}",
        f"detector:{detector}",
        f"algorithm:{condition.get('algorithm') or UNKNOWN}",
        f"payload:{condition.get('payload_bpp') if condition.get('payload_bpp') is not None else UNKNOWN}",
        f"preprocess:{condition.get('preprocess_mode') or UNKNOWN}",
        f"lr:{format_learning_rate(learning_rate)}",
        f"seeds:{len(seeds)}",
    ]
    tags.extend(f"domain:{domain}" for domain in train_domains)
    tags.extend(f"direction:{direction}" for direction in directions)
    for tag in extra or []:
        if tag not in tags:
            tags.append(tag)
    return tags


def run_group(condition: dict[str, object], manifest_sha256: str) -> str:
    """Group every run that used the same cohort and embedding condition.

    Stages run as separate commands, and the group is what puts them back
    together in the W&B interface. The manifest hash is in the group, so two
    cohorts with the same settings never merge.
    """
    algorithm = condition.get("algorithm") or UNKNOWN
    preprocess = condition.get("preprocess_mode") or UNKNOWN
    payload = format_payload(condition.get("payload_bpp"))
    return f"{algorithm}{payload}-{preprocess}-{manifest_sha256[:8]}"
