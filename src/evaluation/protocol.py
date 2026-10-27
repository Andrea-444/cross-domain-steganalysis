"""Combine several single-seed run folders into one protocol summary.

Each run folder holds one invocation of the experiment command. When the
protocol runs one seed and one domain at a time, as it did for S-UNIWARD at
0.4 bpp, no folder holds a cross-seed number: every `summary.json` reports
`seeds: 1` and a standard deviation of 0.0. The means, spreads and paired gaps
that a results chapter reports exist in no file.

This reads the folders, keeps only the runs that share one condition, merges
them by seed, and writes the aggregate.

Merging **by seed** is what makes the paired gaps correct. A gap subtracts a
cross-domain result from a same-domain result inside one seed, and both terms
come from the same trained model. Seed 1 of the protocol is therefore one
record holding every direction that seed produced, across folders.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.metrics import summarize_directions
from src.logging_config import get_logger


LOGGER = get_logger(__name__)

# The reading order of the protocol. `tests/evaluation/test_protocol.py` checks
# that this matches the runner, so the two cannot drift apart.
DIRECTION_ORDER = (
    "real_to_real",
    "ai_to_ai",
    "mixed_to_real",
    "mixed_to_ai",
    "real_to_ai",
    "ai_to_real",
)

# Two runs belong to one condition only when every one of these agrees. A
# difference in any of them makes the numbers incomparable, and averaging them
# would hide the difference rather than report it.
CONDITION_FIELDS = (
    "manifest_sha256",
    "algorithm",
    "payload_bpp",
    "preprocess_mode",
    "detector_commit",
    "learning_rate",
    "epochs",
    "batch_size",
    "train_pairs",
    "mixed_pairs_per_domain",
)

# The training loss of a detector that answers the same thing for every image.
# A loss below this means the network left that state.
PLATEAU_LOSS = 0.6925


def escape_epoch(epochs: list[dict[str, object]]) -> int | None:
    """The first epoch whose training loss leaves the ln(2) plateau.

    A detector that never leaves it has learned nothing, and a detector that
    leaves it late trained for fewer effective epochs than its budget says.
    """
    for entry in epochs:
        if float(entry["train_loss"]) < PLATEAU_LOSS:
            return int(entry["epoch"])
    return None


def read_history(folder: Path) -> dict[str, dict[str, object]]:
    histories: dict[str, dict[str, object]] = {}
    for path in sorted((folder / "history").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        epochs = data["epochs"]
        last = epochs[-1]
        key = f"seed{data['seed']}_{data['train_domain']}"
        histories[key] = {
            "seed": data["seed"],
            "train_domain": data["train_domain"],
            "escape_epoch": escape_epoch(epochs),
            "selected_epoch": data["selection"]["epoch"],
            "total_epochs": len(epochs),
            "best_val_balanced_accuracy": max(
                float(e["val_balanced_accuracy"]) for e in epochs
            ),
            "final_train_minus_val": float(last["train_accuracy"])
            - float(last["val_balanced_accuracy"]),
        }
    return histories


def read_run(folder: str | Path) -> dict[str, object] | None:
    """Read one run folder. A folder with no summary is an unfinished run."""
    folder = Path(folder)
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        LOGGER.warning("Skipping %s: no summary.json, so the run did not finish.", folder)
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    provenance = summary["provenance"]
    embedding = provenance.get("embedding", {})
    condition = config.get("condition", {})
    return {
        "folder": folder.name,
        "runs": summary["runs"],
        "condition": {
            "manifest_sha256": provenance["manifest_sha256"],
            "algorithm": embedding.get("algorithm"),
            "payload_bpp": embedding.get("payload_bpp"),
            "preprocess_mode": condition.get("preprocess_mode"),
            "detector_commit": provenance.get("detector_commit"),
            "learning_rate": config.get("learning_rate"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "train_pairs": config.get("train_pairs"),
            "mixed_pairs_per_domain": config.get("mixed_pairs_per_domain"),
        },
        "repository_commit": provenance.get("repository_commit"),
        "repository_dirty": provenance.get("repository_dirty"),
        "device": provenance.get("device"),
        "duration_seconds": provenance.get("duration_seconds"),
        "history": read_history(folder),
    }


def condition_key(run: dict[str, object]) -> tuple:
    condition = run["condition"]
    return tuple(condition.get(field) for field in CONDITION_FIELDS)


def merge_by_seed(
    runs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Gather every direction a seed produced, across folders.

    Returns the merged records and any conflict. A conflict is the same seed
    and the same direction from two folders, which happens after a re-run. The
    later folder wins, because folder names carry the run timestamp, and the
    conflict is reported rather than hidden.
    """
    collected: dict[int, dict[str, object]] = defaultdict(dict)
    source: dict[tuple[int, str], str] = {}
    conflicts: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda r: str(r["folder"])):
        for record in run["runs"]:
            seed = int(record["seed"])
            for direction, metrics in record["directions"].items():
                previous = source.get((seed, direction))
                if previous is not None:
                    conflicts.append(
                        {
                            "seed": seed,
                            "direction": direction,
                            "kept": run["folder"],
                            "replaced": previous,
                        }
                    )
                collected[seed][direction] = metrics
                source[(seed, direction)] = str(run["folder"])
    merged = [
        {"seed": seed, "directions": collected[seed]} for seed in sorted(collected)
    ]
    return merged, conflicts


def present_directions(merged: list[dict[str, object]]) -> list[str]:
    """Keep only directions every seed produced, in the protocol order.

    A direction that one seed is missing cannot carry a spread, and including
    it would give a mean over a different number of seeds than its neighbours.
    """
    if not merged:
        return []
    shared = set(merged[0]["directions"])
    for record in merged[1:]:
        shared &= set(record["directions"])
    ordered = [name for name in DIRECTION_ORDER if name in shared]
    extra = sorted(shared - set(DIRECTION_ORDER))
    return ordered + extra


def aggregate(folders: list[str | Path]) -> list[dict[str, object]]:
    """One protocol summary for each condition found in the folders."""
    read = [run for run in (read_run(folder) for folder in folders) if run]
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for run in read:
        grouped[condition_key(run)].append(run)

    results: list[dict[str, object]] = []
    for key, runs in grouped.items():
        merged, conflicts = merge_by_seed(runs)
        directions = present_directions(merged)
        partial = sorted(
            {
                direction
                for record in merged
                for direction in record["directions"]
                if direction not in directions
            }
        )
        results.append(
            {
                "condition": dict(zip(CONDITION_FIELDS, key)),
                "folders": sorted(str(run["folder"]) for run in runs),
                "seeds": [record["seed"] for record in merged],
                "directions": directions,
                "directions_with_missing_seeds": partial,
                "conflicts": conflicts,
                "summary": summarize_directions(merged, directions),
                "runs": merged,
                "training": [
                    {"folder": run["folder"], **entry}
                    for run in sorted(runs, key=lambda r: str(r["folder"]))
                    for entry in run["history"].values()
                ],
                "provenance": {
                    "repository_commits": sorted(
                        {str(run["repository_commit"]) for run in runs}
                    ),
                    "any_dirty": any(bool(run["repository_dirty"]) for run in runs),
                    "devices": sorted({str(run["device"]) for run in runs}),
                    "total_gpu_hours": round(
                        sum(float(run["duration_seconds"] or 0) for run in runs) / 3600, 2
                    ),
                    "aggregated_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
    results.sort(key=lambda r: str(r["condition"]))
    return results


def format_condition(condition: dict[str, object]) -> str:
    return (
        f"{condition.get('algorithm')} {condition.get('payload_bpp')} bpp, "
        f"{condition.get('preprocess_mode')}, lr {condition.get('learning_rate')}, "
        f"{condition.get('epochs')} epochs, batch {condition.get('batch_size')}, "
        f"manifest {str(condition.get('manifest_sha256'))[:8]}"
    )


HEADLINE = ("balanced_accuracy", "roc_auc", "detection_error", "mcc", "eer")


def print_report(result: dict[str, object]) -> None:
    summary = result["summary"]
    LOGGER.info("\n%s", format_condition(result["condition"]))
    LOGGER.info("seeds %s from %d folders", result["seeds"], len(result["folders"]))

    LOGGER.info("\n  training")
    LOGGER.info(
        "    %-28s %-6s %7s %10s %9s", "folder", "domain", "escape", "selected", "tr-val"
    )
    for entry in result["training"]:
        LOGGER.info(
            "    %-28s %-6s %7s %6s/%-3s %+9.4f",
            entry["folder"], entry["train_domain"], str(entry["escape_epoch"]),
            entry["selected_epoch"], entry["total_epochs"],
            entry["final_train_minus_val"],
        )

    LOGGER.info("\n  directions (mean +/- standard deviation over seeds)")
    LOGGER.info("    %-15s %s", "", "  ".join(f"{name:>22}" for name in HEADLINE))
    for direction in result["directions"]:
        values = summary[direction]
        cells = []
        for name in HEADLINE:
            mean, spread = values.get(f"{name}_mean"), values.get(f"{name}_std")
            cells.append("".rjust(22) if mean is None else f"{mean:.4f} +/- {spread:.4f}".rjust(22))
        LOGGER.info("    %-15s %s", direction, "  ".join(cells))

    gaps = summary.get("generalization_gaps", {})
    if gaps:
        LOGGER.info("\n  paired gaps (same domain minus cross domain, inside each seed)")
        for name, gap in gaps.items():
            LOGGER.info(
                "    %-34s bal %+.4f +/- %.4f   auc %+.4f +/- %.4f   over %d seeds",
                name, gap["balanced_accuracy_mean"], gap["balanced_accuracy_std"],
                gap.get("roc_auc_mean", float("nan")),
                gap.get("roc_auc_std", float("nan")), gap["seeds"],
            )

    if result["directions_with_missing_seeds"]:
        LOGGER.info(
            "\n  left out, because not every seed produced them: %s",
            ", ".join(result["directions_with_missing_seeds"]),
        )
    for conflict in result["conflicts"]:
        LOGGER.info(
            "\n  conflict: seed %s %s appears twice. Kept %s, replaced %s.",
            conflict["seed"], conflict["direction"], conflict["kept"],
            conflict["replaced"],
        )
    if result["provenance"]["any_dirty"]:
        LOGGER.info("\n  warning: at least one run came from a modified checkout.")
    if len(result["provenance"]["repository_commits"]) > 1:
        LOGGER.info(
            "\n  note: %d repository commits produced this condition: %s",
            len(result["provenance"]["repository_commits"]),
            ", ".join(c[:8] for c in result["provenance"]["repository_commits"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.protocol",
        description="Combine single-seed run folders into one protocol summary.",
    )
    parser.add_argument(
        "--results", default="results",
        help="Folder holding the run folders, or a run folder itself.",
    )
    parser.add_argument(
        "--glob", default="srnet_*", help="Pattern that selects the run folders.",
    )
    parser.add_argument("--output", default=None, help="Write the summary here as JSON.")
    return parser.parse_args()


def main() -> None:
    from src.logging_config import configure_logging

    configure_logging()
    args = parse_args()
    root = Path(args.results)
    folders = [root] if (root / "summary.json").exists() else sorted(root.glob(args.glob))
    if not folders:
        raise SystemExit(f"No run folders matched {root}/{args.glob}")
    results = aggregate(folders)
    for result in results:
        print_report(result)
    if args.output:
        path = Path(args.output)
        if path.is_dir() or args.output.endswith("/"):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = path / f"protocol_{stamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOGGER.info("\nWritten: %s", path.resolve())


if __name__ == "__main__":
    main()
