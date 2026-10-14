"""Repeatable checks on the detector pipeline.

Every check here began as a throwaway script during the investigation of the
stage 1 null result, when `real_to_real` returned exactly 0.5000 and the
confusion matrix held the same row twice. A number that lives only in terminal
scrollback cannot be cited, repeated, or compared against a later run, so the
checks are commands in the repository instead.

Each check writes a JSON report. Copy a report into `evidence/` by hand when it
is worth keeping: no command writes there, so it never collides with a run.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.manifest import read_manifest
from src.logging_config import ProgressReporter, get_logger


LOGGER = get_logger(__name__)

# Second-order residual triples, as in SPAM. T=2 keeps the descriptor at 500
# features, which a few thousand training images can support.
RESIDUAL_CAP = 2
RESIDUAL_SIDE = 2 * RESIDUAL_CAP + 1

DEFAULT_LEARNING_RATES = (1e-3, 1e-4, 1e-5)


def report_path(output: str | None, check: str) -> Path | None:
    if output is None:
        return None
    path = Path(output)
    if path.is_dir() or output.endswith("/"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = path / f"{check}_{stamp}.json"
    return path


def write_report(path: Path | None, result: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info(f"Report: {path.resolve()}")


def load_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def synthetic_stego(cover: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Non-adaptive plus-or-minus-one on a fraction of the pixels.

    This is a positive control, not a steganography method. It touches flat
    regions that an adaptive algorithm avoids, so a detector that cannot find
    it has a problem of its own.
    """
    mask = rng.random(cover.shape) < rate
    step = rng.integers(0, 2, cover.shape).astype(np.int16) * 2 - 1
    return np.clip(cover.astype(np.int16) + mask * step, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# embedding: does the stego image differ from its cover, and by how much?
# --------------------------------------------------------------------------


def check_embedding(manifest: str | Path, sample: int, seed: int) -> dict[str, object]:
    records = read_manifest(manifest)
    shared_paths = sum(
        1 for record in records if record["clean_path"] == record["stego_path"]
    )
    chosen = random.Random(seed).sample(records, min(sample, len(records)))
    rates: list[float] = []
    peaks: set[int] = set()
    for record in chosen:
        cover = load_grayscale(str(record["clean_path"])).astype(np.int16)
        stego = load_grayscale(str(record["stego_path"])).astype(np.int16)
        difference = stego - cover
        rates.append(float(np.count_nonzero(difference) / difference.size))
        peaks.add(int(np.abs(difference).max()))
    identical = sum(1 for rate in rates if rate == 0.0)
    return {
        "check": "embedding",
        "manifest": str(Path(manifest).resolve()),
        "record_count": len(records),
        "sampled": len(chosen),
        "shared_paths": shared_paths,
        "identical_pairs": identical,
        "changed_fraction_min": min(rates) if rates else None,
        "changed_fraction_median": float(np.median(rates)) if rates else None,
        "changed_fraction_max": max(rates) if rates else None,
        "absolute_difference_peaks": sorted(peaks),
        "expected": {
            "shared_paths": 0,
            "identical_pairs": 0,
            "changed_fraction": "0.05 to 0.20 for S-UNIWARD at 0.4 bpp",
            "absolute_difference_peaks": [1],
        },
    }


# --------------------------------------------------------------------------
# provenance: which preprocessing produced the covers behind a stego run?
# --------------------------------------------------------------------------


def check_provenance(run_config: str | Path, max_hops: int = 8) -> dict[str, object]:
    """Follow a stego run back to the preparation step that set the pixels.

    Each step writes its own config beside its manifest, and each names its
    parent under a different key. The preprocessing mode is several steps back,
    so a single read of the stego run cannot answer the question.
    """
    config = json.loads(Path(run_config).read_text(encoding="utf-8"))
    chain: list[dict[str, object]] = []
    manifest = Path(str(config["input_manifest"]))
    for _ in range(max_hops):
        beside = manifest.parent / (manifest.stem + ".config.json")
        step: dict[str, object] = {"manifest": manifest.name, "config_found": beside.exists()}
        if not beside.exists():
            chain.append(step)
            break
        data = json.loads(beside.read_text(encoding="utf-8"))
        for key in ("name", "strategy", "preprocess_mode", "size", "record_count"):
            if key in data:
                step[key] = data[key]
        chain.append(step)
        parent = data.get("source_manifest") or data.get("extraction_manifest")
        if not parent:
            break
        manifest = Path(str(parent))
    modes = [step["preprocess_mode"] for step in chain if step.get("preprocess_mode")]
    return {
        "check": "provenance",
        "run_config": str(Path(run_config).resolve()),
        "embedding": {
            "algorithm": config.get("algorithm"),
            "payload_bpp": config.get("payload_bpp"),
            "record_count": config.get("record_count"),
        },
        "chain": chain,
        "preprocess_mode": modes[0] if modes else None,
    }


# --------------------------------------------------------------------------
# classical: can a classical feature set separate clean from stego?
# --------------------------------------------------------------------------


def residual_triples(difference: np.ndarray, along_rows: bool) -> np.ndarray:
    capped = np.clip(difference, -RESIDUAL_CAP, RESIDUAL_CAP) + RESIDUAL_CAP
    if along_rows:
        first, second, third = capped[:, :-2], capped[:, 1:-1], capped[:, 2:]
    else:
        first, second, third = capped[:-2, :], capped[1:-1, :], capped[2:, :]
    index = (first * RESIDUAL_SIDE + second) * RESIDUAL_SIDE + third
    counts = np.bincount(index.ravel(), minlength=RESIDUAL_SIDE**3).astype(np.float64)
    return counts / max(counts.sum(), 1)


def classical_features(array: np.ndarray) -> np.ndarray:
    """500 features: second-order residual triples in four directions."""
    image = array.astype(np.int16)
    return np.concatenate(
        [
            residual_triples(image[:, :-1] - image[:, 1:], True),
            residual_triples(image[:-1, :] - image[1:, :], False),
            residual_triples(image[:-1, :-1] - image[1:, 1:], True),
            residual_triples(image[:-1, 1:] - image[1:, :-1], True),
        ]
    )


def check_classical(
    manifest: str | Path,
    domain: str,
    train_pairs: int,
    test_pairs: int,
    synthetic: bool,
    synthetic_rate: float,
    seed: int,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = [r for r in read_manifest(manifest) if str(r["domain"]) == domain]
    random.Random(seed).shuffle(records)
    needed = train_pairs + test_pairs
    if len(records) < needed:
        raise ValueError(f"Need {needed} pairs in domain {domain}, found {len(records)}.")
    train_set = records[:train_pairs]
    test_set = records[train_pairs:needed]
    rng = np.random.default_rng(seed)

    def build(pairs: list[dict[str, object]], label: str):
        features, labels = [], []
        progress = ProgressReporter(LOGGER, total=len(pairs), label=f"features {label}")
        for record in pairs:
            cover = load_grayscale(str(record["clean_path"]))
            stego = (
                synthetic_stego(cover, rng, synthetic_rate)
                if synthetic
                else load_grayscale(str(record["stego_path"]))
            )
            features.append(classical_features(cover))
            labels.append(0)
            features.append(classical_features(stego))
            labels.append(1)
            progress.advance(1)
        progress.finish()
        return np.array(features), np.array(labels)

    train_features, train_labels = build(train_set, "train")
    test_features, test_labels = build(test_set, "test")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    model.fit(train_features, train_labels)
    probability = model.predict_proba(test_features)[:, 1]
    auc = float(roc_auc_score(test_labels, probability))
    # Hanley and McNeil: the spread of the area under the curve when the two
    # classes are the same size and the detector is at chance.
    positives = int((test_labels == 1).sum())
    negatives = int((test_labels == 0).sum())
    auc_error = float(
        np.sqrt((positives + negatives + 1) / (12.0 * positives * negatives))
    )
    return {
        "check": "classical",
        "manifest": str(Path(manifest).resolve()),
        "domain": domain,
        "condition": "synthetic" if synthetic else "manifest",
        "synthetic_rate": synthetic_rate if synthetic else None,
        "seed": seed,
        "train_pairs": len(train_set),
        "test_pairs": len(test_set),
        "feature_count": int(train_features.shape[1]),
        "train_accuracy": float(model.score(train_features, train_labels)),
        "test_accuracy": float(accuracy_score(test_labels, model.predict(test_features))),
        "test_roc_auc": auc,
        "roc_auc_chance_error": auc_error,
        "roc_auc_z_score": float((auc - 0.5) / auc_error) if auc_error > 0 else None,
    }


# --------------------------------------------------------------------------
# collapse: does the detector keep its sensitivity to the embedding?
# --------------------------------------------------------------------------


def check_collapse(
    manifest: str | Path,
    domain: str,
    steps: int,
    learning_rates: list[float],
    batch_pairs: int,
    seed: int,
    reference_root: str | Path,
    device_name: str | None,
) -> dict[str, object]:
    """Measure what the detector responds to, over a short run.

    `pair_gap` is the output difference between a cover and its own stego.
    `logit_std` is the output spread across the batch. Their ratio is the part
    of the response that follows the embedding rather than the picture, and it
    is the quantity that matters: the two separately move with the output
    scale, which the initialization sets.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.steganalysis.srnet import (
        ManifestPairDataset,
        initialize_srnet,
        paired_batch,
        resolve_device,
        select_training_records,
    )

    records = read_manifest(manifest)
    chosen = select_training_records(records, domain, "train", seed)
    device = resolve_device(device_name)
    data = ManifestPairDataset(
        chosen, domain, "train", augment=True, preselected=True, paired=True
    )
    marks = sorted({1, 10, steps // 4, steps // 2, steps})
    traces: dict[str, list[dict[str, float]]] = {}
    for rate in learning_rates:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        model = initialize_srnet(reference_root, device)
        optimizer = torch.optim.Adamax(model.parameters(), lr=rate)
        criterion = nn.NLLLoss()
        model.train()
        loader = DataLoader(data, batch_size=batch_pairs, shuffle=True, num_workers=0)
        stream = iter(loader)
        trace: list[dict[str, float]] = []
        LOGGER.info(f"learning rate {rate}: {steps} steps on {len(data)} pairs")
        for step in range(1, steps + 1):
            try:
                clean, stego, _ = next(stream)
            except StopIteration:
                stream = iter(loader)
                clean, stego, _ = next(stream)
            images, labels = paired_batch(clean, stego, device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            if step in marks:
                half = clean.size(0)
                spread = float(output.std(dim=0).mean())
                gap = float((output[:half] - output[half:]).abs().mean())
                entry = {
                    "step": step,
                    "loss": float(loss),
                    "gradient_first_conv": float(
                        model.type1s[0].convbn.conv.weight.grad.norm()
                    ),
                    "gradient_dense": float(model.dense.weight.grad.norm()),
                    "dense_norm": float(model.dense.weight.detach().norm()),
                    "logit_std": spread,
                    "pair_gap": gap,
                    "pair_gap_ratio": gap / spread if spread > 0 else None,
                }
                trace.append(entry)
                LOGGER.info(
                    f"  step {step:4d} loss {entry['loss']:.6f} "
                    f"logit_std {spread:.3e} pair_gap {gap:.3e} "
                    f"ratio {entry['pair_gap_ratio']:.3e}"
                )
            optimizer.step()
        traces[str(rate)] = trace
    return {
        "check": "collapse",
        "manifest": str(Path(manifest).resolve()),
        "domain": domain,
        "steps": steps,
        "batch_pairs": batch_pairs,
        "seed": seed,
        "device": str(device),
        "natural_log_of_two": float(np.log(2)),
        "traces": traces,
        "reading": (
            "A pair_gap_ratio that holds or rises means the detector keeps its "
            "sensitivity to the embedding. A ratio that falls means it is "
            "learning to answer the same thing for every image."
        ),
    }


# --------------------------------------------------------------------------
# synthetic-train: can the detector learn an easy embedding at all?
# --------------------------------------------------------------------------


def check_synthetic_train(
    manifest: str | Path,
    domain: str,
    train_pairs: int,
    val_pairs: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    synthetic_rate: float,
    seed: int,
    reference_root: str | Path,
    device_name: str | None,
) -> dict[str, object]:
    """Train the real detector on an embedding a classical detector can find.

    This separates "the detector cannot learn" from "this embedding is hard".
    The synthetic images go to a temporary folder, so no dataset changes.
    """
    from src.steganalysis.srnet import train_srnet

    records = [r for r in read_manifest(manifest) if str(r["domain"]) == domain]
    train_set = [r for r in records if str(r["split"]) == "train"][:train_pairs]
    val_set = [r for r in records if str(r["split"]) == "val"][:val_pairs]
    if not train_set or not val_set:
        raise ValueError(f"No train or val records for domain {domain}.")
    directory = Path(tempfile.mkdtemp(prefix="synthetic_stego_"))
    rng = np.random.default_rng(seed)
    prepared: list[dict[str, object]] = []
    progress = ProgressReporter(
        LOGGER, total=len(train_set) + len(val_set), label="synthetic stego"
    )
    for record in train_set + val_set:
        cover = load_grayscale(str(record["clean_path"]))
        stego = synthetic_stego(cover, rng, synthetic_rate)
        destination = directory / f"{record['image_id']}_synthetic.png"
        Image.fromarray(stego, mode="L").save(destination)
        prepared.append({**record, "stego_path": str(destination)})
        progress.advance(1)
    progress.finish()
    LOGGER.info(f"Synthetic stego images: {directory}")

    outcome = train_srnet(
        records=prepared,
        train_domain=domain,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        reference_root=reference_root,
        device_name=device_name,
    )
    history = outcome.history
    return {
        "check": "synthetic-train",
        "manifest": str(Path(manifest).resolve()),
        "domain": domain,
        "synthetic_rate": synthetic_rate,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(outcome.device),
        "scratch_directory": str(directory),
        "natural_log_of_two": float(np.log(2)),
        "best_val_balanced_accuracy": max(
            float(entry["val_balanced_accuracy"]) for entry in history
        ),
        "best_train_accuracy": max(float(entry["train_accuracy"]) for entry in history),
        "lowest_train_loss": min(float(entry["train_loss"]) for entry in history),
        "selection": {"epoch": outcome.best_epoch, "val_loss": outcome.best_val_loss},
        "history": history,
        "reading": (
            "A balanced accuracy well above 0.5 means the detector and the "
            "training code work, and the difficulty is in the embedding. A "
            "balanced accuracy at 0.5 means a fault that more training time "
            "does not repair."
        ),
    }


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def add_arguments(subparsers) -> None:
    from src.steganalysis.srnet import DEFAULT_SRNET_ROOT

    embedding = subparsers.add_parser(
        "embedding", help="Confirm the stego images differ from their covers."
    )
    embedding.add_argument("--manifest", required=True)
    embedding.add_argument("--sample", type=int, default=30)
    embedding.add_argument("--seed", type=int, default=0)
    embedding.add_argument("--output", default=None)

    provenance = subparsers.add_parser(
        "provenance", help="Follow a stego run back to its preprocessing mode."
    )
    provenance.add_argument(
        "--run-config", required=True, help="config.json of an embedding run."
    )
    provenance.add_argument("--output", default=None)

    classical = subparsers.add_parser(
        "classical",
        help="Test the covers with a classical feature set and logistic regression.",
    )
    classical.add_argument("--manifest", required=True)
    classical.add_argument("--domain", default="real", choices=["real", "ai"])
    classical.add_argument("--train-pairs", type=int, default=1200)
    classical.add_argument("--test-pairs", type=int, default=300)
    classical.add_argument(
        "--synthetic",
        action="store_true",
        help="Positive control: replace the stego images with non-adaptive +/-1.",
    )
    classical.add_argument("--synthetic-rate", type=float, default=0.5)
    classical.add_argument("--seed", type=int, default=0)
    classical.add_argument("--output", default=None)

    collapse = subparsers.add_parser(
        "collapse",
        help="Measure whether the detector keeps its sensitivity to the embedding.",
    )
    collapse.add_argument("--manifest", required=True)
    collapse.add_argument("--domain", default="real", choices=["real", "ai", "mixed"])
    collapse.add_argument("--steps", type=int, default=200)
    collapse.add_argument(
        "--learning-rates", type=float, nargs="+", default=list(DEFAULT_LEARNING_RATES)
    )
    collapse.add_argument("--batch-pairs", type=int, default=16)
    collapse.add_argument("--seed", type=int, default=1)
    collapse.add_argument("--srnet-root", default=str(DEFAULT_SRNET_ROOT))
    collapse.add_argument("--device", default=None)
    collapse.add_argument("--output", default=None)

    synthetic = subparsers.add_parser(
        "synthetic-train",
        help="Train the detector on an easy embedding, as a positive control.",
    )
    synthetic.add_argument("--manifest", required=True)
    synthetic.add_argument("--domain", default="real", choices=["real", "ai", "mixed"])
    synthetic.add_argument("--train-pairs", type=int, default=1200)
    synthetic.add_argument("--val-pairs", type=int, default=300)
    synthetic.add_argument("--epochs", type=int, default=60)
    synthetic.add_argument("--batch-size", type=int, default=32)
    synthetic.add_argument("--learning-rate", type=float, default=1e-4)
    synthetic.add_argument("--synthetic-rate", type=float, default=0.5)
    synthetic.add_argument("--seed", type=int, default=1)
    synthetic.add_argument("--srnet-root", default=str(DEFAULT_SRNET_ROOT))
    synthetic.add_argument("--device", default=None)
    synthetic.add_argument("--output", default=None)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.check == "embedding":
        result = check_embedding(args.manifest, args.sample, args.seed)
    elif args.check == "provenance":
        result = check_provenance(args.run_config)
    elif args.check == "classical":
        result = check_classical(
            manifest=args.manifest,
            domain=args.domain,
            train_pairs=args.train_pairs,
            test_pairs=args.test_pairs,
            synthetic=args.synthetic,
            synthetic_rate=args.synthetic_rate,
            seed=args.seed,
        )
    elif args.check == "collapse":
        result = check_collapse(
            manifest=args.manifest,
            domain=args.domain,
            steps=args.steps,
            learning_rates=args.learning_rates,
            batch_pairs=args.batch_pairs,
            seed=args.seed,
            reference_root=args.srnet_root,
            device_name=args.device,
        )
    elif args.check == "synthetic-train":
        result = check_synthetic_train(
            manifest=args.manifest,
            domain=args.domain,
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            synthetic_rate=args.synthetic_rate,
            seed=args.seed,
            reference_root=args.srnet_root,
            device_name=args.device,
        )
    else:
        raise ValueError(f"Unknown check: {args.check}")

    summary = {key: value for key, value in result.items() if key not in {"history", "traces"}}
    LOGGER.info("\n" + json.dumps(summary, indent=2))
    write_report(report_path(args.output, str(args.check)), result)
    return result


def main() -> None:
    from src.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m src.steganalysis.diagnostics",
        description="Repeatable checks on the detector pipeline.",
    )
    subparsers = parser.add_subparsers(dest="check", required=True)
    add_arguments(subparsers)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
