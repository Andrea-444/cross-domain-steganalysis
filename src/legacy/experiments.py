"""Legacy custom-CNN and LSB-statistics experiment runner."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.legacy.data import CleanStegoDataset
from src.legacy.lsb_baseline import (
    collect_samples,
    evaluate_lsb_logreg,
    train_lsb_logreg,
)
from src.legacy.train import Metrics, evaluate_model, train_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    dataset_root: Path,
    image_size: int,
    batch_size: int,
    seed: int,
    input_mode: str,
    real_clean_src: str | Path | None = None,
    ai_clean_src: str | Path | None = None,
    train_fraction: float = 0.8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    real_dataset = CleanStegoDataset(
        dataset_root / "real",
        image_size=image_size,
        input_mode=input_mode,
        clean_root=real_clean_src,
    )
    ai_dataset = CleanStegoDataset(
        dataset_root / "ai",
        image_size=image_size,
        input_mode=input_mode,
        clean_root=ai_clean_src,
    )

    rng = random.Random(seed)
    label_to_indices = {0: [], 1: []}
    for index, (_, label) in enumerate(real_dataset.samples):
        label_to_indices[label].append(index)

    min_class_count = min(len(indices) for indices in label_to_indices.values())
    if min_class_count < 2:
        raise ValueError("Real dataset needs at least two clean and two stego images.")

    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in label_to_indices.values():
        rng.shuffle(indices)
        indices = indices[:min_class_count]
        split_at = max(1, int(len(indices) * train_fraction))
        split_at = min(split_at, len(indices) - 1)
        train_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    ai_test_indices = balanced_indices_like(ai_dataset.samples, real_dataset.samples, test_indices, seed)
    real_train = Subset(real_dataset, train_indices)
    real_test = Subset(real_dataset, test_indices)
    ai_test = Subset(ai_dataset, ai_test_indices)

    train_loader = DataLoader(real_train, batch_size=batch_size, shuffle=True, num_workers=0)
    real_test_loader = DataLoader(real_test, batch_size=batch_size, shuffle=False, num_workers=0)
    ai_test_loader = DataLoader(ai_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, real_test_loader, ai_test_loader


def balanced_indices_like(
    samples: list[tuple[Path, int]],
    reference_samples: list[tuple[Path, int]],
    reference_indices: list[int],
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    needed_by_label = {0: 0, 1: 0}
    available_by_label = {0: [], 1: []}

    for index in reference_indices:
        _, label = reference_samples[index]
        needed_by_label[label] += 1
    for index, (_, label) in enumerate(samples):
        available_by_label[label].append(index)

    selected: list[int] = []
    for label, needed in needed_by_label.items():
        available = available_by_label[label]
        if len(available) < needed:
            raise ValueError(
                f"Need {needed} AI samples for label {label}, found {len(available)}."
            )
        rng.shuffle(available)
        selected.extend(available[:needed])

    rng.shuffle(selected)
    return selected


def split_real_samples(
    samples: list[tuple[Path, int]],
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    label_to_samples = {0: [], 1: []}
    for sample in samples:
        label_to_samples[sample[1]].append(sample)

    min_class_count = min(len(class_samples) for class_samples in label_to_samples.values())
    if min_class_count < 2:
        raise ValueError("Dataset needs at least two clean and two stego images.")

    train_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []
    for class_samples in label_to_samples.values():
        rng.shuffle(class_samples)
        class_samples = class_samples[:min_class_count]
        split_at = max(1, int(len(class_samples) * train_fraction))
        split_at = min(split_at, len(class_samples) - 1)
        train_samples.extend(class_samples[:split_at])
        test_samples.extend(class_samples[split_at:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def balanced_subset_like(
    samples: list[tuple[Path, int]],
    reference_samples: list[tuple[Path, int]],
    seed: int,
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    needed_by_label = {0: 0, 1: 0}
    available_by_label = {0: [], 1: []}

    for _, label in reference_samples:
        needed_by_label[label] += 1
    for sample in samples:
        available_by_label[sample[1]].append(sample)

    selected: list[tuple[Path, int]] = []
    for label, needed in needed_by_label.items():
        available = available_by_label[label]
        if len(available) < needed:
            raise ValueError(
                f"Need {needed} AI samples for label {label}, found {len(available)}."
            )
        rng.shuffle(available)
        selected.extend(available[:needed])

    rng.shuffle(selected)
    return selected


def metric_dict(metrics: Metrics) -> dict[str, object]:
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "confusion_matrix": metrics.confusion_matrix,
        "predicted_clean": metrics.predicted_clean,
        "predicted_stego": metrics.predicted_stego,
    }


def available_categories(
    dataset_root: Path,
    real_clean_src: str | Path | None = None,
    ai_clean_src: str | Path | None = None,
) -> list[str]:
    real_clean = Path(real_clean_src) if real_clean_src is not None else dataset_root / "real" / "clean"
    ai_clean = Path(ai_clean_src) if ai_clean_src is not None else dataset_root / "ai" / "clean"
    real_categories = {path.name for path in real_clean.iterdir() if path.is_dir()}
    ai_categories = {path.name for path in ai_clean.iterdir() if path.is_dir()}
    return sorted(real_categories & ai_categories)


def run_lsb_protocol_once(
    dataset_root: Path,
    image_size: int,
    seed: int,
    category: str | None = None,
    real_clean_src: str | Path | None = None,
    ai_clean_src: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    real_samples = collect_samples(
        dataset_root / "real",
        category=category,
        clean_root=real_clean_src,
    )
    ai_samples = collect_samples(
        dataset_root / "ai",
        category=category,
        clean_root=ai_clean_src,
    )
    real_train, real_test = split_real_samples(real_samples, seed=seed)
    ai_train, ai_test = split_real_samples(ai_samples, seed=seed)

    real_scaler, real_model = train_lsb_logreg(real_train, image_size=image_size)
    ai_scaler, ai_model = train_lsb_logreg(ai_train, image_size=image_size)

    return {
        "real_to_real": metric_dict(
            evaluate_lsb_logreg(real_scaler, real_model, real_test, image_size=image_size)
        ),
        "real_to_ai": metric_dict(
            evaluate_lsb_logreg(real_scaler, real_model, ai_test, image_size=image_size)
        ),
        "ai_to_ai": metric_dict(
            evaluate_lsb_logreg(ai_scaler, ai_model, ai_test, image_size=image_size)
        ),
        "ai_to_real": metric_dict(
            evaluate_lsb_logreg(ai_scaler, ai_model, real_test, image_size=image_size)
        ),
    }


def summarize_seed_results(
    seed_results: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    experiment_names = [
        name
        for name in seed_results[0].keys()
        if isinstance(seed_results[0][name], dict)
    ]
    summary: dict[str, dict[str, object]] = {}

    for experiment_name in experiment_names:
        summary[experiment_name] = {}
        for metric_name in metric_names:
            values = [
                float(seed_result[experiment_name][metric_name])
                for seed_result in seed_results
            ]
            summary[experiment_name][f"{metric_name}_mean"] = float(np.mean(values))
            summary[experiment_name][f"{metric_name}_std"] = float(np.std(values))

        matrices = [
            np.asarray(seed_result[experiment_name]["confusion_matrix"], dtype=np.int64)
            for seed_result in seed_results
        ]
        summary[experiment_name]["confusion_matrix_sum"] = np.sum(matrices, axis=0).tolist()

    return summary


def run_full_lsb_protocol(
    dataset_root: Path,
    image_size: int,
    seeds: list[int],
    per_category: bool,
    real_clean_src: str | Path | None = None,
    ai_clean_src: str | Path | None = None,
    include_runs: bool = False,
) -> dict[str, object]:
    overall_runs = []
    for seed in seeds:
        run = run_lsb_protocol_once(
            dataset_root,
            image_size=image_size,
            seed=seed,
            real_clean_src=real_clean_src,
            ai_clean_src=ai_clean_src,
        )
        run["seed"] = seed
        overall_runs.append(run)

    overall: dict[str, object] = {"summary": summarize_seed_results(overall_runs)}
    if include_runs:
        overall["runs"] = overall_runs
    results: dict[str, object] = {"overall": overall}

    if per_category:
        categories: dict[str, object] = {}
        for category in available_categories(
            dataset_root,
            real_clean_src=real_clean_src,
            ai_clean_src=ai_clean_src,
        ):
            category_runs = []
            for seed in seeds:
                run = run_lsb_protocol_once(
                    dataset_root,
                    image_size=image_size,
                    seed=seed,
                    category=category,
                    real_clean_src=real_clean_src,
                    ai_clean_src=ai_clean_src,
                )
                run["seed"] = seed
                category_runs.append(run)
            categories[category] = {"summary": summarize_seed_results(category_runs)}
            if include_runs:
                categories[category]["runs"] = category_runs
        results["categories"] = categories

    return results


def print_summary(results: dict[str, object]) -> None:
    if "overall" not in results:
        print(json.dumps(results, indent=2))
        return

    print("\nOverall summary")
    print("experiment       accuracy        f1")
    print("--------------------------------------")
    overall = results["overall"]["summary"]
    for name, metrics in overall.items():
        print(
            f"{name:<15} "
            f"{metrics['accuracy_mean']:.3f} +/- {metrics['accuracy_std']:.3f}   "
            f"{metrics['f1_mean']:.3f} +/- {metrics['f1_std']:.3f}"
        )

    categories = results.get("categories")
    if categories:
        print("\nPer-category cross-domain accuracy")
        print("category        real->AI    AI->real")
        print("--------------------------------------")
        for category, category_data in categories.items():
            summary = category_data["summary"]
            real_to_ai = summary["real_to_ai"]["accuracy_mean"]
            ai_to_real = summary["ai_to_real"]["accuracy_mean"]
            print(f"{category:<14} {real_to_ai:.3f}      {ai_to_real:.3f}")


def run_experiments(
    dataset: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    image_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    seeds: list[int] | None = None,
    input_mode: str = "rgb",
    baseline: str = "cnn",
    protocol: str = "basic",
    per_category: bool = False,
    real_clean_src: str | Path | None = None,
    ai_clean_src: str | Path | None = None,
    include_runs: bool = False,
    out: str | Path | None = "results",
) -> dict[str, object]:
    seed_everything(seed)
    dataset_root = Path(dataset)
    seeds = seeds or [seed]

    if protocol == "full":
        if baseline != "lsb-logreg":
            raise ValueError("The full protocol currently supports --baseline lsb-logreg.")
        results = {
            "baseline": {"name": baseline},
            "protocol": "full",
            "dataset": str(dataset_root),
            "seeds": seeds,
            "image_size": image_size,
            **run_full_lsb_protocol(
                dataset_root=dataset_root,
                image_size=image_size,
                seeds=seeds,
                per_category=per_category,
                real_clean_src=real_clean_src,
                ai_clean_src=ai_clean_src,
                include_runs=include_runs,
            ),
        }
    elif baseline == "cnn":
        train_loader, real_test_loader, ai_test_loader = make_loaders(
            dataset_root=dataset_root,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            input_mode=input_mode,
            real_clean_src=real_clean_src,
            ai_clean_src=ai_clean_src,
        )

        model = train_model(train_loader, epochs=epochs, lr=lr, input_channels=3)
        experiment_a = evaluate_model(model, real_test_loader)
        experiment_b = evaluate_model(model, ai_test_loader)
        results = {
            "baseline": {"name": baseline, "input_mode": input_mode},
            "protocol": "basic",
            "dataset": str(dataset_root),
            "experiment_a_train_real_test_real": metric_dict(experiment_a),
            "experiment_b_train_real_test_ai": metric_dict(experiment_b),
        }
    elif baseline == "lsb-logreg":
        real_samples = collect_samples(dataset_root / "real", clean_root=real_clean_src)
        ai_samples = collect_samples(dataset_root / "ai", clean_root=ai_clean_src)
        real_train, real_test = split_real_samples(real_samples, seed=seed)
        ai_test = balanced_subset_like(ai_samples, reference_samples=real_test, seed=seed)
        scaler, model = train_lsb_logreg(real_train, image_size=image_size)
        experiment_a = evaluate_lsb_logreg(scaler, model, real_test, image_size=image_size)
        experiment_b = evaluate_lsb_logreg(scaler, model, ai_test, image_size=image_size)
        results = {
            "baseline": {"name": baseline, "input_mode": input_mode},
            "protocol": "basic",
            "dataset": str(dataset_root),
            "experiment_a_train_real_test_real": metric_dict(experiment_a),
            "experiment_b_train_real_test_ai": metric_dict(experiment_b),
        }
    else:
        raise ValueError("baseline must be 'cnn' or 'lsb-logreg'.")

    print_summary(results)

    if out:
        out_path = make_results_path(out, baseline=baseline)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {out_path.resolve()}")

    return results


def make_results_path(out: str | Path, baseline: str) -> Path:
    out_path = Path(out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if out_path.suffix.lower() == ".json":
        if out_path.exists():
            return out_path.with_name(f"{out_path.stem}_{timestamp}{out_path.suffix}")
        return out_path

    return out_path / f"{baseline}_{timestamp}" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small clean-vs-stego experiments.")
    parser.add_argument("--dataset", default="dataset", help="Prepared dataset root.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple random splits and summarize them.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["rgb", "residual"],
        default="rgb",
        help="Use raw RGB images or simple high-pass residual images.",
    )
    parser.add_argument(
        "--baseline",
        choices=["cnn", "lsb-logreg"],
        default="cnn",
        help="Use the small CNN or a simple LSB-statistics logistic regression sanity check.",
    )
    parser.add_argument(
        "--protocol",
        choices=["basic", "full"],
        default="basic",
        help="basic runs real->real and real->AI. full runs all real/AI train-test directions.",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="With --protocol full, also run each shared category separately.",
    )
    parser.add_argument(
        "--real-clean-src",
        default=None,
        help="Original real clean image folder, used when --dataset contains only stego images.",
    )
    parser.add_argument(
        "--ai-clean-src",
        default=None,
        help="Original AI clean image folder, used when --dataset contains only stego images.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Results parent directory, or a .json file path. Directory outputs create timestamped folders.",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="Save every per-seed run. By default only summaries are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        seed=args.seed,
        seeds=args.seeds,
        input_mode=args.input_mode,
        baseline=args.baseline,
        protocol=args.protocol,
        per_category=args.per_category,
        real_clean_src=args.real_clean_src,
        ai_clean_src=args.ai_clean_src,
        include_runs=args.include_runs,
        out=args.out,
    )
