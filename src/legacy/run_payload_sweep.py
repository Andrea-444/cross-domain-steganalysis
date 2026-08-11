from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.legacy.experiments import run_experiments
from src.legacy.prepare_payload_sweep import parse_payloads
from src.legacy.stego_lsb import STEGO_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiments for multiple payload datasets.")
    parser.add_argument("--sweep-root", default=None, help="Folder containing lsb1, lsb2, ...")
    parser.add_argument("--dataset-prefix", default="dataset_lsb")
    parser.add_argument("--payloads", type=parse_payloads, default=[1, 2, 3, 4])
    parser.add_argument("--method", choices=sorted(STEGO_METHODS), default="lsb")
    parser.add_argument("--payload-rate", type=float, default=1.0)
    parser.add_argument("--baseline", choices=["lsb-logreg"], default="lsb-logreg")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--per-category", action="store_true")
    parser.add_argument("--real-clean-src", default=None)
    parser.add_argument("--ai-clean-src", default=None)
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rate_label = f"r{int(args.payload_rate * 100):03d}"
    out_dir = Path(args.out) / f"{args.method}_{rate_label}_sweep_{timestamp}"

    for payload_bits in args.payloads:
        if args.sweep_root is not None:
            dataset = Path(args.sweep_root) / f"{args.method}_b{payload_bits}_{rate_label}"
        else:
            dataset = Path(f"{args.dataset_prefix}{payload_bits}")
        if not dataset.exists():
            print(f"Skipping missing dataset: {dataset}")
            continue
        print(f"\n{args.method} payload {payload_bits}-bit rate {args.payload_rate}: {dataset}")
        run_experiments(
            dataset=dataset,
            baseline=args.baseline,
            protocol="full",
            per_category=args.per_category,
            image_size=args.image_size,
            seeds=args.seeds,
            real_clean_src=args.real_clean_src,
            ai_clean_src=args.ai_clean_src,
            include_runs=False,
            out=out_dir / f"payload_b{payload_bits}_{rate_label}.json",
        )

    print(f"\nSweep results folder: {out_dir.resolve()}")
