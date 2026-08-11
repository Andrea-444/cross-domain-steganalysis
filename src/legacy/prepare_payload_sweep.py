from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.legacy.prepare_dataset import prepare_dataset
from src.legacy.stego_lsb import STEGO_METHODS


def parse_payloads(value: str) -> list[int]:
    payloads = [int(part.strip()) for part in value.split(",") if part.strip()]
    invalid = [payload for payload in payloads if payload < 1 or payload > 4]
    if invalid:
        raise argparse.ArgumentTypeError("Payload bits must be between 1 and 4.")
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for multiple LSB payloads.")
    parser.add_argument("--real-src", required=True, help="Folder with natural images.")
    parser.add_argument("--ai-src", required=True, help="Folder with AI-generated images.")
    parser.add_argument("--out-prefix", default="stego_runs/payload_sweep", help="Output prefix.")
    parser.add_argument("--payloads", type=parse_payloads, default=[1, 2, 3, 4])
    parser.add_argument("--method", choices=sorted(STEGO_METHODS), default="lsb")
    parser.add_argument("--payload-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=300, help="Max images per domain/category.")
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--save-clean", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_timestamp:
        sweep_root = Path(args.out_prefix)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_root = Path(f"{args.out_prefix}_{timestamp}")

    for payload_bits in args.payloads:
        rate_label = f"r{int(args.payload_rate * 100):03d}"
        out = sweep_root / f"{args.method}_b{payload_bits}_{rate_label}"
        print(f"\nPreparing {args.method} {payload_bits}-bit dataset -> {out}")
        prepare_dataset(
            real_src=args.real_src,
            ai_src=args.ai_src,
            out=out,
            limit=args.limit,
            limit_per_category=args.limit_per_category,
            size=args.size,
            preserve_categories=not args.flat,
            payload_bits=payload_bits,
            method=args.method,
            payload_rate=args.payload_rate,
            seed=args.seed,
            save_clean=args.save_clean,
            timestamped=False,
        )

    print(f"\nPayload sweep written to: {sweep_root.resolve()}")
