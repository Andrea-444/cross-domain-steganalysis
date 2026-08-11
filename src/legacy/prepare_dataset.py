from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.legacy.stego_lsb import STEGO_METHODS, list_images, make_stego_pairs


def limited_images(folder: str | Path, limit: int | None = None) -> list[Path]:
    images = list_images(folder)
    if limit is not None:
        images = images[:limit]
    return images


def category_dirs(domain_src: str | Path) -> dict[str, Path]:
    """Return immediate category folders, or one default category for a flat folder."""
    domain_src = Path(domain_src)
    categories = {
        path.name: path
        for path in sorted(domain_src.iterdir())
        if path.is_dir() and list_images(path)
    }
    if categories:
        return categories
    return {"uncategorized": domain_src}


def prepare_domain(
    domain_name: str,
    src: str | Path,
    out: Path,
    limit: int | None,
    limit_per_category: int | None,
    size: int,
    preserve_categories: bool,
    payload_bits: int,
    save_clean: bool,
    method: str,
    payload_rate: float,
    seed: int,
) -> int:
    total = 0
    for category, category_src in category_dirs(src).items():
        category_limit = limit_per_category if limit_per_category is not None else limit
        images = limited_images(category_src, category_limit)
        if len(images) < 2:
            print(f"Skipping {domain_name}/{category}: need at least 2 images, found {len(images)}.")
            continue

        if preserve_categories:
            clean_out = out / domain_name / "clean" / category if save_clean else None
            stego_out = out / domain_name / "stego" / category
            prefix = f"{domain_name}_{category}"
        else:
            clean_out = out / domain_name / "clean" if save_clean else None
            stego_out = out / domain_name / "stego"
            prefix = f"{domain_name}_{category}"

        total += make_stego_pairs(
            images,
            clean_out,
            stego_out,
            size=size,
            prefix=prefix,
            payload_bits=payload_bits,
            save_clean=save_clean,
            method=method,
            payload_rate=payload_rate,
            seed=seed + total,
        )

    if total < 2:
        raise ValueError(f"Need at least two usable {domain_name} images, found {total}.")
    return total


def prepare_dataset(
    real_src: str | Path,
    ai_src: str | Path,
    out: str | Path = "dataset",
    limit: int | None = 300,
    limit_per_category: int | None = None,
    size: int = 128,
    preserve_categories: bool = True,
    payload_bits: int = 4,
    method: str = "lsb",
    payload_rate: float = 1.0,
    seed: int = 42,
    save_clean: bool = False,
    timestamped: bool = True,
) -> None:
    out = (
        timestamped_output_path(out, payload_bits=payload_bits, method=method, payload_rate=payload_rate)
        if timestamped
        else Path(out)
    )
    real_count = prepare_domain(
        domain_name="real",
        src=real_src,
        out=out,
        limit=limit,
        limit_per_category=limit_per_category,
        size=size,
        preserve_categories=preserve_categories,
        payload_bits=payload_bits,
        save_clean=save_clean,
        method=method,
        payload_rate=payload_rate,
        seed=seed,
    )
    ai_count = prepare_domain(
        domain_name="ai",
        src=ai_src,
        out=out,
        limit=limit,
        limit_per_category=limit_per_category,
        size=size,
        preserve_categories=preserve_categories,
        payload_bits=payload_bits,
        save_clean=save_clean,
        method=method,
        payload_rate=payload_rate,
        seed=seed + real_count,
    )

    if save_clean:
        print(f"Prepared {real_count} real clean/stego pairs.")
        print(f"Prepared {ai_count} AI clean/stego pairs.")
    else:
        print(f"Prepared {real_count} real stego images.")
        print(f"Prepared {ai_count} AI stego images.")
    print(f"Dataset written to: {out.resolve()}")


def timestamped_output_path(
    out: str | Path,
    payload_bits: int,
    method: str,
    payload_rate: float,
) -> Path:
    out = Path(out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rate_label = f"r{int(payload_rate * 100):03d}"
    return out / f"{method}_b{payload_bits}_{rate_label}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean/stego image dataset.")
    parser.add_argument("--real-src", required=True, help="Folder with natural images.")
    parser.add_argument("--ai-src", required=True, help="Folder with AI-generated images.")
    parser.add_argument("--out", default="stego_runs", help="Output parent folder.")
    parser.add_argument("--limit", type=int, default=300, help="Max images per domain/category.")
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="Override --limit with a max image count for each category folder.",
    )
    parser.add_argument("--size", type=int, default=128, help="Square resize size.")
    parser.add_argument(
        "--payload-bits",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of low bits used for hiding.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(STEGO_METHODS),
        default="lsb",
        help="Steganography method.",
    )
    parser.add_argument(
        "--payload-rate",
        type=float,
        default=1.0,
        help="Fraction of pixel-channel values modified. Use lower values for harder sparse embedding.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Do not preserve category folders in the prepared dataset.",
    )
    parser.add_argument(
        "--save-clean",
        action="store_true",
        help="Also copy clean images. By default only stego images are saved.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write directly to --out instead of creating a timestamped subfolder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        real_src=args.real_src,
        ai_src=args.ai_src,
        out=args.out,
        limit=args.limit,
        limit_per_category=args.limit_per_category,
        size=args.size,
        preserve_categories=not args.flat,
        payload_bits=args.payload_bits,
        method=args.method,
        payload_rate=args.payload_rate,
        seed=args.seed,
        save_clean=args.save_clean,
        timestamped=not args.no_timestamp,
    )
