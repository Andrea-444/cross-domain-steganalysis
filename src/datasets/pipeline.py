from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from src.datasets.extractor import extract_selection_stage
from src.datasets.index import build_dataset_index
from src.datasets.acquisition import add_arguments as add_acquisition_arguments
from src.datasets.audit import add_arguments as add_audit_arguments
from src.datasets.rebalance import add_arguments as add_rebalance_arguments
from src.datasets.rebalance import run as run_rebalance
from src.datasets.recompress import add_arguments as add_recompress_arguments
from src.datasets.recompress import run as run_recompress
from src.datasets.audit import run as run_audit
from src.datasets.acquisition import run as run_acquisition
from src.datasets.class_mapping import build_genimagepp_class_maps
from src.datasets.manifest import stable_id, write_manifest
from src.datasets.selection import create_selection_manifest


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


GENIMAGE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
)
GENIMAGE_GENERATORS = {
    "stable-diffusion-v1-4": "stable_diffusion_v_1_4/",
    "stable-diffusion-v1-5": "stable_diffusion_v_1_5/",
    "midjourney": "Midjourney/",
    "adm": "ADM/",
    "biggan": "BigGAN/",
    "glide": "glide/",
    "vqdm": "VQDM/",
    "wukong": "wukong/",
}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_names(count: int, seed: int, train_ratio: float, val_ratio: float) -> list[str]:
    if count < 3:
        raise ValueError("At least three images per domain are required.")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    train_count = max(1, int(count * train_ratio))
    val_count = max(1, int(count * val_ratio))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
    names = ["test"] * count
    for index in indices[:train_count]:
        names[index] = "train"
    for index in indices[train_count : train_count + val_count]:
        names[index] = "val"
    return names


def normalize_image(source: Path, destination: Path, size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", optimize=False)


def prepare_domains(
    real_src: str | Path,
    ai_src: str | Path,
    output: str | Path,
    name: str,
    source: str,
    limit_per_domain: int | None = None,
    size: int = 256,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Path:
    real_paths = list_images(real_src)
    ai_paths = list_images(ai_src)
    if not real_paths or not ai_paths:
        raise ValueError(
            f"Both domains need images. Found real={len(real_paths)}, ai={len(ai_paths)}."
        )

    balanced_count = min(len(real_paths), len(ai_paths))
    if limit_per_domain is not None:
        balanced_count = min(balanced_count, limit_per_domain)
    if balanced_count < 3:
        raise ValueError("At least three images per domain are required after balancing.")

    rng = random.Random(seed)
    rng.shuffle(real_paths)
    rng.shuffle(ai_paths)
    selected = {
        "real": sorted(real_paths[:balanced_count]),
        "ai": sorted(ai_paths[:balanced_count]),
    }

    output = Path(output)
    dataset_root = output / name
    records: list[dict[str, object]] = []

    for domain, paths in selected.items():
        splits = split_names(
            len(paths),
            seed=seed + (0 if domain == "real" else 1),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        source_root = Path(real_src if domain == "real" else ai_src)
        for path, split in zip(paths, splits):
            relative = path.relative_to(source_root).as_posix()
            image_id = f"{domain}_{stable_id(source, domain, relative)}"
            clean_path = dataset_root / domain / f"{image_id}.png"
            normalize_image(path, clean_path, size=size)
            records.append(
                {
                    "image_id": image_id,
                    "domain": domain,
                    "source": source,
                    "source_path": str(path.resolve()),
                    "source_relative_path": relative,
                    "clean_path": str(clean_path.resolve()),
                    "split": split,
                    "width": size,
                    "height": size,
                    "color_mode": "L",
                }
            )

    manifest_path = output / "manifests" / f"{name}_clean.jsonl"
    write_manifest(manifest_path, sorted(records, key=lambda record: str(record["image_id"])))
    print(f"Prepared {balanced_count} real and {balanced_count} AI covers.")
    print(f"Processed data: {dataset_root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    return manifest_path


def download_kaggle(dataset: str, output: str | Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not dataset or "/" not in dataset:
        raise ValueError("Kaggle dataset must be written as owner/dataset-name.")

    config_dir = Path(
        os.environ.get("KAGGLE_CONFIG_DIR", "data/.cache/kaggle")
    ).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["KAGGLE_CONFIG_DIR"] = str(config_dir)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(output),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Configure ~/.kaggle/kaggle.json or the "
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        ) from exc
    return output


def _genimage_index(output: Path):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown==5.2.0 to download GenImage.") from exc

    return gdown.download_folder(
        url=GENIMAGE_FOLDER_URL,
        output=str(output),
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )


def download_genimage(
    generator: str,
    output: str | Path,
    accept_large_download: bool,
) -> Path:
    if generator not in GENIMAGE_GENERATORS:
        raise ValueError(f"Unknown generator. Choose from {sorted(GENIMAGE_GENERATORS)}")
    if not accept_large_download:
        raise RuntimeError(
            "Official GenImage generator archives are split into many large files. "
            "Re-run with --accept-large-download after checking available disk space."
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = GENIMAGE_GENERATORS[generator].lower()
    files = _genimage_index(output)
    if not files:
        raise RuntimeError("Could not read the official GenImage Google Drive folder.")
    selected = [file for file in files if file.path.lower().startswith(prefix)]
    if not selected:
        raise RuntimeError(f"No official GenImage files found for {generator}.")

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown==5.2.0 to download GenImage.") from exc

    generator_root = output / generator
    generator_root.mkdir(parents=True, exist_ok=True)
    for file in selected:
        destination = generator_root / Path(file.path).name
        if destination.exists():
            continue
        print(f"Downloading {file.path}")
        result = gdown.download(id=file.id, output=str(destination), quiet=False)
        if result is None:
            raise RuntimeError(f"Failed to download {file.path}")

    _extract_split_zip(generator_root)
    return generator_root


def _extract_split_zip(root: Path) -> None:
    zip_paths = sorted(root.glob("*.zip"))
    if not zip_paths:
        return
    archive = zip_paths[0]
    combined = root / f"{archive.stem}_combined.zip"
    try:
        subprocess.run(
            ["zip", str(archive), "-s", "0", "--out", str(combined)],
            check=True,
        )
        with zipfile.ZipFile(combined) as handle:
            handle.extractall(root / "extracted")
    except (subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            "GenImage uses split ZIP archives. Ensure every .z01/.z02/... part and "
            "the final .zip file downloaded successfully. The system `zip` command "
            "must support conversion of split archives."
        ) from exc


def find_domain_dirs(
    root: str | Path,
    preferred_split: str | None = None,
) -> tuple[Path, Path]:
    root = Path(root)
    search_roots = [root]
    if preferred_split:
        split_roots = [
            path
            for path in root.rglob("*")
            if path.is_dir() and path.name.lower() == preferred_split.lower()
        ]
        if split_roots:
            search_roots = split_roots
    real_candidates = [
        path
        for search_root in search_roots
        for path in search_root.rglob("*")
        if path.is_dir() and path.name.lower() in {"real", "nature"}
    ]
    ai_candidates = [
        path
        for search_root in search_roots
        for path in search_root.rglob("*")
        if path.is_dir() and path.name.lower() in {"ai", "fake"}
    ]
    if not real_candidates or not ai_candidates:
        raise ValueError(
            "Could not detect real/nature and ai/fake directories. Pass --real-src and --ai-src explicitly."
        )
    return real_candidates[0], ai_candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.datasets",
        description="Download and normalize image domains.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download a public dataset.")
    download.add_argument("--source", choices=["kaggle", "genimage"], required=True)
    download.add_argument("--dataset", help="Kaggle owner/dataset-name.")
    download.add_argument(
        "--generator",
        choices=sorted(GENIMAGE_GENERATORS),
        default="stable-diffusion-v1-4",
    )
    download.add_argument("--split", choices=["train", "val"], default="val")
    download.add_argument("--limit-per-domain", type=int, default=1000)
    download.add_argument("--output", default="data/raw")
    download.add_argument("--accept-large-download", action="store_true")

    add_acquisition_arguments(subparsers)
    add_audit_arguments(subparsers)
    add_recompress_arguments(subparsers)
    add_rebalance_arguments(subparsers)

    prepare = subparsers.add_parser("prepare", help="Normalize two local domain folders.")
    prepare.add_argument("--real-src", required=True)
    prepare.add_argument("--ai-src", required=True)
    prepare.add_argument("--name", default="kaggle")
    prepare.add_argument("--source", default="local")
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--limit-per-domain", type=int, default=None)
    prepare.add_argument("--size", type=int, default=256)
    prepare.add_argument("--seed", type=int, default=42)

    index = subparsers.add_parser(
        "index",
        help="Index native image candidates in a directory or archive.",
    )
    index.add_argument("--input", required=True)
    index.add_argument("--dataset-id", required=True)
    index.add_argument("--dataset-version", required=True)
    index.add_argument("--domain", choices=["real", "ai"], required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--class-map", default=None)
    index.add_argument("--class-map-member-column", default="source_member")
    index.add_argument("--class-map-class-column", default="class_id")
    index.add_argument("--no-parent-class", action="store_true")
    index.add_argument("--allow-unclassified", action="store_true")

    select = subparsers.add_parser(
        "select",
        help="Create a deterministic class-matched selection manifest.",
    )
    select.add_argument("--real-index", required=True)
    select.add_argument("--ai-index", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--count-per-domain", type=int, default=6000)
    select.add_argument("--audit-count", type=int, default=100)
    select.add_argument("--pilot-total", type=int, default=500)
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--train-ratio", type=float, default=2 / 3)
    select.add_argument("--val-ratio", type=float, default=1 / 6)

    extract = subparsers.add_parser(
        "extract",
        help="Incrementally materialize audit, pilot, or full selections.",
    )
    extract.add_argument("--selection-manifest", required=True)
    extract.add_argument("--output", default="data/raw/selected")
    extract.add_argument("--stage", choices=["audit", "pilot", "full"], required=True)
    extract.add_argument("--extraction-manifest", default=None)
    extract.add_argument(
        "--materialization",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
    )

    mapping = subparsers.add_parser(
        "map-genimagepp",
        help="Map ImageNet and GenImage++ members to exact ImageNet synset IDs.",
    )
    mapping.add_argument("--real-src", required=True)
    mapping.add_argument("--ai-src", required=True)
    mapping.add_argument("--imagenet-labels", required=True)
    mapping.add_argument("--simple-labels", required=True)
    mapping.add_argument("--class-index", required=True)
    mapping.add_argument("--output", required=True)

    prepare_selection = subparsers.add_parser(
        "prepare-selection",
        help="Normalize an extracted, split-locked selection.",
    )
    prepare_selection.add_argument("--extraction-manifest", required=True)
    prepare_selection.add_argument("--name", required=True)
    prepare_selection.add_argument("--output", required=True)
    prepare_selection.add_argument("--size", type=int, default=256)
    prepare_selection.add_argument(
        "--preprocess",
        choices=["fit", "crop"],
        default="fit",
        help="fit crops then resamples; crop takes a native centre crop only.",
    )
    prepare_selection.add_argument(
        "--skip-undersized",
        action="store_true",
        help="With --preprocess crop, skip images smaller than the crop size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "rebalance-cohort":
        run_rebalance(args)
        return

    if args.command == "recompress-cohort":
        run_recompress(args)
        return

    if args.command == "audit-domains":
        run_audit(args)
        return

    if args.command in {"fetch-archive", "fetch-images", "fetch-labels"}:
        run_acquisition(args)
        return

    if args.command == "map-genimagepp":
        real_map, ai_map = build_genimagepp_class_maps(
            real_source=args.real_src,
            ai_source=args.ai_src,
            imagenet_labels=args.imagenet_labels,
            simple_labels=args.simple_labels,
            class_index=args.class_index,
            output=args.output,
        )
        print(f"Real class map: {real_map.resolve()}")
        print(f"AI class map: {ai_map.resolve()}")
        return

    if args.command == "prepare-selection":
        from src.datasets.preparation import prepare_extracted_selection

        path = prepare_extracted_selection(
            extraction_manifest=args.extraction_manifest,
            name=args.name,
            output=args.output,
            size=args.size,
            mode=args.preprocess,
            skip_undersized=args.skip_undersized,
        )
        print(f"Manifest: {path.resolve()}")
        return

    if args.command == "index":
        path = build_dataset_index(
            source=args.input,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            domain=args.domain,
            output=args.output,
            class_map_path=args.class_map,
            class_map_member_column=args.class_map_member_column,
            class_map_class_column=args.class_map_class_column,
            infer_class_from_parent=not args.no_parent_class,
            allow_unclassified=args.allow_unclassified,
        )
        print(f"Dataset index: {path.resolve()}")
        return

    if args.command == "select":
        path = create_selection_manifest(
            real_index=args.real_index,
            ai_index=args.ai_index,
            output=args.output,
            count_per_domain=args.count_per_domain,
            audit_count=args.audit_count,
            pilot_total=args.pilot_total,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        print(f"Selection manifest: {path.resolve()}")
        return

    if args.command == "extract":
        result = extract_selection_stage(
            selection_manifest=args.selection_manifest,
            output=args.output,
            stage=args.stage,
            extraction_manifest=args.extraction_manifest,
            materialization=args.materialization,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "prepare":
        prepare_domains(
            real_src=args.real_src,
            ai_src=args.ai_src,
            output=args.output,
            name=args.name,
            source=args.source,
            limit_per_domain=args.limit_per_domain,
            size=args.size,
            seed=args.seed,
        )
        return

    raw_root = Path(args.output) / args.source
    if args.source == "kaggle":
        if not args.dataset:
            raise ValueError("--dataset owner/dataset-name is required for Kaggle.")
        downloaded = download_kaggle(args.dataset, raw_root)
    else:
        downloaded = download_genimage(
            generator=args.generator,
            output=raw_root,
            accept_large_download=args.accept_large_download,
        )

    try:
        real_src, ai_src = find_domain_dirs(
            downloaded,
            preferred_split=args.split if args.source == "genimage" else None,
        )
    except ValueError:
        print(f"Downloaded to {downloaded.resolve()}.")
        print("Run `python -m src.datasets prepare` with explicit domain folders.")
        return

    name = "kaggle" if args.source == "kaggle" else f"genimage_{args.generator}"
    prepare_domains(
        real_src=real_src,
        ai_src=ai_src,
        output="data/processed",
        name=name,
        source=args.source,
        limit_per_domain=args.limit_per_domain,
        size=256,
        seed=42,
    )


if __name__ == "__main__":
    main()
