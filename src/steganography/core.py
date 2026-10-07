from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.manifest import (
    deterministic_seed,
    read_manifest,
    validate_clean_manifest,
    write_manifest,
)
from src.logging_config import (
    ProgressReporter,
    configure_logging,
    get_logger,
    success_line,
)

LOGGER = get_logger(__name__)


CONSEAL_ALGORITHMS = {"wow", "suniward", "hill", "mipod"}
OPENSTEGO_RELEASE_API = "https://api.github.com/repos/syvaidya/openstego/releases/latest"


def _load_conseal():
    cache_root = Path(os.environ.get("STEGANALYSIS_CACHE", "data/.cache")).resolve()
    numba_cache = cache_root / "numba"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
            import conseal as cl
    except ImportError as exc:
        raise RuntimeError(
            "conseal is not installed. Run `pip install conseal==2024.12`."
        ) from exc
    return cl, cache_root


def _configure_mipod_cache(cl, cache_root: Path) -> None:
    definitions = cl.mipod._defs
    lookup_path = cache_root / "conseal" / "mipod_ixlnx3.npz"
    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    def load_lookup_table():
        if lookup_path.exists():
            return np.load(lookup_path)["ixlnx3"]
        LOGGER.info("Preparing MiPOD lookup table. This one-time step can take a few minutes.")
        values = definitions.prepare_lookup_table(max_val=1000, resolution=100)
        np.savez(lookup_path, ixlnx3=values)
        return values

    definitions.load_lookup_table = load_lookup_table


def embed_conseal(
    image: np.ndarray,
    algorithm: str,
    payload_bpp: float,
    seed: int,
) -> np.ndarray:
    if algorithm not in CONSEAL_ALGORITHMS:
        raise ValueError(f"Unknown algorithm. Choose from {sorted(CONSEAL_ALGORITHMS)}")
    if payload_bpp <= 0 or payload_bpp > 1:
        raise ValueError("payload_bpp must be in the interval (0, 1].")
    if image.ndim != 2:
        raise ValueError("conseal benchmark images must be grayscale.")

    cl, cache_root = _load_conseal()
    if algorithm == "mipod":
        _configure_mipod_cache(cl, cache_root)
    simulator = getattr(cl, algorithm).simulate_single_channel
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="optimization might not have converged")
        stego = simulator(x0=image, alpha=payload_bpp, seed=seed)
    return np.asarray(stego, dtype=np.uint8)


def embed_legacy(
    cover: np.ndarray,
    secret: np.ndarray,
    payload_bits: int,
) -> np.ndarray:
    from src.legacy.stego_lsb import hide_array

    return hide_array(cover, secret, payload_bits=payload_bits, method="lsb")


def timestamped_run_dir(output: str | Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output) / f"{label}_{timestamp}"
    suffix = 1
    while path.exists():
        path = Path(output) / f"{label}_{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def cohort_identifier(manifest: str | Path, length: int = 12) -> str:
    """Short content hash of a clean manifest.

    Image IDs are stable across cohorts, so two cohorts prepared differently
    from the same sources share them while holding different pixels. Keying the
    stego cache on image_id alone would let one cohort silently reuse the
    other's stego images, pairing a stego image with a clean cover it was not
    derived from. The cohort identifier keeps their caches apart.
    """
    digest = hashlib.sha256()
    with open(manifest, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def embedding_ledger(
    runs_directory: str | Path, cohort_id: str | None = None
) -> list[dict[str, object]]:
    """Report which embedding conditions have already been run.

    Reads the configuration each run writes, so the history is derived from the
    runs themselves rather than from separate bookkeeping that could drift.
    """
    runs_directory = Path(runs_directory)
    if not runs_directory.is_dir():
        return []
    entries: list[dict[str, object]] = []
    for config_path in sorted(runs_directory.glob("*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if cohort_id is not None and config.get("cohort_id") != cohort_id:
            continue
        entries.append({
            "run_dir": str(config_path.parent.resolve()),
            "cohort_id": config.get("cohort_id"),
            "backend": config.get("backend"),
            "algorithm": config.get("algorithm"),
            "payload_bpp": config.get("payload_bpp"),
            "seed": config.get("seed"),
            "record_count": config.get("record_count"),
            "generated_count": config.get("generated_count"),
            "reused_count": config.get("reused_count"),
            "input_manifest": config.get("input_manifest"),
        })
    return entries


def benchmark_manifest(
    manifest: str | Path,
    algorithm: str,
    payload_bpp: float,
    output: str | Path,
    seed: int,
    backend: str = "conseal",
    legacy_payload_bits: int = 1,
    cache_dir: str | Path | None = None,
    limit_per_domain: int | None = None,
) -> Path:
    records = read_manifest(manifest)
    validate_clean_manifest(records)
    if limit_per_domain is not None:
        if limit_per_domain <= 0:
            raise ValueError("limit_per_domain must be greater than zero.")
        selected: list[dict[str, object]] = []
        for domain in ("real", "ai"):
            domain_records = sorted(
                (record for record in records if record["domain"] == domain),
                key=lambda record: str(record["image_id"]),
            )
            selected.extend(domain_records[:limit_per_domain])
        records = selected
    payload_label = f"p{int(round(payload_bpp * 100)):03d}"
    label = f"{backend}_{algorithm}_{payload_label}"
    run_dir = timestamped_run_dir(output, label)
    cohort_id = cohort_identifier(manifest)
    cache_root = (
        Path(cache_dir) / label / f"cohort_{cohort_id}" / f"seed_{seed}"
        if cache_dir
        else None
    )
    stego_records: list[dict[str, object]] = []
    generated_count = 0
    reused_count = 0
    LOGGER.info(
        "Embedding %s at %s bpp, seed %s, over %d covers.",
        algorithm, payload_bpp, seed, len(records),
    )
    progress = ProgressReporter(LOGGER, total=len(records), label="embed")

    for index, record in enumerate(records):
        clean_path = Path(str(record["clean_path"]))
        embedding_seed = deterministic_seed(
            seed,
            str(record["image_id"]),
            algorithm,
            str(payload_bpp),
        )

        if cache_root:
            destination = (
                cache_root
                / str(record["domain"])
                / "stego"
                / f"{record['image_id']}.png"
            )
        else:
            destination = (
                run_dir
                / str(record["domain"])
                / "stego"
                / f"{record['image_id']}.png"
            )

        if destination.exists():
            reused_count += 1
        else:
            with Image.open(clean_path) as image:
                cover = np.array(image.convert("L"), dtype=np.uint8)

            if backend == "conseal":
                stego = embed_conseal(
                    cover,
                    algorithm=algorithm,
                    payload_bpp=payload_bpp,
                    seed=embedding_seed,
                )
            elif backend == "legacy":
                next_path = Path(str(records[(index + 1) % len(records)]["clean_path"]))
                with Image.open(next_path) as image:
                    secret = np.array(image.convert("L"), dtype=np.uint8)
                stego = embed_legacy(cover, secret, payload_bits=legacy_payload_bits)
            else:
                raise ValueError("backend must be 'conseal' or 'legacy'.")

            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(stego, mode="L").save(destination, format="PNG")
            generated_count += 1

        progress.advance()
        stego_records.append(
            {
                **record,
                "stego_path": str(destination.resolve()),
                "algorithm": algorithm if backend == "conseal" else "legacy-lsb",
                "backend": backend,
                "payload_bpp": payload_bpp,
                "embedding_seed": embedding_seed,
            }
        )

    progress.finish()
    output_manifest = write_manifest(run_dir / "dataset_manifest.jsonl", stego_records)
    config = {
        "backend": backend,
        "algorithm": algorithm,
        "payload_bpp": payload_bpp,
        "seed": seed,
        "input_manifest": str(Path(manifest).resolve()),
        "input_manifest_sha256": cohort_identifier(manifest, length=64),
        "cohort_id": cohort_id,
        "limit_per_domain": limit_per_domain,
        "record_count": len(records),
        "generated_count": generated_count,
        "reused_count": reused_count,
        "cache_dir": str(cache_root.resolve()) if cache_root else None,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        success_line(
            f"Generated {generated_count} stego images. "
            f"Reused {reused_count} cached stego images."
        )
    )
    LOGGER.info("Run folder: %s", run_dir.resolve())
    LOGGER.info("Manifest: %s", output_manifest.resolve())
    return output_manifest


def resolve_openstego_jar(explicit_path: str | Path | None = None) -> Path:
    candidates = [
        Path(explicit_path) if explicit_path else None,
        Path(os.environ["OPENSTEGO_JAR"]) if os.environ.get("OPENSTEGO_JAR") else None,
        Path("external/openstego/openstego.jar"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        "OpenStego JAR not found. Run `python -m src.steganography setup-openstego` "
        "or set OPENSTEGO_JAR."
    )


def setup_openstego(output: str | Path = "external/openstego") -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        OPENSTEGO_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "steganalysis-research"},
    )
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    assets = release.get("assets", [])
    archives = [
        asset for asset in assets if str(asset.get("name", "")).lower().endswith(".zip")
    ]
    if not archives:
        raise RuntimeError("The latest OpenStego release does not contain a ZIP asset.")

    archive_path = output / str(archives[0]["name"])
    urllib.request.urlretrieve(str(archives[0]["browser_download_url"]), archive_path)
    extract_dir = output / "release"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)
    jars = list(extract_dir.rglob("openstego*.jar"))
    if not jars:
        raise RuntimeError("Could not find openstego.jar in the downloaded release.")
    destination = output / "openstego.jar"
    shutil.copy2(jars[0], destination)
    (output / "VERSION").write_text(str(release.get("tag_name", "unknown")), encoding="utf-8")
    LOGGER.info(f"OpenStego installed at: {destination.resolve()}")
    return destination


def validate_payload_file(payload_type: str, payload_file: Path) -> None:
    if not payload_file.exists():
        raise ValueError(f"Payload file does not exist: {payload_file}")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if payload_type == "image" and payload_file.suffix.lower() not in image_extensions:
        raise ValueError("--payload-type image requires an image payload file.")


def estimated_openstego_capacity(cover: Path, max_bits: int) -> int:
    with Image.open(cover) as image:
        channels = 4 if image.mode == "RGBA" else 3
        raw_bytes = image.width * image.height * channels * max_bits // 8
    return max(0, raw_bytes - 4096)


def openstego_java_prefix(jar: str | Path | None) -> list[str]:
    home = Path(os.environ.get("STEGANALYSIS_CACHE", "data/.cache")) / "openstego-home"
    home.mkdir(parents=True, exist_ok=True)
    return [
        "java",
        f"-Duser.home={home.resolve()}",
        "-jar",
        str(resolve_openstego_jar(jar)),
    ]


def openstego_embed(
    payload_type: str,
    payload_file: str | Path,
    cover: str | Path,
    output: str | Path,
    jar: str | Path | None,
    max_bits: int,
    password_env: str,
) -> Path:
    payload_file = Path(payload_file)
    cover = Path(cover)
    output = Path(output)
    validate_payload_file(payload_type, payload_file)
    if not cover.exists():
        raise ValueError(f"Cover file does not exist: {cover}")
    capacity = estimated_openstego_capacity(cover, max_bits=max_bits)
    if payload_file.stat().st_size > capacity:
        raise ValueError(
            f"Payload is approximately too large ({payload_file.stat().st_size} bytes) "
            f"for the estimated capacity ({capacity} bytes)."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = openstego_java_prefix(jar) + [
        "embed",
        "-a",
        "randomlsb",
        "-mf",
        str(payload_file),
        "-cf",
        str(cover),
        "-sf",
        str(output),
        "-b",
        str(max_bits),
    ]
    password = os.environ.get(password_env)
    if password:
        command.extend(["-e", "-p", password])
    subprocess.run(command, check=True)
    if not output.exists():
        raise RuntimeError("OpenStego did not create the requested stego file.")
    LOGGER.info(f"Recoverable stego image: {output.resolve()}")
    return output


def openstego_extract(
    stego: str | Path,
    output_dir: str | Path,
    jar: str | Path | None,
    password_env: str,
) -> Path:
    stego = Path(stego)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in output_dir.iterdir()}
    command = openstego_java_prefix(jar) + [
        "extract",
        "-a",
        "randomlsb",
        "-sf",
        str(stego),
        "-xd",
        str(output_dir),
    ]
    password = os.environ.get(password_env)
    if password:
        command.extend(["-p", password])
    subprocess.run(command, check=True)
    after = {path.resolve() for path in output_dir.iterdir()}
    if not after - before:
        raise RuntimeError("OpenStego did not extract a payload file.")
    LOGGER.info(f"Extracted payload directory: {output_dir.resolve()}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.steganography",
        description="Established steganography adapters.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Generate research stego images.")
    benchmark.add_argument("--manifest", required=True)
    benchmark.add_argument("--algorithm", choices=sorted(CONSEAL_ALGORITHMS), default="wow")
    benchmark.add_argument("--payload-bpp", type=float, choices=[0.1, 0.2, 0.4], default=0.2)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--output", default="stego_runs")
    benchmark.add_argument("--backend", choices=["conseal", "legacy"], default="conseal")
    benchmark.add_argument("--legacy-payload-bits", type=int, choices=[1, 2, 3, 4], default=1)
    benchmark.add_argument(
        "--limit-per-domain",
        type=int,
        default=None,
        help="Process at most this many real and this many AI covers.",
    )
    benchmark.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Optional stable cache for stego PNGs. Existing image_ids for the same "
            "backend/algorithm/payload/seed are reused instead of regenerated."
        ),
    )

    ledger = subparsers.add_parser(
        "ledger",
        help="List embedding conditions already run, so work is not repeated.",
    )
    ledger.add_argument("--runs", default="data/stego/runs")
    ledger.add_argument(
        "--manifest",
        default=None,
        help="Restrict to runs for this clean manifest's cohort.",
    )

    setup = subparsers.add_parser("setup-openstego")
    setup.add_argument("--output", default="external/openstego")

    embed = subparsers.add_parser("embed", help="Embed a recoverable file with OpenStego.")
    embed.add_argument("--payload-type", choices=["message", "image"], required=True)
    embed.add_argument("--payload-file", required=True)
    embed.add_argument("--cover", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--jar", default=None)
    embed.add_argument("--max-bits", type=int, choices=[1, 2, 3, 4], default=1)
    embed.add_argument("--password-env", default="OPENSTEGO_PASSWORD")

    extract = subparsers.add_parser("extract", help="Extract a recoverable OpenStego payload.")
    extract.add_argument("--stego", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--jar", default=None)
    extract.add_argument("--password-env", default="OPENSTEGO_PASSWORD")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.command == "benchmark":
        benchmark_manifest(
            manifest=args.manifest,
            algorithm=args.algorithm,
            payload_bpp=args.payload_bpp,
            output=args.output,
            seed=args.seed,
            backend=args.backend,
            legacy_payload_bits=args.legacy_payload_bits,
            cache_dir=args.cache_dir,
            limit_per_domain=args.limit_per_domain,
        )
    elif args.command == "ledger":
        cohort = cohort_identifier(args.manifest) if args.manifest else None
        entries = embedding_ledger(args.runs, cohort_id=cohort)
        if cohort:
            LOGGER.info(f"Cohort {cohort} ({Path(args.manifest).name})")
        if not entries:
            LOGGER.info("No embedding runs recorded.")
            return
        LOGGER.info(
            f"{'algorithm':<12}{'bpp':>6}{'seed':>6}{'covers':>8}"
            f"{'new':>6}{'reused':>8}  cohort       run"
        )
        for entry in entries:
            LOGGER.info(
                f"{str(entry['algorithm']):<12}"
                f"{str(entry['payload_bpp']):>6}"
                f"{str(entry['seed']):>6}"
                f"{str(entry['record_count']):>8}"
                f"{str(entry['generated_count']):>6}"
                f"{str(entry['reused_count']):>8}"
                f"  {str(entry['cohort_id'] or '-'):<12} "
                f"{Path(str(entry['run_dir'])).name}"
            )
        LOGGER.info(f"\n{len(entries)} run(s). Re-running a listed condition reuses its cache.")
        return
    elif args.command == "setup-openstego":
        setup_openstego(args.output)
    elif args.command == "embed":
        openstego_embed(
            payload_type=args.payload_type,
            payload_file=args.payload_file,
            cover=args.cover,
            output=args.output,
            jar=args.jar,
            max_bits=args.max_bits,
            password_env=args.password_env,
        )
    else:
        openstego_extract(
            stego=args.stego,
            output_dir=args.output_dir,
            jar=args.jar,
            password_env=args.password_env,
        )


if __name__ == "__main__":
    main()
