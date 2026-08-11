from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ALETHEIA_REPOSITORY = "https://github.com/daniellerch/aletheia.git"
ALETHEIA_VERSION = "v0.3"


def setup_aletheia(output: str | Path = "external/aletheia") -> Path:
    output = Path(output)
    if (output / "aletheia.py").exists():
        print(f"Aletheia already exists at: {output.resolve()}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ALETHEIA_VERSION,
            ALETHEIA_REPOSITORY,
            str(output),
        ],
        check=True,
        env=environment,
    )
    print(f"Aletheia {ALETHEIA_VERSION} cloned to: {output.resolve()}")
    return output


def aletheia_auto(images: str | Path, root: str | Path = "external/aletheia") -> None:
    root = Path(root)
    entrypoint = root / "aletheia.py"
    if not entrypoint.exists():
        raise RuntimeError(
            "Aletheia is not installed. Run "
            "`python -m src.steganalysis.aletheia setup-aletheia`."
        )
    subprocess.run([sys.executable, str(entrypoint), "auto", str(images)], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.steganalysis.aletheia",
        description="External steganalysis tool adapters.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup-aletheia")
    setup.add_argument("--output", default="external/aletheia")
    auto = subparsers.add_parser("aletheia-auto")
    auto.add_argument("--images", required=True)
    auto.add_argument("--root", default="external/aletheia")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "setup-aletheia":
        setup_aletheia(args.output)
    else:
        aletheia_auto(args.images, args.root)
