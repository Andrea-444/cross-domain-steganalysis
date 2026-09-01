from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.configuration import PipelineConfiguration
from src.pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run the non-interactive cross-domain steganalysis pipeline.",
    )
    parser.add_argument("--config", required=True, help="Path to pipeline JSON configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration_path = Path(args.config)
    configuration = PipelineConfiguration.from_json(configuration_path)
    run_directory = PipelineRunner(configuration, configuration_path).run()
    print(f"Pipeline run: {run_directory.resolve()}")


if __name__ == "__main__":
    main()
