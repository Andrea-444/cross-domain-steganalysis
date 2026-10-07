from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.logging_config import get_logger
from src.pipeline.configuration import PipelineConfiguration
from src.pipeline.context import PipelineContext, StageRecord, utc_timestamp
from src.pipeline.steps import DatasetStage, DetectorStage, EmbeddingStage, TrackingStage
from src.pipeline.steps.base import PipelineStage

LOGGER = get_logger(__name__)


class PipelineRunner:
    """Compose pipeline stages while persisting state around every transition."""

    def __init__(
        self,
        configuration: PipelineConfiguration,
        configuration_source: str | Path,
        stages: Iterable[PipelineStage] | None = None,
    ):
        self.configuration = configuration
        self.configuration_source = Path(configuration_source)
        self.stages = list(
            stages
            if stages is not None
            else [DatasetStage(), EmbeddingStage(), DetectorStage(), TrackingStage()]
        )

    def run(self) -> Path:
        run_directory = create_pipeline_run_directory(
            self.configuration.run.output,
            self.configuration.run.name,
        )
        context = PipelineContext(
            configuration=self.configuration,
            run_directory=run_directory,
            configuration_source=self.configuration_source,
        )
        (run_directory / "pipeline_config.json").write_text(
            json.dumps(self.configuration.to_dict(), indent=2),
            encoding="utf-8",
        )
        context.persist()

        for stage in self.stages:
            record = StageRecord(
                name=stage.name,
                status="running",
                started_at=utc_timestamp(),
            )
            context.stages.append(record)
            context.persist()
            LOGGER.info("Stage %s: start.", stage.name)
            try:
                result = stage.execute(context)
            except Exception as error:
                record.status = "failed"
                record.finished_at = utc_timestamp()
                record.error = str(error)
                context.persist()
                LOGGER.error("Stage %s failed: %s", stage.name, error)
                raise
            LOGGER.info("Stage %s: completed.", stage.name)
            record.status = "completed"
            record.finished_at = utc_timestamp()
            record.outputs = result.outputs
            context.persist()

        return run_directory


def create_pipeline_run_directory(output: str | Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "experiment"
    base = Path(output)
    path = base / f"{safe_name}_{timestamp}"
    suffix = 1
    while path.exists():
        path = base / f"{safe_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path
