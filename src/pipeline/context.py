from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.configuration import PipelineConfiguration


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageRecord:
    name: str
    status: str
    started_at: str
    finished_at: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineContext:
    configuration: PipelineConfiguration
    run_directory: Path
    configuration_source: Path
    clean_manifest: Path | None = None
    stego_manifest: Path | None = None
    detector_result_directory: Path | None = None
    stages: list[StageRecord] = field(default_factory=list)

    @property
    def state_path(self) -> Path:
        return self.run_directory / "pipeline_state.json"

    def persist(self) -> Path:
        payload = {
            "schema_version": self.configuration.schema_version,
            "run_directory": str(self.run_directory.resolve()),
            "configuration_source": str(self.configuration_source.resolve()),
            "clean_manifest": _resolved(self.clean_manifest),
            "stego_manifest": _resolved(self.stego_manifest),
            "detector_result_directory": _resolved(self.detector_result_directory),
            "stages": [asdict(stage) for stage in self.stages],
            "updated_at": utc_timestamp(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.state_path


def _resolved(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None
