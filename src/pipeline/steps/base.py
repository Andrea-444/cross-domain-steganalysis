from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.pipeline.context import PipelineContext


@dataclass(frozen=True)
class StageResult:
    outputs: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    name: str

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute one bounded pipeline stage."""
