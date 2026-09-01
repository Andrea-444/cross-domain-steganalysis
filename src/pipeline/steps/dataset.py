from __future__ import annotations

from pathlib import Path

from src.datasets.manifest import read_manifest, validate_clean_manifest
from src.datasets.pipeline import prepare_domains
from src.pipeline.context import PipelineContext
from src.pipeline.steps.base import StageResult


class DatasetStage:
    """Resolve or prepare the validated clean-cover manifest."""

    name = "dataset"

    def execute(self, context: PipelineContext) -> StageResult:
        configuration = context.configuration.dataset
        if configuration.clean_manifest:
            manifest = Path(configuration.clean_manifest)
            source = "existing"
        else:
            manifest = prepare_domains(
                real_src=str(configuration.real_src),
                ai_src=str(configuration.ai_src),
                output=configuration.output,
                name=configuration.name,
                source=configuration.source,
                size=configuration.size,
                seed=configuration.seed,
                limit_per_domain=configuration.limit_per_domain,
            )
            source = "prepared"

        records = read_manifest(manifest)
        validate_clean_manifest(records)
        context.clean_manifest = manifest.resolve()
        return StageResult(
            outputs={
                "clean_manifest": str(context.clean_manifest),
                "record_count": len(records),
                "source": source,
            }
        )
