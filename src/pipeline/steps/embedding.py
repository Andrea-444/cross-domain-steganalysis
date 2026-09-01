from __future__ import annotations

from pathlib import Path

from src.datasets.manifest import read_manifest, validate_stego_manifest
from src.pipeline.context import PipelineContext
from src.pipeline.steps.base import StageResult
from src.steganography.core import benchmark_manifest


class EmbeddingStage:
    """Resolve or generate the pair-safe steganographic manifest."""

    name = "embedding"

    def execute(self, context: PipelineContext) -> StageResult:
        configuration = context.configuration.steganography
        if configuration.stego_manifest:
            manifest = Path(configuration.stego_manifest)
            source = "existing"
        else:
            if context.clean_manifest is None:
                raise RuntimeError("Embedding requires a clean manifest.")
            manifest = benchmark_manifest(
                manifest=context.clean_manifest,
                algorithm=configuration.algorithm,
                payload_bpp=configuration.payload_bpp,
                output=configuration.output,
                seed=configuration.seed,
                backend=configuration.backend,
                cache_dir=configuration.cache_dir,
                limit_per_domain=configuration.limit_per_domain,
            )
            source = "generated"

        records = read_manifest(manifest)
        validate_stego_manifest(records)
        context.stego_manifest = manifest.resolve()
        return StageResult(
            outputs={
                "stego_manifest": str(context.stego_manifest),
                "record_count": len(records),
                "source": source,
            }
        )
