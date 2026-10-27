from __future__ import annotations

from src.pipeline.context import PipelineContext
from src.pipeline.steps.base import StageResult
from src.steganalysis.experiments import run_srnet_protocol


class DetectorStage:
    """Train and evaluate the configured detector protocol."""

    name = "detector"

    def execute(self, context: PipelineContext) -> StageResult:
        configuration = context.configuration.detector
        if not configuration.enabled:
            return StageResult(outputs={"status": "skipped"})
        if context.stego_manifest is None:
            raise RuntimeError("Detector execution requires a stego manifest.")

        result_directory = run_srnet_protocol(
            manifest=context.stego_manifest,
            seeds=configuration.seeds,
            epochs=configuration.epochs,
            batch_size=configuration.batch_size,
            learning_rate=configuration.learning_rate,
            output=configuration.output,
            reference_root=configuration.reference_root,
            device=configuration.device,
        )
        context.detector_result_directory = result_directory.resolve()
        return StageResult(
            outputs={
                "detector": configuration.detector,
                "result_directory": str(context.detector_result_directory),
            }
        )
