from __future__ import annotations

from src.pipeline.context import PipelineContext
from src.pipeline.steps.base import StageResult
from src.tracking.wandb import log_pipeline_run


class TrackingStage:
    """Publish already-persisted local run metadata to optional W&B tracking."""

    name = "tracking"

    def execute(self, context: PipelineContext) -> StageResult:
        configuration = context.configuration.tracking
        if not configuration.enabled or configuration.mode == "disabled":
            return StageResult(outputs={"status": "disabled"})
        try:
            outcome = log_pipeline_run(
                run_directory=context.run_directory,
                detector_result_directory=context.detector_result_directory,
                project=configuration.project,
                entity=configuration.entity,
                mode=configuration.mode,
                upload_pipeline_artifact=configuration.upload_pipeline_artifact,
            )
        except Exception as error:
            outcome = {"status": "failed", "error": str(error)}
        return StageResult(outputs=outcome)
