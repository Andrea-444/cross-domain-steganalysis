from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.logging_config import get_logger


LOGGER = get_logger(__name__)


class WandbExperimentTracker:
    """Optional live tracking for one complete SRNet protocol invocation."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        mode: str,
        name: str,
        config: dict[str, Any],
        tags: list[str] | None = None,
        upload_artifact: bool = False,
    ) -> None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking is enabled but wandb is not installed. "
                "Install it with `pip install wandb` or use --wandb-mode disabled."
            ) from error

        self._wandb = wandb
        self._upload_artifact = upload_artifact
        self._failed = False
        self.run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=name,
            config=config,
            tags=tags or None,
        )
        if self.run is None:
            raise RuntimeError("W&B did not create a run.")

    def _try(self, action: str, callback) -> bool:
        if self._failed:
            return False
        try:
            callback()
            return True
        except Exception as error:  # W&B must not invalidate canonical local results.
            self._failed = True
            LOGGER.warning("W&B %s failed; continuing with local results: %s", action, error)
            return False

    def log_epoch(
        self,
        seed: int,
        train_domain: str,
        epoch_record: dict[str, float],
    ) -> None:
        prefix = f"training/seed_{seed}/{train_domain}"

        def log() -> None:
            epoch_key = f"{prefix}/epoch"
            self.run.define_metric(epoch_key)
            self.run.define_metric(f"{prefix}/*", step_metric=epoch_key)
            self.run.log(
                {
                    epoch_key: int(epoch_record["epoch"]),
                    **{
                        f"{prefix}/{key}": float(value)
                        for key, value in epoch_record.items()
                        if key != "epoch"
                    },
                }
            )

        self._try("epoch logging", log)

    def log_evaluation(
        self,
        seed: int,
        direction: str,
        metrics: dict[str, Any],
    ) -> None:
        prefix = f"evaluation/seed_{seed}/{direction}"
        values = {
            f"{prefix}/{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
        matrix = metrics.get("confusion_matrix")
        if isinstance(matrix, list) and len(matrix) == 2:
            values.update(
                {
                    f"{prefix}/true_clean": float(matrix[0][0]),
                    f"{prefix}/false_stego": float(matrix[0][1]),
                    f"{prefix}/missed_stego": float(matrix[1][0]),
                    f"{prefix}/true_stego": float(matrix[1][1]),
                }
            )
        self._try("evaluation logging", lambda: self.run.log(values))

    def finish(self, result: dict[str, Any], run_directory: str | Path) -> dict[str, Any]:
        run_directory = Path(run_directory)

        def log_final() -> None:
            self.run.log(_flatten_summary(result.get("summary", {})))
            columns = [
                "seed", "direction", "accuracy", "balanced_accuracy", "precision",
                "recall", "f1", "roc_auc", "pr_auc", "true_clean",
                "false_stego", "missed_stego", "true_stego",
            ]
            table = self._wandb.Table(columns=columns)
            for seed_run in result.get("runs", []):
                for direction, metrics in seed_run.get("directions", {}).items():
                    matrix = metrics.get("confusion_matrix", [[None, None], [None, None]])
                    table.add_data(
                        seed_run.get("seed"), direction,
                        metrics.get("accuracy"), metrics.get("balanced_accuracy"),
                        metrics.get("precision"), metrics.get("recall"), metrics.get("f1"),
                        metrics.get("roc_auc"), metrics.get("pr_auc"),
                        matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1],
                    )
            self.run.log({"evaluation/results": table})
            if self._upload_artifact:
                artifact = self._wandb.Artifact(
                    name=f"{run_directory.name}-metadata",
                    type="steganalysis-run",
                )
                for filename in ("config.json", "dataset_manifest.jsonl", "summary.json"):
                    path = run_directory / filename
                    if path.exists():
                        artifact.add_file(str(path))
                self.run.log_artifact(artifact)

        self._try("final result logging", log_final)
        metadata = {
            "status": "failed" if self._failed else "logged",
            "run_id": getattr(self.run, "id", None),
            "run_url": getattr(self.run, "url", None),
        }
        try:
            self.run.finish()
        except Exception as error:
            metadata["status"] = "failed"
            LOGGER.warning("W&B run finalization failed; local results are intact: %s", error)
        return metadata


def start_experiment_run(
    *,
    mode: str,
    project: str,
    entity: str | None,
    name: str,
    config: dict[str, Any],
    tags: list[str] | None,
    upload_artifact: bool,
) -> WandbExperimentTracker | None:
    if mode == "disabled":
        return None
    return WandbExperimentTracker(
        project=project,
        entity=entity,
        mode=mode,
        name=name,
        config=config,
        tags=tags,
        upload_artifact=upload_artifact,
    )


def log_pipeline_run(
    run_directory: str | Path,
    detector_result_directory: str | Path | None,
    project: str,
    entity: str | None,
    mode: str,
    upload_pipeline_artifact: bool,
) -> dict[str, Any]:
    """Log canonical local run metadata to W&B as an optional final stage."""
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed. "
            "Install it with `pip install wandb` or disable tracking."
        ) from error

    run_directory = Path(run_directory)
    pipeline_config = _read_json(run_directory / "pipeline_config.json")
    pipeline_state = _read_json(run_directory / "pipeline_state.json")
    detector_summary = _detector_summary(detector_result_directory)
    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=str(pipeline_config.get("run", {}).get("name", run_directory.name)),
        config=pipeline_config,
    )
    if run is None:
        raise RuntimeError("W&B did not create a run.")

    run.log(
        {
            "pipeline/completed_stages": sum(
                stage.get("status") == "completed"
                for stage in pipeline_state.get("stages", [])
            ),
            **_flatten_summary(detector_summary),
        }
    )
    if upload_pipeline_artifact:
        artifact = wandb.Artifact(
            name=f"{run_directory.name}-metadata",
            type="pipeline-run",
        )
        artifact.add_file(str(run_directory / "pipeline_config.json"))
        artifact.add_file(str(run_directory / "pipeline_state.json"))
        if detector_result_directory is not None:
            summary_path = Path(detector_result_directory) / "summary.json"
            if summary_path.exists():
                artifact.add_file(str(summary_path))
        run.log_artifact(artifact)
    run_url = getattr(run, "url", None)
    run_id = getattr(run, "id", None)
    run.finish()
    return {"status": "logged", "run_id": run_id, "run_url": run_url}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detector_summary(directory: str | Path | None) -> dict[str, Any]:
    if directory is None:
        return {}
    path = Path(directory) / "summary.json"
    if not path.exists():
        return {}
    return _read_json(path).get("summary", {})


def _flatten_summary(summary: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for direction, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                flattened[f"metrics/{direction}/{metric}"] = float(value)
    return flattened
