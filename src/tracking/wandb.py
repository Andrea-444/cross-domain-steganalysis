from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
