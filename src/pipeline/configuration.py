from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfiguration:
    name: str = "experiment"
    phase: str = "pilot"
    output: str = "results/pipeline"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("run.name must not be empty.")
        if self.phase not in {"smoke", "pilot", "final"}:
            raise ValueError("run.phase must be smoke, pilot, or final.")


@dataclass(frozen=True)
class DatasetConfiguration:
    clean_manifest: str | None = None
    real_src: str | None = None
    ai_src: str | None = None
    name: str = "covers"
    source: str = "local"
    output: str = "data/processed"
    size: int = 256
    seed: int = 42
    limit_per_domain: int | None = None

    def validate(self) -> None:
        if self.clean_manifest:
            return
        if not self.real_src or not self.ai_src:
            raise ValueError(
                "dataset requires clean_manifest or both real_src and ai_src."
            )
        if self.size <= 0:
            raise ValueError("dataset.size must be greater than zero.")
        if self.limit_per_domain is not None and self.limit_per_domain <= 0:
            raise ValueError("dataset.limit_per_domain must be greater than zero.")


@dataclass(frozen=True)
class SteganographyConfiguration:
    stego_manifest: str | None = None
    algorithm: str = "suniward"
    payload_bpp: float = 0.2
    seed: int = 42
    backend: str = "conseal"
    output: str = "stego_runs"
    cache_dir: str | None = None
    limit_per_domain: int | None = None

    def validate(self) -> None:
        if self.stego_manifest:
            return
        if self.algorithm not in {"suniward", "mipod", "hill", "wow"}:
            raise ValueError("Unsupported steganography.algorithm.")
        if self.backend != "conseal":
            raise ValueError("The research pipeline supports only the conseal backend.")
        if not 0 < self.payload_bpp <= 1:
            raise ValueError("steganography.payload_bpp must be in (0, 1].")
        if self.limit_per_domain is not None and self.limit_per_domain <= 0:
            raise ValueError("steganography.limit_per_domain must be greater than zero.")


@dataclass(frozen=True)
class DetectorConfiguration:
    enabled: bool = True
    detector: str = "srnet"
    seeds: list[int] = field(default_factory=lambda: [1])
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: str | None = None
    reference_root: str = "external/srnet-reference"
    output: str = "results"

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.detector != "srnet":
            raise ValueError("detector.detector must currently be srnet.")
        if not self.seeds:
            raise ValueError("detector.seeds must contain at least one seed.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("detector.epochs and batch_size must be greater than zero.")
        if self.learning_rate <= 0:
            raise ValueError("detector.learning_rate must be greater than zero.")


@dataclass(frozen=True)
class TrackingConfiguration:
    enabled: bool = False
    mode: str = "disabled"
    project: str = "cross-domain-steganalysis"
    entity: str | None = None
    upload_pipeline_artifact: bool = False

    def validate(self) -> None:
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("tracking.mode must be online, offline, or disabled.")


@dataclass(frozen=True)
class PipelineConfiguration:
    run: RunConfiguration
    dataset: DatasetConfiguration
    steganography: SteganographyConfiguration
    detector: DetectorConfiguration
    tracking: TrackingConfiguration
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PipelineConfiguration":
        if not isinstance(raw, dict):
            raise ValueError("Pipeline configuration must be a JSON object.")
        configuration = cls(
            schema_version=int(raw.get("schema_version", 1)),
            run=RunConfiguration(**_mapping(raw.get("run"), "run")),
            dataset=DatasetConfiguration(**_mapping(raw.get("dataset"), "dataset")),
            steganography=SteganographyConfiguration(
                **_mapping(raw.get("steganography"), "steganography")
            ),
            detector=DetectorConfiguration(**_mapping(raw.get("detector", {}), "detector")),
            tracking=TrackingConfiguration(**_mapping(raw.get("tracking", {}), "tracking")),
        )
        configuration.validate()
        return configuration

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfiguration":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Pipeline configuration does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid pipeline configuration JSON: {error}") from error
        return cls.from_mapping(raw)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported pipeline configuration schema_version.")
        self.run.validate()
        self.dataset.validate()
        self.steganography.validate()
        self.detector.validate()
        self.tracking.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value
