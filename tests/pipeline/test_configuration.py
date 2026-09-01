from __future__ import annotations

import unittest

from src.pipeline.configuration import PipelineConfiguration


def valid_configuration() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "test", "phase": "smoke", "output": "results/pipeline"},
        "dataset": {"clean_manifest": "clean.jsonl"},
        "steganography": {"stego_manifest": "stego.jsonl"},
        "detector": {"enabled": False},
        "tracking": {"enabled": False, "mode": "disabled"},
    }


class PipelineConfigurationTests(unittest.TestCase):
    def test_loads_typed_sections(self) -> None:
        configuration = PipelineConfiguration.from_mapping(valid_configuration())
        self.assertEqual(configuration.run.phase, "smoke")
        self.assertEqual(configuration.dataset.clean_manifest, "clean.jsonl")
        self.assertFalse(configuration.detector.enabled)

    def test_requires_dataset_source_or_manifest(self) -> None:
        raw = valid_configuration()
        raw["dataset"] = {}
        with self.assertRaisesRegex(ValueError, "clean_manifest"):
            PipelineConfiguration.from_mapping(raw)

    def test_rejects_unknown_research_backend(self) -> None:
        raw = valid_configuration()
        raw["steganography"] = {"backend": "legacy"}
        with self.assertRaisesRegex(ValueError, "conseal"):
            PipelineConfiguration.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
