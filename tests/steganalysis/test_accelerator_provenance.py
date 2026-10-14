from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.steganalysis.experiments import accelerator_state


NVRM_LINE = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.84  "
    "Release Build  (dvs-builder@U22-I3-AM25-26-2)  Wed Jun 10 21:06:37 UTC 2026"
)


class AcceleratorStateTests(unittest.TestCase):
    """The driver belongs in provenance: a library version does not move with it."""

    def test_no_cuda_leaves_the_gpu_fields_empty(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            state = accelerator_state()
        self.assertIsNone(state["device_name"])
        self.assertIsNone(state["nvml_available"])
        self.assertIn("mps_available", state)

    def test_kernel_driver_is_read_from_the_proc_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            version_file = Path(raw) / "version"
            version_file.write_text(NVRM_LINE + "\nGCC version: 13.3.0\n", encoding="utf-8")
            with patch("src.steganalysis.experiments.Path", side_effect=lambda p: (
                version_file if str(p) == "/proc/driver/nvidia/version" else Path(p)
            )), patch("torch.cuda.is_available", return_value=False):
                state = accelerator_state()
        self.assertIn("595.84", str(state["kernel_driver"]))
        self.assertNotIn("GCC", str(state["kernel_driver"]))

    def test_missing_proc_file_is_not_an_error(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            state = accelerator_state()
        self.assertIsNone(state["kernel_driver"])

    def test_absent_reporting_library_is_unknown_not_broken(self) -> None:
        """A machine without pynvml must not look like a machine with a fault."""
        import builtins

        real_import = builtins.__import__

        def refuse_pynvml(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("no pynvml")
            return real_import(name, *args, **kwargs)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 5080"
        ), patch.object(builtins, "__import__", refuse_pynvml):
            state = accelerator_state()
        self.assertEqual(state["device_name"], "NVIDIA GeForce RTX 5080")
        self.assertIsNone(state["nvml_available"])

    def test_broken_reporting_library_is_recorded_as_false(self) -> None:
        import builtins
        import types

        broken = types.ModuleType("pynvml")

        def explode() -> None:
            raise RuntimeError("Driver/library version mismatch")

        broken.nvmlInit = explode
        broken.nvmlShutdown = lambda: None
        real_import = builtins.__import__

        def give_broken(name, *args, **kwargs):
            if name == "pynvml":
                return broken
            return real_import(name, *args, **kwargs)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 5080"
        ), patch.object(builtins, "__import__", give_broken):
            state = accelerator_state()
        self.assertFalse(state["nvml_available"])


if __name__ == "__main__":
    unittest.main()
