---
type: development-log
status: active
updated: 2026-09-01
---

# Development Log

Use this as a chronological lab and engineering notebook. Capture what changed, what was tested, what happened, and what should happen next.

## Entry template

### YYYY-MM-DD — Short title

- **Goal:**
- **Work completed:**
- **Commands or experiment configuration:**
- **Result:**
- **Problems or surprises:**
- **Next step:**
- **Related files/results:**

---

## 2026-08-20 — Project notes initialized

- **Goal:** Establish shared project context that is readable in Obsidian and available to Codex.
- **Work completed:** Added project overview, architecture, decision log, development log, ideas, and task notes.
- **Result:** The notes vault now has a starting structure for research and engineering context.
- **Next step:** Review the inferred project description and decisions, then correct or expand them.

## 2026-08-31 — Pipeline readiness and W&B plan documented

- **Goal:** Create a central summary of the current solution architecture and a concrete path to repeatable final experiments.
- **Work completed:** Added [[Pipeline Summary and Experiment Plan]], expanded [[Tasks]] with scientific-protocol, pipeline-hardening, W&B, preflight, and final-execution work, and updated [[Architecture]] and [[Project Overview]].
- **Commands or experiment configuration:** Ran the repository unit tests with the `experimental_1` environment; no training experiment was run.
- **Result:** All 8 unit tests passed. The documented code skeleton is operational under its current tests, while the final dataset, experiment matrix, statistical analysis, provenance capture, checkpoints, per-image outputs, and W&B integration remain open.
- **Problems or surprises:** The generic `python` command is unavailable in the current non-interactive shell; the project environment's Python executable was used. Current SRNet training prints epoch metrics but does not preserve structured history or a checkpoint, and the experiment runner discards returned test probabilities.
- **Next step:** Review the proposed protocol with the project team, record accepted choices in [[Decisions]], then implement local structured experiment outputs before adding W&B as an optional mirror.
- **Related files/results:** [[Pipeline Summary and Experiment Plan]], [[Tasks]], [[Architecture]].

## 2026-09-01 — Typed pipeline architecture implemented

- **Goal:** Connect the existing dataset, embedding, detector, evaluation, and tracking boundaries through one reproducible workflow.
- **Work completed:** Added typed JSON configuration, persisted pipeline context, explicit stages, one runner/CLI, shared evaluation metrics, optional artifact-first W&B logging, a smoke configuration, and focused stage/composition tests.
- **Commands or experiment configuration:** Validated `configs/pipeline.smoke.example.json`, checked the pipeline CLI, and ran the full unit-test suite. No research training run was executed.
- **Result:** All 15 tests passed. Existing domain CLIs remain available, while `python -m src.pipeline --config <file>` now provides composed non-interactive execution.
- **Problems or surprises:** W&B is intentionally optional and was not installed or contacted during verification.
- **Next step:** Review and adjust the smoke configuration paths, then run a small end-to-end smoke test only when the intended local inputs and compute environment are ready.
- **Related files/results:** [[Architecture]], [[Decisions]], `src/pipeline/`, `src/evaluation/`, `src/tracking/`.

## 2026-09-01 — Incremental dataset selection implemented

- **Goal:** Select the audit, pilot, and full image cohorts once so later stages reuse the same images without overlap or test-set leakage.
- **Work completed:** Added native-source indexing, deterministic class-matched selection manifests, locked pair-safe splits, cumulative cohort labels, checksum-verified incremental extraction, CLI commands, and synthetic tests for directories, ZIP, and TAR.ZST archives.
- **Commands or experiment configuration:** Ran `python -m unittest discover -s tests -v`; no dataset was downloaded or extracted.
- **Result:** All 21 tests passed. Audit uses 100 images per domain, pilot cumulatively uses 500, and the planned full cohort uses 6,000 by default. Final-test images are selected at the start but excluded from audit and pilot materialization.
- **Problems or surprises:** The actual GenImage++ archive layout and the ImageNet validation-label mapping still need verification against downloaded source files before creating the real indexes.
- **Next step:** Obtain the source archives and label metadata, build the indexes and selection manifest, then extract and inspect the 100-image-per-domain audit cohort.
- **Related files/results:** [[Architecture]], [[Decisions]], `src/datasets/index.py`, `src/datasets/selection.py`, `src/datasets/extractor.py`.
