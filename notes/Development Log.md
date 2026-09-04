---
type: development-log
status: active
updated: 2026-09-04
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

## 2026-09-03 — Gitflow workflow skill added

- **Goal:** Record the repository's branching, verification, and artifact rules as an executable workflow instead of re-deriving them each session.
- **Work completed:** Added the `gitflow-feature-workflow` skill under `.claude/skills/` with feature/hotfix/release playbooks, a plan-handoff reference, a project-context reference for the `AGENTS.md` + `agent-instructions/` + `notes/` tiers, and a self-check script. Initialized the `dev` branch from `main` and published it.
- **Commands or experiment configuration:** `sh .claude/skills/gitflow-feature-workflow/scripts/check_skill.sh`; full unit suite run bare to check its exit code.
- **Result:** All 21 tests passed and the skill self-check passed. Branch prefixes are `feat`, `bug`, `spec`, and `junk`; `wip` was dropped because it labels a state rather than a kind of change and encourages long-lived divergence from `dev`.
- **Problems or surprises:** Two environment gotchas are now recorded in the skill: a bare `python` and `conda` are unavailable in non-interactive shells, so the environment interpreter must be called by absolute path, and `head` on this machine resolves to LWP's HTTP tool rather than GNU `head`. Piping the test suite also hides its exit code.
- **Next step:** Use the skill for the detector observability work — preserved test probabilities, epoch history, checkpoints, per-seed outputs, and MPS device selection.
- **Related files/results:** `.claude/skills/gitflow-feature-workflow/`, `.gitignore`.

## 2026-09-04 — Detector run observability implemented

- **Goal:** Make one SRNet protocol run interpretable after the fact instead of discarding the evidence needed to analyse it.
- **Work completed:** Device selection now prefers CUDA, then MPS, then CPU. Training returns structured epoch history including validation accuracy and balanced accuracy, and writes the selected checkpoint with its selection metadata. The runner keeps per-image predictions and probabilities for all four directions, preserves per-seed metrics next to the aggregate summary, and records provenance: manifest SHA-256, dataset counts, embedding condition, detector commit, repository commit and dirty flag, device, platform, dependency versions, and timing. Verified that the reference model ends in `LogSoftmax`, so the retained score is a true `P(stego)`; ROC-AUC and PR-AUC are therefore reported.
- **Commands or experiment configuration:** Full suite run bare; plus a two-epoch integration run on the existing 20-record S-UNIWARD 0.2 bpp smoke manifest to exercise the unmocked path.
- **Result:** 36 tests passed, exit code 0. The smoke run selected `mps` automatically and produced the full `history/`, `checkpoints/`, and `predictions/` layout. Its metrics are meaningless (7 training images) and are development verification only, not evidence.
- **Problems or surprises:** `embedding_seed` is assigned per image, so the first provenance implementation recorded a list of every distinct seed. It now summarises a varying seed as a count; the run-level master seed remains in the stego run's own `config.json`. Measured MPS at roughly 19x CPU throughput for a batch-16 training step, which is what makes the full matrix feasible locally.
- **Next step:** Resume/idempotency for interrupted multi-seed runs, the protocol-level aggregator, and mirroring the new fields into W&B. Separately, the dataset archives still need acquiring before any real cohort exists.
- **Related files/results:** `src/steganalysis/srnet.py`, `src/steganalysis/experiments.py`, `src/evaluation/metrics.py`, `tests/steganalysis/test_observability.py`.

## 2026-09-04 — End-to-end pipeline verified at 100 covers per domain

- **Goal:** Exercise the composed pipeline on real images before any research cohort exists, to prove the stages connect and the new run outputs are produced.
- **Work completed:** Added `configs/pipeline.verify100.json` and ran `python -m src.pipeline` over the existing local Kaggle images at 100 covers per domain, S-UNIWARD 0.4 bpp, one seed, fifteen epochs.
- **Commands or experiment configuration:** `python -m src.pipeline --config configs/pipeline.verify100.json`.
- **Result:** All four stages completed in 1m40s, exit code 0. 200 covers normalized to 256x256 grayscale, split 70/15/15 per domain. Zero pair-split leaks. Embedding altered 4,422-6,281 pixels per image with every change exactly +/-1, as adaptive embedding requires. The detector auto-selected `mps` and produced the full `history/`, `checkpoints/`, and `predictions/` layout. All four directions scored 0.500 balanced accuracy with ROC-AUC 0.49-0.51.
- **Problems or surprises:** The chance-level result is expected and is not evidence: 70 training covers gives SRNet 140 training images, far too few to learn. More importantly, this Kaggle pair carries the exact confound the project exists to avoid — natural covers are JPEG at 99 distinct resolutions (mostly 1080x720), AI covers are lossless PNG at a uniform 512x512. The two domains therefore differ in compression history and in downsampling ratio (about 4.2x versus 2x) before any embedding happens. This data is suitable for pipeline verification only. The exact Kaggle `owner/dataset-name` is not recorded anywhere in the repository; only `source: local-kaggle` survives in the manifests.
- **Next step:** Record the Kaggle dataset identifier. Keep using this pair for pipeline work only, and acquire the GenImage++ FLUX and ImageNet archives for any run intended as evidence.
- **Related files/results:** `configs/pipeline.verify100.json`, `results/pipeline/verify100-suniward-p040_20260904_130956`, `results/srnet_20260904_131025`.
