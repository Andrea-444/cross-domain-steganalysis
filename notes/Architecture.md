---
type: architecture
status: evolving
phase: initial-design
protocol-frozen: false
updated: 2026-09-01
---

# Architecture

> [!important] Initial architecture
> This describes the current working design, not a frozen final architecture. Dataset audits, dependency checks, and pilot experiments may lead to documented changes.

Editable diagram: [[Solution Architecture.drawio]]

## System flow

```text
Natural and AI image sources
          |
          v
Typed JSON run configuration
configs/
          |
          v
Pipeline runner and explicit stages
src/pipeline/
          |
          v
Dataset preparation and normalization
src/datasets/
          |
          v
Clean manifest with stable IDs and pair-safe splits
          |
          v
Steganographic embedding
src/steganography/
          |
          v
Stego manifest containing clean/stego pairs
          |
          v
Detector training and four-direction evaluation
src/steganalysis/
          |
          v
Metrics and repeated-seed summaries
src/evaluation/
          |
          +------> Local JSON/JSONL artifacts (canonical)
          |
          +------> Optional W&B tracking
                    src/tracking/
```

## Main components

### Pipeline orchestration

- `src/pipeline/configuration.py` — typed schema for a non-interactive JSON run configuration.
- `src/pipeline/context.py` — shared manifest/result paths and persisted stage records.
- `src/pipeline/runner.py` — composes stages and saves state before and after every transition.
- `src/pipeline/steps/` — explicit dataset, embedding, detector, and tracking boundaries.
- `configs/pipeline.smoke.example.json` — executable smoke-stage configuration template, not a frozen final protocol.
- Existing domain CLIs remain available for focused work; the runner calls their Python APIs.

### Dataset pipeline

- `src/datasets/pipeline.py` — acquisition, balancing, preprocessing, and normalization.
- `src/datasets/manifest.py` — stable identifiers, manifest I/O, and validation.

### Steganography

- `src/steganography/core.py` — `conseal` benchmark embedding and OpenStego demonstrations.
- Primary embedding conditions: S-UNIWARD at 0.2 and 0.4 bpp, with three seeds.
- First robustness condition: MiPOD at 0.4 bpp, with three seeds.
- Optional extensions: HILL at 0.4 bpp if compute permits; WOW only as a separately labelled cross-algorithm test; S-UNIWARD at 0.1 bpp as a later difficult condition.
- Legacy LSB implementations remain under `src/legacy/` and are not primary research methods.

### Steganalysis

- `src/steganalysis/srnet.py` — adapter for the external PyTorch SRNet implementation pinned at commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`; SRNet is not rewritten locally.
- `src/steganalysis/experiments.py` — repeated-seed detector training and four-direction evaluation.
- `src/evaluation/metrics.py` — classification metrics and repeated-seed summaries.
- `sealwatch==2025.9` — planned SRM/SRMQ1 + FLD classical baseline, pending installation and compatibility validation.
- `src/steganalysis/aletheia.py` — optional secondary screening and analysis tools.

### Experiment observability — initial implementation

- Local manifests, normalized configuration, stage state, metrics, and summaries remain the portable record of each experiment.
- `src/tracking/wandb.py` provides optional online/offline W&B logging after local artifacts exist.
- W&B failures are recorded as tracking failures and do not invalidate local scientific results.
- Epoch histories, result tables, confusion-matrix dashboards, and retention policy remain future tracking work.
- Define one tracked training run by embedding condition, training domain, and seed; attach evaluation on both test domains to that run.
- Aggregate the two training domains and repeated seeds into one protocol-level four-direction summary.
- Support disabled and offline tracking so tests and local development do not require credentials or network access.
- See [[Pipeline Summary and Experiment Plan]] for the proposed logging schema and dashboard.

## Data invariants

- Natural and AI domains must receive equivalent preprocessing.
- Clean/stego pairs must share the same `image_id` and split.
- A clean cover and its stego counterpart must never cross train/test boundaries.
- Final comparisons should change the intended experimental variable, not image format, size, or preprocessing.
- Runs should record their configuration, input manifest, seed, and outputs.
- Pipeline stages pass typed results through a shared context; orchestration does not own domain algorithms.

## External dependencies

- `conseal==2024.12`
- Pinned SRNet reference implementation under `external/srnet-reference` at commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`
- Planned `sealwatch==2025.9` installation for the classical baseline
- Optional Aletheia installation
- Optional OpenStego installation for recoverable demonstrations

## Open architecture questions

- Where should large raw data, processed data, caches, models, and results live?
- Which generated artifacts are reproducible and which require backup?
- Which configuration values will be frozen for final experiment sweeps?
- How will final tables and figures be generated from run summaries?
- Which artifacts may be uploaded to W&B, and what retention/storage policy applies?
