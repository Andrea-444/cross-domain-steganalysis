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

## System flow

```text
Natural and AI image sources
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
Run configuration, metrics, and summaries
          |
          +------> Local JSON/JSONL artifacts (canonical)
          |
          +------> Optional W&B tracking and dashboards (proposed)
```

## Main components

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
- `src/steganalysis/experiments.py` — repeated-seed experiment protocol, metrics, and summaries.
- `sealwatch==2025.9` — planned SRM/SRMQ1 + FLD classical baseline, pending installation and compatibility validation.
- `src/steganalysis/aletheia.py` — optional secondary screening and analysis tools.

### Experiment observability — proposed

- Keep local manifests, configuration, predictions, metrics, and summaries as the portable record of each experiment.
- Add optional W&B logging for epoch histories, run comparison, result tables, confusion matrices, and artifacts.
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

## External dependencies

- `conseal==2024.12`
- Pinned SRNet reference implementation under `external/srnet-reference` at commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`
- Planned `sealwatch==2025.9` installation for the classical baseline
- Optional Aletheia installation
- Optional OpenStego installation for recoverable demonstrations

## Open architecture questions

- Where should large raw data, processed data, caches, models, and results live?
- Which generated artifacts are reproducible and which require backup?
- What is the canonical configuration format for final experiment sweeps?
- How will final tables and figures be generated from run summaries?
- Which artifacts may be uploaded to W&B, and what retention/storage policy applies?
- What is the exact mapping between a local run directory, a W&B run ID, and a protocol-level summary?
