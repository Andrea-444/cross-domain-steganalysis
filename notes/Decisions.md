---
type: decision-log
status: active
updated: 2026-09-01
---

# Decisions

Record decisions that would otherwise need to be rediscovered. Keep entries short and link supporting evidence when available.

## Decision template

### YYYY-MM-DD — Decision title

- **Status:** proposed / accepted / superseded
- **Context:** Why a decision was needed.
- **Decision:** What was chosen.
- **Reasoning:** Why this option was selected.
- **Consequences:** Trade-offs, follow-up work, or constraints.
- **Related:** Links to notes, code, experiments, or results.

---

## Current decisions inferred from the project

### 2026-08-20 — Treat steganalysis as the target task

- **Status:** accepted
- **Decision:** Predict clean versus steganographic images; do not frame the classifier as an AI-image detector.
- **Reasoning:** The research variable is the cover domain and its effect on hidden-payload detectability.

### 2026-08-20 — Use pair-safe dataset splits

- **Status:** accepted
- **Decision:** Clean/stego pairs inherit the same stable image ID and split.
- **Reasoning:** This prevents data leakage between training and testing.

### 2026-08-20 — Normalize both cover domains identically

- **Status:** accepted
- **Decision:** Balance both domains and convert images using the same crop, grayscale, resize, and PNG pipeline.
- **Reasoning:** This reduces format and resolution shortcuts that could confound the comparison.

### 2026-08-20 — Use established adaptive embedding methods

- **Status:** superseded on 2026-09-01
- **Decision:** Use `conseal` implementations of WOW, S-UNIWARD, HILL, and MiPOD for primary research experiments.
- **Consequences:** Custom LSB code remains educational/legacy rather than primary evidence.

### 2026-08-20 — Use pinned external SRNet as the provisional primary detector

- **Status:** accepted
- **Decision:** Use the external PyTorch SRNet implementation pinned at commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355` as the provisional primary steganalyzer. Load it through `src/steganalysis/srnet.py` rather than rewriting the architecture.
- **Reasoning:** An existing recognized implementation is more reproducible and reviewable than a manual reimplementation.
- **Consequences:** The current adapter is initial architecture, not final-ready. Validate the training recipe, convergence, checkpointing, metric history, per-image scores, W&B integration, and device behavior before freezing the protocol. Use `sealwatch==2025.9` for the planned SRM/SRMQ1 + FLD classical baseline after compatibility validation.
- **Related:** [[Experiment Decisions]], [[Architecture]].

### 2026-08-31 — Prefer a pre-generated AI-image dataset initially

- **Status:** accepted
- **Context:** The available local machine is an M4 Pro MacBook with 24 GB unified memory and approximately 200 GB free storage. Large controlled image-generation runs are not a practical default on this machine, while access to an external A100 or RTX 5080 is possible but not confirmed.
- **Decision:** Select an existing pre-generated AI-image dataset for the main initial protocol. Keep self-generation from matched prompts as an optional refinement if suitable external GPU access becomes available.
- **Reasoning:** This allows dataset selection, embedding design, detector selection, evaluation design, and pipeline development to proceed without depending on uncertain compute access.
- **Consequences:** The selected public dataset must be audited carefully for generator provenance, licensing, semantic mismatch, compression history, resizing, and other natural-versus-AI preprocessing confounders. The exact dataset remains undecided.
- **Related:** [[Pipeline Summary and Experiment Plan]], [[Tasks]].

### 2026-08-31 — Use GenImage++ and ImageNet as the working dataset pair

- **Status:** accepted
- **Decision:** Start with GenImage++ FLUX covers and class-matched ImageNet ILSVRC natural covers. Prefer 6,000 covers per domain, then adjust after a pilot.
- **Reasoning:** This provides modern pre-generated AI images and a shared ImageNet class vocabulary while fitting the current storage plan.
- **Consequences:** The exact FLUX subset and whether to use all 1,000 classes remain open. JPEG-versus-lossless history must pass a residual/compression audit before final experiments.
- **Related:** [[Experiment Decisions]].

### 2026-09-01 — Focus the embedding matrix

- **Status:** accepted
- **Decision:** Use S-UNIWARD at 0.2 and 0.4 bpp with three seeds as the primary matrix. Use MiPOD at 0.4 bpp with three seeds as the first robustness condition. Add HILL at 0.4 bpp only if compute permits.
- **Reasoning:** The main variable is cover domain. A focused matrix limits compute and avoids diluting the research question with a broad algorithm comparison.
- **Consequences:** Exclude LSB from the research matrix. Keep WOW only as a separately labelled optional cross-algorithm test. Keep S-UNIWARD at 0.1 bpp as a later optional difficult condition.
- **Related:** [[Experiment Decisions]], [[Pipeline Summary and Experiment Plan]].

### 2026-09-01 — Compose the workflow through a typed pipeline runner

- **Status:** accepted
- **Decision:** Use one non-interactive JSON configuration and an explicit pipeline runner for dataset preparation, embedding, detector execution, and optional tracking. Keep scientific logic in its domain modules, persist local configuration and stage state before W&B, and test both individual stages and composition.
- **Reasoning:** This makes complete runs reproducible without coupling domain algorithms to orchestration or making W&B the source of truth.
- **Consequences:** Existing focused CLIs remain supported. W&B is optional and non-fatal; local artifacts are canonical. Final sweep values are still provisional.
- **Related:** [[Architecture]], `src/pipeline/`, `configs/pipeline.smoke.example.json`.
