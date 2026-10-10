---
type: decision-log
status: active
updated: 2026-10-10
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

### 2026-09-01 — Select once and materialize dataset cohorts incrementally

- **Status:** accepted
- **Decision:** Build immutable source indexes and one deterministic class-matched selection manifest before extraction. Materialize 100-image audit, 500-image cumulative pilot, and full cohorts without selecting or storing duplicate source images.
- **Reasoning:** Later stages can add only missing images while preserving reproducibility and preventing accidental selection overlap.
- **Consequences:** Final-test membership is fixed at selection time and excluded from audit/pilot work. Source archives are downloaded once; checksums guard the extracted corpus.
- **Related:** [[Architecture]], `src/datasets/index.py`, `src/datasets/selection.py`, `src/datasets/extractor.py`.

### 2026-10-07 — Normalize covers by native crop, without resampling

- **Status:** accepted
- **Context:** The ImageNet and GenImage++ sources differ in resolution: natural covers are mostly around 500x375 and AI covers are uniformly 1024x1024. `ImageOps.fit` crops to the target aspect ratio using the short side and then resamples, so reducing both to 256x256 applied a median factor of 1.46x to natural covers (mean 1.66x, range 0.66x to 7.59x, measured over 100 covers) against exactly 4x for AI covers. Resampling attenuates the high-frequency residuals steganalysis measures, so unequal factors attenuate the domains unequally. Worse, a factor below 1.0 means the cover was enlarged, and interpolation invents high-frequency content that was never captured; the factor also varied widely *within* the natural domain.
- **Decision:** Prepare covers with `--preprocess crop`, a centred native crop with no resampling. Exclude source images smaller than the crop size with `--skip-undersized`; excluded covers are recorded in the run configuration.
- **Reasoning:** Resampling demonstrably alters the residual statistics steganalysis measures, and it did so unequally here: natural covers were resampled by a median 1.46x with a range of 0.66x to 7.59x, against exactly 4x for AI covers. A factor below 1.0 enlarged the cover, so interpolation invented high-frequency content that was never captured. The decision rests on removing a known processing artifact, not on a measured separability difference. **The supporting measurement was superseded on 2026-10-07**: at 100 covers per domain the probe gave 0.745 for `fit` against 0.675 for `crop`, but at 5,830 covers per domain `crop` alone reaches 0.9954, so the probe saturates and can no longer discriminate between the two preprocessing paths.
- **Consequences:** Undersized covers must be filtered at selection time for larger cohorts, or covers will be lost unevenly across classes; one of 100 natural covers was dropped at 100-per-domain. This supersedes the provisional "prefer crop-only, confirm after checking source dimensions" note in [[Experiment Decisions]].
- **Related:** [[Development Log]] 2026-09-29, `src/datasets/preparation.py`, `results/audits/srnet-probe-fit/`, `results/audits/srnet-probe-crop/`.

### 2026-10-07 — Do not equalize compression history between domains

- **Status:** accepted
- **Context:** Natural covers come from JPEG sources (quality 96 by their quantization tables) and AI covers from lossless PNG. JPEG history cannot be removed from covers that already carry it, so the only available intervention is adding matching history to the AI covers.
- **Decision:** Do not apply JPEG recompression to either domain. Keep `recompress-cohort` available as a diagnostic.
- **Reasoning:** Adding JPEG at quality 96 to the AI covers moved SRNet domain separability from 0.675 to 0.661, and applying it to both domains moved it to 0.688. Both sit within about one standard deviation of the untreated crop cohort across three seeds, so the intervention changed nothing measurable while adding a processing stage that would itself need justifying.
- **Consequences:** The residual domain separability of roughly 0.675 survives compression equalization and is therefore most likely intrinsic to the generated images rather than an artifact of preparation. This is an inference from one failed intervention, not proof: semantic content, generation resolution, and tonal statistics under grayscale conversion remain untested as alternative explanations. **Whether a residual separability of about 0.675 is acceptable for the thesis remains open and is not settled by this decision.**
- **Related:** [[Development Log]] 2026-09-29, `src/datasets/recompress.py`, `results/audits/srnet-probe-smoke100-crop-jpegai96/`, `results/audits/srnet-probe-smoke100-crop-jpegboth96/`.

### 2026-10-07 — Gate cover pairs with the SRNet probe, not the feature screen

- **Status:** accepted
- **Context:** Two tools now test whether clean covers reveal their domain: a three-feature screen with a permutation test (`audit-domains`) and SRNet itself (`domain_probe`).
- **Decision:** A cohort is gated by `src.steganalysis.domain_probe`, always paired with a cohort the probe does detect as a positive control. `audit-domains` is a cheap screen only and must not be used to clear a cohort.
- **Reasoning:** The screen gave a false all-clear twice. It scored the crop cohort at 0.532 with permutation p = 0.2736, and the two recompressed variants at 0.502 and 0.497 with p of 0.54 and 0.60, all of which read as chance. SRNet separated the same covers at 0.675, 0.661 and 0.688. Hand-crafted features cannot bound what a deep network will find.
- **Consequences:** Gating costs GPU time rather than seconds. A near-chance probe result is only meaningful alongside a positive control, because a null may instead mean the probe lacked power at that cohort size.
- **Related:** [[Development Log]] 2026-09-29, `src/steganalysis/domain_probe.py`, `src/datasets/audit.py`.

### 2026-10-07 — Treat the cover-domain gap as near-maximal, not subtle

- **Status:** accepted
- **Context:** The SRNet domain probe was run on the full `imagenet-flux-6000-crop-balanced` cohort, 5,830 covers per domain, 3 seeds, 30 epochs, on CUDA.
- **Decision:** Record the cover-domain gap as near-maximal and report it as a measured property of the dataset pair. Do not treat small-scale separability figures as estimates of it.
- **Reasoning:** The probe reached balanced accuracy 0.9954 (sd 0.0007) and ROC-AUC 0.9997 (sd 0.0001), with 8 to 11 errors per seed out of 1,948 test covers. The 100-cover figure of 0.675 was a limit of 132 training covers, not a measurement of the gap. This is also consistent with the sources: GenImage++ is published as a benchmark **for** AI-image detection, so real against AI separability is a design property of the data rather than a surprise.
- **Consequences:** Research question 1, comparing `ai_to_ai` against `real_to_real`, is **unaffected**: domain separability says nothing about clean-versus-stego detection within a single domain. Research question 2, transfer via `real_to_ai`, remains measurable but its interpretation narrows: a transfer failure occurs across a near-maximal domain gap, so it cannot be attributed to steganographic traces differing between domains without a further control that separates that from ordinary distribution shift. Small-scale separability numbers in [[Methods and Findings]] are superseded.
- **Related:** `results/audits/srnet-probe-6000-crop-balanced/summary.json`, [[Methods and Findings]].

### 2026-10-10 — Native crop is the primary protocol, `fit` is a sensitivity test

- **Status:** accepted
- **Context:** The two sources hold different resolutions. A native 256x256 crop keeps 35.0 per cent of a natural cover and 6.2 per cent of an AI cover, a ratio of 5.6. `ImageOps.fit` matches the field of view instead, but resamples the natural covers by a median 1.46x (range 0.66x to 7.59x) against exactly 4x for AI covers. One choice matches the field of view. The other matches the resampling. No choice matches both.
- **Decision:** Run the primary experiment with `--preprocess crop`. Run the same experiment with `fit` as a secondary sensitivity condition. State the field-of-view limitation in the thesis rather than treating either mode as the single correct answer.
- **Reasoning:** Crop preserves the original pixel values, applies the same literal operation to both domains, and adds no interpolation pattern. S-UNIWARD embeds into those unchanged pixels. Classical spatial steganalysis avoids unnecessary resizing. The `fit` condition then tests whether the conclusion survives a different preprocessing choice. If both modes give the same conclusion, the result is stronger. If they disagree, preprocessing drives the result and no general claim about AI covers against natural covers is available.
- **Consequences:** Every embedding and detector condition runs twice, once for each preprocessing mode. The AI crop holds a smaller part of the scene and more concentrated texture, which is a stated limitation and not a solved problem. S-UNIWARD hides bits in complex texture, so this limitation reaches research question 1 and not only the domain gap.
- **Related:** `evidence/domain-probes/`, [[Methods and Findings]]. Suggested by a second opinion from Codex on 2026-10-10.

### 2026-10-10 — Add a mixed-domain control and order the experiments by priority

- **Status:** accepted
- **Context:** The SRNet domain probe separates the two domains at 0.9954 balanced accuracy on clean covers alone. A poor `real_to_ai` result would therefore have two possible causes, and the four-direction design cannot separate them. The first cause is that embedding traces differ between the domains. The second is that the AI covers simply lie outside the detector's training distribution.
- **Decision:** Train a third detector on an equal mixture of natural and AI covers, and evaluate it on each domain. This adds two directions, `mixed_to_real` and `mixed_to_ai`. Run the experiments in this order: same-domain first, mixed-domain second, cross-domain last.
- **Reasoning:** The mixed detector sees both domains during training, so neither test domain is out of distribution for it. If `real_to_ai` is poor while `mixed_to_ai` is good, the first cause is distribution shift. If `mixed_to_ai` and `mixed_to_real` still differ substantially, that is stronger evidence that detectability itself differs between the domains. The ordering puts the interpretable results first and the ambiguous one last.
- **Consequences:** The mixed training set must hold the **same number of covers** as each single-domain set, with half drawn from each domain. A mixed model trained on the union would see twice the data, and any advantage could come from data volume instead of domain coverage. Training cost rises by half, because each seed now trains three models rather than two.
- **Related:** [[Methods and Findings]]. Suggested by a second opinion from Codex on 2026-10-10.

### 2026-10-10 — Resolution matching is an optional last pilot

- **Status:** accepted
- **Decision:** Reducing the AI covers to approximately the natural-image size before cropping stays an option. Test it last, and only after the crop and `fit` conditions are complete. Test it on approximately 500 covers for each domain, not on the full cohort.
- **Reasoning:** The option improves how much of the scene the AI crop keeps, but it resamples one domain and not the other. It exchanges one asymmetry for another, so a pilot must measure whether it is actually better before any larger run.
- **Consequences:** No work on this option begins until the crop and `fit` conditions have results.
- **Related:** [[Decisions]] entry above on preprocessing.
