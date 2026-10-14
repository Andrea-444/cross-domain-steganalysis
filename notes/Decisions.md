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

### 2026-10-12 — Run the six directions in three stages, grouped by train domain

- **Status:** accepted
- **Context:** The full protocol trains three detectors for each seed. It then evaluates six directions. At three seeds that is nine trainings in one command. A single command gives no intermediate result. A failure late in the command also loses the earlier work. The user asked to run the directions one at a time.
- **Decision:** Add a `--directions` option that runs a subset of the six directions. Run the protocol in three stages, each stage taking both directions of one train domain: `real_to_real real_to_ai`, then `ai_to_ai ai_to_real`, then `mixed_to_real mixed_to_ai`. Each stage writes its own result folder and its own W&B run.
- **Reasoning:** The train domain, and not the direction, is the natural unit of work. One trained model supplies both of its directions. A stage that keeps the pair together therefore costs exactly the same training as a stage that takes one direction. A stage that splits the pair trains the same model twice. It also loses the paired generalization gap. The gap subtracts the cross-domain result from the same-domain result inside one seed of one model, and one stage must hold both terms. The staged form keeps every gap computable. It also gives a readable result after each stage.
- **Consequences:** The results arrive in three folders instead of one, and in three W&B runs instead of one. A comparison across stages must confirm that the manifest SHA-256 and the seeds match. The `config.json` of each stage records both. The stage order differs from the reading order. Stage 1 already holds a same-domain and a cross-domain direction. Read the same-domain numbers from stages 1 and 2 first. Read the mixed numbers from stage 3 second. Interpret the cross-domain numbers last, and against stage 3.
- **Related:** [[Decisions]] entry of 2026-10-10 on the mixed-domain control and experiment priority. The `--directions` option is in `src/steganalysis/experiments.py`.

### 2026-10-12 — Declare `wandb` as a project dependency

- **Status:** accepted
- **Context:** The experiment runner imports `wandb` only when the command asks for tracking. The runner catches a failed import. The run then continues and writes complete local results, with `tracking.status` set to `failed`. The package was absent from `requirements.txt`. A machine that installed the stated requirements would therefore lose tracking silently, except for one warning line.
- **Decision:** Add `wandb>=0.19` to `requirements.txt`.
- **Reasoning:** Tracking is now part of the documented workflow, so the environment must state it. The lower bound and not a pin, because the package records results and does not affect them.
- **Consequences:** Every machine needs `pip install -r requirements.txt` again before an online run. The local results stay canonical whether tracking succeeds or fails. Confirm `tracking.status` in `summary.json` after each tracked run.

### 2026-10-13 — Train SRNet on paired batches, as the reference implementation does

- **Status:** accepted
- **Context:** Stage 1 of the six-direction protocol returned exactly 0.5000 accuracy in both directions and all three seeds. The train loss sat at 0.693, which is the natural logarithm of 2 and the loss of a classifier that guesses. The confusion matrix held the same row twice, so a cover and its stego always got the same answer. The control direction `real_to_real` failed with the rest, so the run carried no information about the research question. A check of the images excluded a data cause. No record shared a path, and no pair was identical. S-UNIWARD changed 7.65 per cent of the pixels, by exactly plus or minus one.
- **Decision:** Load the covers and stego images as pairs during training and validation. One dataset index gives a cover and its own stego. Both get the same augmentation draw, and the batch holds both. `batch_size` continues to count images, so the images for each optimizer step do not change. Evaluation keeps the flat mode, because a prediction belongs to one image.
- **Reasoning:** The reference implementation returns the pair from one dataset index, so every batch holds matched content. Our loader built a flat list of 2N images and shuffled it. That put a cover and its stego in different batches. The pair constraint is what makes a batch informative for steganalysis. The two classes then hold the same pictures, and the only systematic difference inside the batch is the embedding. Without it the batch statistics of the normalization layers follow the picture instead of the label.
- **Consequences:** Stage 1 must run again. Its numbers describe the old loader and belong in the thesis as a recorded failure, not as a result. This change is necessary, and it is not proven sufficient. The run also did 7,290 optimizer steps, and the SRNet paper uses 400,000. The next run is a pilot on one domain and one seed. It must show that the loss moves at all, before any full stage repeats.
- **Related:** `external/srnet-reference/dataset/dataset.py` returns `{"cover": ..., "stego": ...}` from one index. [[Methods and Findings]].

### 2026-10-13 — Use the reference initialization, after measuring the collapse

- **Status:** accepted
- **Context:** The paired-batch pilot ran 150 epochs and 36,000 optimizer steps on `real_to_real`. The train loss held at 0.693147, which is ln(2) to six decimals, from epoch 2 to epoch 150. A 200-step probe at three learning rates then measured the network directly. The results follow. The gradient at the first convolution stayed between 5.9e-3 and 0.74, so gradients reach the input layer. The output difference between a cover and its own stego started at 1.3e-3. The embedding therefore survives the forward pass. The norm of the fully connected layer held near 2.0. The collapse is therefore not that layer going to zero. The output spread across a batch fell from 0.273 to 0.017 at learning rate 1e-3. It fell to 0.035 at 1e-4, and to 0.078 at 1e-5. The cover and stego output difference fell with it, and reached 6.7e-5 at 1e-3.
- **Decision:** Initialize with the reference scheme. Xavier uniform on the convolutions, and normal with standard deviation 0.01 on the fully connected layer. The earlier code applied Xavier normal to both.
- **Reasoning:** The measurement shows the network learning to be constant, and losing its sensitivity to the embedding while it does so. Every learning rate shows the same direction, so the learning rate alone does not explain it. Xavier on a 512-to-2 layer gives a standard deviation near 0.06. The network then starts above ln(2). On a structured batch in train mode it gives 0.6998, against ln(2) of 0.6931. From above ln(2), the quickest descent is to make the output constant. The reference scheme starts at 0.6922, which is at ln(2), so that route down is already used up. The loss can then fall only by separating the two classes.
- **Consequences:** The pilot and stage 1 both describe the earlier initialization. Both stay in the record as failures, not as results. This is the fourth candidate of four, and three are now closed. The data is sound. The pair constraint is in place. The learning rate is not the deciding factor on its own. One explanation remains if the network still holds at ln(2) after this change. That explanation is the training budget. The project then needs a curriculum or a different detector.
- **Related:** `external/srnet-reference/utils/utils.py`, function `weights_init`. [[Methods and Findings]].

### 2026-10-14 — Diagnostics are commands in the repository, not scripts in a terminal

- **Status:** accepted
- **Context:** The investigation of the stage 1 null result used a sequence of scripts pasted into a terminal. Those scripts produced the numbers that now carry the investigation. They include the embedding check and the chain walk to the preprocessing mode. They also include the classical detector with its control, the collapse probe, and the synthetic training control. None of them was in the repository. The user asked for them as documented commands instead.
- **Decision:** Add `src/steganalysis/diagnostics.py`, with the subcommands `embedding`, `provenance`, `classical`, `collapse` and `synthetic-train`. Each writes a JSON report. The module follows the convention already in the repository. A module gives `add_arguments(subparsers)` and `run(args)`, as `src/datasets/audit.py` does.
- **Reasoning:** A number in terminal scrollback cannot go into a thesis. Nobody can repeat it, cite it, or compare it against a later run. The diagnostics also hold two definitions that the detector work depends on. One is the synthetic embedding used as a positive control. The other is the classical feature set. Each must have one home. A second copy of either would make two results that look comparable and are not.
- **Consequences:** Every number from the investigation must be produced again through the commands before it enters the thesis. The reports hold their own settings, so a later report states what it measured. The scripts in the conversation record stay as history, and no result cites them.
- **Related:** `README.md`, section "Diagnose a Detector Run". [[Methods and Findings]].

### 2026-10-14 — Record the GPU driver, and treat the 2026-10-10 upgrade as an open confound

- **Status:** accepted for the provenance change. The confound is open and not resolved.
- **Context:** The GPU machine stopped running `nvidia-smi` on 2026-10-14. The cause is a driver upgrade. The kernel module in memory is 595.84, and the module on disk is 595.99.02. The machine booted on 2026-10-08 11:26, and `dpkg` installed `nvidia-driver-595-open` 595.99.02 on 2026-10-10 06:37. A loaded kernel module does not change while a machine runs, so the two versions parted on that date. A later check showed that CUDA arithmetic still works and only the reporting library, NVML, is broken.
- **Decision:** Record the accelerator in the provenance of every run. The fields are the CUDA runtime of torch, the kernel driver line from `/proc/driver/nvidia/version`, whether NVML starts, the device name, and whether MPS is present. A missing reporting library records None, and a broken one records False. Those are different states.
- **Reasoning:** The provenance recorded the torch version and the kernel release, and neither moves when a driver changes. Two runs could therefore hold identical provenance and come from different environments. The project is a controlled comparison, so an environment change that nothing records is a hole in the method.
- **Consequences:** Every run from this date states its driver. Runs before this date do not, and the driver for those must come from this note.
- **Open confound:** Every successful CUDA run in the project predates 2026-10-10, and every failed CUDA run postdates it. The domain probe reached 0.9959 on 2026-10-07. Stage 1, the paired pilot and the three collapse probes all returned chance results on 2026-10-12 and 2026-10-13. This is a correlation with no exceptions, and the project has no explanation for the collapse after four separate fixes. Three facts argue against the driver as the cause. The failures held plausible gradients and losses. A mismatched driver usually fails loudly rather than computing wrong answers. The classical detector ran on CPU and found the real embedding at ROC-AUC 0.5578. That result says the task is hard for reasons that do not involve the GPU. The question stays open.
- **Test that closes it:** Run the domain probe again after the machine restarts. It is the only task on that machine with a known CUDA answer. A result near 0.99 means the card computes correctly and the detector failures are real. A much lower result means every GPU number after 2026-10-10 is void. One difference must stay in view during that comparison: the initialization changed on 2026-10-13, so the repeat is not identical to the original.
- **Related:** [[Methods and Findings]].
