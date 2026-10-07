---
type: development-log
status: active
updated: 2026-10-07
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

## 2026-09-04 — Hugging Face acquisition path added

- **Goal:** Replace the local Kaggle images with the recorded working pair, and make both sources reachable without downloading corpora whole.
- **Work completed:** Added `src/datasets/acquisition.py` with `fetch-archive` (one file from a Hugging Face dataset repo, reused if already present) and `fetch-images` (streams an image dataset into `<class>/` directories). Both write the original encoded bytes rather than re-saving. Added `huggingface_hub` and `datasets` to `requirements.txt`. Cleared `data/processed` and `data/stego`; kept `src/raw_data`, `results/`, and `stego_runs/`.
- **Commands or experiment configuration:** `python -m unittest discover -s tests -t .`
- **Result:** 45 tests passed, exit code 0. Confirmed GenImage++ `archives/flux.tar.zst` is 7.22 GB of a 204 GB repository and is not gated; ImageNet-1k is gated behind the ImageNet Terms of Access and needs an authenticated account.
- **Problems or surprises:** `tests/datasets/` shadowed the installed `datasets` library, because discovery placed `tests/` itself on `sys.path`. The documented test command now passes `-t .` so the repository root is the top-level directory instead. This changed a documented command in `README.md` and `agent-instructions/DEVELOPMENT_WORKFLOW.md`. Also note the internal directory layout of the GenImage++ archives is not documented on the dataset card, so class inference must be verified by listing the archive before selection.
- **Next step:** Download the FLUX archive, list its layout, and confirm whether it carries ImageNet synsets before choosing the class scope for 1,000 covers per domain.
- **Related files/results:** `src/datasets/acquisition.py`, `tests/datasets/test_acquisition.py`, `requirements.txt`.

## 2026-09-29 — Domain confound audit implemented and run on the smoke100 cohort

- **Goal:** Test the gate that everything downstream depends on: are natural and AI covers distinguishable from clean images alone, before any embedding?
- **Work completed:** Added `src/datasets/audit.py` and the `audit-domains` command. It scores three interpretable features per cover (8x8 blockiness contrast, Laplacian high-pass energy, neighbour-residual spread) and cross-validates a logistic regression that predicts the domain from clean covers only.
- **Commands or experiment configuration:** `python -m src.datasets audit-domains --manifest data/processed/manifests/imagenet-flux-smoke100_clean.jsonl`
- **Result:** 55 tests passed. On the 100-per-domain cohort the domains are **partially separable with no steganography present**: balanced accuracy 0.630, ROC-AUC 0.691 against a 0.5 chance baseline. Feature means: blockiness real=0.0042 / ai=0.0130; high-pass energy real=45.77 / ai=34.96; residual spread real=11.44 / ai=8.74.
- **Problems or surprises:** Two findings run against expectation. The AI covers are *blockier* than the JPEG-sourced natural covers, not less — plausibly the 8x8 latent-patch structure of the FLUX VAE decoder, which survives where JPEG blocking did not. And the natural covers retain markedly more high-frequency energy, consistent with their much gentler downsampling (about 1.95x from ~500x375) versus the AI covers' exact 4x (1024 to 256). The 0.630 figure is a **lower bound**: it comes from three crude hand-crafted features, and SRNet is purpose-built to exploit exactly this kind of residual difference.
- **Next step:** Decide whether 0.630 is acceptable, and test crop-only preprocessing as the candidate remedy for the resampling mismatch. This is a protocol decision and is not settled here.
- **Related files/results:** `src/datasets/audit.py`, `tests/datasets/test_audit.py`, `results/audits/imagenet-flux-smoke100_domain_separability.json`.

## 2026-09-29 — Crop-only preprocessing removes the measurable domain confound

- **Goal:** Test whether native-crop preprocessing removes the pre-embedding domain separability found earlier today.
- **Work completed:** Added `normalize_image` with `fit` and `crop` modes, exposed `--preprocess` and `--skip-undersized` on `prepare-selection`, and added a label-shuffle permutation test to `audit-domains` so a near-chance score can be judged against its null rather than by eye.
- **Commands or experiment configuration:** `prepare-selection --preprocess crop --skip-undersized` to build `imagenet-flux-smoke100-crop`, then `audit-domains --permutations 200` on both cohorts.
- **Result:** 62 tests passed. With `fit`, the domains are significantly separable from clean covers alone: balanced accuracy 0.630, ROC-AUC 0.691, permutation p = 0.0050. With `crop`, they are not: balanced accuracy 0.532, ROC-AUC 0.567, permutation p = 0.2736. Crop-only costs one natural cover, which is 170 px on its short side and is recorded in the run configuration.
- **Problems or surprises:** The blockiness hypothesis recorded this morning was wrong. AI blockiness was attributed to FLUX's 8x8 VAE latent structure, but it falls from 0.0130 under `fit` to 0.0008 under `crop`, so it was produced by the exact-4x LANCZOS downsampling, not by the generator. High-pass energy rises in both domains under crop (real 45.8 to 49.6, ai 35.0 to 42.5) and the gap between them narrows, which is what removing resampling should do.
- **Next step:** Decide whether to adopt crop-only as the accepted preprocessing and record it in [[Decisions]]. The audit remains a necessary rather than sufficient check: it uses three hand-crafted features, and SRNet may exploit differences they miss.
- **Related files/results:** `src/datasets/preparation.py`, `src/datasets/audit.py`, `results/audits/imagenet-flux-smoke100_domain_separability.json`, `results/audits/imagenet-flux-smoke100-crop_domain_separability.json`.

## 2026-09-29 — SRNet domain probe overturns the feature-based all-clear

- **Goal:** Ask the actual detector, rather than three hand-crafted features, whether it can identify the cover domain from clean covers alone.
- **Work completed:** Added `src/steganalysis/domain_probe.py`, which trains SRNet on clean covers labelled by domain with no stego images present. Ran it with three seeds and thirty epochs on both cohorts, using the `fit` cohort as a positive control so that a null on `crop` could be interpreted.
- **Commands or experiment configuration:** `python -m src.steganalysis.domain_probe --manifest <clean manifest> --seeds 1 2 3 --epochs 30 --batch-size 16`.
- **Result:** 69 tests passed. Positive control succeeded: on `fit`, SRNet reached balanced accuracy 0.745 (sd 0.050) and ROC-AUC 0.843 (sd 0.034), higher than the three-feature probe's 0.630, confirming the probe has power at this data volume. On `crop`, SRNet still reached balanced accuracy 0.675 (sd 0.056) and ROC-AUC 0.766 (sd 0.049). Per-seed balanced accuracy was [0.676, 0.794, 0.765] for `fit` and [0.632, 0.640, 0.754] for `crop`.
- **Problems or surprises:** **The feature-based audit gave a false all-clear.** It scored `crop` at 0.532 with permutation p = 0.2736, which reads as chance, while SRNet separates the same covers at 0.675 balanced accuracy. Crop-only genuinely helped, reducing separability from 0.745 to 0.675, but it did not remove it. The documented caveat that the three-feature audit is necessary rather than sufficient proved to be the operative one, and any future gate decision should use the SRNet probe rather than the cheap screen.
- **Next step:** Separate the two possible causes of the residual separability. Compression history is untouched so far: natural covers are JPEG-sourced and AI covers are lossless PNG, and crop-only does nothing about that. Genuine generative-model pixel statistics are the other candidate, and unlike compression history they are a legitimate property of the domains rather than a confound. Equalizing compression across domains and re-running the probe would distinguish them. This is a protocol decision and is not settled here.
- **Related files/results:** `src/steganalysis/domain_probe.py`, `results/audits/srnet-probe-fit/summary.json`, `results/audits/srnet-probe-crop/summary.json`.

## 2026-09-29 — Compression equalization does not reduce domain separability

- **Goal:** Determine whether the residual SRNet domain separability under crop-only preprocessing comes from compression history or from intrinsic generative pixel statistics.
- **Work completed:** Measured the natural sources' JPEG quality from their quantization tables (median 96; 86 of 100 covers at 96). Added `src/datasets/recompress.py` and `recompress-cohort`, which re-encodes chosen domains through JPEG at a given quality and decodes back, storing the result as lossless PNG so only the pixel content carries the new history. Built two variants at quality 96, one adding JPEG to the AI covers only and one to both domains, then ran the SRNet probe with three seeds on each.
- **Commands or experiment configuration:** `recompress-cohort --quality 96 --domains ai` and `--domains ai real`, then `domain_probe --seeds 1 2 3 --epochs 30`.
- **Result:** 74 tests passed. Equalizing compression had no meaningful effect on SRNet separability. Balanced accuracy: crop-only 0.675 (sd 0.056), JPEG on AI only 0.661 (sd 0.049), JPEG on both 0.688 (sd 0.086). All three sit within roughly one standard deviation of each other, against 0.745 for the resampled `fit` cohort.
- **Problems or surprises:** The three-feature audit was fooled again, and more completely than before: it scored the two recompressed variants at 0.502 and 0.497 with permutation p of 0.54 and 0.60, while SRNet separated the same covers at 0.661 and 0.688. It should not be used as a gate. The substantive finding is that the residual separability survives compression equalization, so it is most likely an intrinsic property of FLUX-generated images rather than a preprocessing artifact. That is an inference from one failed intervention, not proof: semantic content, generation resolution, and tonal statistics remain untested as alternative explanations. The test split is only 33 to 34 covers, so each prediction moves balanced accuracy by about three points.
- **Next step:** Record a preprocessing decision in [[Decisions]]. The evidence supports crop-only, since it removed a real resampling confound, and does not support adding JPEG to the AI covers, since it changed nothing measurable. Whether an intrinsic domain separability of about 0.675 is acceptable for the thesis is a protocol judgement and remains open.
- **Related files/results:** `src/datasets/recompress.py`, `results/audits/srnet-probe-smoke100-crop-jpegai96/`, `results/audits/srnet-probe-smoke100-crop-jpegboth96/`.

## 2026-10-07 — Preprocessing decisions recorded; duplicate branch retired

- **Goal:** Record the preprocessing and gating conclusions in [[Decisions]] so a second machine reads them as accepted, and remove the superseded duplicate branch.
- **Work completed:** Added three accepted entries to [[Decisions]]: normalize by native crop without resampling, do not equalize compression history, and gate cohorts with the SRNet probe rather than the three-feature screen. Updated the provisional preprocessing row in [[Experiment Decisions]] and removed its now-settled crop/resize open item. Deleted the remote branch `feat/imagenet-genimagepp-class-matched-cohort`.
- **Commands or experiment configuration:** No experiment run; notes and branch housekeeping only.
- **Result:** `origin` now carries only `dev` and `main`. The deleted branch held one commit, `6731390`, an independent implementation of the class-mapping and preparation workflow that `c09bea2` on `dev` had already superseded. It is recoverable by that SHA if ever needed.
- **Problems or surprises:** None. Note that [[Experiment Decisions]] still lists "Confirm the accessible ImageNet validation archive and its download size" as open, although the gated `ILSVRC/imagenet-1k` streaming path is settled and documented in `README.md`. That item was already stale before this change and was left for the user rather than edited silently.
- **Next step:** Embed on the crop cohort and run the first real four-direction experiment. SRNet convergence on embedding traces at realistic scale is still unmeasured.
- **Related files/results:** `notes/Decisions.md`, `notes/Experiment Decisions.md`.

## 2026-10-07 — Untracked the regenerable data artifacts

- **Goal:** Correct yesterday's decision to track dataset indexes and selection manifests.
- **Work completed:** Removed `data/indexes/` and `data/selections/` from version control with `git rm --cached`, leaving the files in place on disk. Kept the class maps and the audit summaries. Rewrote the `.gitignore` comments, which had recorded the reasoning that this change reverses.
- **Commands or experiment configuration:** No experiment run.
- **Result:** Nine files and about 6,350 lines removed from the repository. Three class-map files and six audit summaries remain tracked.
- **Problems or surprises:** The original justification did not hold. Indexes and selections are regenerable by `index` and `select`, they describe whatever is on one machine's disk rather than a property of the datasets, and selection manifests record absolute local paths so another machine cannot consume them. The claim that they preserved a provenance chain was also wrong on its own terms: the audit summaries reference `data/processed/manifests/*_clean.jsonl`, which was never tracked. `agent-instructions/DATA_AND_ARTIFACTS.md` already says not to commit generated artifacts merely because they are useful locally.
- **Next step:** Unchanged. Embed on the crop cohort and run the first real four-direction experiment.
- **Related files/results:** `.gitignore`, `notes/Decisions.md`.

## 2026-10-07 — Embedding isolated per cohort; history ledger added

- **Goal:** Embed the crop cohort without repeating work across methods, and make the embedding history auditable.
- **Work completed:** The stego cache key now includes a cohort identifier derived from the clean manifest's content hash, and each run records `cohort_id` and `input_manifest_sha256`. Added `python -m src.steganography ledger`, which reports the conditions already run by reading each run's own configuration. Embedded S-UNIWARD at 0.4 bpp, seed 42, on `imagenet-flux-smoke100-crop`.
- **Commands or experiment configuration:** `benchmark --algorithm suniward --payload-bpp 0.4 --seed 42 --cache-dir data/stego/cache`, run twice, then the same condition on the `fit` cohort.
- **Result:** 82 tests passed. First run generated 199 stego images, the identical re-run generated 0 and reused 199. The `fit` cohort with the same algorithm, payload and seed generated all 200 rather than reusing, confirming the cohorts no longer share a cache. Verified on the crop run: zero pair-split leaks, every change exactly +/-1, and 6.3 to 10.1 per cent of pixels altered per image.
- **Problems or surprises:** The previous cache key was backend, algorithm, payload and seed only. Since `fit` and `crop` share image IDs but differ in pixels, embedding both with one cache would have paired stego images with covers they were not derived from, silently. No such run had been made, so no existing artifact is affected.
- **Next step:** Run the four-direction SRNet experiment on the embedded crop cohort. This is the first real test of whether the detector learns embedding traces, as opposed to the domain signal already measured by the probe.
- **Related files/results:** `src/steganography/core.py`, `tests/steganography/test_cohort_isolation.py`.

## 2026-10-07 — Consolidated methods and findings record for thesis writing

- **Goal:** Pull the methodological decisions and their supporting numbers into one place usable when writing the thesis, rather than leaving them spread across log entries, decision records, README tables and result JSON.
- **Work completed:** Added [[Methods and Findings]]. It carries the confound investigation with exact figures read from the result artifacts, the per-seed values, the feature means, the false-negative finding about the cheap screen, a threats-to-validity section, the pipeline provenance summary, and the reproduction commands. Measured observations and interpretation are separated throughout, per `agent-instructions/RESULTS_AND_WRITING.md`.
- **Commands or experiment configuration:** No experiment run; figures were read from `results/audits/*/summary.json` rather than transcribed from memory.
- **Result:** The note states prominently that no experimental result exists yet and that every figure in it comes from methodological validation rather than the research protocol. It also addresses the open [[Tasks]] item on recording threats to validity.
- **Problems or surprises:** None. Four candidate causes of the residual domain separability remain unexcluded and are listed rather than glossed: semantic content, generation resolution, tonal statistics under grayscale conversion, and single-generator scope.
- **Next step:** Unchanged. Re-prepare the 6,000-per-domain cohort with crop-only, gate it with the probe, then run the first four-direction experiment.
- **Related files/results:** `notes/Methods and Findings.md`.

## 2026-10-07 — Corrected the recorded resampling factor

- **Goal:** Verify the downsample factor quoted for natural covers after a question about what `fit` actually does.
- **Work completed:** Measured the effective factor over the 100 natural covers in the smoke cohort and corrected `README.md`, [[Decisions]] and [[Methods and Findings]]. Earlier entries in this log are left as written, since they record what was believed at the time.
- **Commands or experiment configuration:** Short script over `data/raw/selected/smoke100/real` reading each source's dimensions.
- **Result:** The recorded figure of about 1.95x was wrong. `ImageOps.fit` crops to the target aspect ratio using the **short** side and then resamples, so the factor is the short side over 256, not the long side. Measured: median 1.46x, mean 1.66x, range 0.66x to 7.59x, against exactly 4.00x for the uniformly 1024x1024 AI covers.
- **Problems or surprises:** Two findings that strengthen the crop-only decision rather than weaken it. A minimum of 0.66x means some natural covers were **enlarged** under `fit`, and interpolation invents high-frequency content instead of attenuating it, which is worse for a steganalyzer than smoothing. And the factor varied from 0.66x to 7.59x *within* the natural domain, so `fit` applied inconsistent resampling across natural covers as well as unequal resampling between domains. The measured separability figures are unaffected; only the explanatory factor was misquoted.
- **Next step:** Unchanged. Re-prepare the 6,000-per-domain cohort with crop-only.
- **Related files/results:** `README.md`, `notes/Decisions.md`, `notes/Methods and Findings.md`.

## 2026-10-07 — Cohort rebalancing after one-sided skip

- **Goal:** Restore per-class domain matching after crop-only preprocessing dropped undersized natural covers from the 6,000-per-domain cohort.
- **Work completed:** Added `src/datasets/rebalance.py` and the `rebalance-cohort` command. It equalizes domain counts per class **and** per split, choosing which surplus cover to drop deterministically by sorted `image_id`, and records every dropped cover in the run configuration.
- **Commands or experiment configuration:** `rebalance-cohort --manifest data/processed/manifests/imagenet-flux-smoke100-crop_clean.jsonl --name smoke100-crop-balanced`.
- **Result:** 89 tests passed. On the GPU machine's full cohort, crop-only kept 11,830 of 12,000 covers: 170 natural covers were under 256 px on the short side, all on the natural side, leaving 5,830 against 6,000 and 98 of 1,000 classes unmatched. Verified locally on the 199-cover crop cohort: 198 kept, 1 AI cover dropped, zero class-and-split mismatches remaining.
- **Problems or surprises:** The imbalance matters for a reason beyond the recorded class-matching rule. Unequal domain totals would have given the real-trained and AI-trained detectors different training-set sizes (3,887 against 4,000 covers), so a `real_to_real` versus `ai_to_ai` difference could have reflected data volume rather than domain difficulty. Also worth recording for future cohorts: about 2.8 per cent of ImageNet validation covers are below 256 px on the short side, and the index stores `size_bytes` but not dimensions, so these cannot yet be filtered at selection time, which is where the fix belongs.
- **Next step:** Rebalance the full cohort, gate it with the SRNet domain probe at full scale, then embed and run the four-direction protocol.
- **Related files/results:** `src/datasets/rebalance.py`, `tests/datasets/test_rebalance.py`.

## 2026-10-07 — Full-scale domain probe: the cover domains are near-perfectly separable

- **Goal:** Gate the full `imagenet-flux-6000-crop-balanced` cohort with the SRNet domain probe before embedding.
- **Work completed:** Ran the probe on the GPU machine with 3 seeds and 30 epochs on CUDA. Copied the summary into `results/audits/srnet-probe-6000-crop-balanced/`. Updated [[Decisions]] and [[Methods and Findings]], and marked the small-scale separability figures as superseded.
- **Commands or experiment configuration:** `domain_probe --manifest data/processed/manifests/imagenet-flux-6000-crop-balanced_clean.jsonl --seeds 1 2 3 --epochs 30 --batch-size 32`.
- **Result:** Balanced accuracy 0.9954 (sd 0.0007) and ROC-AUC 0.9997 (sd 0.0001). Per-seed balanced accuracy 0.9959, 0.9944, 0.9959, with 8, 11 and 8 errors per seed out of 1,948 test covers. Selected epochs 29, 23 and 25. Duration 66 minutes on CUDA.
- **Problems or surprises:** The small-scale figure of 0.675 was not an estimate of the domain gap; it was the limit of 132 training covers. The documented power caveat was the operative one for a second time. Two consequences follow. First, the crop-only decision can no longer be justified by the measured 0.745 against 0.675 difference, because at full scale `crop` alone saturates the probe; the justification is now that unequal and sometimes magnifying resampling is a known processing artifact worth removing on principle. Running the `fit` probe at full scale would be uninformative, since both variants sit at the ceiling, so it is not worth 66 minutes. Second, the result is consistent with the sources rather than alarming: GenImage++ is published as a benchmark **for** AI-image detection, so separability is a design property of the data.
- **Next step:** Embed S-UNIWARD at 0.4 bpp on the rebalanced cohort and run the four-direction protocol. Research question 1, comparing `ai_to_ai` with `real_to_real`, is unaffected by domain separability. Research question 2 needs its interpretation narrowed, and a control that separates steganographic-trace mismatch from ordinary distribution shift remains undesigned.
- **Related files/results:** `results/audits/srnet-probe-6000-crop-balanced/summary.json`.
