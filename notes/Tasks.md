---
type: task-list
status: active
updated: 2026-09-04
---

# Tasks

## Protocol decisions — do before implementation locks in assumptions

- [ ] Review and correct [[Project Overview]].
- [ ] Review [[Pipeline Summary and Experiment Plan]] and record accepted protocol choices in [[Decisions]].
- [ ] Review the inferred entries in [[Decisions]].
- [ ] Define the exact final research deliverable and deadline.
- [ ] Audit the selected GenImage++ FLUX subset and ImageNet source pool before confirming them for final experiments.
- [ ] Decide whether to use all 1,000 ImageNet classes or a justified subset.
- [ ] Reconsider matched self-generation only if suitable external GPU access is confirmed.
- [x] Confirm the final embedding algorithms, payloads, and random seeds.
- [ ] Decide whether the primary AI domain uses one generator and whether a second generator is an external robustness test.
- [ ] Decide whether semantic class matching is required and define the policy.
- [ ] Define the primary outcome, cross-domain contrasts, uncertainty intervals, and statistical tests.
- [ ] Decide the minimum evidence required to label a run `final` rather than `smoke`, `pilot`, or `exploratory`.

## Dataset and manifest readiness

- [ ] Audit candidate datasets for usable counts, corrupt files, duplicates, licenses, storage size, image dimensions, and class/domain balance.
- [ ] Verify pair-safe splitting on a full candidate manifest, not only on synthetic test fixtures.
- [ ] Add or document cross-split duplicate/source-identity checks beyond matching `image_id` values.
- [ ] Record dataset source, subset/version, preprocessing, counts by domain/class/split, and manifest hash.
- [ ] Estimate generated stego storage for the accepted algorithm × payload × seed matrix.

## Local experiment pipeline

- [ ] Create one frozen experiment configuration that covers dataset, embedding, training, evaluation, and tracking settings.
- [x] Save structured epoch history instead of only printing training/validation loss.
- [x] Add validation accuracy and any accepted validation metrics.
- [x] Save the selected best checkpoint and selection metadata.
- [x] Preserve per-seed metrics rather than writing only aggregate summaries.
- [x] Preserve `image_id`, labels, predictions, and probabilities for every test direction.
- [x] Add accepted probability-based metrics and plots, such as ROC-AUC and PR-AUC, only after verifying score semantics. Score semantics verified: the reference model ends in `LogSoftmax`. Plots remain open.
- [ ] Record full provenance: manifest hash, dataset counts, embedding condition, detector commit, repository state, dependencies, hardware, timing, and exit status.
- [ ] Define stable run IDs and safe resume/idempotency behavior for interrupted multi-seed experiments.
- [ ] Document the final experiment configuration and compute requirements.
- [ ] Create a machine-readable protocol-level aggregator across training domains, directions, seeds, algorithms, and payloads.

## W&B integration

- [ ] Decide the W&B entity, project name, access policy, retention policy, and which collaborator accounts need access.
- [ ] Decide and pin a compatible `wandb` dependency before adding it to the environment.
- [ ] Add optional W&B configuration with `online`, `offline`, and `disabled` modes; keep W&B disabled in automated tests by default.
- [ ] Ensure no API key or private credential is written to source files, configuration committed to Git, logs, or artifacts.
- [ ] Define the run identity: one embedding condition × training domain × seed, evaluated on both test domains.
- [ ] Define consistent W&B `name`, `group`, `job_type`, and tags for protocol version, stage, algorithm, payload, domain, and seed.
- [ ] Initialize exactly one W&B run for each tracked training run and always finish it cleanly on success or failure.
- [ ] Log complete reproducibility configuration, including manifest hash and detector commit.
- [ ] Log per-epoch training loss/accuracy, validation loss/accuracy, learning rate, and epoch duration.
- [ ] Log final metrics for both test domains using unambiguous direction-prefixed names.
- [ ] Add W&B Tables for per-direction/per-seed results and per-image predictions.
- [ ] Add confusion matrices and, when probabilities are retained, ROC/PR curves and score distributions.
- [ ] Log small reproducibility artifacts: frozen config, manifest or manifest reference/hash, per-seed metrics, prediction table, summary, and selected checkpoint.
- [ ] Avoid uploading full datasets or large image collections by default; document any approved exception.
- [ ] Store the W&B run ID and URL in the corresponding local result directory while keeping local outputs canonical.
- [ ] Add deterministic run IDs and documented resume behavior so retries do not silently create misleading duplicates.
- [ ] Add tests using disabled/no-op or mocked tracking to prove that metrics are unchanged when W&B is unavailable.
- [ ] Verify an offline tracked smoke run, inspect its local files, then sync one approved test run.
- [ ] Build a dashboard showing four-direction metrics, transfer gaps, seed variability, learning curves, confusion matrices, and algorithm/payload comparisons.
- [ ] Mark runs visibly as `smoke`, `pilot`, `exploratory`, or `final` and filter the dashboard to exclude non-final evidence.
- [ ] Document setup, login, offline sync, dashboard use, and recovery from interrupted runs in `README.md`.
- [ ] Consider W&B Sweeps only after a single tracked run and the frozen protocol are validated.

## Preflight and analysis

- [ ] Define an end-to-end smoke test with a very small balanced dataset.
- [ ] Run a representative pilot to estimate time, memory, GPU use, and disk use without treating its output as final.
- [ ] Verify that local summaries and W&B values match for the same run.
- [ ] Define the statistical analysis used to compare experiment directions.
- [ ] Define the tables and figures needed for the final report.
- [ ] Define a failure-case review protocol using preserved per-image predictions.

## Final execution and writing

- [ ] Run the frozen final matrix without changing methods midway.
- [ ] Record failed and repeated runs explicitly; do not overwrite or silently omit them.
- [ ] Aggregate repeated seeds and calculate the accepted uncertainty/statistical comparisons.
- [ ] Review failure cases by domain, direction, embedding method, and payload.
- [ ] Record threats to validity and known confounders.
- [ ] Make final experiments reproducible from documented commands or configuration.
- [ ] Generate final tables and figures from preserved machine-readable results.
- [ ] Archive final manifests, configurations, metrics, and environment information.

## Blocked

- Nothing recorded.

## Done

- [x] Create the initial Obsidian-compatible project note structure.
- [x] Implement the current dataset → embedding → SRNet → four-direction evaluation skeleton.
- [x] Implement immutable dataset indexes, deterministic class-matched selection, and incremental audit/pilot/full extraction.
- [x] Run the repository unit test suite successfully on 2026-08-31 (8 tests; development verification only).
- [x] Create the central pipeline and experiment-planning summary.
