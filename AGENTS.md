# Project Instructions

## Purpose

Work on the cross-domain steganalysis project described in `README.md`. The target task is binary steganalysis (`clean` versus `stego`), not AI-generated-image detection. Preserve the controlled comparison between natural and AI-generated cover domains.

## Sources of truth

Use sources in this order when they disagree:

1. Executable code and tests for current behavior.
2. `README.md` for the documented workflow and intended research design.
3. `notes/Decisions.md` for decisions explicitly accepted by the user.
4. Other files under `notes/` for evolving context, plans, and ideas.

Do not silently resolve a meaningful conflict. Report it and either fix the stale documentation as part of the requested work or ask the user which direction is authoritative.

## Instruction routing

Read the relevant supporting instructions before substantive work:

- Research design or experiment changes: `agent-instructions/RESEARCH_PROTOCOL.md`
- Code, tests, dependencies, or commands: `agent-instructions/DEVELOPMENT_WORKFLOW.md`
- Datasets, generated images, models, caches, or results: `agent-instructions/DATA_AND_ARTIFACTS.md`
- Analysis, figures, tables, or prose claims: `agent-instructions/RESULTS_AND_WRITING.md`
- Steganography demonstrations, credentials, downloads, or external tools: `agent-instructions/SECURITY_AND_SCOPE.md`

For project orientation, consult `notes/Project Overview.md` and `notes/Architecture.md`. Consult `notes/Tasks.md` for user-maintained priorities, but treat the user's current request as authoritative.

## Non-negotiable research constraints

- Keep clean/stego pairs in the same train, validation, or test split.
- Apply equivalent preprocessing to natural and AI cover domains in controlled comparisons.
- Do not present legacy LSB or recoverable OpenStego demonstrations as evidence from the primary adaptive-steganography benchmark.
- Do not present smoke-test output as a final research result.
- Preserve seeds, manifests, configuration, detector version, algorithm, payload, and domain direction needed to interpret or reproduce a result.
- Distinguish facts observed in artifacts from hypotheses, interpretations, and pending decisions.

## Working rules

- Preserve unrelated user changes and generated artifacts.
- Inspect before editing; make the smallest change that satisfies the task.
- Do not download large datasets, run full training sweeps, delete artifacts, or replace external dependencies unless the user requests or approves it.
- Never expose credentials, passwords, API keys, or private payload contents.
- Use the repository's existing commands and conventions. Do not invent a formatter, linter, configuration system, or dependency manager that is not present.
- Validate changes in proportion to their scope and report what was and was not verified.
- Update project notes when requested or when a completed change would otherwise leave a directly related note materially false. Do not convert unapproved ideas into accepted decisions.

## Repository boundaries

- Primary implementation: `src/datasets/`, `src/steganography/`, and `src/steganalysis/`.
- Educational and historical prototypes: `src/legacy/`.
- Tests: `tests/`.
- Living project notes: `notes/`.
- External reference implementations and tools: `external/`.
- Generated datasets, stego images, caches, and results: `data/`, `stego_runs/`, and `results/`.

Read nested instructions if additional `AGENTS.md` files are introduced later.
