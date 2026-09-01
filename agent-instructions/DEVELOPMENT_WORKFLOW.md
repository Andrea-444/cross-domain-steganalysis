# Development Workflow

## Current environment

The README identifies the existing conda environment as `experimental_1` and installs Python dependencies from `requirements.txt`:

```bash
conda activate experimental_1
pip install -r requirements.txt
```

Do not change the environment manager or add packaging, formatting, or linting tools unless the task requires it and the user agrees to the project-level change.

## Main command groups

Run commands from the repository root:

```bash
python -m src.datasets
python -m src.steganography
python -m src.steganalysis.experiments
python -m src.steganalysis.aletheia
```

Use each command's `--help` output or its parser implementation as the authority for current flags.

## Tests

The documented full test command is:

```bash
python -m unittest discover -s tests -v
```

Some tests are conditionally skipped when optional SRNet or OpenStego dependencies are absent. Report skips separately from passes and failures.

For a focused change, run the smallest relevant test first. Run the full suite when the change crosses modules, affects manifests or experiment validity, changes a public command, or is otherwise high risk.

## Code organization

- `src/datasets/`: acquisition, normalization, manifest construction, and validation.
- `src/steganography/`: adaptive benchmark embedding and recoverable OpenStego demonstrations.
- `src/steganalysis/`: SRNet integration, cross-domain experiments, metrics, and optional Aletheia support.
- `src/legacy/`: earlier LSB and CNN prototypes; keep them isolated from primary results.

Prefer the shared manifest helpers in `src/datasets/manifest.py` over duplicating manifest parsing, identifiers, deterministic seeds, or validation.

## Change discipline

- Preserve existing CLI behavior unless the task explicitly requires a breaking change.
- Keep deterministic behavior deterministic; include relevant seeds in generated configuration.
- Add or update tests for bug fixes, new invariants, public behavior, and reproducibility-sensitive logic.
- Use temporary directories and synthetic small images in tests. Do not make ordinary unit tests depend on large datasets or full training runs.
- Keep optional external tools behind clear setup checks and actionable errors.
- Avoid editing vendored or cloned code in `external/` when an adapter in `src/` can express the integration.
- Do not modify unrelated dirty files.

## Definition of done

A change is complete when, as applicable:

1. The requested behavior exists in the primary code path.
2. Relevant tests pass, or unrun verification is clearly disclosed.
3. Research invariants and artifact provenance remain intact.
4. User-facing commands or behavior are reflected in `README.md` when changed.
5. Directly affected instructions or notes are not left materially false.
6. The final report identifies changed files, validation performed, and remaining limitations.

## Not currently established

No repository-wide formatter, linter, type checker, continuous-integration configuration, or formal release process was found when these instructions were created. Do not imply that one exists.
