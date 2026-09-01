# Security and Scope

## Authorized scope

Keep work aligned with the repository's documented purposes:

- defensive research on detecting steganographic images;
- controlled generation of stego images for evaluation;
- recoverable OpenStego demonstrations using user-authorized covers and payloads;
- dataset preparation, detector evaluation, reproducibility, and scientific reporting.

Do not silently expand the project into deployment for concealment, evasion, unauthorized data transfer, or targeting systems or data outside the user's stated scope.

## Payload handling

- Prefer synthetic, non-sensitive payloads for tests and demonstrations.
- Do not inspect, reproduce, transmit, or expose private payload contents unless the user explicitly asks and the action is necessary for the task.
- Keep OpenStego demonstrations separate from the primary adaptive-algorithm research conclusions.
- Do not place a password directly in a recorded command. The current workflow supports the `OPENSTEGO_PASSWORD` environment variable or another explicitly named password environment variable.
- Never write secret values into Markdown notes, source files, result JSON, logs, or version control.

## Credentials

- Kaggle access may use `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
- Do not display credential files or echo secret environment variables.
- Do not copy credentials into the repository.
- If authentication is missing, report the required configuration without inventing or requesting that the user paste secrets into project files.

## External code and downloads

- Treat network downloads and external repositories as supply-chain inputs.
- Use the documented source and pinned version when one exists.
- Do not update SRNet, Aletheia, OpenStego, `conseal`, or dataset sources as incidental maintenance.
- GenImage downloads can be very large; verify user intent, target, and disk capacity before starting.
- OpenStego setup downloads and executes a Java archive. Keep it in the documented `external/openstego` boundary and report setup failures clearly.

## Filesystem and execution safety

- Inspect exact targets before any deletion, replacement, or bulk generation.
- Prefer new timestamped output directories over overwriting prior work.
- Do not run full dataset downloads or training sweeps merely to validate a small code change.
- Use temporary directories and generated fixtures for tests where possible.
- Preserve evidence from failed or partial research runs unless the user requests cleanup.

## Reporting security concerns

If a request would materially depart from defensive research or authorized demonstration, pause and ask for the intended scope. When reporting a security or privacy risk, identify the affected artifact or workflow and suggest a bounded mitigation without exposing secret material.
