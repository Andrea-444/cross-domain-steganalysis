# Research Protocol

## Established objective

The project studies whether hidden information added to an AI-generated image is detected differently from hidden information added to a natural image.

The detector predicts:

- `0`: clean
- `1`: stego

The current research questions are:

1. Are hidden payloads intrinsically easier or harder to detect in AI-generated covers?
2. Does a steganalyzer trained on natural images generalize to AI-generated covers?

## Controlled evaluation matrix

Preserve all four directions unless the user explicitly changes the protocol:

| Direction | Current role |
|---|---|
| `ai_to_ai` | AI-domain detectability |
| `real_to_ai` | Transfer from conventional training to AI covers |
| `real_to_real` | Conventional same-domain control |
| `ai_to_real` | Reverse-transfer control |

## Current primary pipeline

1. Acquire or identify clean natural and AI cover images.
2. Balance the two domains.
3. Apply identical center-crop, grayscale conversion, resize, and PNG encoding.
4. Assign stable image IDs and train/validation/test splits in a clean manifest.
5. Generate stego images with the same embedding method and payload policy across domains.
6. Preserve each clean/stego pair's `image_id` and split.
7. Train SRNet separately on the natural and AI training domains.
8. Evaluate the two trained models in all four domain directions.
9. Store the run configuration, manifest, per-seed results, and summary metrics.

## Current methods

- Primary embedding library: `conseal==2024.12`.
- Implemented adaptive algorithms: WOW, S-UNIWARD, HILL, and MiPOD.
- CLI-supported payload values: `0.1`, `0.2`, and `0.4` bits per pixel.
- Primary detector: the pinned `brijeshiitg/Pytorch-implementation-of-SRNet` reference at commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`.
- Optional secondary tool: Aletheia. Treat its pretrained screening as secondary evidence because cover-source mismatch may apply.
- OpenStego RandomLSB and code under `src/legacy/` are demonstrations or educational baselines, not substitutes for the primary adaptive benchmark.

## Required validity checks

- Verify every manifest record contains the required fields and valid domain/split values.
- Reject duplicate clean `image_id` values.
- Reject a clean/stego pair assigned to different splits.
- Keep preprocessing, embedding algorithm, payload, split policy, and detector training procedure fixed when attributing a difference to cover domain.
- Use deterministic seeds where the implementation supports them.
- Check class balance and sample counts before interpreting accuracy.
- Treat one-seed, one-epoch, or deliberately tiny runs as pipeline smoke tests only.
- Retain negative, chance-level, and unexpected results; do not select runs based on favorable outcomes.

## Protocol changes

Before changing a scientific variable, state:

- the hypothesis or failure that motivates the change;
- which variable changes and which controls remain fixed;
- whether the change creates a new experiment or supersedes an old one;
- what artifacts are required to reproduce the comparison.

Record an accepted consequential change in `notes/Decisions.md`. Do not rewrite prior result artifacts to match a new protocol.

## Pending decisions

The repository does not yet establish all of the following as final:

- final dataset source, subset, and sample size;
- final algorithms and payloads included in the reported experiment;
- final seed count, epoch count, and compute budget;
- statistical tests, confidence-interval method, or decision thresholds;
- primary endpoint and criteria for supporting or rejecting the hypotheses;
- treatment of generator-specific AI subdomains.

Label these as pending until the user accepts a decision and it is documented.
