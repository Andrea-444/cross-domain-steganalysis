# Data and Artifacts

## Current locations

The code and README currently use these locations:

| Path | Purpose |
|---|---|
| `src/raw_data/` | Existing local natural and AI source images described by the README |
| `data/raw/` | Default programmatic download destination |
| `data/indexes/` | Immutable native-source candidate indexes and index configuration |
| `data/selections/` | Deterministic class-matched selection manifests and configuration |
| `data/raw/selected/` | Incrementally materialized native audit, pilot, and full selections |
| `data/processed/` | Normalized images and clean manifests |
| `data/.cache/` | Tool caches, including `conseal` and OpenStego runtime data |
| `stego_runs/` | Timestamped steganographic benchmark runs |
| `results/` | Timestamped detector configurations, manifests, and summaries |
| `external/` | Pinned reference implementations and optional external tools |

The `.gitignore` currently excludes `data/`, `external/`, `results/`, `stego_runs/`, `raw_data/`, and `dataset/` patterns. Check Git status rather than assuming every nested path is ignored exactly as intended.

## Artifact handling rules

- Treat raw images, processed datasets, stego images, trained weights, caches, manifests, and result directories as user data.
- Do not delete, rename, overwrite, or bulk-regenerate them unless the user requests it and the exact target is verified.
- Preserve timestamped run directories. Create a new run instead of rewriting an old result.
- Preserve dataset indexes and selection manifests once they are used. Create a new named selection instead of overwriting one.
- Keep source archives immutable. Incremental extraction must reuse checksum-matched files and reject changed content.
- Do not hand-edit scientific result JSON or manifests to make a run appear successful.
- Keep the input manifest with a result run when the current workflow does so.
- Preserve absolute paths already recorded in artifacts, but note that they reduce portability when interpreting or moving a run.
- Do not commit large or generated artifacts merely because they are useful locally.

## Dataset preparation

For controlled comparisons:

- search source folders recursively;
- balance natural and AI domains according to the pipeline behavior;
- apply the same normalization parameters to both domains;
- preserve source provenance and relative source paths in the clean manifest;
- verify output counts, image mode, dimensions, and split membership;
- never reuse a clean cover across conflicting splits through its stego counterpart.
- assign the final test split before audit/pilot extraction and do not expose final-test images to method-selection pilots.

The current default normalization size is 256×256 grayscale PNG. Treat color, variable-size, or JPEG-domain work as a new protocol unless the user explicitly makes it part of the controlled experiment.

## Downloads and external dependencies

- Kaggle download requires an `owner/dataset-name` and user-provided Kaggle credentials.
- GenImage archives are large. The CLI requires `--accept-large-download`; still confirm disk and scope before initiating a download.
- The primary SRNet reference is pinned to commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355` according to the README and adapter.
- Aletheia and OpenStego are optional and installed separately.
- Do not update pins or replace external sources as incidental cleanup.

## Run provenance

For a result intended as research evidence, retain or report at least:

- source dataset and subset;
- clean and stego manifest paths;
- sample counts by domain, class, and split;
- preprocessing size and mode;
- embedding backend, algorithm, payload, and seed;
- detector source/version, training domain, test domain, and training settings;
- experiment seeds;
- output directory and generated configuration/summary files;
- software and hardware details when they materially affect reproducibility.

The current code records part, but not necessarily all, of this information. Do not claim complete reproducibility without checking the actual artifacts.

## Pending policy decisions

The project has not yet established a final retention policy, storage budget, dataset licensing record, artifact checksum policy, model-weight policy, or archival destination. Escalate these choices instead of inventing them.
