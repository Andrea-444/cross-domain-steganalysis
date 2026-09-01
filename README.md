# Cross-Domain Steganalysis of Natural and AI-Generated Images

This project studies whether hidden information added to an already AI-generated image is detected differently from hidden information added to a natural image.

It is **not** an AI-generated-image detector. The classifier receives clean and steganographic images and predicts:

```text
0 = clean
1 = stego
```

The central questions are:

1. Are hidden payloads intrinsically easier or harder to detect in AI-generated covers?
2. Does a steganalyzer trained on natural images generalize to AI-generated covers?

The four controlled train-test directions are:

```text
AI   -> AI      primary: AI-domain detectability
real -> AI      primary: conventional training transferred to AI covers
real -> real    control: conventional same-domain performance
AI   -> real    control: reverse transfer
```

## Research Architecture

```text
Kaggle API / GenImage / local folders
                 |
                 v
balanced 256x256 grayscale PNG covers
                 |
         clean manifest + fixed split
                 |
                 v
conseal: WOW / S-UNIWARD / HILL / MiPOD
                 |
           stego manifest
                 |
                 v
SRNet train(real) + SRNet train(AI)
                 |
                 v
four cross-domain evaluations and metrics
```

Clean/stego pairs inherit the same `image_id` and split. This prevents a clean cover from appearing in training while its stego counterpart appears in testing.

## Project Layout

```text
src/
  pipeline/
    runner.py         end-to-end stage composition and persisted run state
    configuration.py typed non-interactive JSON configuration
    context.py        shared paths and stage execution records
    steps/            dataset, embedding, detector, and tracking stages
  datasets/
    pipeline.py       acquisition, flattening, balancing, and preprocessing
    manifest.py       stable IDs and pair-safe manifest validation
  steganography/
    core.py           conseal benchmarks and OpenStego demonstrations
  steganalysis/
    experiments.py    four-direction detector training and evaluation protocol
    srnet.py           adapter for the pinned SRNet implementation
    aletheia.py        optional Aletheia setup and screening adapter
  evaluation/
    metrics.py        detector metrics and repeated-seed summaries
  tracking/
    wandb.py          optional artifact-first W&B integration
  legacy/
    ...               original custom LSB, CNN, and sweep scripts
```

Each folder is a Python package. The primary command groups are:

```text
python -m src.datasets
python -m src.steganography
python -m src.steganalysis.experiments
python -m src.steganalysis.aletheia
python -m src.pipeline --config configs/pipeline.smoke.example.json
```

The first four commands remain available for focused stage work. The pipeline command composes them from one validated JSON configuration and saves `pipeline_config.json` and `pipeline_state.json` under a timestamped local run directory. W&B runs only after local state is saved and is disabled by default.

## Established Tools

### Research embedding

The primary benchmark uses [`conseal==2024.12`](https://conseal.readthedocs.io/) implementations of:

- WOW
- S-UNIWARD
- HILL
- MiPOD

Payloads are controlled in bits per pixel (`bpp`): `0.1`, `0.2`, or `0.4`.

`conseal` is an embedding simulator for steganalysis research. It models randomized secret bits at a controlled payload; it is not used to recover a particular text or image file.

### Deep steganalysis

The primary detector uses the established PyTorch SRNet implementation:

```text
repository: brijeshiitg/Pytorch-implementation-of-SRNet
commit: 2c838cf7d26dd0b8e45eda79ebf103ca312c0355
```

It is stored outside the project source under `external/srnet-reference`.

### Secondary steganalysis

The project includes an adapter for [`daniellerch/aletheia`](https://github.com/daniellerch/aletheia), pinned to `v0.3`. Aletheia provides established automatic detectors, rich-model feature extraction, and ensemble classifiers. Its additional Octave/model dependencies are installed separately.

### Recoverable demonstration

Text and complete image files can be embedded and recovered through the official [OpenStego](https://www.openstego.com/) RandomLSB command-line tool.

This is a demonstration mode and must not be mixed with the modern adaptive-algorithm benchmark.

### Legacy code

The original custom LSB, LSB-statistics classifier, and small CNN remain available in:

```text
src/legacy/stego_lsb.py
src/legacy/experiments.py
src/legacy/lsb_baseline.py
src/legacy/model.py
```

They are educational prototypes and are excluded from primary research results.

## Installation

The tested environment is the existing conda environment:

```bash
conda activate experimental_1
pip install -r requirements.txt
```

Clone the pinned SRNet reference implementation:

```bash
git clone https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet.git \
  external/srnet-reference
git -C external/srnet-reference checkout \
  2c838cf7d26dd0b8e45eda79ebf103ca312c0355
```

Optional Aletheia setup:

```bash
python -m src.steganalysis.aletheia setup-aletheia
```

Optional OpenStego setup:

```bash
python -m src.steganography setup-openstego
```

Java is required only for recoverable OpenStego demonstrations.

## Dataset Strategy

### Current Kaggle data

The current local folders are:

```text
src/raw_data/
  real/
    any nested folders...
  ai/
    any nested folders...
```

Nested folders are searched recursively and then ignored. The research dataset has only two domains: `real` and `ai`.

The current raw data is unbalanced and format-mismatched:

```text
real: 745 variable-resolution JPEG files
AI:   250 mostly 512x512 PNG files
```

Training directly on those files risks learning JPEG, resolution, or aspect-ratio differences. The preparation command balances the domains and applies identical center-crop, grayscale conversion, resizing, and PNG encoding:

```bash
python -m src.datasets prepare \
  --real-src src/raw_data/real \
  --ai-src src/raw_data/ai \
  --name kaggle \
  --source local-kaggle \
  --output data/processed \
  --size 256 \
  --seed 42
```

Output:

```text
data/processed/kaggle/
  real/
  ai/
data/processed/manifests/kaggle_clean.jsonl
```

### Programmatic Kaggle download

Configure either `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`, then run:

```bash
python -m src.datasets download \
  --source kaggle \
  --dataset owner/dataset-name \
  --output data/raw \
  --limit-per-domain 1000
```

If the downloaded structure contains recognizable `real`/`nature` and `ai`/`fake` directories, normalization starts automatically. Otherwise, run `src.datasets prepare` with explicit paths.

### What GenImage is

[GenImage](https://github.com/GenImage-Dataset/GenImage) is a million-scale dataset of natural ImageNet images and AI-generated images from generators including Stable Diffusion, Midjourney, ADM, GLIDE, Wukong, VQDM, and BigGAN.

GenImage does **not** contain steganographic images. This project uses it as a source of clean natural and AI covers, then applies the same `conseal` algorithm and payload to both domains.

The official Google Drive release is very large and each generator archive is split into many parts. The downloader therefore requires explicit confirmation:

```bash
python -m src.datasets download \
  --source genimage \
  --generator stable-diffusion-v1-4 \
  --split val \
  --limit-per-domain 1000 \
  --output data/raw \
  --accept-large-download
```

`--limit-per-domain` controls the processed subset, but the official archive still has to be downloaded before sampling. Check disk space first.

## Generate Research Stego Images

Example with WOW at `0.2 bpp`:

```bash
python -m src.steganography benchmark \
  --manifest data/processed/manifests/kaggle_clean.jsonl \
  --algorithm wow \
  --payload-bpp 0.2 \
  --seed 42 \
  --output stego_runs
```

Other algorithms:

```bash
--algorithm suniward
--algorithm hill
--algorithm mipod
```

Each run receives a timestamped folder:

```text
stego_runs/conseal_wow_p020_YYYYMMDD_HHMMSS/
  config.json
  dataset_manifest.jsonl
  real/stego/
  ai/stego/
```

The manifest printed by this command is the input to the experiment runner.

## Run Cross-Domain SRNet Experiments

```bash
python -m src.steganalysis.experiments \
  --manifest stego_runs/conseal_wow_p020_YYYYMMDD_HHMMSS/dataset_manifest.jsonl \
  --detector srnet \
  --protocol full \
  --seeds 1 2 3 4 5 \
  --epochs 5 \
  --batch-size 16 \
  --output results
```

Each result run contains:

```text
results/srnet_YYYYMMDD_HHMMSS/
  config.json
  dataset_manifest.jsonl
  summary.json
```

SRNet is computationally heavier than the original prototype. Start with one seed and one epoch to verify a new environment, then use the full configuration for actual results.

## Run The Composed Pipeline

Copy and edit the example configuration, then run it non-interactively:

```bash
cp configs/pipeline.smoke.example.json configs/pipeline.local.json
python -m src.pipeline --config configs/pipeline.local.json
```

The configuration can either point to existing clean and stego manifests or define how to prepare and embed them. A pipeline run records every stage as `running`, `completed`, or `failed`. Tracking supports `online`, `offline`, and `disabled` modes; enabling it requires the optional `wandb` package. A W&B failure is recorded but does not invalidate canonical local results.

## Metrics

The initial metrics are:

- **Accuracy:** proportion of all correct predictions.
- **Balanced accuracy:** average clean-class and stego-class recall.
- **Precision:** among images predicted as stego, the proportion that are stego.
- **Recall:** proportion of stego images detected.
- **F1:** harmonic mean of stego precision and recall.
- **Confusion matrix:**

```text
[[correct clean, clean falsely flagged as stego],
 [missed stego,  correctly detected stego]]
```

For multiple seeds, `summary.json` reports metric mean and standard deviation.

Primary interpretation:

- Compare `AI -> AI` with `real -> real` to study intrinsic domain difficulty.
- Compare `real -> AI` with `real -> real` to measure transfer from conventional training.
- Use `AI -> real` as the reverse-transfer control.

A result near `0.5` balanced accuracy indicates chance-level binary detection. A lower cross-domain result is evidence of domain mismatch only when the same algorithm, payload, preprocessing, split policy, and detector training procedure are used.

## Recoverable Text and Image Demonstrations

Embed a text file:

```bash
python -m src.steganography embed \
  --payload-type message \
  --payload-file secret.txt \
  --cover cover.png \
  --output text_stego.png
```

Embed an image file:

```bash
python -m src.steganography embed \
  --payload-type image \
  --payload-file secret.png \
  --cover cover.png \
  --output image_stego.png
```

Extract:

```bash
python -m src.steganography extract \
  --stego image_stego.png \
  --output-dir extracted
```

Set `OPENSTEGO_PASSWORD` to enable OpenStego encryption without placing a password directly in shell history.

## Aletheia Secondary Screening

After installing Aletheia and its external dependencies:

```bash
python -m src.steganalysis.aletheia aletheia-auto \
  --images path/to/images
```

This uses Aletheia's pretrained automatic screening. The controlled four-direction train-test matrix remains SRNet-based.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- recursive Kaggle folder flattening;
- domain balancing;
- normalized grayscale dimensions and formats;
- train/validation/test assignment;
- deterministic `conseal` embedding;
- ±1 adaptive embedding changes;
- pair-leakage detection;
- metric layout;
- reference SRNet forward pass;
- byte-for-byte text and image recovery through OpenStego.

## Current Scope and Limitations

- The first benchmark uses grayscale spatial steganography for compatibility with established spatial methods and SRNet.
- Color and JPEG-domain experiments are later extensions.
- GenImage download size is substantial.
- OpenStego RandomLSB is a recoverable demonstration, not a replacement for WOW, S-UNIWARD, HILL, or MiPOD.
- Aletheia pretrained predictions may suffer cover-source mismatch; they are secondary evidence, not a substitute for controlled training.
