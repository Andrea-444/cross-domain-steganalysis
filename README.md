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

### Candidate indexing and incremental extraction

The current working candidates are GenImage++ FLUX for AI covers and ImageNet ILSVRC validation images for natural covers. This pair is not frozen until the native-image audit passes. The older `download --source genimage` command targets the original GenImage release and is not the GenImage++ acquisition path.

### Acquiring the source data

Both cover domains come from Hugging Face, and each needs a different access
shape.

**AI covers — GenImage++ (`Lunahera/genimagepp`).** Freely downloadable under
CC-BY-NC-4.0. The repository holds roughly 204 GB across all generators, but the
working pair needs only `archives/flux.tar.zst` (7.22 GB):

```bash
python -m src.datasets fetch-archive \
  --repo Lunahera/genimagepp \
  --file archives/flux.tar.zst \
  --output data/raw/genimagepp
```

An archive already present is reused rather than downloaded again, so the file
stays immutable for checksum-guarded extraction.

**Natural covers — ImageNet-1k (`ILSVRC/imagenet-1k`).** This dataset is
**gated**: sign in to Hugging Face, accept the ImageNet Terms of Access on the
dataset page, then authenticate locally with `hf auth login`. Streaming
transfers only the rows consumed, so a class-balanced sample costs a small
fraction of the 50,000-image validation split:

```bash
python -m src.datasets fetch-images \
  --repo ILSVRC/imagenet-1k \
  --split validation \
  --limit-per-class 10 \
  --max-classes 100 \
  --output data/raw/imagenet-val
```

Images land in one directory per Hugging Face class label. Build exact
synset maps from the pinned Hugging Face label order, the ImageNet simple-label
order, and the official class index:

```bash
python -m src.datasets fetch-labels \
  --repo ILSVRC/imagenet-1k \
  --split validation \
  --revision <pinned-revision> \
  --output data/raw/imagenet-hf-labels.json

python -m src.datasets map-genimagepp \
  --real-src data/raw/imagenet-val \
  --ai-src data/raw/genimagepp/flux.tar.zst \
  --imagenet-labels data/raw/imagenet-hf-labels.json \
  --simple-labels data/raw/imagenet-simple-labels.json \
  --class-index data/raw/imagenet_class_index.json \
  --output data/metadata/imagenet-genimagepp
```

The mapping joins the ordered metadata by class index and requires exact label
equality; it does not use fuzzy class-name matching. Pass the generated JSONL
maps to `index` with `--no-parent-class`.

Both commands write the **original encoded bytes**. Decoding and re-saving an
image would add a second JPEG generation, and those artifacts sit in the same
high-frequency band that steganalysis measures.

Download each source archive once, then index its image members without normalizing or extracting the whole corpus:

```bash
python -m src.datasets index \
  --input data/raw/genimagepp/flux.tar.zst \
  --dataset-id genimagepp-flux \
  --dataset-version hf-4160fbc \
  --domain ai \
  --output data/indexes/genimagepp-flux.jsonl

python -m src.datasets index \
  --input data/raw/imagenet/ILSVRC2012_img_val.tar \
  --dataset-id imagenet-ilsvrc2012-val \
  --dataset-version ilsvrc2012 \
  --domain real \
  --class-map data/metadata/imagenet-val-class-map.csv \
  --class-map-member-column filename \
  --class-map-class-column class_id \
  --no-parent-class \
  --output data/indexes/imagenet-val.jsonl
```

The class-map example represents a normalized `filename,class_id` file that must be derived and verified from the official ImageNet metadata. If an archive already stores images under class directories, parent-directory class inference is the default.

Create one immutable, class-count-matched selection before extracting anything:

```bash
python -m src.datasets select \
  --real-index data/indexes/imagenet-val.jsonl \
  --ai-index data/indexes/genimagepp-flux.jsonl \
  --count-per-domain 6000 \
  --audit-count 100 \
  --pilot-total 500 \
  --seed 42 \
  --output data/selections/working-pair.jsonl
```

Materialize cumulative stages from that same selection:

```bash
python -m src.datasets extract --selection-manifest data/selections/working-pair.jsonl --stage audit --output data/raw/selected
python -m src.datasets extract --selection-manifest data/selections/working-pair.jsonl --stage pilot --output data/raw/selected
python -m src.datasets extract --selection-manifest data/selections/working-pair.jsonl --stage full --output data/raw/selected
```

Normalize the extracted locked selection without changing its class or split
assignments:

```bash
python -m src.datasets prepare-selection \
  --extraction-manifest data/raw/selected/extraction_manifest.jsonl \
  --name working-pair \
  --output data/processed \
  --size 256
```

`audit` contains 100 images per domain. `pilot` reuses those files and adds 400 per domain. `full` reuses both earlier stages and adds the remaining selected images. Content checksums prevent silent replacement or duplicate extraction. Final test members are assigned from the beginning and never enter audit or pilot cohorts.

For active experiments, keep the selected native corpus on a local or external NVMe filesystem. Compressed archives are suitable for cold storage; network/object storage is better used as a source or backup than as the direct image-training path.

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
  summary.json                      per-seed metrics, aggregates, and provenance
  history/seed<N>_<domain>.json     per-epoch training and validation metrics
  checkpoints/seed<N>_<domain>.pt   selected weights and selection metadata
  predictions/seed<N>_<direction>.jsonl   per-image label, prediction, probability
```

`summary.json` keeps each seed's metrics under `runs` alongside the aggregate
`summary`, and records provenance: manifest SHA-256, dataset counts by domain and
split, embedding condition read from the manifest, detector commit, repository
commit and dirty flag, resolved device, platform, dependency versions, and timing.

Because the reference model ends in `LogSoftmax`, the retained probability is a
genuine `P(stego)`, so `summary.json` also reports ROC-AUC and PR-AUC. Both are
`null` for a direction whose test labels contain only one class.

The training device is chosen in the order explicit `--device`, CUDA, MPS, CPU.
On Apple Silicon this selects the GPU: an SRNet training step measured roughly
nineteen times faster on MPS than on CPU on the reference machine.

Checkpoint selection uses validation loss. Validation accuracy and balanced
accuracy are recorded per epoch but do not currently affect selection.

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
python -m unittest discover -s tests -t . -v
```

The `-t .` flag sets the top-level directory to the repository root. Without
it, discovery puts `tests/` itself on `sys.path` and `tests/datasets/`
shadows the installed `datasets` library, which breaks the Hugging Face
acquisition imports.

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
