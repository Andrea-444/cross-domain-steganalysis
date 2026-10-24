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

The six controlled train-test directions, in the order the accepted plan runs
them:

```text
real -> real    1. same-domain control
AI   -> AI      1. same-domain AI detectability
mixed -> real   2. mixed-domain control
mixed -> AI     2. mixed-domain control
real -> AI      3. transfer from conventional training
AI   -> real    3. reverse transfer
```

The `mixed` detector trains on an equal mixture of both domains, so neither
test domain is outside its training distribution. It separates two causes of a
poor `real -> AI` result: embedding traces that differ between the domains, and
a detector that is simply out of distribution. The mixed training set holds the
**same number of covers** as each single-domain set, half from each domain, so
data volume cannot explain any advantage.

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

### Embedding history and not repeating work

Pass `--cache-dir` so a condition is embedded once. Stego images are keyed by
backend, algorithm, payload, **cohort** and seed, so re-running a condition
reuses its images instead of regenerating them, and a different algorithm
writes to its own cache without disturbing the first.

The cohort component matters. Image IDs are stable across cohorts, so two
cohorts prepared differently from the same sources (for example `fit` versus
`crop`) share IDs while holding different pixels. Without the cohort key, one
cohort would silently reuse the other's stego images and pair a stego image
with a clean cover it was not derived from.

Review what has already been embedded:

```bash
python -m src.steganography ledger --runs data/stego/runs

# restricted to one cohort
python -m src.steganography ledger \
  --runs data/stego/runs \
  --manifest data/processed/manifests/<name>_clean.jsonl
```

The ledger is derived from the `config.json` each run writes, so it cannot
drift from what actually happened.

## Aggregate a Protocol Across Seeds

Each experiment command writes one run folder. When the protocol runs one seed
and one domain at a time, no folder holds a cross-seed number: every
`summary.json` reports `seeds: 1` and a standard deviation of 0.0. This reads
the folders and writes the means, spreads and paired gaps.

```bash
python -m src.evaluation.protocol --results results --output results/protocol/
```

It groups the runs by condition first. Two runs combine only when the manifest
hash, algorithm, payload, preprocessing, detector commit, learning rate, epoch
count and batch size all agree. A difference in any of them produces a separate
group, so a change of learning rate is reported and never averaged away.

Inside a group it merges **by seed**. Seed 1 of the protocol becomes one record
holding every direction that seed produced, across folders. That is what keeps
the paired gaps correct: a gap subtracts a cross-domain result from a
same-domain result inside one seed, and both terms must come from the same
trained model.

The report holds three parts:

- **training** — the escape epoch, the selected epoch against the budget, and
  the final train-minus-validation difference, for each trained model. Read
  this before the metrics. A detector that left the ln(2) plateau late trained
  for fewer effective epochs than its budget states, and a selected epoch at
  the end of the budget means the result is a lower bound.
- **directions** — the mean and spread across seeds, for each direction.
- **paired gaps** — each gap averaged over the seeds that produced both of its
  terms.

Two cases it reports rather than hides. A direction that not every seed
produced is **left out**, because a mean over a different seed count than its
neighbours misleads. The same seed and direction appearing in two folders is a
**conflict**: the later folder wins, and the report names both.

## Diagnose a Detector Run

When a detector run returns chance accuracy, these checks narrow the cause before
any GPU time is spent again. Each one writes a JSON report. Copy a report into
`evidence/` by hand when it is worth keeping.

```bash
# 1. Do the stego images differ from their covers, and by how much?
python -m src.steganalysis.diagnostics embedding \
  --manifest <stego-run>/dataset_manifest.jsonl \
  --output results/diagnostics/

# 2. Which preprocessing produced the covers behind this run?
python -m src.steganalysis.diagnostics provenance \
  --run-config <stego-run>/config.json

# 3. Can a classical feature set find the embedding?
#    Run the control first: --synthetic replaces the stego images with a
#    non-adaptive embedding that any working detector finds.
python -m src.steganalysis.diagnostics classical \
  --manifest <stego-run>/dataset_manifest.jsonl --synthetic
python -m src.steganalysis.diagnostics classical \
  --manifest <stego-run>/dataset_manifest.jsonl

# 4. Does the detector keep its sensitivity to the embedding while it trains?
python -m src.steganalysis.diagnostics collapse \
  --manifest <stego-run>/dataset_manifest.jsonl --steps 200

# 5. Can the detector learn an embedding that is easy to see?
python -m src.steganalysis.diagnostics synthetic-train \
  --manifest <stego-run>/dataset_manifest.jsonl --epochs 60
```

Read them in this order. Checks 1 and 2 test the data, check 3 tests the data
without the detector, and checks 4 and 5 test the detector. A check that tests
the detector means nothing until the checks before it have passed.

Two readings carry most of the weight:

- **`classical`** reports `test_roc_auc` with `roc_auc_chance_error` and
  `roc_auc_z_score` beside it. A z-score above 2 means the area under the curve
  is far enough from 0.5 to take seriously. Always run `--synthetic` first: it
  says whether the feature code works, and without it a result of 0.5 has two
  possible meanings.
- **`collapse`** reports `pair_gap_ratio`, which is how much of the output of
  the detector follows the embedding rather than the picture. A ratio that holds
  or rises means the detector keeps its sensitivity. A ratio that falls means it
  is learning to answer the same thing for every image, and more training time
  does not repair that.

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

### Staged runs

`--directions` runs a subset of the six directions, so a long protocol can be
split into stages that each produce their own result folder and their own W&B
run. The default is all six directions.

```bash
# Stage 1: the natural-cover detector
python -m src.steganalysis.experiments --manifest <manifest> \
  --directions real_to_real real_to_ai --seeds 1 2 3 --output results

# Stage 2: the AI-cover detector
python -m src.steganalysis.experiments --manifest <manifest> \
  --directions ai_to_ai ai_to_real --seeds 1 2 3 --output results

# Stage 3: the mixed-domain control
python -m src.steganalysis.experiments --manifest <manifest> \
  --directions mixed_to_real mixed_to_ai --seeds 1 2 3 --output results
```

Each train domain supplies two directions from one trained model, so keep both
directions of a domain in the same stage. A stage that splits them trains the
same model twice and reports no paired generalization gap, because a gap needs
the same-domain and cross-domain result from one seed of one model.

Because the reference model ends in `LogSoftmax`, the retained probability is a
genuine `P(stego)`, so `summary.json` also reports ROC-AUC and PR-AUC. Both are
`null` for a direction whose test labels contain only one class.

The training device is chosen in the order explicit `--device`, CUDA, MPS, CPU.
On Apple Silicon this selects the GPU: an SRNet training step measured roughly
nineteen times faster on MPS than on CPU on the reference machine.

Checkpoint selection uses validation loss. Validation accuracy and balanced
accuracy are recorded per epoch but do not currently affect selection.

SRNet is computationally heavier than the original prototype. Start with one seed and one epoch to verify a new environment, then use the full configuration for actual results.

The primary mixed-domain control deliberately uses the same total number of
training pairs as a specialist. To test whether more mixed-domain data helps,
use an explicit, separately reported full-union condition:

```bash
python -m src.steganalysis.experiments --manifest <manifest> \
  --directions mixed_to_real mixed_to_ai --seeds 1 --epochs 300 \
  --batch-size 32 --learning-rate 1e-4 \
  --mixed-pairs-per-domain 3887 --output results
```

This uses 3,887 natural and 3,887 AI pairs, or 7,774 total. The option cannot
be combined with `--train-pairs`, and it fails if either domain is too small
instead of silently changing the requested condition.

### Visualize an SRNet run with W&B

W&B is optional. Install and authenticate it on the machine that performs the
training; do not put an API key in the repository or in the experiment command:

```bash
source .venv/bin/activate
pip install wandb
wandb login
```

Enable live tracking by adding these flags to the SRNet command:

```bash
python -m src.steganalysis.experiments \
  --manifest data/stego/runs/conseal_suniward_p040_<timestamp>/dataset_manifest.jsonl \
  --seeds 1 2 3 \
  --epochs 30 \
  --batch-size 32 \
  --output results \
  --wandb-mode online \
  --wandb-project stego-cross-domain \
  --wandb-run-name suniward-p040 \
  --wandb-tags suniward p040 srnet
```

### Run names in W&B

The run name, tags and group are **derived from the run**, not typed. A name
that was typed can disagree with what ran; a derived one cannot.

```text
<role>-<domains>-<algorithm><payload>-<preprocess>-lr<rate>-s<seeds>-<stamp>

pilot-real-suniward040-crop-lr1e-4-s1-20261014_153201
stage-ai-suniward040-crop-lr1e-4-s1-2-3-20261015_090412
full-real+ai+mixed-suniward040-crop-lr1e-4-s1-2-3-20261016_221015
```

The stamp is the timestamp of the results folder the run wrote, so
`...-20261014_153201` wrote `results/srnet_20261014_153201/`. It also separates
two runs that used identical settings, which the rest of the name cannot do. A
plain date would not: two runs on one day collide.

| Part | Where it comes from |
|---|---|
| `role` | `pilot` for one seed, `full` for all six directions, otherwise `stage` |
| `domains` | the domains actually trained, joined by `+` |
| `algorithm`, `payload` | the embedding run's `config.json`, beside the manifest |
| `preprocess` | the preparation config, two steps back along the manifest chain |
| `rate` | `--learning-rate` |
| `seeds` | `--seeds` |
| `stamp` | the results folder this run wrote |

A part that cannot be read prints `unknown` rather than being omitted, so a
missing chain is visible in the name instead of silently absent.

**Tags** carry one fact each, as `key:value`: `role:`, `detector:`,
`algorithm:`, `payload:`, `preprocess:`, `lr:`, `seeds:`, one `domain:` for
each trained domain and one `direction:` for each evaluated direction. Tags
passed with `--wandb-tags` are added to these, not instead of them.

**Group** is `<algorithm><payload>-<preprocess>-<first 8 of the manifest
SHA-256>`, for example `suniward040-crop-f271f135`. Stages run as separate
commands, and the group is what puts them back together. The manifest hash is
in the group, so two cohorts with the same settings never merge.

`--wandb-run-name` still overrides the derived name. Use it only when you need
a name the convention cannot express, and expect to explain the name later.

### Where each number goes in W&B

Three places, three kinds of number. W&B draws a panel for every key in a run's
**history**, so a final metric logged there becomes a chart with a single bar.

| Place | Written with | Holds | Why |
|---|---|---|---|
| history | `run.log` | the training curves | they change over epochs, so a chart shows something |
| summary | `run.summary.update` | the aggregate for each direction, and each paired gap | columns of the runs table, which compare **across** runs |
| table | `evaluation/results` | one row for each seed and direction, every metric and the four confusion cells | per-seed detail, in the shape detail belongs in |

A single-seed, two-direction run used to create 137 panels. It now creates 8,
and 95 values move to the runs table instead. A full three-seed, six-direction
run would have created about 615.

To compare runs, add the summary keys as columns in the runs table, or build a
bar chart across runs from one of them. A panel inside one run cannot make that
comparison, because it holds one value.

One W&B run represents the complete command. It records separate training and
validation curves for every seed and training domain, per-seed metrics for all
six evaluation directions, an evaluation table with confusion-matrix cells,
and aggregate mean/standard-deviation metrics. Local `summary.json`, histories,
predictions, and checkpoints remain the canonical results.

Use `--wandb-mode offline` when the training host cannot reach W&B. Upload the
result later with the `wandb sync` command printed by W&B. Add
`--wandb-upload-artifact` only when the manifest and result metadata should also
be copied to W&B; checkpoints and image files are never uploaded by this flag.
If W&B initialization or logging fails, training continues and the failure is
recorded in the local `summary.json`.

## Run The Composed Pipeline

Copy and edit the example configuration, then run it non-interactively:

```bash
cp configs/pipeline.smoke.example.json configs/pipeline.local.json
python -m src.pipeline --config configs/pipeline.local.json
```

The configuration can either point to existing clean and stego manifests or define how to prepare and embed them. A pipeline run records every stage as `running`, `completed`, or `failed`. Tracking supports `online`, `offline`, and `disabled` modes; enabling it requires the optional `wandb` package. A W&B failure is recorded but does not invalidate canonical local results.

## Cover-Domain Confound Audit

The cross-domain claim requires that natural and AI covers differ only in how
they were generated. If they also differ in compression history or resampling
ratio, a steganalyzer can learn that instead. Audit a clean manifest before
using it for any experiment:

```bash
python -m src.datasets audit-domains \
  --manifest data/processed/manifests/<name>_clean.jsonl \
  --permutations 200 \
  --output results/audits/<name>_domain_separability.json
```

The audit uses clean covers only, with no stego images. It scores three
features per cover (8x8 blockiness contrast, Laplacian high-pass energy,
neighbour-residual spread) and cross-validates a classifier predicting the
domain. Balanced accuracy near 0.5 with a non-significant permutation p-value
means the domains are indistinguishable before embedding. A significant result
means the pair is confounded and no cross-domain number from it is trustworthy.

This is a necessary check, not a sufficient one: it uses three hand-crafted
features, while SRNet may exploit differences they do not capture. On the
first cohort audited it **did** give a false all-clear, so treat it as a cheap
screen and confirm any pass with the SRNet probe below.

### Rebalancing after skipped covers

Crop-only cannot fill the target size from a smaller source, and the loss is
one-sided: natural covers vary in resolution while AI covers are uniformly
large. On the 6,000-per-domain ImageNet/GenImage++ cohort, 170 natural covers
were below 256 px on the short side, leaving 5,830 natural against 6,000 AI and
98 of 1,000 classes unmatched.

```bash
python -m src.datasets rebalance-cohort \
  --manifest data/processed/manifests/<name>_clean.jsonl \
  --name <name>-balanced --output data/processed
```

Matching is applied per class **and** per split, since splits are locked at
selection time and a cover must never move between them. This is a
manifest-only operation: no image is read or written, and `clean_path` values
are unchanged.

Unequal domain totals matter beyond the recorded class-matching rule. They
would give the two detectors different amounts of training data, so a
`real_to_real` versus `ai_to_ai` difference could reflect training-set size
rather than domain difficulty.

### SRNet domain probe

Ask the detector itself the same question, using clean covers only:

```bash
python -m src.steganalysis.domain_probe \
  --manifest data/processed/manifests/<name>_clean.jsonl \
  --seeds 1 2 3 --epochs 30 --batch-size 16 \
  --output results/audits/srnet-probe-<name>
```

Always pair a near-chance result with a positive-control cohort the probe does
detect; otherwise the null may only mean the probe lacked power. Measured on
the 100-per-domain cohort with three seeds:

| Cohort | 3-feature balanced accuracy | SRNet balanced accuracy | SRNet ROC-AUC |
|---|---|---|---|
| `fit` (crop + resample) | 0.630 | 0.745 | 0.843 |
| `crop` (native crop) | 0.532 | 0.675 | 0.766 |

Crop-only reduces domain separability but does not remove it. Equalizing
compression does not reduce it further:

| Cohort | 3-feature | SRNet balanced accuracy | SRNet ROC-AUC |
|---|---|---|---|
| `fit` (crop + resample) | 0.630 | 0.745 (sd 0.050) | 0.843 |
| `crop` (native crop) | 0.532 | 0.675 (sd 0.056) | 0.766 |
| `crop` + JPEG q96 on AI | 0.502 | 0.661 (sd 0.049) | 0.798 |
| `crop` + JPEG q96 on both | 0.497 | 0.688 (sd 0.086) | 0.751 |

Removing the resampling mismatch mattered; equalizing compression did not. The
residual separability therefore appears intrinsic to the generated images
rather than an artifact of preparation. Note how far the three-feature screen
diverges from the probe on the recompressed variants: it reports chance where
SRNet reports two-thirds accuracy.

Compression history is equalized with:

```bash
python -m src.datasets recompress-cohort \
  --manifest data/processed/manifests/<name>_clean.jsonl \
  --name <variant-name> --output data/processed \
  --quality 96 --domains ai
```

JPEG history cannot be removed from covers that already have it, so the only
available direction is to add matching history to the other domain. The output
is stored as lossless PNG; only the pixel content carries the quantization.

### Preprocessing modes

`prepare-selection` accepts `--preprocess fit` (default: crop to aspect, then
resample with LANCZOS) or `--preprocess crop` (centred native crop, no
resampling). Resampling attenuates the high-frequency residuals steganalysis
depends on, and domains whose sources differ in resolution are attenuated by
different amounts, which is itself a confound.

On the 100-per-domain ImageNet/GenImage++ cohort this mattered. `fit` crops to
the target aspect ratio using the short side, then resamples, so the effective
factor for natural covers was a median of 1.46x (mean 1.66x, range 0.66x to
7.59x) against exactly 4x for the uniformly 1024x1024 AI covers. A factor below
1.0 means the cover was *enlarged*, and interpolation invents high-frequency
content rather than attenuating it:

| Preprocessing | Balanced accuracy | ROC-AUC | Permutation p |
|---|---|---|---|
| `fit` (crop + resample) | 0.630 | 0.691 | 0.0050 |
| `crop` (native crop) | 0.532 | 0.567 | 0.2736 |

Crop-only requires every source image to be at least the target size. Use
`--skip-undersized` to exclude smaller images; they are recorded in the run
configuration rather than silently dropped.

## Metrics

The initial metrics are:

- **Accuracy:** proportion of all correct predictions.
- **Balanced accuracy:** average clean-class and stego-class recall.
- **Precision:** among images predicted as stego, the proportion that are stego.
- **Recall:** proportion of stego images detected.
- **F1:** harmonic mean of stego precision and recall.
- **Specificity (true negative rate):** proportion of clean images called clean.
- **False positive rate:** proportion of clean images called stego. Equals
  `1 - specificity`.
- **False negative rate:** proportion of stego images called clean.
- **Detection error `P_E`:** `0.5 * (FPR + FNR)`. The standard single figure in
  the steganalysis literature. Lower is better, and `0.5` is chance.
- **Matthews correlation coefficient:** a balanced single figure from all four
  cells of the confusion matrix. `1.0` is perfect, `0.0` is chance, and `-1.0`
  is perfectly inverted. It stays informative when the classes are uneven.
- **Equal error rate:** the rate where the false positive and false negative
  rates meet on the ROC curve. It needs probabilities, and it is interpolated
  between the two points that bracket the crossing.
- **Confusion matrix:**

```text
[[correct clean, clean falsely flagged as stego],
 [missed stego,  correctly detected stego]]
```

A rate with a zero denominator is `null`, not `0.0`. A direction with no actual
negatives has no specificity, and reporting `0.0` would read as a perfect
score. `P_E` is `null` when either of its two rates is. The equal error rate is
`null` when the labels hold a single class, because the ROC curve is undefined.

### Paired domain-generalization gaps

`summary.json` carries a `generalization_gaps` block:

```text
real_to_real_minus_real_to_ai     kind: transfer
ai_to_ai_minus_ai_to_real         kind: transfer
mixed_to_real_minus_mixed_to_ai   kind: test_domain
```

Each gap is `first - second`, **computed inside one seed and only then
averaged**, so the same trained model supplies both terms. Differencing the
means instead would discard the pairing and let seed-to-seed variation inflate
the spread. Each entry reports the per-seed values, their mean, their standard
deviation, the two direction names and its `kind`.

The two kinds answer different questions, and the difference matters.

**`transfer`** holds the detector fixed and moves the test domain away from
its own. It measures what the detector loses when the cover source changes.

**`test_domain`** holds fixed a detector that trained on both domains, and
moves the test domain between them. Neither domain is foreign to it and
neither carries a training advantage, so the difference isolates the cover
domain itself.

That isolation is why the mixed pair exists. Comparing `real_to_real` with
`ai_to_ai` changes the training domain **and** the test domain at once, so a
difference between them mixes the difficulty of the cover domain with how much
each detector gained from its own training data. The mixed pair changes one
thing.

Read the sign with the metric in mind. For a metric where higher is better,
such as balanced accuracy, a positive `transfer` gap means the detector
degraded across domains, and a positive `test_domain` gap means the second
domain is the harder one. For an error metric such as `detection_error` or
`eer`, the sign reverses.

For multiple seeds, `summary.json` reports metric mean and standard deviation.

Primary interpretation:

- Compare `AI -> AI` with `real -> real` for domain difficulty, and read it
  together with `mixed_to_real_minus_mixed_to_ai`. The first changes the
  training domain and the test domain together; the second changes only the
  test domain. If they disagree, the training domain is carrying part of the
  difference.
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
