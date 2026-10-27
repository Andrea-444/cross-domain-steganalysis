# Cross-Domain Steganalysis of Natural and AI-Generated Images

This project studies whether steganographic content is detected differently
in natural and AI-generated images. It is a binary steganalysis task:

```text
0 = clean image
1 = stego image
```

It is **not** an AI-generated-image detector.

## Research questions

1. Is the same hidden payload easier or harder to detect in AI-generated
   covers than in natural covers?
2. Does a detector trained on one cover domain generalize to the other?
3. Does mixed-domain training improve robustness across both domains?

The controlled protocol evaluates six train-test directions:

| Training domain | Test domain | Purpose |
|---|---|---|
| Natural | Natural | Natural same-domain control |
| AI-generated | AI-generated | AI same-domain detectability |
| Natural | AI-generated | Natural-to-AI transfer |
| AI-generated | Natural | Reverse-transfer control |
| Mixed | Natural | Mixed-domain control |
| Mixed | AI-generated | Mixed-domain control |

Clean/stego pairs always remain in the same train, validation, or test split.
Natural and AI-generated images receive equivalent preprocessing.

## Current experimental setup

- **Natural covers:** ImageNet validation images.
- **Primary AI covers:** GenImage++ FLUX images, class-matched to ImageNet.
- **External AI evaluation:** Stable Diffusion 3 and FLUX Realistic subsets.
- **Preprocessing:** 256×256 grayscale PNG images.
- **Primary detector:** SRNet, using the pinned reference implementation at
  commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`.
- **Embedding conditions:** S-UNIWARD at 0.4 and 0.2 bpp, followed by MiPOD at
  0.4 bpp.
- **Tracking:** local JSON/JSONL artifacts, selected checkpoints, predictions,
  and optional Weights & Biases logging.

The repository also supports WOW, HILL, optional Aletheia screening, and an
OpenStego demonstration. These are not all part of the current primary
experiment.

## Repository structure

```text
src/
  datasets/        acquisition, class matching, selection and preprocessing
  steganography/   adaptive embedding and stego-manifest generation
  steganalysis/    SRNet training, checkpoint evaluation and diagnostics
  evaluation/      metrics and cross-seed aggregation
  pipeline/        optional composed pipeline runner
  tracking/        run naming and optional W&B integration
  legacy/          historical prototypes, excluded from primary findings
tests/              unit and integration tests
configs/            example pipeline configurations
evidence/           small, deliberately retained research evidence
notes/              experiment decisions, run reviews and evolving findings
```

Generated datasets, models, checkpoints and result folders are intentionally
excluded from Git.

## Installation

The project currently uses the existing `experimental_1` conda environment:

```bash
conda activate experimental_1
pip install -r requirements.txt
```

Clone the pinned SRNet implementation:

```bash
git clone https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet.git \
  external/srnet-reference
git -C external/srnet-reference checkout \
  2c838cf7d26dd0b8e45eda79ebf103ca312c0355
```

Large datasets and external tools are downloaded separately and remain
untracked. Run a command with `--help` to see its complete options.

## Main workflow

### 1. Prepare a controlled cover dataset

The dataset tools can acquire or index source images, create a deterministic
class-matched selection, extract it, and normalize both domains. The main
entry point is:

```bash
python -m src.datasets --help
```

The resulting clean manifest is the input to the embedding stage.

### 2. Generate stego images

Example: S-UNIWARD at 0.4 bpp.

```bash
python -m src.steganography benchmark \
  --manifest data/processed/manifests/<clean-manifest>.jsonl \
  --algorithm suniward \
  --payload-bpp 0.4 \
  --seed 42 \
  --output data/stego/runs \
  --cache-dir data/stego/cache
```

The command writes a timestamped run and a paired `dataset_manifest.jsonl`.

### 3. Train and evaluate SRNet

```bash
python -m src.steganalysis.experiments \
  --manifest data/stego/runs/<embedding-run>/dataset_manifest.jsonl \
  --detector srnet \
  --protocol full \
  --seeds 1 2 3 \
  --epochs 300 \
  --batch-size 32 \
  --output results
```

Long runs can be split by training domain with `--directions`. Keep both test
directions for one trained model together, for example
`real_to_real real_to_ai`.

### 4. Evaluate saved checkpoints on another cover set

```bash
python -m src.steganalysis.evaluate \
  --manifest <external-stego-manifest>.jsonl \
  --checkpoints <checkpoint-a>.pt <checkpoint-b>.pt \
  --test-domain ai \
  --test-set-name <name> \
  --batch-size 32 \
  --output results
```

This evaluates existing models without retraining and records per-image
predictions plus aggregate metrics.

### 5. Aggregate repeated seeds

```bash
python -m src.evaluation.protocol \
  --results results \
  --output results/protocol
```

## Outputs and reproducibility

Research runs retain the applicable configuration, copied input manifest,
embedding condition, seeds, selected checkpoints, metric summaries and
predictions. Results should be interpreted by direction; a single combined
score would hide the domain-generalization question.

Detailed decisions and findings are maintained in:

- [`notes/Methods and Findings.md`](notes/Methods%20and%20Findings.md)
- [`notes/Run Reviews.md`](notes/Run%20Reviews.md)
- [`notes/Decisions.md`](notes/Decisions.md)
- [`notes/Architecture.md`](notes/Architecture.md)

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Some tests are skipped when optional SRNet or OpenStego dependencies are not
installed.

## Scope and limitations

- Current primary experiments use grayscale spatial steganography.
- Color and JPEG-domain experiments are outside the present scope.
- External AI datasets test generator sensitivity but do not establish
  generator-independent conclusions.
- OpenStego RandomLSB is a recoverable demonstration, not evidence for the
  adaptive-steganography benchmark.
- Datasets and external model implementations retain their own licenses and
  terms of use.

## License

The original source code in this repository is available under the
[MIT License](LICENSE). This license does not replace the licenses of external
datasets, models, or third-party tools.
