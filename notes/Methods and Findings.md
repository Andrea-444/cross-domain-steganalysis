---
type: thesis-source-record
status: active
updated: 2026-10-07
audience: thesis-writing
---

# Methods and Findings

## Why this note exists

[[Development Log]] is chronological and [[Decisions]] records choices without
their supporting numbers. This note is the **consolidated source record for
thesis writing**: the methodological narrative, the measured evidence behind
each decision, and the known threats to validity, in one place.

Per `agent-instructions/RESULTS_AND_WRITING.md`, **measured observations and
interpretation are kept separate throughout**. Anything labelled
*Interpretation* is an inference, not a result.

> [!warning] No experimental result exists yet
> Every number below comes from **methodological validation**, not from the
> research experiment. The four-direction clean-versus-stego protocol has never
> been run on audited data. Nothing here answers the thesis questions; it
> establishes whether the dataset and pipeline can answer them.

## 1. Cover-domain confound investigation

### 1.1 The problem

The thesis compares steganalysis across natural and AI-generated covers. The
comparison is only valid if the domains differ in *how the images were
produced*, not in incidental processing history. Both candidate sources differ
in ways that touch the exact signal a steganalyzer reads:

| Property | Natural (ImageNet ILSVRC val) | AI (GenImage++ FLUX) |
|---|---|---|
| Format | JPEG, lossy | PNG, lossless |
| Source JPEG quality | median 96 (86 of 100 covers at 96) | not applicable |
| Dimensions | 50 distinct in a 100-cover sample, mostly 500x375 | uniformly 1024x1024 |
| Downsample factor to 256 under `fit` | median 1.46x, mean 1.66x, range 0.66x to 7.59x | exactly 4.00x |

Resampling attenuates high-frequency residuals; unequal factors attenuate the
domains unequally. JPEG quantization disturbs the same band.

`ImageOps.fit` crops to the target aspect ratio using the short side and then
resamples, so the factor is governed by the short side rather than the long
one. Two consequences are easy to miss. A factor below 1.0 means the cover was
**enlarged**, and interpolation invents high-frequency content rather than
attenuating it; 100-cover measurement puts the natural minimum at 0.66x. And
the factor ranged from 0.66x to 7.59x **within** the natural domain, so `fit`
applied inconsistent resampling across natural covers as well as unequal
resampling between domains.

### 1.2 Method

Two instruments, both using **clean covers only, with no stego images present**,
predicting the cover's domain:

1. **Feature screen** (`src/datasets/audit.py`): three interpretable features
   per cover (8x8 blockiness contrast, Laplacian high-pass energy,
   neighbour-residual spread), cross-validated logistic regression, with a
   label-shuffle permutation test.
2. **SRNet probe** (`src/steganalysis/domain_probe.py`): the project's actual
   detector, trained on clean covers labelled by domain.

Chance is 0.5 balanced accuracy. A near-chance probe result is only meaningful
alongside a positive-control cohort the probe *does* detect, since a null may
instead mean the probe lacked power.

### 1.3 Measured results

Cohort: 100 covers per domain, 100 ImageNet classes, split 66/17/17 per domain.
SRNet probe: 3 seeds, 30 epochs, batch 16, MPS. Test split 33-34 covers.

| Cohort variant | Feature screen | Permutation p | SRNet balanced acc. | sd | SRNet ROC-AUC |
|---|---|---|---|---|---|
| `fit` (crop + LANCZOS resample) | 0.6300 | 0.0050 | **0.7451** | 0.0500 | 0.8431 |
| `crop` (native centre crop) | 0.5324 | 0.2736 | **0.6752** | 0.0555 | 0.7659 |
| `crop` + JPEG q96 on AI only | 0.5020 | 0.5423 | **0.6605** | 0.0491 | 0.7978 |
| `crop` + JPEG q96 on both | 0.4970 | 0.5970 | **0.6875** | 0.0864 | 0.7512 |

Per-seed SRNet balanced accuracy:

- `fit`: 0.6765, 0.7941, 0.7647
- `crop`: 0.6324, 0.6397, 0.7537
- `crop` + JPEG q96 on AI: 0.6415, 0.6121, 0.7279
- `crop` + JPEG q96 on both: 0.7904, 0.6930, 0.5790

Feature means under `fit` versus `crop`:

| Feature | `fit` real | `fit` ai | `crop` real | `crop` ai |
|---|---|---|---|---|
| Blockiness contrast | 0.0042 | 0.0130 | 0.0048 | 0.0008 |
| High-pass energy | 45.77 | 34.96 | 49.57 | 42.51 |
| Residual spread | 11.44 | 8.74 | 12.39 | 10.63 |

Artifacts: `results/audits/srnet-probe-{fit,crop,smoke100-crop-jpegai96,smoke100-crop-jpegboth96}/summary.json`
and `results/audits/imagenet-flux-smoke100{,-crop}_domain_separability.json`.

### 1.4 Interpretation

- Removing resampling reduced separability by about 0.07 balanced accuracy.
  **The resampling mismatch was a genuine, measurable confound.**
- The three compression variants (0.6752, 0.6605, 0.6875) lie within roughly
  one standard deviation of each other. **Equalizing compression history had no
  measurable effect.**
- The residual separability of about 0.675 therefore does not appear to come
  from compression history. The most likely remaining explanation is intrinsic
  pixel statistics of the generated images.
  **This is an inference from one failed intervention, not a demonstration.**
- The AI covers' elevated blockiness under `fit` (0.0130) collapses under
  `crop` (0.0008). An earlier hypothesis attributing it to FLUX's 8x8 VAE latent
  structure is **contradicted**: a native crop would have preserved a generator
  artifact. It was produced by the exact-4x LANCZOS downsampling.

### 1.5 Methodological finding worth reporting

**The feature screen produced false negatives.** It scored the three
crop-based variants at 0.5324, 0.5020 and 0.4970 with permutation p-values of
0.27, 0.54 and 0.60 — all reading as chance — while SRNet separated the same
covers at 0.6752, 0.6605 and 0.6875. Hand-crafted features cannot bound what a
deep network will find, so a cheap screen is not a substitute for probing with
the detector actually used. Recorded as an accepted decision in [[Decisions]].

## 2. Threats to validity

Addresses the open [[Tasks]] item "Record threats to validity and known
confounders."

### Not yet excluded as causes of the residual domain separability

- **Semantic content.** Class matching is by ImageNet synset, but FLUX renders
  a prompt while ImageNet photographs a scene. Composition, framing and object
  scale may differ systematically within a shared class label.
- **Generation resolution.** FLUX emits 1024x1024; a native 256 crop samples a
  smaller field of view from the AI covers than from a 500x375 natural image.
- **Tonal statistics under grayscale conversion.** Colour distribution differs
  between domains and the conversion is a fixed linear map.
- **Single generator.** One FLUX subset supports no claim about AI-generated
  imagery in general.

### Statistical power

- Domain-probe test splits held 33-34 covers, so one prediction moves balanced
  accuracy by about three percentage points.
- Three seeds give a coarse standard-deviation estimate.
- The feature screen's permutation test used 200 shuffles; the SRNet probe has
  **no significance test**, only seed-to-seed spread.

### Preprocessing cost

Crop-only cannot fill 256x256 from a smaller source. One natural cover of 100
was excluded (170x226). AI covers are uniformly 1024x1024 and never excluded,
so **the loss is one-sided** and erodes the class-count matching between
domains. The index records `size_bytes` but not image dimensions, so undersized
candidates cannot currently be filtered at selection time, which is where the
fix belongs.

### Unmeasured

**Whether SRNet converges on embedding traces at a realistic cohort size is
unknown.** The only completed detector runs are a 140-image Kaggle smoke run
(chance-level, confounded source) and the domain probes, which detect domain
rather than embedding.

## 3. Pipeline provenance

Each detector run records: manifest SHA-256, dataset counts by domain and
split, embedding condition read from the manifest, pinned SRNet commit,
repository commit and dirty flag, resolved device, platform, dependency
versions and timing. Per-epoch history, the selected checkpoint with its
selection criterion, and per-image predictions with probabilities are preserved.

Verified properties of the embedding step on the 199-cover crop cohort
(S-UNIWARD, 0.4 bpp, seed 42): zero pair-split leaks, every pixel change
exactly +/-1, and 6.3 to 10.1 per cent of pixels altered per image.

Stego images are cached by backend, algorithm, payload, **cohort** and seed.
The cohort component was added after finding that cohorts prepared differently
from the same sources share image IDs while holding different pixels, so a
shared cache could pair a stego image with a cover it was not derived from.

## 4. Reproduction

```bash
# Prepare covers (crop-only is the accepted preprocessing)
python -m src.datasets prepare-selection \
  --extraction-manifest data/raw/selected/<cohort>/extraction_manifest.jsonl \
  --name <cohort>-crop --output data/processed \
  --size 256 --preprocess crop --skip-undersized

# Gate the cohort before using it
python -m src.steganalysis.domain_probe \
  --manifest data/processed/manifests/<cohort>-crop_clean.jsonl \
  --seeds 1 2 3 --epochs 30 --batch-size 16 \
  --output results/audits/srnet-probe-<cohort>-crop

# Embed, then review what has already been embedded
python -m src.steganography benchmark \
  --manifest data/processed/manifests/<cohort>-crop_clean.jsonl \
  --algorithm suniward --payload-bpp 0.4 --seed 42 \
  --output data/stego/runs --cache-dir data/stego/cache
python -m src.steganography ledger --runs data/stego/runs

# Four-direction protocol
python -m src.steganalysis.experiments \
  --manifest data/stego/runs/<run>/dataset_manifest.jsonl \
  --seeds 1 2 3 --epochs 30 --batch-size 16 --output results

# Tests
python -m unittest discover -s tests -t . -v
```

## Related notes

- [[Decisions]] — accepted protocol choices; the authority.
- [[Development Log]] — chronological record with dates and commands.
- [[Pipeline Summary and Experiment Plan]] — gates and open questions.
- [[Experiment Decisions]] — concise experiment sheet.
- [[Tasks]] — outstanding work.
