---
type: thesis-source-record
status: active
updated: 2026-10-14
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
> research experiment. Stage 1 of the six-direction protocol ran on 2026-10-12
> and returned chance accuracy from a fault in the detector configuration. See
> section 6. Nothing here answers the thesis questions. It establishes whether
> the dataset and pipeline can answer them.

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

### 1.3 Measured results at full scale (authoritative)

Cohort: `imagenet-flux-6000-crop-balanced`, 5,830 covers per domain, 1,000
classes. SRNet probe: 3 seeds, 30 epochs, batch 32, CUDA, 66 minutes. Test
split 1,948 covers (974 per domain).

| Metric | Value |
|---|---|
| Balanced accuracy | **0.9954** (sd 0.0007) |
| ROC-AUC | **0.9997** (sd 0.0001) |
| Per-seed balanced accuracy | 0.9959, 0.9944, 0.9959 |
| Errors per seed | 8, 11, 8 of 1,948 |
| Selected epochs | 29, 23, 25 |

Artifact: `results/audits/srnet-probe-6000-crop-balanced/summary.json`.

**Interpretation.** The two cover domains are near-perfectly separable from
clean covers alone, after crop-only preprocessing and after domain rebalancing.
This is consistent with the sources rather than surprising: GenImage++ is
published as a benchmark **for** AI-image detection, so separability is a design
property of the data.

Consequences, stated precisely:

- Research question 1 (`ai_to_ai` against `real_to_real`) is **unaffected**.
  Domain separability says nothing about clean-versus-stego detection inside a
  single domain.
- Research question 2 (`real_to_ai`) remains measurable, but a transfer failure
  occurs across a near-maximal domain gap. Attributing it to steganographic
  traces differing between domains, rather than to ordinary distribution shift,
  requires a further control that this design does not currently provide.

### 1.4 Measured results at 100 covers per domain (superseded)

> [!warning] These figures understate the domain gap
> At 132 training covers the probe was limited by sample size, not by the data.
> The `crop` figure of 0.675 below becomes 0.9954 at 5,830 covers per domain.
> Retained because the preprocessing and compression comparisons were only ever
> run at this scale.


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

### 1.5 Interpretation of the small-scale comparison

- Removing resampling reduced separability by about 0.07 balanced accuracy at
  this scale. **At full scale the probe saturates (0.9954 for `crop` alone), so
  it can no longer discriminate between the two preprocessing paths.** The
  crop-only decision therefore rests on removing a known processing artifact —
  unequal and sometimes magnifying resampling — rather than on this measured
  difference. Running the `fit` probe at full scale would be uninformative,
  since both variants sit at the ceiling.
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

### 1.6 Methodological finding worth reporting

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
- **Generation resolution — now measured, and large.** A native 256x256 crop
  keeps **35.0 per cent** of a natural cover and **6.2 per cent** of an AI
  cover, a ratio of 5.6. In practice the AI covers become close-ups of texture
  such as fur, while the natural covers still hold most of the scene including
  smooth areas. This reaches research question 1 and not only the domain gap:
  S-UNIWARD hides bits in complex texture, so a result of "AI covers are harder
  to detect" could mean "textured crops are harder to detect". No longer an
  untested hypothesis. See §1.7.
- **Tonal statistics under grayscale conversion.** Colour distribution differs
  between domains and the conversion is a fixed linear map.
- **Single generator.** One FLUX subset supports no claim about AI-generated
  imagery in general.

### Statistical power

- **This caveat was borne out.** The 100-cover probe reported 0.675; the same
  probe at 5,830 covers per domain reports 0.9954. The small-scale figure
  measured the limit of 132 training covers, not the domain gap. Treat every
  remaining small-scale number in this note as a lower bound only.
- Domain-probe test splits at 100 covers per domain held 33-34 covers, so one
  prediction moved balanced accuracy by about three percentage points. The
  full-scale split holds 1,948.
- Three seeds give a coarse standard-deviation estimate, though at full scale
  the spread is very small (sd 0.0007).
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

## 1.7 Preprocessing: a tradeoff with no clean answer

The two sources hold different resolutions. One preprocessing choice matches
the field of view. The other matches the resampling. No choice matches both.

| Mode | Field of view | Resampling |
|---|---|---|
| `fit` | Equal. Both keep the whole short side. | Unequal. Natural median 1.46x, range 0.66x to 7.59x. AI exactly 4x. |
| `crop` | Unequal. 35.0 per cent against 6.2 per cent. | Equal. Neither domain gets resampled. |

Measured over 100 covers for each domain.

**Accepted protocol.** `crop` is primary. `fit` runs as a sensitivity test. If
both give the same conclusion, the result is stronger. If they disagree,
preprocessing drives the result and no general claim about AI against natural
covers is available. Resolution matching is an optional last pilot.

This is a property of the two sources, not a defect in the code, and the thesis
should state it as a limitation rather than present either mode as correct.

## 4. The mixed-domain control

A poor `real_to_ai` result has two possible causes, and the four-direction
design cannot separate them:

1. The embedding traces differ between the domains.
2. The AI covers lie outside the detector's training distribution.

Cause 2 is likely at a domain separability of 0.9954. A third detector trains
on an equal mixture of both domains, so neither test domain is out of
distribution for it.

| Outcome | Reading |
|---|---|
| `real_to_ai` poor, `mixed_to_ai` good | Cause 2. The failure is distribution shift. |
| `mixed_to_ai` and `mixed_to_real` still differ | Stronger evidence for cause 1. |
| Both mixed directions good and equal | Detectability does not differ between domains. |

**The mixed training set must hold the same number of covers as each
single-domain set**, half drawn from each domain. A model trained on the union
would see twice the data, and any advantage could come from data volume rather
than from domain coverage.

### Experiment order

1. Same-domain: `real_to_real` and `ai_to_ai`. These answer research question 1
   and domain separability does not affect them.
2. Mixed-domain: `mixed_to_real` and `mixed_to_ai`.
3. Cross-domain: `real_to_ai` and `ai_to_real`. Interpret these last, and only
   against the mixed-domain result.

## 5. Where the work runs

Acquisition, preparation, embedding and all detector runs happen on the GPU
machine, at `~/stego-analysis`, with the interpreter at
`~/stego-analysis/.venv/bin/python`. That machine holds the full datasets. The
development machine holds a 100-cover cohort for tests only.

## 6. The detector null result, and what caused it

### 6.1 What happened

Stage 1 of the six-direction protocol ran on 2026-10-12. It trained on natural
covers and tested on both domains, with three seeds, 30 epochs and a learning
rate of 1e-3. Both directions returned **accuracy exactly 0.5000**, with a
standard deviation of 0.0 across the seeds.

Three observations fixed the shape of the failure.

The train loss held at 0.693 from epoch 2 to epoch 30, in every seed. That
number is the natural logarithm of 2, and it is the loss of a classifier that
answers at random.

The confusion matrix held the same row twice. For seed 1 it was 366 and 608 on
the clean row, and 366 and 608 on the stego row. A cover and its stego received
the same answer every time.

The control direction failed with the rest. `real_to_real` trains and tests on
one domain, so it must work before any cross-domain number carries meaning.

> [!warning] Stage 1 and the paired pilot are failures, not results
> No number from either run describes AI covers against natural covers. Both
> stay in this record as a methodological finding.

### 6.2 The diagnostic sequence

Each check is a command in the repository. See `README.md`, section "Diagnose a
Detector Run", and `src/steganalysis/diagnostics.py`.

| # | Check | Measurement | Reading |
|---|---|---|---|
| 1 | `embedding` | 0 shared paths, 0 identical pairs, 7.65 % of pixels changed, peak difference 1 | the embedding is present and matches S-UNIWARD at 0.4 bpp |
| 2 | `classical --synthetic` | ROC-AUC 0.9522, z = 19.17 | the classical instrument works |
| 3 | `classical` | ROC-AUC 0.5578, z = 2.45 | the real embedding is detectable, and weakly so |
| 4 | `collapse` | see 6.3 | the detector gives up its sensitivity to the embedding |
| 5 | `synthetic-train` | balanced accuracy 0.8917 | the detector and the training code work |

Checks 2 and 3 ran on CPU, so neither depends on the GPU.

### 6.3 The measurement that identified the cause

The `collapse` check trains for a short run and records two quantities at
fixed steps. `logit_std` is the spread of the output across a batch.
`pair_gap` is the output difference between a cover and its own stego. Their
ratio is the part of the response that follows the embedding rather than the
picture, and the ratio is the quantity that matters: both parts move with the
output scale, which the initialization sets.

Ratio of `pair_gap` to `logit_std`, after 200 steps:

| Initialization | step 1 | lr 1e-3 | lr 1e-4 | lr 1e-5 |
|---|---|---|---|---|
| Xavier normal (earlier) | 4.6e-3 | 3.8e-3 | 1.5e-2 | 1.0e-2 |
| reference scheme | 3.5e-3 | 4.3e-3 | 1.8e-2 | 1.3e-2 |

At 1e-3 the ratio holds flat. At 1e-4 and 1e-5 it becomes four to five times
larger over the same 200 steps. **Both rows show it**, which is what separates
the learning rate from the initialization.

### 6.4 Cause

*Interpretation.* At a learning rate of 1e-3 the detector reduces its loss by
making its output constant. The loss then settles at ln(2), and every image
receives the same answer. A cover and its stego therefore receive the same
answer, and the accuracy is exactly 0.5 by construction rather than by chance.
The paired pilot held a train loss of 0.693147 for 150 epochs, which is ln(2)
to six decimals, and the output spread fell toward zero at every learning rate
tested.

At 1e-4 that route down is not available quickly enough, and the loss falls by
separating the two classes instead.

### 6.5 Two defects fixed during the investigation

Both were deviations from the reference implementation, and both are fixed.

**The batches were not paired.** The loader built a flat list of 2N images and
shuffled it, so a cover and its stego went into different batches. The
reference returns the pair from one dataset index. The pair constraint is what
makes a batch informative here: both classes then hold the same pictures, so
the only systematic difference inside a batch is the embedding.

**The initialization did not match the reference.** Xavier normal on the
512-to-2 output layer gives a standard deviation near 0.06. The reference uses
normal with a standard deviation of 0.01. Measured on a structured batch in
train mode, the earlier scheme starts at 0.6998 and the reference scheme at
0.6922, against ln(2) of 0.6931.

> [!important] Attribution limit
> Both fixes preceded the learning-rate test, so the successful control run
> carries all three changes together. The table in 6.3 is the only measurement
> that isolates the learning rate, because it holds for both initializations.
> The contribution of each fix is **not** measured.

### 6.6 What the positive control established

`synthetic-train` builds a non-adaptive embedding of plus or minus one on half
the pixels, trains the real detector through the real training code, and
reports balanced accuracy. The embedding is not steganography. It touches flat
regions that an adaptive algorithm avoids, and a classical detector finds it at
ROC-AUC 0.9522.

Result on 2026-10-14: balanced accuracy **0.8917**, train loss from 0.6939 to
0.1875, over 4,500 optimizer steps at a learning rate of 1e-4.

The paired pilot did 36,450 steps at 1e-3 and stayed at ln(2). The control did
**eight times fewer steps** and learned. The training budget was therefore not
the limiting factor, although the SRNet paper uses 400,000 iterations.

### 6.7 A GPU driver confound, raised and then closed

The GPU machine lost `nvidia-smi` on 2026-10-14. The kernel module in memory
was 595.84 and the module on disk was 595.99.02. The machine booted on
2026-10-08 11:26, and `dpkg` installed `nvidia-driver-595-open` 595.99.02 on
2026-10-10 06:37. A loaded kernel module does not change while a machine runs.

This mattered because the dates split the project exactly. The domain probe
reached 0.9959 on 2026-10-07. Stage 1, the paired pilot and the collapse probes
all returned chance results on 2026-10-12 and 2026-10-13.

**The control closes the correlation.** `synthetic-train` ran on 2026-10-14,
four days after the upgrade, on the same mismatched machine, and converged to
0.8917. CUDA arithmetic gives the right answers. Only NVML is broken, so `nvidia-smi` does
not run and some PyTorch paths abort at `nvmlInit_v2`. No GPU result is
discarded.

Runs now record the accelerator in their provenance: the CUDA runtime, the
kernel driver line from `/proc/driver/nvidia/version`, the device name, whether
MPS is present, and whether NVML starts. Runs before 2026-10-14 do not, and
their driver is the one recorded above.

### 6.8 Methodological findings worth reporting

**A control direction that returns exactly 0.5000 is a diagnostic, not a
result.** Exactly 0.5 with zero standard deviation across seeds, together with
identical confusion rows, identifies a constant-output detector. Approximately
0.5 would have meant something else.

**A positive control needs a known answer.** The real stego images could not
decide between "the detector is broken" and "the task is hard", because nobody
knew whether the detector should have found that signal. An embedding that a
500-feature linear model finds at 0.95 removes the ambiguity.

**A diagnostic must have the power to decide.** Two checks run during this
investigation used 375 and 1,250 optimizer steps. Both returned chance results,
and I read both as evidence of a fault. Neither had the power to show anything.
The second one sent the investigation toward the initialization, which was not
the cause.

**An instrument needs its own control.** The first classical measurement
returned 0.4975, and I nearly read it as evidence against the data. A second
run used a synthetic embedding and returned 0.9522, which showed that the
instrument works. The 0.5578 on real data then became interpretable.

### 6.9 Open

The control used an easy embedding. S-UNIWARD at 0.4 bpp is much harder, and
the classical detector finds it at ROC-AUC 0.5578. Whether a learning rate of
1e-4 is enough for the real task is **not yet measured**. The pilot that tests
it repeats stage 1 exactly, with one change: `real_to_real`, one seed, 150
epochs, batch size 32, learning rate 1e-4.

Until that pilot finishes, the established claim is narrow: the detector learns
an easy embedding, and the learning rate of 1e-3 prevented it from learning
anything.
