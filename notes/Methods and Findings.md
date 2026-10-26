---
type: thesis-source-record
status: active
updated: 2026-10-26
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

> [!warning] Status on 2026-10-21
> **Quote section 10.** It holds the complete protocol: six directions, three
> seeds, nine runs, with means, spreads and paired gaps.
>
> Sections 1 to 6 are methodological validation. Section 7 is the first
> working detector, from a single-seed pilot. Sections 8 and 9 report seed 1
> alone, and section 10 supersedes their numbers. All three stay as the record
> of what the project knew on each date.
>
> Two limits apply to every number in section 10. Each detector stopped while
> still improving, so each value is a lower bound. The condition covers one
> algorithm, one payload, one generator and one detector.

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

### 6.9 Closed on 2026-10-14

The pilot ran and the question is answered. A learning rate of 1e-4 is enough
for the real task: `real_to_real` reached 0.7228 balanced accuracy and 0.8323
ROC-AUC on S-UNIWARD at 0.4 bpp. Section 7 holds that result.

The cause stated in 6.4 is confirmed by the learning curve. At 1e-4 the network
held ln(2) for 17 epochs and then escaped. At 1e-3 it did 36,450 steps and never
escaped. The learning rate did not slow the escape. It prevented it.

## 7. First working detector: the 1e-4 pilot

### 7.1 The run

One seed, `real_to_real` and `real_to_ai`, 150 epochs, batch size 32, learning
rate 1e-4. The two directions come from one trained model, so the second is an
evaluation and not a second training.

| | |
|---|---|
| manifest | `conseal_suniward_p040_20261007_143130`, SHA-256 `f271f135...` |
| embedding | S-UNIWARD, 0.4 bpp, a separate seed for each of 11,660 images |
| covers | 3,887 train, 969 validation, 974 test, for each domain |
| preprocessing | native centre crop, 256x256, grayscale PNG |
| detector | SRNet at commit `2c838cf7` |
| repository | commit `8b14ca35`, clean |
| hardware | RTX 5080, kernel driver 595.84, CUDA runtime 13.0 |
| duration | 6,670 seconds |
| W&B | run `o3quu45i` |
| checkpoint | epoch 148 of 150, by validation loss |

### 7.2 Same-domain result

| Metric | `real_to_real` |
|---|---|
| accuracy | 0.7228 |
| balanced accuracy | 0.7228 |
| precision | 0.7136 |
| recall | 0.7444 |
| F1 | 0.7286 |
| specificity | 0.7012 |
| false positive rate | 0.2988 |
| false negative rate | 0.2556 |
| detection error `P_E` | 0.2772 |
| MCC | 0.4460 |
| ROC-AUC | 0.8323 |
| PR-AUC | 0.8434 |
| EER | 0.2752 |
| confusion matrix | 683, 291 / 249, 725 |

The same command at learning rate 1e-3 gave exactly 0.5000. This is the first
evidence in the project that S-UNIWARD at 0.4 bpp is detectable in these covers.

### 7.3 Cross-domain result

| Metric | `real_to_ai` |
|---|---|
| accuracy | 0.5000 |
| balanced accuracy | 0.5000 |
| specificity | 0.0503 |
| false positive rate | 0.9497 |
| detection error `P_E` | 0.5000 |
| MCC | 0.0000 |
| ROC-AUC | 0.4970 |
| PR-AUC | 0.4971 |
| EER | 0.5010 |
| confusion matrix | 49, 925 / 49, 925 |

Two separate failures sit inside that column, and they must stay separate.

**The decision threshold does not hold.** The detector calls 95 per cent of AI
images stego, with or without a payload. A detector trained on photographs
treats AI covers as suspicious before any embedding.

**The ranking carries no information either.** ROC-AUC is 0.4970. A threshold
fault alone leaves the area under the curve high, and recalibration repairs it.
This one cannot be repaired that way.

The confusion matrix holds the same row twice. In the AI domain a cover and its
stego twin receive the same answer every time, so the accuracy is 0.5000 by
construction. The features this detector learned do not respond to S-UNIWARD
traces in AI covers.

### 7.4 Paired generalization gaps

Each gap is `real_to_real` minus `real_to_ai`, inside the one seed.

| Metric | Gap |
|---|---|
| balanced accuracy | +0.2228 |
| ROC-AUC | +0.3352 |
| PR-AUC | +0.3463 |
| MCC | +0.4460 |
| specificity | +0.6509 |
| false positive rate | -0.6509 |
| detection error | -0.2228 |
| EER | -0.2259 |

### 7.5 The learning curve

Two features of the curve matter more than the final number.

**A plateau of 17 epochs.** The train loss held 0.6931 from epoch 1 to epoch
17. It moved at epoch 18 and the network learned from there. At 1e-3 the same
model did 36,450 steps and never left that plateau.

**The training did not finish.** The selected checkpoint is epoch 148 of 150.
Validation balanced accuracy was 0.663 at epoch 100, 0.707 at epoch 130 and
0.724 at epoch 150. Train accuracy was 0.7336 against validation 0.7239, so the
model was not overfitting and had room to continue.

> [!important] 0.7228 is a lower bound
> A best checkpoint at the end of the budget means the budget was too small.
> The ceiling of this detector on this data is not measured.

### 7.6 Interpretation

*Interpretation, not measurement.* The cross-domain failure is consistent with
cover-source mismatch at full strength. The same detector that separates clean
from stego at 0.83 ROC-AUC inside its own domain carries no usable signal in
the other domain, and it treats almost every AI cover as stego. The domain
probe measured 0.9959 balanced accuracy between the domains on clean covers
alone, so the detector sees a large domain difference before any embedding.

### 7.7 What this does not establish

**One seed.** Every standard deviation in the summary is 0.0 because one value
has no spread. No claim in the thesis comes from this run.

**The cause of the cross-domain failure is still undetermined.** Two
explanations remain. The embedding traces differ between the domains, or the AI
covers lie outside the training distribution of this detector. This pilot
cannot separate them. The mixed-domain detector was built for exactly this, and
it is now required rather than optional.

**No AI-domain result exists.** `ai_to_ai` has never run, so research question
1 has no evidence at all.

### 7.8 Consequence for the protocol

The stages run at 300 epochs rather than 150, at learning rate 1e-4, with three
seeds. See [[Decisions]], 2026-10-15.

One risk to watch in the per-seed histories: each seed escapes the plateau at
its own epoch. A seed that escapes late receives less effective training, and
the spread across seeds would then measure the escape epoch rather than the
variance of the detector.

## 8. The six directions: first evidence for both research questions

> [!note] Seed 1 only. Section 10 supersedes these numbers.
> This section records what the project knew on 2026-10-19, when natural
> covers had two seeds and the other domains had one.

### 8.1 What exists

Four trained detectors, all at 300 epochs, learning rate 1e-4, batch size 32,
on manifest `f271f135` from repository commit `d5168c07`.

| Run | Domain | Seed | Folder |
|---|---|---|---|
| 1 | real | 1 | `srnet_20261015_112246` |
| 2 | ai | 1 | `srnet_20261015_182123` |
| 3 | mixed | 1 | `srnet_20261016_173640` |
| 4 | real | 2 | `srnet_20261017_121320` |

Each detector is evaluated on both test domains, so four trainings give eight
evaluations and cover all six directions.

### 8.2 Validity of the runs

| Run | Escape epoch | Selected epoch | Train minus validation |
|---|---|---|---|
| real seed 1 | 19 | 287 / 300 | +0.0384 |
| real seed 2 | 33 | 289 / 300 | +0.0362 |
| ai seed 1 | 7 | 244 / 300 | +0.0366 |
| mixed seed 1 | 18 | 298 / 300 | +0.0382 |

**The escape-epoch risk did not occur.** Section 7.8 warned that a domain that
leaves the ln(2) plateau late would receive less effective training, and that
the spread would then measure the escape epoch. AI left the plateau first, at
epoch 7. No domain was handicapped.

**Every detector is still undertrained.** Each selected checkpoint sits near
the end of its budget, so every value below is a lower bound. The train minus
validation difference is close to +0.037 in all four, so they are undertrained
by the same amount and the comparison between them stays fair.

The budget, and not the quantity of data, is the limit that binds. The same
3,887 training pairs gave 0.7228 balanced accuracy at 150 epochs and 0.7608 at
300. A limit of data volume would show as a large train minus validation
difference, and the measured difference is small.

**Reproducibility is high.** Two independent trainings on natural covers:

| | seed 1 | seed 2 | difference |
|---|---|---|---|
| balanced accuracy | 0.7608 | 0.7613 | 0.0005 |
| ROC-AUC | 0.8780 | 0.8781 | 0.0001 |
| MCC | 0.5238 | 0.5238 | 0.0000 |

### 8.3 All six directions, seed 1

| Direction | Balanced acc | ROC-AUC | `P_E` | MCC | EER | Specificity | Recall |
|---|---|---|---|---|---|---|---|
| `real_to_real` | 0.7608 | 0.8780 | 0.2392 | 0.5238 | 0.2361 | 0.7146 | 0.8070 |
| `ai_to_ai` | 0.6242 | 0.7034 | 0.3758 | 0.2485 | 0.3747 | 0.6170 | 0.6314 |
| `mixed_to_real` | 0.6427 | 0.7553 | 0.3573 | 0.3199 | 0.3368 | 0.8686 | 0.4168 |
| `mixed_to_ai` | 0.6304 | 0.7192 | 0.3696 | 0.2617 | 0.3634 | 0.6715 | 0.5893 |
| `real_to_ai` | 0.5077 | 0.5288 | 0.4923 | 0.0292 | 0.4774 | 0.0832 | 0.9322 |
| `ai_to_real` | 0.5072 | 0.5262 | 0.4928 | 0.0162 | 0.4856 | 0.7392 | 0.2751 |

Confusion matrices, as clean row then stego row:

```text
real_to_real   696  278 / 188  786
ai_to_ai       601  373 / 359  615
mixed_to_real  846  128 / 568  406
mixed_to_ai    654  320 / 400  574
real_to_ai      81  893 /  66  908
ai_to_real     720  254 / 706  268
```

### 8.4 Research question 1

Under the same algorithm, payload, preprocessing, detector and budget, only the
cover domain changes.

| | balanced acc | ROC-AUC | `P_E` |
|---|---|---|---|
| `real_to_real` | 0.7608 | 0.8780 | 0.2392 |
| `ai_to_ai` | 0.6242 | 0.7034 | 0.3758 |
| difference | **-0.1366** | **-0.1746** | **+0.1366** |

*Measurement.* S-UNIWARD at 0.4 bpp is harder to detect in AI covers than in
natural covers, by 13.7 points of balanced accuracy and 17.5 points of area
under the curve.

The seed-to-seed difference on natural covers is 0.0005, so the effect is
roughly 250 times the seed noise. The AI domain has one seed at the time of
writing, so the claim waits for its repeat.

### 8.5 Research question 2, and what the mixed control decides

| Direction | balanced acc | ROC-AUC |
|---|---|---|
| `real_to_ai` | 0.5077 | 0.5288 |
| `ai_to_real` | 0.5072 | 0.5262 |
| `mixed_to_real` | 0.6427 | 0.7553 |
| `mixed_to_ai` | 0.6304 | 0.7192 |

Both transfers fall to chance, and they fall by the same amount in each
direction. A detector trained on one domain carries no usable signal into the
other, at any threshold.

**The mixed detector works in both domains.** This is the measurement the
control was built to supply, and it separates the two explanations that
section 1 of this note could not separate.

*Interpretation.* The embedding traces are **not** incompatible between the
domains. One detector that has seen both finds the embedding in both. The
cross-domain collapse is distribution shift, and not a difference in what
S-UNIWARD leaves behind.

### 8.6 The two transfer failures are mirror images

| | specificity | recall |
|---|---|---|
| `real_to_ai` | 0.0832 | 0.9322 |
| `ai_to_real` | 0.7392 | 0.2751 |

A detector trained on natural covers calls 93 per cent of AI images stego. A
detector trained on AI covers calls 74 per cent of natural images clean.

*Interpretation.* One cause seen from two sides. The decision boundary sits at
the wrong place for the residual statistics of the other domain. The natural
detector reads the residual of an AI cover as evidence of embedding. The AI
detector reads the residual of a natural cover as evidence of a clean image.

### 8.7 A design effect that must not be read as a finding

`mixed_to_real` at 0.6427 is well below `real_to_real` at 0.7608. Part of that
distance is built into the design and is not a result.

The mixed training set holds 3,886 pairs, one fewer than the 3,887-pair
single-domain set because the domain-balanced selection takes the integer half.
It holds exactly 1,943 pairs from each domain, against the 3,887 of a
specialist. The mixed detector therefore trains on half the natural data.

The same design makes the AI side more interesting. `mixed_to_ai` at 0.6304
matches `ai_to_ai` at 0.6242 while using half the AI data. Detection in the AI
domain appears to be limited by something other than the quantity of data.

### 8.8 What is still missing

**Seeds.** AI and mixed have one seed each. Two more runs were started on
2026-10-19. Nothing in sections 8.4 to 8.7 is a final claim until the repeats
exist.

**An aggregator.** Each run folder holds one seed, so every standard deviation
in every `summary.json` is 0.0. No file holds the cross-seed means, spreads or
paired gaps that the results chapter reports.

**A ceiling.** Every detector stopped while still improving. The absolute
values are floors, and the size of each effect is measured between floors.

## 9. A second reading of research question 1

> [!note] Seed 1 only. Section 10.5 supersedes these numbers.
> The argument of this section holds at three seeds, and the measurement that
> separates the two readings, in 9.5, has still not run.

### 9.1 The problem with the first reading

Section 8.4 compares `real_to_real` with `ai_to_ai` and reports that AI covers
are 13.7 points of balanced accuracy harder. That comparison moves two things
at once. The training domain changes and the test domain changes together, so
the difference holds both the difficulty of the cover domain and whatever each
detector gained from training on its own domain.

The mixed detector gives a comparison that moves one thing. It trains on both
domains, so neither test domain is foreign to it and neither carries a
training advantage. The difference between `mixed_to_real` and `mixed_to_ai`
is therefore the cover domain alone.

### 9.2 The two readings disagree by a factor of ten

| Comparison | What moves | Gap in balanced accuracy |
|---|---|---|
| `real_to_real` against `ai_to_ai` | training domain and test domain | **+0.1366** |
| `mixed_to_real` against `mixed_to_ai` | test domain only | **+0.0123** |

Both are seed 1 at 300 epochs. The sign agrees: AI covers are the harder side
in both. The size does not.

### 9.3 What else the mixed detector shows

| | specialist | mixed | change |
|---|---|---|---|
| natural covers | 0.7608 | 0.6427 | **-0.1181** |
| AI covers | 0.6242 | 0.6304 | **+0.0062** |

The mixed detector holds 1,943 pairs from each domain, against 3,887 for each
specialist. Halving the natural data costs 0.118 of balanced accuracy.
Halving the AI data costs nothing.

### 9.4 Two readings, and the project does not choose between them yet

*Interpretation, and both of these fit the measurements.*

**Reading A: the cover domains are close in difficulty, and the training data
is what differs.** Detection of natural covers improves with more training
data and had not saturated at 3,887 pairs. Detection of AI covers saturated
below 1,944 pairs. The 0.1366 of section 8.4 would then be mostly a difference
in how each domain rewards data, and the 0.0123 would be the difficulty of the
cover domain.

**Reading B: the mixed detector is limited by something common to both
domains, and that limit hides a real difference.** It reaches 0.6427 and
0.6304, and both sit far below the 0.7608 of the natural specialist. A
detector held near one ceiling in both domains would report a small difference
whatever the true one is.

### 9.5 What separates them

A measurement of how each domain responds to training-set size. Train a
single-domain detector on 1,943 pairs, which is the count the mixed detector
holds, and compare it with the 3,887-pair result in the same domain.

- If natural covers lose approximately 0.118 and AI covers lose approximately
  nothing, reading A holds, and the mixed detector is at its data limit rather
  than a common ceiling.
- If both domains lose a similar amount, the saturation claim of reading A
  fails.

Two trainings for each domain at one seed, so about 7.4 hours.

> [!warning] Do not report 0.1366 as the answer to research question 1
> Until the readings separate, state both numbers and what each one controls.
> A reviewer who sees only the same-domain comparison will ask what the mixed
> detector showed, because the protocol already contains it.

## 10. The complete protocol: three seeds, six directions

> [!note] This section supersedes the numbers in sections 8 and 9
> Sections 8 and 9 report seed 1 alone. They stay as the record of what the
> project knew on 2026-10-19. Every number for the thesis comes from here.

### 10.1 The condition

S-UNIWARD at 0.4 bpp, native centre crop, learning rate 1e-4, 300 epochs,
batch size 32, manifest `f271f135`. Nine run folders: three training domains,
three seeds each. Produced by `python -m src.evaluation.protocol`.

Two repository commits produced the condition, `d5168c07` and `4bc1044f`. The
only difference between them is where the tracking code stores its metrics.
No training or evaluation code changed, and `summary.json` does not depend on
it. The aggregator reports the two commits rather than hiding them.

### 10.2 Validity

| Folder | Domain | Seed | Escape | Selected | Train minus validation |
|---|---|---|---|---|---|
| `srnet_20261015_112246` | real | 1 | 19 | 287 / 300 | +0.0384 |
| `srnet_20261017_121320` | real | 2 | 33 | 289 / 300 | +0.0362 |
| `srnet_20261021_070017` | real | 3 | 15 | 293 / 300 | +0.0225 |
| `srnet_20261015_182123` | ai | 1 | 7 | 244 / 300 | +0.0366 |
| `srnet_20261019_180835` | ai | 2 | 9 | 268 / 300 | +0.0773 |
| `srnet_20261020_233927` | ai | 3 | 6 | 273 / 300 | +0.0619 |
| `srnet_20261016_173640` | mixed | 1 | 18 | 298 / 300 | +0.0382 |
| `srnet_20261019_215609` | mixed | 2 | 11 | 291 / 300 | +0.0503 |
| `srnet_20261021_031924` | mixed | 3 | 10 | 298 / 300 | +0.0638 |

**No domain started late.** AI left the ln(2) plateau first in every seed, at
epochs 6 to 9. Natural covers left last, at 15 to 33.

**Every detector is still undertrained.** Each selected checkpoint sits
between epoch 244 and 298 of 300, so **every value below is a lower bound**.

**The AI and mixed detectors start to overfit where the natural one does
not.** The natural detector reaches +0.0225 at seed 3. The AI detector reaches
+0.0773 at seed 2 and the mixed detector +0.0638 at seed 3. The differences
are modest, and they point in a consistent direction.

### 10.3 All six directions

Mean and spread over three seeds.

| Direction | Balanced acc | ROC-AUC | `P_E` | MCC | EER |
|---|---|---|---|---|---|
| `real_to_real` | 0.7680 +/- 0.0098 | 0.8824 +/- 0.0062 | 0.2320 +/- 0.0098 | 0.5374 +/- 0.0192 | 0.2300 +/- 0.0073 |
| `mixed_to_real` | 0.6448 +/- 0.0040 | 0.7430 +/- 0.0118 | 0.3552 +/- 0.0040 | 0.3017 +/- 0.0145 | 0.3480 +/- 0.0085 |
| `ai_to_ai` | 0.6343 +/- 0.0071 | 0.7208 +/- 0.0122 | 0.3657 +/- 0.0071 | 0.2709 +/- 0.0162 | 0.3662 +/- 0.0061 |
| `mixed_to_ai` | 0.6302 +/- 0.0061 | 0.7145 +/- 0.0095 | 0.3698 +/- 0.0061 | 0.2652 +/- 0.0072 | 0.3665 +/- 0.0059 |
| `ai_to_real` | 0.5142 +/- 0.0085 | 0.5380 +/- 0.0118 | 0.4858 +/- 0.0085 | 0.0323 +/- 0.0160 | 0.4808 +/- 0.0042 |
| `real_to_ai` | 0.5067 +/- 0.0022 | 0.5297 +/- 0.0007 | 0.4933 +/- 0.0022 | 0.0332 +/- 0.0180 | 0.4781 +/- 0.0026 |

> [!warning] The spread is a population standard deviation
> `numpy` computes it with `ddof=0`. Over three seeds that is 1.22 times
> smaller than the sample standard deviation. The `real_to_real` spread would
> read 0.0120 rather than 0.0098. State which one the thesis reports.

### 10.4 Paired gaps

Each gap is differenced inside one seed and only then averaged.

| Gap | Kind | Balanced acc | ROC-AUC |
|---|---|---|---|
| `real_to_real` minus `real_to_ai` | transfer | +0.2613 +/- 0.0086 | +0.3527 +/- 0.0057 |
| `ai_to_ai` minus `ai_to_real` | transfer | +0.1201 +/- 0.0075 | +0.1827 +/- 0.0091 |
| `mixed_to_real` minus `mixed_to_ai` | test domain | **+0.0145 +/- 0.0028** | +0.0286 +/- 0.0054 |

### 10.5 Research question 1

| Comparison | What moves | Gap |
|---|---|---|
| `real_to_real` against `ai_to_ai` | training domain and test domain | +0.1337 |
| `mixed_to_real` minus `mixed_to_ai` | test domain only | +0.0145 +/- 0.0028 |

*Measurement.* Both comparisons give the same sign. AI covers are the harder
side. The sizes differ by a factor of ten.

The mixed gap is positive in every seed, and its mean is about five spreads
from zero, so the direction of the effect is not an artifact of one run.

*Interpretation.* A detector with no training advantage in either domain finds
S-UNIWARD at 0.4 bpp approximately 1.5 points of balanced accuracy harder in
AI covers. The 13.4 points of the same-domain comparison hold that difficulty
together with the different amount each domain gains from training on itself.
Section 9.5 gives the measurement that separates the two, and it has not run.

### 10.6 Research question 2

| | balanced acc | ROC-AUC |
|---|---|---|
| `real_to_ai` | 0.5067 +/- 0.0022 | 0.5297 +/- 0.0007 |
| `ai_to_real` | 0.5142 +/- 0.0085 | 0.5380 +/- 0.0118 |
| `mixed_to_real` | 0.6448 +/- 0.0040 | 0.7430 +/- 0.0118 |
| `mixed_to_ai` | 0.6302 +/- 0.0061 | 0.7145 +/- 0.0095 |

*Measurement.* Both transfers sit near 0.53 area under the curve. They are
above chance, and they carry no practical value: a detector at 0.53 separates
almost nothing. The same two test sets give 0.74 and 0.71 to a detector that
trained on both domains.

*Interpretation.* The embedding traces are not incompatible between the
domains. One detector that has seen both finds the embedding in both. The
collapse of a single-domain detector on the other domain is distribution
shift.

### 10.7 What remains open

**The two readings of research question 1.** Section 9.5 names the experiment,
which is a single-domain detector trained on 1,944 pairs, and it costs about
7.4 hours.

**The ceiling.** The 300-epoch values remain the primary three-seed condition,
but the mixed seed-1 budget sensitivity has now run to 500 epochs. Section 11
records that result.

**One algorithm, one payload, one primary generator, one detector.** The
external SD3 evaluation in section 12 adds a second AI generator, but no model
has trained on it. The accepted matrix still adds S-UNIWARD at 0.2 bpp, MiPOD
at 0.4 bpp, and WOW as a cross-algorithm test.

## 11. Mixed-domain training-budget sensitivity: 500 epochs

One mixed-domain detector was trained from scratch with seed 1 for 500 epochs.
Everything except the epoch budget matches the 300-epoch mixed seed-1
condition: manifest `f271f135`, S-UNIWARD at 0.4 bpp, native centre crop,
3,886 mixed training pairs, batch size 32 and learning rate 1e-4.

The selected checkpoint moved from epoch 298 of 300 to epoch 472 of 500. The
500-epoch run left the chance plateau at epoch 12, reached a minimum validation
loss of 0.5646, and finished with train accuracy 0.716 against validation
balanced accuracy 0.669.

| Direction | 300-epoch seed 1 | 500-epoch seed 1 | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | 0.7192 | **+0.0765** |
| `mixed_to_real` ROC-AUC | 0.7553 | 0.8220 | **+0.0667** |
| `mixed_to_ai` balanced accuracy | 0.6304 | 0.6371 | +0.0067 |
| `mixed_to_ai` ROC-AUC | 0.7192 | 0.7260 | +0.0068 |
| natural minus AI balanced-accuracy gap | +0.0123 | +0.0821 | +0.0698 |

**Observation.** The extra 200 epochs help the natural-cover side much more
than the AI-generated-cover side. The mixed detector gains 7.65 percentage
points on natural covers and 0.67 points on AI-generated covers.

**Interpretation.** The small mixed-domain gap at 300 epochs was partly a
training-budget effect. At a longer budget, the same model finds the embedding
more easily in natural covers. This supports the direction of research question
1, but the 8.21-point gap is a one-seed sensitivity result and not a replacement
for the three-seed primary estimate.

The minimum validation loss occurs at epoch 472 and the final loss at epoch 500
is only 0.0010 higher. For this seed, 500 epochs is close to a useful stopping
point. The next controlled step is the full-union mixed model at 300 epochs,
which changes data volume rather than training duration.

Detailed review: [[run-reviews/Run 11 - Mixed seed 1 500 epochs]]. W&B run:
`j4o9fc8u`. Result folder: `results/srnet_20261024_155602`.

## 12. External-generator evaluation: Stable Diffusion 3

The saved checkpoints were evaluated without retraining on 1,000 held-out
GenImage++ Stable Diffusion 3 covers and their S-UNIWARD stego counterparts.
The embedding remains 0.4 bpp, and preprocessing remains a native centre crop
to 256x256 grayscale PNG. The external test therefore changes the AI-image
generator and the individual images while holding the detector checkpoints
and steganographic condition fixed.

| Training condition | SD3 balanced acc. | SD3 ROC-AUC | Change from FLUX balanced acc. | Change from FLUX ROC-AUC |
|---|---:|---:|---:|---:|
| Natural, 300 epochs | 0.5073 +/- 0.0017 | 0.5290 +/- 0.0076 | +0.0006 | -0.0007 |
| FLUX, 300 epochs | 0.5590 +/- 0.0109 | 0.6289 +/- 0.0163 | **-0.0753** | **-0.0919** |
| Mixed, 300 epochs | **0.5797 +/- 0.0088** | **0.6516 +/- 0.0143** | **-0.0505** | **-0.0629** |
| Mixed, 500 epochs, seed 1 | 0.5585 | 0.6247 | **-0.0786** | **-0.1013** |

**Observation.** Natural-trained detectors remain near chance on both FLUX
and SD3. FLUX-trained and mixed detectors retain above-chance ranking
information on SD3, but both fall relative to the FLUX test. The 300-epoch
mixed condition loses less and gives the highest mean SD3 result.

The errors change character by training condition. Natural-trained detectors
label nearly every SD3 image as stego: mean specificity is 0.0857 and recall
is 0.9290. FLUX-trained detectors instead miss most stego images: specificity
is 0.8397 and recall is 0.2783. Mixed training improves the balance to 0.7290
specificity and 0.4303 recall.

**Interpretation.** The original AI-cover result is partly
generator-specific. Training on both cover domains gives better external
transfer than either specialist under the tested 300-epoch condition. This
does not establish that embedding is intrinsically harder in SD3, because no
detector trained on SD3. Ordinary cover-distribution shift remains the direct
explanation.

For seed 1, extending mixed training from 300 to 500 epochs improved FLUX
slightly but reduced SD3 balanced accuracy from 0.5905 to 0.5585. Increased
specialization is a hypothesis, not a finding, because the longer condition
has one seed.

Detailed review: [[run-reviews/Run 12 - SD3 external checkpoint evaluation]].
Result folder:
`results/srnet_eval_genimagepp-sd3-suniward040_20261025_155824`.

## 13. Same-generator prompt shift: realistic FLUX

The same saved checkpoints were evaluated on 1,000 held-out GenImage++
`flux_realistic` covers and their S-UNIWARD stego counterparts. This condition
keeps the FLUX generator but replaces the base ImageNet-label prompts with
detailed prompts. Embedding remains S-UNIWARD at 0.4 bpp, and preprocessing
remains a native centre crop to 256x256 grayscale PNG.

| Training condition | Base FLUX balanced acc. | Realistic FLUX balanced acc. | SD3 balanced acc. | Base FLUX ROC-AUC | Realistic FLUX ROC-AUC | SD3 ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Natural, 300 epochs | 0.5067 | 0.5037 | 0.5073 | 0.5297 | 0.5531 | 0.5290 |
| FLUX, 300 epochs | 0.6343 | **0.6997 +/- 0.0041** | 0.5590 | 0.7208 | **0.8111 +/- 0.0079** | 0.6289 |
| Mixed, 300 epochs | 0.6302 | **0.7018 +/- 0.0050** | 0.5797 | 0.7145 | **0.8058 +/- 0.0070** | 0.6516 |
| Mixed, 500 epochs, seed 1 | 0.6371 | **0.7075** | 0.5585 | 0.7260 | **0.8141** | 0.6247 |

**Observation.** Realistic-prompt FLUX is easier for every checkpoint that
trained on FLUX. Against base FLUX, balanced accuracy rises by 0.0654 for the
FLUX specialist and 0.0716 for the mixed 300-epoch detector. ROC-AUC rises by
0.0903 and 0.0913. Natural-trained detectors remain at chance and label almost
every realistic FLUX image as stego.

**Interpretation.** The tested realistic-prompt shift does not explain the
loss on SD3. With the generator held at FLUX, this prompt shift improves rather
than harms steganalysis. The SD3 decrease is therefore more consistent with
generator-specific cover-distribution shift.

This remains an inference from three datasets, not a causal isolation of one
image property. A matching realistic-prompt SD3 evaluation would complete the
2x2 comparison and test whether the FLUX-SD3 difference persists under both
prompt conditions.

Detailed review:
[[run-reviews/Run 13 - FLUX realistic external checkpoint evaluation]]. Result
folder:
`results/srnet_eval_genimagepp-flux-realistic-suniward040_20261025_184816`.

## 14. Full-union mixed-domain sensitivity

One mixed-domain detector was trained with seed 1 for 300 epochs on every
available training pair: 3,887 natural and 3,887 AI-generated pairs, or 7,774
pairs total. The equal-total mixed control used 3,886 pairs total. The
manifest, S-UNIWARD 0.4 bpp condition, native centre crop, seed, batch size and
learning rate remain fixed.

| Direction | Equal-total 300 epochs | Full union 300 epochs | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | **0.7305** | +0.0878 |
| `mixed_to_real` ROC-AUC | 0.7553 | **0.8388** | +0.0835 |
| `mixed_to_ai` balanced accuracy | 0.6304 | **0.6555** | +0.0251 |
| `mixed_to_ai` ROC-AUC | 0.7192 | **0.7520** | +0.0328 |
| Natural minus AI balanced-accuracy gap | +0.0123 | **+0.0749** | +0.0626 |

**Observation.** The larger mixed recipe improves both domains, with a much
larger improvement on natural covers. It also slightly exceeds the 500-epoch
equal-total seed-1 run on both domains. The selected checkpoint is epoch 214
of 300, at validation loss 0.5285, so this run does not show the same
end-of-budget behavior as the original 300-epoch mixed control.

**Interpretation.** The two sensitivity runs now agree that the very small
natural-minus-AI gap from the equal-total 300-epoch seed-1 model was not stable
under more training exposure. Their gaps are similar: 7.49 points for the
full-union run and 8.21 points for the 500-epoch equal-total run. This supports
the direction that the current AI-generated covers are harder, but both values
remain one-seed sensitivity results.

**Design limitation.** Holding epochs fixed does not hold optimizer updates
fixed. The full-union run performs about 145,800 updates, compared with about
72,900 for the equal-total 300-epoch run and 121,500 for the equal-total
500-epoch run. The gain therefore combines more unique training pairs with
more optimization steps. It cannot be attributed to data volume alone without
a step-matched control.

Detailed review:
[[run-reviews/Run 14 - Full-union mixed seed 1 300 epochs]]. W&B run:
`hgsctm2p`. The local result-folder path was not present in the supplied W&B
export and remains to be recorded.
