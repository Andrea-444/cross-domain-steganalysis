# Thesis Figure Generation Plan

## Purpose

Generate publication-quality figures for the cross-domain steganalysis thesis
from the completed experiment artifacts. The figures must be derived from the
canonical local histories, summaries, predictions, and external-evaluation
outputs. W&B may be used to locate or cross-check runs, but it is not the source
from which the final figures must be rendered.

The target task is binary steganalysis (clean versus stego). Natural and
AI-generated images are cover domains, not detector labels.

## Non-negotiable rules

- Do not rerun training, embedding, preprocessing, or checkpoint evaluation.
- Do not modify, rename, move, or overwrite an existing result directory.
- Do not invent or interpolate missing values.
- Verify every run's algorithm, payload, preprocessing mode, training domain,
  seed, epoch budget, manifest hash, and sample counts before combining it with
  another run.
- Keep observations separate from interpretation. The plotting task should not
  add causal or statistical-significance claims.
- Do not add uncertainty bars to a one-seed condition.
- For the primary three-seed results, report the arithmetic mean and population
  standard deviation, matching the thesis protocol.
- Compute a transfer gap within each seed before aggregating it.
- Preserve chance-level and negative results.
- Do not treat the 150-epoch pilot or development runs as thesis evidence.
- Use the project instructions in `AGENTS.md` and the relevant files under
  `agent-instructions/` before changing code or interpreting artifacts.

## Required artifact audit

Before plotting, resolve each W&B run below to its canonical local result
directory. For every run, inspect at least:

- `config.json`;
- `dataset_manifest.jsonl` or the recorded manifest reference;
- `summary.json`;
- the relevant file under `history/`;
- selected-checkpoint metadata when it affects a figure;
- the matching review under `notes/run-reviews/`.

If a canonical local artifact is missing, stop that figure and report the exact
missing item. A W&B CSV export may be used only as an explicitly documented
fallback. Do not silently mix local JSON and W&B-derived values.

## Run sets and short labels

### A. Primary natural-domain training

| Legend label | Exact W&B run name |
|---|---|
| Seed 1 | `pilot-real-suniward040-crop-lr1e-4-s1-20261015_112246` |
| Seed 2 | `pilot-real-suniward040-crop-lr1e-4-s2-20261017_121320` |
| Seed 3 | `pilot-real-suniward040-crop-lr1e-4-s3-20261021_070017` |

### B. Primary AI-domain training

| Legend label | Exact W&B run name |
|---|---|
| Seed 1 | `pilot-ai-suniward040-crop-lr1e-4-s1-20261015_182123` |
| Seed 2 | `pilot-ai-suniward040-crop-lr1e-4-s2-20261019_180835` |
| Seed 3 | `pilot-ai-suniward040-crop-lr1e-4-s3-20261020_233927` |

### C. Primary mixed-domain training

| Legend label | Exact W&B run name |
|---|---|
| Seed 1 | `pilot-mixed-suniward040-crop-lr1e-4-s1-20261016_173640` |
| Seed 2 | `pilot-mixed-suniward040-crop-lr1e-4-s2-20261019_215609` |
| Seed 3 | `pilot-mixed-suniward040-crop-lr1e-4-s3-20261021_031924` |

### D. Seed-1 embedding-condition sensitivity

#### S-UNIWARD, 0.4 bpp

- Natural: `pilot-real-suniward040-crop-lr1e-4-s1-20261015_112246`
- AI: `pilot-ai-suniward040-crop-lr1e-4-s1-20261015_182123`
- Mixed: `pilot-mixed-suniward040-crop-lr1e-4-s1-20261016_173640`

#### S-UNIWARD, 0.2 bpp

- Natural: `pilot-real-suniward020-crop-lr1e-4-s1-20261026_160301`
- AI: `pilot-ai-suniward020-crop-lr1e-4-s1-20261027_195919`
- Mixed: `pilot-mixed-suniward020-crop-lr1e-4-s1-20261031_131809`

#### MiPOD, 0.4 bpp

- Natural: `pilot-real-mipod040-crop-lr1e-4-s1-20261031_170217`
- AI: `pilot-ai-mipod040-crop-lr1e-4-s1-20261031_204727`
- Mixed: `pilot-mixed-mipod040-crop-lr1e-4-s1-20261101_002949`

Use these legend labels:

- `S-UNIWARD · 0.4 bpp`
- `S-UNIWARD · 0.2 bpp`
- `MiPOD · 0.4 bpp`

### E. Mixed-training controls

| Legend label | Exact W&B run name |
|---|---|
| Equal-total · 300 ep. | `pilot-mixed-suniward040-crop-lr1e-4-s1-20261016_173640` |
| Equal-total · 500 ep. | `pilot-mixed-suniward040-crop-lr1e-4-s1-20261024_155602` |
| Full-union · 300 ep. | `pilot-mixed-suniward040-crop-lr1e-4-s1-20261025_145314` |

The full-union condition contains more unique training pairs and more optimizer
updates per epoch. Do not interpret it as a training-duration-only comparison.
Do not directly overlay its epoch curve with the equal-total curves as though
one epoch represented the same exposure.

### F. External AI-cover evaluation

Use the completed artifacts reviewed in:

- `notes/run-reviews/Run 12 - SD3 external checkpoint evaluation.md`;
- `notes/run-reviews/Run 13 - FLUX realistic external checkpoint evaluation.md`;
- the primary base-FLUX evaluations from run sets A--C.

The primary external figure uses the three natural-trained, three FLUX-trained,
and three equal-total mixed 300-epoch checkpoints. The 500-epoch and full-union
checkpoints are single-seed controls and must be shown separately if included.
They must not be included in the three-seed mean or standard deviation.

Use these legend labels:

- `Natural-trained`
- `FLUX-trained`
- `Mixed-trained`

Use these dataset labels:

- `Base FLUX`
- `Realistic FLUX`
- `Stable Diffusion 3`

## Runs to exclude

Do not use the following in thesis figures:

- `pilot-real-suniward040-crop-lr1e-4-s1-20261014_130157` (150-epoch pilot);
- `pilot-paired-real`;
- `stage1-real`;
- `rich-dew-1`.

## Required figures

### F01 — Primary validation loss

Create one figure with three aligned subplots:

1. natural-domain training, seeds 1--3;
2. AI-domain training, seeds 1--3;
3. mixed-domain training, seeds 1--3.

Requirements:

- x-axis: epoch;
- y-axis: validation loss;
- no smoothing;
- identical axis policy across the three subplots;
- short seed legend;
- mark the selected checkpoint epoch subtly if the exact selection metadata is
  available for every plotted run; otherwise omit checkpoint markers entirely.

### F02 — Primary validation balanced accuracy

Use the same three-subplot layout as F01, with validation balanced accuracy on
the y-axis. Include a horizontal chance reference at 0.5. Do not describe this
training-time validation measure as the final test result.

### F03 — Primary six-direction balanced accuracy

Plot the final test balanced accuracy for:

- `N→N`;
- `N→A`;
- `A→A`;
- `A→N`;
- `M→N`;
- `M→A`.

Show the three individual seed values and the mean with population-standard-
deviation error bars. Prefer a point-range plot over a dense bar chart. Include
a horizontal reference at 0.5. Do not collapse the six directions into one
headline value.

### F04 — Primary six-direction ROC-AUC

Use the same organization as F03, replacing balanced accuracy with ROC-AUC.
Show individual seeds, mean, and population standard deviation. Include the
0.5 chance reference.

### F05 — Paired cross-domain generalization gaps

For each seed, calculate:

```text
G_N→A = BA_N→N - BA_N→A
G_A→N = BA_A→A - BA_A→N
```

Plot the three seed-level gaps for both transfer directions and show the mean
with population standard deviation. Do not label the difference between
`M→N` and `M→A` as a transfer gap because both domains were present during
mixed training.

### F06 — Embedding-condition sensitivity

Using seed 1 only, compare:

- S-UNIWARD at 0.4 bpp;
- S-UNIWARD at 0.2 bpp;
- MiPOD at 0.4 bpp.

Create:

1. balanced accuracy across all six directions;
2. ROC-AUC across all six directions.

Do not add error bars. State `seed 1 only` in the subtitle or caption. Keep the
direction ordering identical to F03 and F04.

### F07 — Mixed-training controls

Compare the three mixed conditions in run set E on:

- `M→N` balanced accuracy;
- `M→A` balanced accuracy;
- `M→N` ROC-AUC;
- `M→A` ROC-AUC.

Use a compact two-panel figure or another layout that remains readable. Do not
add uncertainty bars. Make the data-volume and epoch-budget distinction clear
in the legend labels.

### F08 — External AI-cover evaluation

For the three primary checkpoint families (natural-trained, FLUX-trained, and
mixed-trained), compare:

- Base FLUX;
- Realistic FLUX;
- Stable Diffusion 3.

Create:

1. balanced accuracy;
2. ROC-AUC.

Show the three checkpoint seeds and the mean with population standard
deviation when all three seed results are available. State clearly that no
detector was trained or threshold-tuned on the two external datasets.

If the 500-epoch or full-union checkpoint is included, add it as a clearly
separated single-seed marker or a separate supplemental figure. Do not merge it
into the primary three-seed aggregates.

## Diagnostic data to preserve but not plot by default

Export the following to the figure data tables, but do not automatically create
a separate main-text figure for each metric:

- accuracy;
- precision;
- recall;
- specificity;
- F1;
- MCC;
- detection error;
- EER;
- confusion-matrix cells.

Recall and specificity may be used in a supplemental error-asymmetry figure if
the thesis discussion explicitly needs to explain false-positive versus
false-negative behavior. Do not add that figure without first checking whether
the existing tables already communicate the result.

## Visual style

### General

- Use Matplotlib, optionally with Seaborn for styling.
- Use a restrained academic layout with a white background and light gridlines.
- Use a readable serif or neutral sans-serif font that embeds correctly in PDF.
- Avoid 3D effects, gradients, shadows, and decorative backgrounds.
- Use sufficiently large text for insertion at thesis page width.
- Keep titles concise; put methodological detail in captions or provenance.
- Prefer direct category labels and short legends over W&B run names.
- Place legends outside the plotting area when they would cover data.
- Avoid redundant legends when the x-axis already identifies every category.
- Use the same direction order everywhere: `N→N`, `N→A`, `A→A`, `A→N`,
  `M→N`, `M→A`.
- Use `N`, `A`, and `M` only after the figure or caption defines them as
  natural, AI-generated, and mixed training domains.

### Seed colors for training curves

Use the same seed colors in the natural, AI, and mixed training panels:

| Seed | Color | Line style | Marker, if used |
|---|---|---|---|
| Seed 1 | blue `#0072B2` | solid | circle |
| Seed 2 | orange `#E69F00` | dashed | square |
| Seed 3 | purple `#CC79A7` | dotted | triangle |

Do not place markers at every epoch. If markers are used, space them sparsely.

### Domain colors for result figures

| Training domain | Legend label | Color |
|---|---|---|
| Natural | Natural-trained | blue `#0072B2` |
| AI/FLUX | FLUX-trained | vermilion `#D55E00` |
| Mixed | Mixed-trained | green `#009E73` |

Use these colors consistently in primary and external-evaluation figures.

### Embedding-condition colors

| Condition | Color |
|---|---|
| S-UNIWARD · 0.4 bpp | blue `#0072B2` |
| S-UNIWARD · 0.2 bpp | purple `#CC79A7` |
| MiPOD · 0.4 bpp | orange `#E69F00` |

### Mixed-control colors

| Condition | Color |
|---|---|
| Equal-total · 300 ep. | blue `#0072B2` |
| Equal-total · 500 ep. | orange `#E69F00` |
| Full-union · 300 ep. | green `#009E73` |

### Axes and uncertainty

- Keep comparable panels on consistent scales.
- Show the 0.5 chance reference for balanced accuracy and ROC-AUC.
- Avoid misleadingly narrow axes. If an axis is truncated, make the truncation
  explicit and retain the chance reference where relevant.
- Use population standard deviation only for the completed three-seed primary
  comparisons.
- Overlay individual seed points so the error bars do not hide the number of
  completed repetitions.

## Output requirements

Create a new timestamped output directory under `results/`, for example:

```text
results/thesis_figures_YYYYMMDD_HHMMSS/
```

Do not overwrite an earlier figure export. The directory must contain:

- one high-resolution PNG for every figure, at least 300 dpi;
- one vector PDF for every figure;
- one CSV containing the exact plotted values for every figure;
- the plotting script or notebook used to generate the figures;
- `PROVENANCE.md`;
- `figure_manifest.json`.

Use stable filenames such as:

```text
F01_primary_validation_loss.pdf
F01_primary_validation_loss.png
F01_primary_validation_loss.csv
```

`PROVENANCE.md` must map every short legend label to:

- the exact W&B run name, when available;
- the local result directory;
- source JSON/JSONL files;
- seed, training domain, algorithm, payload, preprocessing, and epoch budget;
- any missing artifact or W&B fallback;
- the command used to generate the figures.

`figure_manifest.json` must record, for each figure:

- figure ID and title;
- source files;
- source file SHA-256 values;
- selected metric fields;
- aggregation rule;
- output filenames;
- plotting-script version or repository commit;
- repository dirty state.

## Validation before handoff

Before reporting completion:

1. Confirm that every plotted number matches its source artifact.
2. Confirm that clean/stego class metrics and domain directions were not
   interchanged.
3. Confirm that all three primary seeds are present where three-seed uncertainty
   is shown.
4. Confirm that one-seed figures have no uncertainty bars.
5. Confirm that F03--F06 use the same six-direction ordering.
6. Confirm that colors and short labels follow this file.
7. Open every PNG and PDF and check for clipped labels, overlapping legends,
   unreadable text, and missing glyphs.
8. Report which figures were produced, which were blocked, and why.
9. Do not copy figures into `thesis/` or modify LaTeX until the user reviews and
   approves them.
