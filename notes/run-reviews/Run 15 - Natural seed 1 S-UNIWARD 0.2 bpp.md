---
type: run-review
run: 15
status: reviewed
updated: 2026-10-27
---

# Run 15 — Natural-domain seed 1 with S-UNIWARD at 0.2 bpp

## Identity and purpose

- Date: 2026-10-26
- Result folder: `results/srnet_20261026_160301`
- W&B run: `jto6cuff`
- Scientific role: first seed of the lower-payload sensitivity condition
- Detector: SRNet
- Training domain: natural covers (`real`)
- Evaluations: `real_to_real` and `real_to_ai`
- Embedding: S-UNIWARD, 0.2 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs and 974 test
  pairs
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f895975ae0f58a0ac2eb49117e9337259a93ae2b639cb1d70f081d165e8465ea`
- Embedding run:
  `data/stego/runs/conseal_suniward_p020_20261025_193049`
- Repository commit: `d7f0bd2a295170140f5720c1a8cece9dd57de797`
- Selected checkpoint: epoch 286 by minimum validation loss
- Runtime: 13,905 seconds, approximately 3 hours 52 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | EER | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `real_to_real` | 0.5965 | 0.6600 | 0.5638 | 0.1952 | 0.4004 | 654, 320 / 466, 508 |
| `real_to_ai` | 0.5015 | 0.5065 | 0.6089 | 0.0037 | 0.4959 | 221, 753 / 218, 756 |

- Training loss first fell below 0.69 at epoch 119, much later than in the
  0.4 bpp natural seed-1 run.
- Minimum validation loss was 0.6756 at epoch 286. Validation balanced
  accuracy at that epoch was 0.592.
- At epoch 300, train accuracy was 0.622, validation balanced accuracy was
  0.589 and validation loss was 0.7042.
- On natural covers, specificity was 0.6715 and stego recall was 0.5216.
- On AI-generated covers, specificity was 0.2269 and stego recall was 0.7762.
  The high cross-domain F1 is produced by predicting stego frequently, not by
  useful separation; ROC-AUC is 0.5065 and MCC is 0.0037.

## Controlled payload comparison

Run 2 is the matching natural-domain seed-1 condition at 0.4 bpp. The covers,
splits, algorithm, preprocessing, detector, seed, epoch budget, batch size and
learning rate remain fixed. The intended change is the payload.

| Direction and metric | 0.4 bpp | 0.2 bpp | Change |
|---|---:|---:|---:|
| `real_to_real` balanced accuracy | 0.7608 | 0.5965 | **-0.1643** |
| `real_to_real` ROC-AUC | 0.8780 | 0.6600 | **-0.2180** |
| `real_to_real` F1 | 0.7713 | 0.5638 | -0.2075 |
| `real_to_real` MCC | 0.5238 | 0.1952 | -0.3286 |
| `real_to_ai` balanced accuracy | 0.5077 | 0.5015 | -0.0062 |
| `real_to_ai` ROC-AUC | 0.5288 | 0.5065 | -0.0223 |

The same-domain detector remains above chance at 0.2 bpp, but its separation
is much weaker. Cross-domain transfer remains at chance under both payloads.

## Plain-language meaning

Hiding fewer bits makes the steganography substantially harder to detect in
natural covers. The detector still finds some signal at 0.2 bpp, but much less
than at 0.4 bpp. Reducing the payload does not repair transfer to AI-generated
covers: the model still cannot reliably separate clean from stego images after
the cover domain changes.

The smaller transfer gap at 0.2 bpp must not be described as better transfer.
It is smaller because performance on the original natural domain fell sharply,
while the AI-domain result stayed near chance.

## Limitations and status

- This is one seed. It establishes feasibility and an initial payload effect,
  not the final uncertainty of the 0.2 bpp condition.
- The selected checkpoint is epoch 286 of 300. The result remains close to the
  training limit and should be treated as a lower bound.
- The model learned slowly: training loss remained near chance for roughly the
  first 118 epochs. Equal nominal epoch budgets therefore contain fewer useful
  learning epochs at 0.2 bpp than at 0.4 bpp.
- The AI-trained and mixed-domain 0.2 bpp runs are still required before this
  payload can answer the project's domain-difficulty questions.
- No significance claim is made from one seed.

## Evidence used

- `/Users/andrea/Downloads/config.yaml`
- `/Users/andrea/Downloads/output.log`
- `/Users/andrea/Downloads/wandb-summary.json`
- [[run-reviews/Run 02 - Natural seed 1 300 epochs]]
