---
type: run-review
run: 9
status: reviewed
updated: 2026-10-24
---

# Run 9 — Natural-domain seed 3 at 300 epochs

## Identity and purpose

- Date: 2026-10-21
- Result folder: `results/srnet_20261021_070017`
- Scientific role: natural-domain seed 3, completing the 300-epoch natural condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: natural covers (`real`)
- Evaluations: `real_to_real` and `real_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 3, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `4bc1044f359665d6da30b4b7df94e585a19511c7`
- Selected checkpoint: epoch 293 by minimum validation loss
- Detailed source record: [[Methods and Findings#10. Complete three-seed protocol]]

The repository commit differs from Runs 2 and 5. Inspection of the executable
diff shows a change to where evaluation scalars are placed in W&B: final values
are written to the run summary rather than the step history. The model training,
checkpoint selection and metric calculations are unchanged, so this does not
change the scientific condition. It does explain why this run's exported
`wandb-summary.json` has a different metric layout.

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `real_to_real` | 0.7818 | 0.8911 | 0.7880 | 0.5646 | 733, 241 / 184, 790 |
| `real_to_ai` | 0.5087 | 0.5304 | 0.6671 | 0.0569 | 32, 942 / 15, 959 |

- The model left its initial chance-level plateau around epoch 15.
- The selected checkpoint was epoch 293 of 300.
- Final train accuracy was 0.796 and final validation accuracy was 0.773, a
  gap of about 2.3 percentage points.
- On natural test covers, the model correctly classified 733/974 clean images
  and 790/974 stego images.
- On AI-generated test covers, it correctly classified only 32/974 clean
  images and 959/974 stego images.
- It predicted stego for 1,901/1,948, or 97.6%, of all AI-domain test images.
  That extreme class bias makes the transfer F1 score look much better than
  the balanced accuracy and ROC-AUC warrant.

## Comparison with natural seeds 1 and 2

All three runs share the full experimental condition except for the seed.

| `real_to_real` metric | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Balanced accuracy | 0.7608 | 0.7613 | 0.7818 |
| ROC-AUC | 0.8780 | 0.8781 | 0.8911 |
| MCC | 0.5238 | 0.5238 | 0.5646 |

Seed 3 is the strongest natural-domain result, exceeding seed 2 by about 2.05
percentage points in balanced accuracy. It changes the size of the estimate,
but not the qualitative result established by the first two seeds.

| `real_to_ai` metric | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Balanced accuracy | 0.5077 | 0.5036 | 0.5087 |
| ROC-AUC | 0.5288 | 0.5300 | 0.5304 |
| MCC | 0.0292 | 0.0134 | 0.0569 |
| Test images predicted stego | 92.5% | 92.2% | 97.6% |

Cross-domain performance is almost unchanged across seeds and remains close to
chance. All three natural-trained models also show the same strong tendency to
label AI-domain images as stego.

## Completed three-seed natural condition

The protocol aggregator merges this run with Runs 2 and 5.

| Direction | Balanced accuracy | ROC-AUC | Detection error | MCC | EER |
|---|---:|---:|---:|---:|---:|
| `real_to_real` | 0.7680 +/- 0.0098 | 0.8824 +/- 0.0062 | 0.2320 +/- 0.0098 | 0.5374 +/- 0.0192 | 0.2300 +/- 0.0073 |
| `real_to_ai` | 0.5067 +/- 0.0022 | 0.5297 +/- 0.0007 | 0.4933 +/- 0.0022 | 0.0332 +/- 0.0180 | 0.4781 +/- 0.0026 |

The displayed spreads are population standard deviations (`ddof=0`), matching
the current aggregator output.

## Plain-language meaning

Run 9 completes a reproducible natural-domain result. Across three seeds,
SRNet detects S-UNIWARD at 0.4 bpp in natural photographs with 76.80% average
balanced accuracy and 0.8824 average ROC-AUC.

The same models do not transfer meaningfully to AI-generated covers. Their
average balanced accuracy is 50.67%, essentially chance, and their average
ROC-AUC is only 0.5297. The tiny cross-seed spread shows that this failure is
not an unlucky initialization: it is a stable result under the tested setup.

## Interpretation

The within-domain natural signal is repeatable, and seed 3 provides the best
individual estimate without changing the conclusion. In contrast, the
cross-domain collapse is more stable than the same-domain score: balanced
accuracy varies by only about 0.22 percentage points across the three natural
seeds.

The high transfer F1 of 0.6671 is misleading in isolation. The model predicts
nearly every AI-domain image as stego, so it obtains very high stego recall
while almost completely failing on clean images. Balanced accuracy, MCC,
ROC-AUC and the confusion matrix expose that failure.

Together with the mixed-domain controls, this supports the interpretation that
the natural specialist relies heavily on cover-domain-specific evidence rather
than learning a detector that transfers cleanly across the two cover sources.

## Limitations and status

- The checkpoint was selected very late, at epoch 293. The 300-epoch budget
  therefore does not establish a clear performance ceiling.
- The result completes the natural seed count, but final claims should be based
  on the protocol-level comparison of natural, AI and mixed conditions.
- The transfer F1 must not be presented without the confusion matrix or
  class-balanced metrics because the threshold is extremely biased.
- The reported standard deviation uses `ddof=0`; a thesis must state that
  convention or replace it consistently with an accepted alternative.
- The result covers one algorithm, payload, AI generator, preprocessing policy
  and detector. It should not be generalized beyond that scope.
