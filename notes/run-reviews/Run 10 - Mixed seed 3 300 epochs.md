---
type: run-review
run: 10
status: reviewed
updated: 2026-10-24
---

# Run 10 — Mixed-domain seed 3 at 300 epochs

## Identity and purpose

- Date: 2026-10-21
- Result folder: `results/srnet_20261021_031924`
- Scientific role: mixed-domain control, seed 3, completing the 300-epoch mixed condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 3,886 pairs total, with 1,943 selected from each domain
- Training: seed 3, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `4bc1044f359665d6da30b4b7df94e585a19511c7`
- Selected checkpoint: epoch 298 by minimum validation loss
- Detailed source record: [[Methods and Findings#10. Complete three-seed protocol]]

The repository commit differs from mixed seeds 1 and 2. The executable change
only moves final evaluation scalars from the W&B step history to the run
summary; training, checkpoint selection and metric calculations are unchanged.
The scientific condition therefore remains comparable.

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.6504 | 0.7466 | 0.6541 | 0.3009 | 623, 351 / 330, 644 |
| `mixed_to_ai` | 0.6376 | 0.7230 | 0.6361 | 0.2752 | 625, 349 / 357, 617 |

- The model left its initial chance-level plateau around epoch 10.
- The selected checkpoint was epoch 298 of 300, at the end of the training
  budget.
- Final train accuracy was 0.681 and final validation accuracy was 0.617, a
  gap of about 6.4 percentage points.
- On natural test covers, specificity was 0.6396 and stego recall was 0.6612.
  The model correctly classified 623/974 clean images and 644/974 stego images.
- On AI-generated test covers, specificity was 0.6417 and stego recall was
  0.6335. It correctly classified 625/974 clean images and 617/974 stego images.
- The decision threshold was unusually balanced in this seed: it predicted
  stego for 51.1% of natural test images and 49.6% of AI-generated test images.

## Comparison with mixed seeds 1 and 2

All three runs share the experimental condition except for the seed. The seed
also changes the deterministic selection of 1,943 training pairs from each
domain.

| `mixed_to_real` metric | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Balanced accuracy | 0.6427 | 0.6412 | 0.6504 |
| ROC-AUC | 0.7553 | 0.7272 | 0.7466 |
| F1 | 0.5385 | 0.6182 | 0.6541 |
| MCC | 0.3199 | 0.2844 | 0.3009 |

| `mixed_to_ai` metric | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Balanced accuracy | 0.6304 | 0.6227 | 0.6376 |
| ROC-AUC | 0.7192 | 0.7012 | 0.7230 |
| F1 | 0.6146 | 0.6743 | 0.6361 |
| MCC | 0.2617 | 0.2587 | 0.2752 |

Seed 3 gives the strongest balanced accuracy in both test domains. The change
is modest and remains within the cross-seed variation. Its more balanced
confusion matrices also show why F1 varies independently of the ranking and
balanced-accuracy results.

## The controlled seed-3 test-domain comparison

The same trained model is evaluated on both test domains.

| Metric | `mixed_to_real` | `mixed_to_ai` | Natural minus AI |
|---|---:|---:|---:|
| Balanced accuracy | 0.6504 | 0.6376 | +0.0128 |
| ROC-AUC | 0.7466 | 0.7230 | +0.0236 |
| Detection error | 0.3496 | 0.3624 | -0.0128 |
| MCC | 0.3009 | 0.2752 | +0.0257 |

Seed 3 repeats the direction seen in seeds 1 and 2: AI-generated covers are
slightly harder for the same mixed-trained model. The gap remains small.

## Comparison with the seed-3 specialist transfers

| AI test direction | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| `real_to_ai` | 0.5087 | 0.5304 |
| `mixed_to_ai` | 0.6376 | 0.7230 |

| Natural test direction | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| `ai_to_real` | 0.5092 | 0.5338 |
| `mixed_to_real` | 0.6504 | 0.7466 |

Mixed training restores about 12.9 balanced-accuracy points on the AI test and
14.1 points on the natural test relative to the corresponding seed-3 specialist
transfers.

## Completed three-seed mixed condition

The protocol aggregator merges this run with Runs 4 and 7.

| Direction | Balanced accuracy | ROC-AUC | Detection error | MCC | EER |
|---|---:|---:|---:|---:|---:|
| `mixed_to_real` | 0.6448 +/- 0.0040 | 0.7430 +/- 0.0118 | 0.3552 +/- 0.0040 | 0.3017 +/- 0.0145 | 0.3480 +/- 0.0085 |
| `mixed_to_ai` | 0.6302 +/- 0.0061 | 0.7145 +/- 0.0095 | 0.3698 +/- 0.0061 | 0.2652 +/- 0.0072 | 0.3665 +/- 0.0059 |

The paired natural-minus-AI gap is +0.0145 +/- 0.0028 in balanced accuracy and
+0.0286 +/- 0.0054 in ROC-AUC. The displayed spreads are population standard
deviations (`ddof=0`), matching the current aggregator output.

## Plain-language meaning

Run 10 completes a consistent result across three seeds: a detector trained on
both cover sources can detect steganography in both. Average balanced accuracy
is 64.48% on natural covers and 63.02% on AI-generated covers.

AI-generated covers are slightly harder for the same detector, by about 1.45
balanced-accuracy points on average. This controlled gap is much smaller than
the 13.37-point difference between the two separately trained specialists.

## Interpretation

The mixed-domain control shows that the failed specialist transfers are not
evidence that the embedding traces require incompatible detectors. When SRNet
sees both cover distributions during training, it recovers useful
discrimination on both test domains. This supports distribution shift as the
explanation for the transfer collapse under the tested setup.

The mixed detector also provides the cleaner comparison of domain difficulty
because the trained model is held fixed. All three seeds give natural covers a
small advantage, but this does not support treating the much larger specialist
gap as pure domain difficulty. Training-set size and specialization remain
confounded in that comparison.

## Limitations and status

- This run completes the mixed seed count and the ten-run review set, but final
  claims should use the protocol-level aggregate rather than this seed alone.
- The seed changes both model randomness and the selected mixed-domain training
  subset, so those two sources of variation are not separated.
- The selected checkpoint is epoch 298 of 300, so the performance ceiling is
  not established.
- The final train-validation gap of 6.4 percentage points is the largest among
  the three mixed seeds and indicates some overfitting sensitivity.
- Each mixed detector receives only 1,943 training pairs from each domain,
  versus 3,887 for a specialist. Specialist-versus-mixed differences therefore
  retain a data-volume confound.
- The reported standard deviation uses `ddof=0`; a thesis must state that
  convention or replace it consistently with an accepted alternative.
- The result covers one algorithm, payload, AI generator, preprocessing policy
  and detector. It should not be generalized beyond that scope.
