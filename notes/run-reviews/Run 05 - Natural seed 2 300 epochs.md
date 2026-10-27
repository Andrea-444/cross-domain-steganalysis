---
type: run-review
run: 5
status: reviewed
updated: 2026-10-24
---

# Run 5 — Natural-domain seed 2 at 300 epochs

## Identity and purpose

- Date: 2026-10-17
- Result folder: `results/srnet_20261017_121320`
- Scientific role: natural-domain seed 2 of the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: natural covers (`real`)
- Evaluations: `real_to_real` and `real_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 2, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 289 by minimum validation loss
- Detailed source record: [[Methods and Findings#8. The six directions: first evidence for both research questions]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `real_to_real` | 0.7613 | 0.8781 | 0.7692 | 0.5238 | 708, 266 / 199, 775 |
| `real_to_ai` | 0.5036 | 0.5300 | 0.6510 | 0.0134 | 79, 895 / 72, 902 |

- The model left its initial chance-level plateau around epoch 33.
- The selected checkpoint was epoch 289 of 300, close to the end of the
  training budget.
- Final train accuracy was 0.791 and final validation accuracy was 0.755.
- On natural test covers, the model correctly classified 708/974 clean images
  and 775/974 stego images.
- On AI-generated test covers, it predicted stego for 895/974 clean images and
  902/974 stego images. It therefore called 1,797/1,948, or 92.2%, of all AI
  test images stego.

## Replication of Run 2

Runs 2 and 5 share the manifest, training and test domains, embedding condition,
preprocessing, detector implementation, epoch budget, batch size, learning rate
and repository commit. Only the experiment seed changes.

| Same-domain metric | Run 2: seed 1 | Run 5: seed 2 | Seed 2 minus seed 1 |
|---|---:|---:|---:|
| Balanced accuracy | 0.7608 | 0.7613 | +0.0005 |
| ROC-AUC | 0.8780 | 0.8781 | +0.00003 |
| F1 | 0.7713 | 0.7692 | -0.0021 |
| MCC | 0.5238 | 0.5238 | +0.00003 |

The same-domain result is effectively unchanged between the two seeds.

| Transfer metric | Run 2: seed 1 | Run 5: seed 2 | Seed 2 minus seed 1 |
|---|---:|---:|---:|
| Balanced accuracy | 0.5077 | 0.5036 | -0.0041 |
| ROC-AUC | 0.5288 | 0.5300 | +0.0013 |
| MCC | 0.0292 | 0.0134 | -0.0157 |

The natural-to-AI failure also repeats. Both seeds remain near chance and call
more than 92% of AI test images stego.

## Plain-language meaning

Run 5 confirms that Run 2 was not a lucky initialization. A second independently
seeded natural detector reaches virtually the same same-domain performance and
reproduces the same failure on AI-generated covers.

The replication is especially useful because the learning curves differed.
Seed 1 left the chance plateau around epoch 19, while seed 2 left around epoch
33. Despite that 14-epoch difference, their final natural-domain balanced
accuracies differ by only 0.05 percentage points.

## Interpretation

The natural-domain detector is stable across these first two seeds. The
natural-to-AI collapse is also stable: it is not explained by one unusual
random initialization or by the exact epoch at which the model begins learning.

This strengthens the distribution-shift interpretation established by Run 4.
The natural specialist reliably learns its own domain, but its decision boundary
systematically treats almost every AI-generated cover as stego.

## Limitations and status

- This is the second seed, not the final cross-seed aggregate. A third seed is
  still needed for the accepted primary condition.
- The selected checkpoint is epoch 289 of 300, so both natural-domain seed runs
  remain lower-bound estimates rather than measured performance ceilings.
- Near-identical results across two seeds demonstrate repeatability for this
  condition only. They do not generalize to other algorithms, payloads,
  generators or detectors.
- The high `real_to_ai` F1 score is not evidence of transfer. It is produced by
  predicting stego for 92.2% of AI test images.
- Final claims should use the later three-seed protocol aggregate rather than
  averaging these two values by hand.
