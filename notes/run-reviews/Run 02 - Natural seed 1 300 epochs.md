---
type: run-review
run: 2
status: reviewed
updated: 2026-10-24
---

# Run 2 — Natural-domain seed 1 at 300 epochs

## Identity and purpose

- Date: 2026-10-15
- Result folder: `results/srnet_20261015_112246`
- Scientific role: seed 1 of the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: natural covers (`real`)
- Evaluations: `real_to_real` and `real_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 287 by minimum validation loss
- Detailed source record: [[Methods and Findings#8. The six directions: first evidence for both research questions]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `real_to_real` | 0.7608 | 0.8780 | 0.7713 | 0.5238 | 696, 278 / 188, 786 |
| `real_to_ai` | 0.5077 | 0.5288 | 0.6544 | 0.0292 | 81, 893 / 66, 908 |

- The model left its initial chance-level plateau around epoch 19.
- The selected checkpoint was epoch 287 of 300, close to the training limit.
- Final train accuracy was 0.793 and final validation accuracy was 0.755.
- On natural test covers, the model correctly classified 696/974 clean images
  and 786/974 stego images.
- On AI-generated test covers, it predicted stego for 893/974 clean images and
  908/974 stego images. It therefore called 1,801/1,948, or 92.5%, of all AI
  test images stego, largely regardless of the true class.

## Comparison with Run 1

The training budget changed from 150 to 300 epochs. The manifest, seed, domains,
embedding condition, batch size and learning rate stayed fixed. The two runs
record different repository commits, but a direct comparison of those commits
shows changes only in project notes and no change to training or evaluation
code. This is therefore a controlled training-budget comparison.

| Metric | Run 1: 150 epochs | Run 2: 300 epochs | Change |
|---|---:|---:|---:|
| `real_to_real` balanced accuracy | 0.7228 | 0.7608 | +0.0380 |
| `real_to_real` ROC-AUC | 0.8323 | 0.8780 | +0.0458 |
| `real_to_ai` balanced accuracy | 0.5000 | 0.5077 | +0.0077 |
| `real_to_ai` ROC-AUC | 0.4970 | 0.5288 | +0.0318 |

## Plain-language meaning

Giving the natural-image detector twice as many epochs made it clearly better
on natural covers. It did not make the detector useful on AI-generated covers.
The model still treated almost every AI image as stego and remained close to
chance at distinguishing clean from stego AI images.

## Interpretation

The added training budget improves learning inside the training domain but does
not solve cross-domain transfer. This makes insufficient training an unlikely
explanation for the natural-to-AI collapse. The result instead supports a
strong domain-mismatch explanation, although this run alone still cannot
separate ordinary distribution shift from domain-specific embedding traces.

## Limitations and status

- This file describes one seed. Cross-seed uncertainty must come from the
  protocol aggregate, not the zero standard deviations in this run summary.
- The selected checkpoint is near the end of the 300-epoch budget, so the
  natural-domain score is still best treated as a lower bound.
- The natural-to-AI ROC-AUC of 0.5288 is slightly above 0.5 but remains too weak
  to support useful transfer. No significance claim is made from this one seed.
- The high cross-domain F1 score is not evidence of successful transfer. It is
  produced by predicting the stego class for 92.5% of AI test images.
- This run contributes the natural seed-1 record to the later three-seed
  aggregate. Final claims should use that aggregate rather than this run alone.
