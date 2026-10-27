---
type: run-review
run: 1
status: reviewed
updated: 2026-10-24
---

# Run 1 — 150-epoch natural-domain pilot

## Identity and purpose

- Date: 2026-10-14
- W&B run: `o3quu45i`
- Result folder: `results/srnet_20261014_130157`
- Role: pilot; one seed, not a final cross-seed result
- Detector: SRNet
- Training domain: natural covers (`real`)
- Evaluations: `real_to_real` and `real_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 1, 150 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Selected checkpoint: epoch 148 by minimum validation loss
- Detailed source record: [[Methods and Findings#7. First working detector: the 1e-4 pilot]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `real_to_real` | 0.7228 | 0.8323 | 0.7286 | 0.4460 | 683, 291 / 249, 725 |
| `real_to_ai` | 0.5000 | 0.4970 | 0.6551 | 0.0000 | 49, 925 / 49, 925 |

- The detector remained at chance-level loss for the first 17 epochs and began
  learning at epoch 18.
- On natural test covers, it correctly classified 683/974 clean images and
  725/974 stego images.
- On AI-generated test covers, it labelled 925/974 clean images and 925/974
  stego images as stego. The two confusion-matrix rows are identical.
- Training was still productive near the end: validation accuracy was 0.663 at
  epoch 100, 0.707 at epoch 130, and 0.724 at epoch 150. The selected checkpoint
  was epoch 148 of 150.
- Train accuracy at epoch 150 was 0.734 and validation accuracy was 0.724, so
  the recorded curve does not show a large train-validation overfitting gap.

## Plain-language meaning

The run established that the corrected training setup can learn the clean-versus-stego
task on natural photographs. The same trained detector did not transfer that
ability to AI-generated covers: it treated almost every AI image as suspicious,
whether the image was clean or stego. Because its AI-domain ROC-AUC was also
approximately 0.5, changing only the classification threshold would not recover
a useful detector.

## Interpretation

The natural-to-AI failure is consistent with severe cover-source mismatch. It
does not yet tell us whether the failure comes from ordinary domain shift or
from S-UNIWARD traces behaving differently in AI-generated covers. The
mixed-domain control is needed to separate those explanations.

## Limitations and status

- This is a one-seed pilot. Its reported standard deviations of zero mean that
  no between-seed variation was measured.
- The checkpoint lies at the end of the training budget, so 0.7228 should be
  treated as a lower-bound natural-domain result, not the detector's ceiling.
- It provides no `ai_to_ai` evidence and therefore cannot answer whether
  steganography is intrinsically easier or harder to detect in AI-generated
  covers.
- The later 300-epoch, multi-seed protocol supersedes this pilot for final
  numerical claims. This run remains useful as the first successful detector
  and as evidence motivating the mixed-domain control.
