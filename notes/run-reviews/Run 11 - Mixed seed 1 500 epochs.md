---
type: run-review
run: 11
status: reviewed
updated: 2026-10-25
---

# Run 11 — Mixed-domain seed 1 at 500 epochs

## Identity and purpose

- Date: 2026-10-24 to 2026-10-25
- Result folder: `results/srnet_20261024_155602`
- W&B run: `j4o9fc8u`
- Scientific role: training-budget sensitivity check
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 3,886 pairs total, with 1,943 from each domain
- Training: seed 1, 500 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `32baab2e9ae4227294a4e1ca0722418fbfbc8608`
- Selected checkpoint: epoch 472 by minimum validation loss
- Runtime: 22,204 seconds, approximately 6 hours 10 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.7192 | 0.8220 | 0.6983 | 0.4427 | 768, 206 / 341, 633 |
| `mixed_to_ai` | 0.6371 | 0.7260 | 0.6350 | 0.2741 | 626, 348 / 359, 615 |

- The model left the chance-level training-loss plateau at epoch 12.
- Minimum validation loss was 0.5646 at epoch 472. Validation balanced
  accuracy at that epoch was 0.674.
- Final train accuracy was 0.716 and final validation balanced accuracy was
  0.669, a gap of 0.047.
- The selected checkpoint is late, but not at the final epoch. The final
  validation loss, 0.5656, was very close to the minimum.

## Direct comparison with the 300-epoch mixed seed 1

Both runs use the same seed, manifest, mixed training size, batch size and
learning rate. The planned change is the training budget.

| Direction | 300 epochs | 500 epochs | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | 0.7192 | **+0.0765** |
| `mixed_to_real` ROC-AUC | 0.7553 | 0.8220 | **+0.0667** |
| `mixed_to_ai` balanced accuracy | 0.6304 | 0.6371 | +0.0067 |
| `mixed_to_ai` ROC-AUC | 0.7192 | 0.7260 | +0.0068 |
| natural minus AI balanced-accuracy gap | +0.0123 | +0.0821 | +0.0698 |

The longer budget helped the natural-cover side much more than the
AI-generated-cover side. The mixed detector's natural-cover balanced accuracy
rose by 7.65 percentage points, while its AI-cover score rose by only 0.67
points.

## Plain-language meaning

At 300 epochs, the mixed model made the two cover domains look almost equally
difficult. At 500 epochs, it became much better on natural covers but improved
only slightly on AI-generated covers. The earlier small domain gap was
therefore partly a consequence of stopping training early; it was not a stable
estimate of the model's eventual behavior.

This result strengthens the evidence that steganography is harder to detect in
the current AI-generated covers. It does not yet establish the final size of
that difference because this is one seed.

## Interpretation and next step

The 500-epoch checkpoint was selected at epoch 472 and the final validation
loss was nearly identical. For this seed, 500 epochs appears much closer to a
useful stopping point than 300 epochs. There is no clear reason from this curve
to extend immediately beyond 500.

The next experiment should be the separately planned full-union mixed model at
300 epochs, with 3,887 pairs per domain. That changes training-set size while
returning to the primary training budget. It must be interpreted separately
from this budget sensitivity run.

## Limitations and status

- This is one seed. Its zero standard deviations are bookkeeping values, not
  uncertainty estimates.
- The comparison uses two separate CUDA runs. The nominal seed and data are
  the same, but GPU training is not guaranteed to reproduce every update
  bit-for-bit.
- The repository commits differ because later code added tracking and an
  inactive training-pair cap. With no cap supplied, the mixed selection and
  training path remain unchanged.
- This sensitivity run does not replace the three-seed 300-epoch primary
  condition.
- The full-union experiment is still needed to separate training duration from
  training-data quantity.

## Evidence used

- `/Users/andrea/Downloads/config.yaml`
- `/Users/andrea/Downloads/output.log`
- `/Users/andrea/Downloads/wandb-summary.json`
- [[run-reviews/Run 04 - Mixed seed 1 300 epochs]]
