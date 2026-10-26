---
type: run-review
run: 14
status: reviewed
updated: 2026-10-26
---

# Run 14 — Full-union mixed-domain seed 1 at 300 epochs

## Identity and purpose

- Date: 2026-10-25 to 2026-10-26
- Result folder: `results/srnet_20261025_145314`
- W&B run: `hgsctm2p`
- Scientific role: mixed-training-volume sensitivity check
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 7,774 pairs total, with all 3,887 pairs from each
  domain; 15,548 images per epoch
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `46f1cb889096aa85476e591e4036f7edd706456e`
- Selected checkpoint: epoch 214 by minimum validation loss
- Runtime: 25,711 seconds, approximately 7 hours 9 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.7305 | 0.8388 | 0.7059 | 0.4676 | 793, 181 / 344, 630 |
| `mixed_to_ai` | 0.6555 | 0.7520 | 0.6203 | 0.3166 | 729, 245 / 426, 548 |

- The minimum validation loss was 0.5285 at epoch 214. Validation balanced
  accuracy at that epoch was 0.693.
- At epoch 300, train accuracy was 0.743, validation balanced accuracy was
  0.680 and validation loss was 0.6101.
- The selected checkpoint is well before the final epoch. Unlike the
  equal-total 300-epoch run, this curve does not suggest that the current run
  was simply stopped too early.
- On natural covers, specificity was 0.8142 and stego recall was 0.6468.
- On AI-generated covers, specificity was 0.7485 and stego recall was 0.5626.

## Comparison with the equal-total mixed seed 1

Both runs use the same seed, manifest, 300-epoch budget, batch size and
learning rate. The intended change is the mixed training-set size: 3,886 pairs
in the equal-total control against 7,774 pairs here.

| Direction | Equal-total 300 epochs | Full union 300 epochs | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | 0.7305 | **+0.0878** |
| `mixed_to_real` ROC-AUC | 0.7553 | 0.8388 | **+0.0835** |
| `mixed_to_ai` balanced accuracy | 0.6304 | 0.6555 | **+0.0251** |
| `mixed_to_ai` ROC-AUC | 0.7192 | 0.7520 | **+0.0328** |
| Natural minus AI balanced-accuracy gap | +0.0123 | +0.0749 | +0.0626 |

The larger run improves both domains, but the improvement is much larger on
natural covers. The fixed-model natural-minus-AI gap grows from 1.23 to 7.49
percentage points.

## Comparison with the 500-epoch equal-total sensitivity

| Direction | Equal-total 500 epochs | Full union 300 epochs | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.7192 | 0.7305 | +0.0113 |
| `mixed_to_real` ROC-AUC | 0.8220 | 0.8388 | +0.0168 |
| `mixed_to_ai` balanced accuracy | 0.6371 | 0.6555 | +0.0184 |
| `mixed_to_ai` ROC-AUC | 0.7260 | 0.7520 | +0.0260 |
| Natural minus AI balanced-accuracy gap | +0.0821 | +0.0749 | -0.0072 |

The full-union run is slightly better on both domains than the 500-epoch
equal-total run. The two sensitivity runs also agree closely on the size of
the natural-minus-AI gap: 7.49 against 8.21 percentage points.

## Plain-language meaning

Giving the mixed model all available training pairs helped. The model became
better on both natural and AI-generated covers, while keeping useful
performance on both. The largest gain was on natural covers, so the almost
equal scores from the smaller 300-epoch mixed run were not stable under a
larger training exposure.

For this seed, AI-generated covers remain the harder side: balanced accuracy
is 65.55% on AI-generated covers and 73.05% on natural covers. This agrees
with the direction seen in the 500-epoch sensitivity run.

## Important design limitation

This is not a pure data-volume comparison. Training is specified in epochs.
Doubling the dataset also doubles the number of optimizer updates per epoch:

- Equal-total, 300 epochs: about 72,900 updates.
- Equal-total, 500 epochs: about 121,500 updates.
- Full union, 300 epochs: about 145,800 updates.

The improvement can therefore come from seeing more unique pairs, receiving
more optimization steps, or both. A step-matched control would be required to
separate those explanations. The result is still useful as a practical test
of the larger mixed-training recipe.

## External AI-cover evaluations

The selected full-union checkpoint was evaluated without retraining on the
same 1,000-pair SD3 and realistic-prompt FLUX test sets used in Runs 12 and
13.

| AI test set | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| Base FLUX | 0.6555 | 0.7520 | 0.6203 | 0.3166 | 729, 245 / 426, 548 |
| Realistic-prompt FLUX | **0.7350** | **0.8412** | 0.7254 | 0.4712 | 770, 230 / 300, 700 |
| SD3 | 0.5880 | 0.6841 | 0.4425 | 0.2063 | 849, 151 / 673, 327 |

Relative to the equal-total 300-epoch mixed seed-1 checkpoint:

| Test set | Balanced-accuracy change | ROC-AUC change |
|---|---:|---:|
| Realistic-prompt FLUX | **+0.0280** | **+0.0293** |
| SD3 | -0.0025 | **+0.0193** |

**Observation.** The larger recipe improves realistic-prompt FLUX on both
metrics. On SD3, balanced accuracy is essentially unchanged while ROC-AUC
improves. The SD3 confusion matrix explains the difference: specificity is
high at 0.849, but stego recall is only 0.327. The model ranks SD3 images more
usefully than before, but its fixed decision threshold still misses most
stego images.

**Interpretation.** More mixed training exposure helps within the FLUX
generator family, including the prompt-shifted set. It does not remove the
generator shift to SD3. Threshold calibration may contribute to the low SD3
balanced accuracy, but choosing a threshold on this test set would be test
leakage; any calibration experiment would need a separate validation set.

Evaluation result folders:

- SD3:
  `results/srnet_eval_genimagepp-sd3-full-union-suniward040_20261026_154127`
- Realistic-prompt FLUX:
  `results/srnet_eval_genimagepp-flux-realistic-full-union-suniward040_20261026_154150`

## Limitations and status

- This is one seed. Its zero standard deviations are bookkeeping values, not
  uncertainty estimates.
- The result does not replace the three-seed equal-total primary condition.
- Both external-evaluation summaries record `repository_dirty: true`. The
  checkpoint hash and evaluation commit are preserved, but the uncommitted GPU
  files were not enumerated in these artifacts.
- The SD3 threshold behavior is an observation, not permission to tune on the
  final test set.

## Evidence used

- `/Users/andrea/Downloads/config (1).yaml`
- `/Users/andrea/Downloads/output (1).log`
- `/Users/andrea/Downloads/wandb-summary (1).json`
- `/Users/andrea/Pictures/config.json`
- `/Users/andrea/Pictures/summary.json`
- `/Users/andrea/Desktop/config.json`
- `/Users/andrea/Desktop/summary.json`
- [[run-reviews/Run 04 - Mixed seed 1 300 epochs]]
- [[run-reviews/Run 11 - Mixed seed 1 500 epochs]]
- [[run-reviews/Run 12 - SD3 external checkpoint evaluation]]
- [[run-reviews/Run 13 - FLUX realistic external checkpoint evaluation]]
