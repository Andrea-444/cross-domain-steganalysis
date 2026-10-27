---
type: run-review
run: 7
status: reviewed
updated: 2026-10-24
---

# Run 7 — Mixed-domain seed 2 at 300 epochs

## Identity and purpose

- Date: 2026-10-19
- Result folder: `results/srnet_20261019_215609`
- Scientific role: mixed-domain control, seed 2 of the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 3,886 pairs total, with 1,943 selected from each domain
- Training: seed 2, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 291 by minimum validation loss
- Detailed source record: [[Methods and Findings#10. Complete three-seed protocol]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.6412 | 0.7272 | 0.6182 | 0.2844 | 683, 291 / 408, 566 |
| `mixed_to_ai` | 0.6227 | 0.7012 | 0.6743 | 0.2587 | 452, 522 / 213, 761 |

- The model left its initial chance-level plateau around epoch 11.
- The selected checkpoint was epoch 291 of 300, close to the end of the
  training budget.
- Final train accuracy was 0.666 and final validation accuracy was 0.616.
- On natural test covers, specificity was 0.7012 and stego recall was 0.5811.
  The model correctly classified 683/974 clean images and 566/974 stego images.
- On AI-generated test covers, specificity was 0.4641 and stego recall was
  0.7813. It correctly classified 452/974 clean images and 761/974 stego images.

## Replication of Run 4

Runs 4 and 7 share the manifest, mixed-domain design, embedding condition,
preprocessing, detector implementation, epoch budget, batch size, learning rate
and repository commit. The seed changes both initialization and the deterministic
mixed-domain sample selection.

| Direction and metric | Run 4: seed 1 | Run 7: seed 2 | Seed 2 minus seed 1 |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | 0.6412 | -0.0015 |
| `mixed_to_real` ROC-AUC | 0.7553 | 0.7272 | -0.0281 |
| `mixed_to_ai` balanced accuracy | 0.6304 | 0.6227 | -0.0077 |
| `mixed_to_ai` ROC-AUC | 0.7192 | 0.7012 | -0.0179 |

Balanced accuracy is stable across the two seeds, while ranking performance
varies somewhat more.

The decision threshold is less stable. Seed 1 predicted stego for 27.4% of
natural and 45.9% of AI test images; seed 2 predicted stego for 44.0% and 65.9%,
respectively. This changes precision, recall and F1 even though the headline
balanced accuracies remain similar.

## The controlled seed-2 test-domain comparison

The same trained model is evaluated on both test domains.

| Metric | `mixed_to_real` | `mixed_to_ai` | Natural minus AI |
|---|---:|---:|---:|
| Balanced accuracy | 0.6412 | 0.6227 | +0.0185 |
| ROC-AUC | 0.7272 | 0.7012 | +0.0259 |
| Detection error | 0.3588 | 0.3773 | -0.0185 |
| MCC | 0.2844 | 0.2587 | +0.0257 |

Seed 2 repeats the seed-1 result: AI-generated covers are the harder test side,
but the fixed-model balanced-accuracy gap remains small.

## Comparison with the seed-2 specialist transfers

| AI test direction | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| `real_to_ai` | 0.5036 | 0.5300 |
| `mixed_to_ai` | 0.6227 | 0.7012 |

| Natural test direction | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| `ai_to_real` | 0.5262 | 0.5541 |
| `mixed_to_real` | 0.6412 | 0.7272 |

Mixed training restores approximately 11.9 balanced-accuracy points on the AI
test and 11.5 points on the natural test relative to the corresponding seed-2
specialist transfers.

## Plain-language meaning

Run 7 replicates the central mixed-control result from Run 4. One model trained
on both cover sources again detects steganography in both. The two test-domain
scores are close: 64.12% on natural covers and 62.27% on AI-generated covers.

This is the second seed showing that the dramatic specialist transfer failures
do not mean the embedding is impossible to detect across the two domains. The
model needs exposure to both cover distributions.

## Interpretation

The distribution-shift explanation is now supported by two mixed-control seeds.
Both recover useful discrimination in both domains, and both show only a small
fixed-model disadvantage on the AI test set.

The change in confusion-matrix balance between seeds indicates threshold or
calibration instability. ROC-AUC also changes more than balanced accuracy.
Therefore, class-specific metrics from a single mixed seed should not be treated
as stable behavior even though the broader control conclusion replicates.

## Limitations and status

- This is the second mixed seed, not the final cross-seed aggregate. A third
  seed is still needed for the accepted primary condition.
- The seed changes both model randomness and which 1,943 covers are selected
  from each domain, so those sources of variation are not separated.
- The final train-validation accuracy gap is 5.0 percentage points, larger than
  mixed seed 1's 3.8-point gap. Late validation behavior is noisy.
- The selected checkpoint is epoch 291 of 300, so the performance ceiling is
  not established.
- The mixed detector still receives half as many domain-specific training pairs
  as a specialist. Specialist-versus-mixed differences retain that data-volume
  confound.
- Final claims should use the later three-seed protocol aggregate rather than
  averaging the first two seeds by hand.
