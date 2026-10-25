---
type: run-review
run: 8
status: reviewed
updated: 2026-10-24
---

# Run 8 — AI-domain seed 3 at 300 epochs

## Identity and purpose

- Date: 2026-10-20
- Result folder: `results/srnet_20261020_233927`
- Scientific role: AI-domain seed 3, completing the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: AI-generated covers (`ai`)
- Evaluations: `ai_to_ai` and `ai_to_real`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 3, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 273 by minimum validation loss
- Detailed source record: [[Methods and Findings#10. Complete three-seed protocol]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `ai_to_ai` | 0.6396 | 0.7294 | 0.5966 | 0.2859 | 727, 247 / 455, 519 |
| `ai_to_real` | 0.5092 | 0.5338 | 0.2364 | 0.0264 | 844, 130 / 826, 148 |

- The model left its initial chance-level plateau around epoch 6.
- The selected checkpoint was epoch 273 of 300.
- Final train accuracy was 0.685 and final validation accuracy was 0.623.
- On AI-generated test covers, specificity was 0.7464 and stego recall was
  0.5329. The model correctly classified 727/974 clean images and 519/974
  stego images.
- On natural test covers, specificity was 0.8665 but stego recall was only
  0.1520. It correctly classified 844/974 clean images but only 148/974 stego
  images.
- The model predicted stego for only 278/1,948, or 14.3%, of natural test
  images. This is the strongest clean-class bias among the three AI seeds.

## Comparison with AI seeds 1 and 2

All three runs share the full experimental condition except for the seed.

| `ai_to_ai` metric | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Balanced accuracy | 0.6242 | 0.6391 | 0.6396 |
| ROC-AUC | 0.7034 | 0.7295 | 0.7294 |
| MCC | 0.2485 | 0.2784 | 0.2859 |
| F1 | 0.6269 | 0.6455 | 0.5966 |

Seeds 2 and 3 are nearly identical in balanced accuracy and ROC-AUC. Seed 3's
lower F1 comes from a more conservative decision threshold, not weaker ranking
performance.

| `ai_to_real` metric | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Balanced accuracy | 0.5072 | 0.5262 | 0.5092 |
| ROC-AUC | 0.5262 | 0.5541 | 0.5338 |
| MCC | 0.0162 | 0.0542 | 0.0264 |

All three reverse-transfer results remain weak. Their decision thresholds vary:
the share of natural test images predicted as stego is 26.8%, 37.1% and 14.3%
for seeds 1, 2 and 3, respectively.

## Completed three-seed AI condition

The protocol aggregator merges this run with Runs 3 and 6.

| Direction | Balanced accuracy | ROC-AUC | MCC | EER |
|---|---:|---:|---:|---:|
| `ai_to_ai` | 0.6343 +/- 0.0071 | 0.7208 +/- 0.0122 | 0.2709 +/- 0.0162 | 0.3662 +/- 0.0061 |
| `ai_to_real` | 0.5142 +/- 0.0085 | 0.5380 +/- 0.0118 | 0.0323 +/- 0.0160 | 0.4808 +/- 0.0042 |

The displayed spreads are population standard deviations (`ddof=0`), matching
the current aggregator output.

## Plain-language meaning

Run 8 completes a consistent three-seed result: SRNet can detect S-UNIWARD at
0.4 bpp inside the AI-generated cover domain, averaging 63.43% balanced
accuracy. This is a moderate but repeatable detection signal.

The same AI-trained models do not generalize to natural photographs. Their
average reverse-transfer balanced accuracy is 51.42%, and their prediction bias
changes markedly from seed to seed.

## Interpretation

Same-domain AI detection is real and reproducible under this condition. It is
not caused by a single favorable initialization. All AI models also leave the
initial plateau early, between epochs 6 and 9, so their lower performance than
the natural specialists is not explained by a delayed start to learning.

The reverse-transfer result is consistently weak even though its class bias is
not consistent. This distinction matters: the threshold behavior varies, but
none of the three models develops strong domain-invariant discrimination. The
mixed-control recovery in Runs 4 and 7 remains the evidence that exposure to
both cover sources is necessary.

## Limitations and status

- This run completes the AI seed count, but final research claims still require
  the protocol-level comparison with the completed natural and mixed conditions.
- The three AI runs show final train-validation gaps of approximately 3.7, 7.7
  and 6.2 percentage points. The later seeds show more overfitting sensitivity
  than seed 1.
- Selected checkpoints fall between epochs 244 and 273. The budget does not
  establish a clean performance ceiling, and late validation behavior is noisy.
- The reported standard deviation uses `ddof=0`; a thesis must state that
  convention or replace it consistently with an accepted alternative.
- The result covers one algorithm, payload, AI generator, preprocessing policy
  and detector. It should not be generalized beyond that scope.
