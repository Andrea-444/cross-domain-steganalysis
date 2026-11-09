---
type: run-review
run: 17
status: reviewed
updated: 2026-11-01
---

# Run 17 — Mixed-domain seed 1 with S-UNIWARD at 0.2 bpp

## Identity and purpose

- Date: 2026-10-31
- Result folder: not present in the supplied W&B downloads; recover from the
  GPU host before final reporting
- W&B run ID: not present in the supplied downloads
- Scientific role: mixed-domain control for the lower-payload sensitivity
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: S-UNIWARD, 0.2 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 3,886 pairs total, with 1,943 selected from each domain
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f895975ae0f58a0ac2eb49117e9337259a93ae2b639cb1d70f081d165e8465ea`
- Embedding run:
  `data/stego/runs/conseal_suniward_p020_20261025_193049`
- Repository commit: `940109e0fcaf7b47c4eb175587fccf1956c8bcfc`
- Lowest rounded validation loss in the supplied log: 0.6927 at epoch 31;
  exact checkpoint-selection metadata was not supplied
- W&B runtime: 13,439 seconds, approximately 3 hours 44 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | EER | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.5026 | 0.5042 | 0.5483 | 0.0052 | 0.4979 | 391, 583 / 386, 588 |
| `mixed_to_ai` | 0.5046 | 0.5129 | 0.5971 | 0.0104 | 0.4938 | 268, 706 / 259, 715 |

- Both balanced accuracy and ROC-AUC are close to the 0.5 chance reference.
- The natural-domain evaluation predicts stego for 1,171/1,948 images, or
  60.1%. The AI-domain evaluation predicts stego for 1,421/1,948 images, or
  72.9%. The F1 values are therefore inflated by frequent stego predictions.
- The fixed-model natural-minus-AI gap is -0.0021 in balanced accuracy and
  -0.0087 in ROC-AUC. This small gap is not evidence of equal difficulty,
  because the detector has no useful discrimination in either domain.
- Training loss first fell below 0.69 at epoch 49. It reached 0.6641 at epoch
  300, while final validation loss was 1.2457 and validation accuracy was
  0.519. The highest logged validation accuracy was 0.520 at epoch 264.

## Controlled payload comparison

Run 4 is the matching mixed-domain seed-1 condition at 0.4 bpp. The covers,
splits, algorithm, preprocessing, detector and training settings remain fixed;
the intended change is the payload.

| Direction and metric | 0.4 bpp | 0.2 bpp | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | 0.5026 | **-0.1401** |
| `mixed_to_real` ROC-AUC | 0.7553 | 0.5042 | **-0.2511** |
| `mixed_to_ai` balanced accuracy | 0.6304 | 0.5046 | **-0.1258** |
| `mixed_to_ai` ROC-AUC | 0.7192 | 0.5129 | **-0.2063** |

Halving the payload removes the useful discrimination that the mixed detector
showed at 0.4 bpp. The lower-payload condition is at chance on both test
domains under this one-seed, 300-epoch training recipe.

## Plain-language meaning

At 0.2 bpp, the mixed model does not reliably tell clean and stego images
apart in either natural or AI-generated covers. Training on both cover domains
does not recover the weak signal at this payload.

The nearly zero difference between its two test scores should not be read as
proof that the domains are equally difficult. The comparison has reached a
floor: both scores are approximately random.

## Limitations and status

- This is one seed and does not measure uncertainty across initializations.
- The supplied downloads do not include the local result-folder path, W&B run
  ID or exact checkpoint-selection metadata. These should be recovered from
  the GPU host before final archival.
- The model learns some training-set signal while validation performance stays
  near chance. The run does not identify whether more training, a different
  detector or the payload itself is the limiting factor.
- The result supports a practical finding about this SRNet recipe at 0.2 bpp;
  it is not evidence that 0.2-bpp S-UNIWARD is generally undetectable.

## Evidence used

- `/Users/andrea/Downloads/config (2).yaml`
- `/Users/andrea/Downloads/output (2).log`
- `/Users/andrea/Downloads/wandb-summary (2).json`
- [[run-reviews/Run 04 - Mixed seed 1 300 epochs]]
- [[run-reviews/Run 15 - Natural seed 1 S-UNIWARD 0.2 bpp]]
- [[run-reviews/Run 16 - AI seed 1 S-UNIWARD 0.2 bpp]]
