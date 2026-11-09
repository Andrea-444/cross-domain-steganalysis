---
type: run-review
run: 20
status: reviewed
updated: 2026-11-01
---

# Run 20 — Mixed-domain seed 1 with MiPOD at 0.4 bpp

## Identity and purpose

- Date: 2026-11-01
- Result folder: not present in the supplied W&B downloads; recover from the
  GPU host before final reporting
- W&B run ID: not present in the supplied downloads
- Scientific role: mixed-domain control for the first embedding-method
  robustness condition
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: MiPOD, 0.4 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 3,886 pairs total, with 1,943 selected from each domain
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f7d579f51f6c673519c651b1d8f2f3603f3714b7fc52138a07c40e2933884a2d`
- Embedding run: `data/stego/runs/conseal_mipod_p040_20261026_161657`
- Repository commit: `940109e0fcaf7b47c4eb175587fccf1956c8bcfc`
- Lowest rounded validation loss in the supplied log: 0.5603 at epoch 287;
  exact checkpoint-selection metadata was not supplied
- W&B runtime: 13,391 seconds, approximately 3 hours 43 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | EER | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.7284 | 0.8388 | 0.7050 | 0.4628 | 0.2710 | 787, 187 / 342, 632 |
| `mixed_to_ai` | 0.6073 | 0.6850 | 0.6463 | 0.2200 | 0.3871 | 484, 490 / 275, 699 |

- The same trained model detects MiPOD substantially better in natural covers
  than in AI-generated covers.
- On natural covers, specificity is 0.8080 and stego recall is 0.6489. On
  AI-generated covers, specificity falls to 0.4969 while stego recall rises to
  0.7177. The AI result therefore has a strong bias toward predicting stego.
- Training loss first fell below 0.69 at epoch 17. At epoch 300, train accuracy
  was 0.700, validation accuracy was 0.660 and validation loss was 0.5753.
- The lowest rounded validation loss occurs at epoch 287. The maximum logged
  validation accuracy is 0.669 at epoch 283.

## Fixed-model test-domain comparison

The detector, training data, checkpoint and evaluation procedure stay fixed;
only the test cover domain changes.

| Metric | Natural covers | AI-generated covers | Natural minus AI |
|---|---:|---:|---:|
| Balanced accuracy | 0.7284 | 0.6073 | **+0.1211** |
| ROC-AUC | 0.8388 | 0.6850 | **+0.1538** |
| F1 | 0.7050 | 0.6463 | +0.0586 |
| MCC | 0.4628 | 0.2200 | +0.2428 |

The fixed-model gap closely matches the MiPOD specialist gap from Runs 18 and
19: +0.1278 in balanced accuracy and +0.1573 in ROC-AUC. Under MiPOD, both
ways of comparing the cover domains therefore point to a substantial natural
advantage at seed 1.

## Controlled algorithm comparison

Run 4 is the matching mixed-domain S-UNIWARD 0.4 bpp condition. Dataset,
payload, preprocessing, detector, seed, training size and budget stay fixed;
the embedding algorithm changes.

| Direction and metric | S-UNIWARD | MiPOD | Change |
|---|---:|---:|---:|
| `mixed_to_real` balanced accuracy | 0.6427 | 0.7284 | **+0.0857** |
| `mixed_to_real` ROC-AUC | 0.7553 | 0.8388 | **+0.0835** |
| `mixed_to_ai` balanced accuracy | 0.6304 | 0.6073 | **-0.0231** |
| `mixed_to_ai` ROC-AUC | 0.7192 | 0.6850 | **-0.0342** |

MiPOD is easier for this mixed detector to detect in natural covers, but
slightly harder in AI-generated covers. The embedding method therefore changes
the size of the domain gap rather than shifting both domains equally.

## Plain-language meaning

This run changes the overall conclusion in an important way. With S-UNIWARD
at 0.4 bpp, the mixed model performed similarly on natural and AI-generated
covers. With MiPOD at the same payload, the same type of mixed model performs
about 12 percentage points better on natural covers.

The difference between natural and AI-generated covers is therefore not one
fixed number. It depends on which steganography method is used.

## Limitations and status

- This is one seed. The algorithm-by-domain interaction is a strong descriptive
  result, but it has no cross-seed uncertainty estimate.
- The selected checkpoint remains near the end of the 300-epoch budget, so the
  result may still be a lower-bound estimate.
- The comparison covers two embedding algorithms at one shared 0.4-bpp
  payload. It does not establish behavior for steganography methods in general.
- The local result-folder path, W&B run ID and exact checkpoint metadata were
  not included in the supplied downloads.
- By user decision on 2026-11-01, this is the final training run in the current
  thesis experiment set. Further experiments are deferred unless later review
  identifies a necessary gap.

## Evidence used

- `/Users/andrea/Downloads/config (5).yaml`
- `/Users/andrea/Downloads/output (5).log`
- `/Users/andrea/Downloads/wandb-summary (5).json`
- [[run-reviews/Run 04 - Mixed seed 1 300 epochs]]
- [[run-reviews/Run 18 - Natural seed 1 MiPOD 0.4 bpp]]
- [[run-reviews/Run 19 - AI seed 1 MiPOD 0.4 bpp]]
