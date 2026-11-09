---
type: run-review
run: 18
status: reviewed
updated: 2026-11-01
---

# Run 18 — Natural-domain seed 1 with MiPOD at 0.4 bpp

## Identity and purpose

- Date: 2026-10-31
- Result folder: not present in the supplied W&B downloads; recover from the
  GPU host before final reporting
- W&B run ID: not present in the supplied downloads
- Scientific role: natural-domain seed 1 of the first embedding-method
  robustness condition
- Detector: SRNet
- Training domain: natural covers (`real`)
- Evaluations: `real_to_real` and `real_to_ai`
- Embedding: MiPOD, 0.4 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs and 974 test pairs
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f7d579f51f6c673519c651b1d8f2f3603f3714b7fc52138a07c40e2933884a2d`
- Embedding run: `data/stego/runs/conseal_mipod_p040_20261026_161657`
- Repository commit: `940109e0fcaf7b47c4eb175587fccf1956c8bcfc`
- Lowest rounded validation loss in the supplied log: 0.4273 at epoch 289;
  exact checkpoint-selection metadata was not supplied
- W&B runtime: 13,502 seconds, approximately 3 hours 45 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | EER | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `real_to_real` | 0.7633 | 0.8784 | 0.7680 | 0.5271 | 0.2320 | 724, 250 / 211, 763 |
| `real_to_ai` | 0.5159 | 0.5430 | 0.6259 | 0.0394 | 0.4713 | 216, 758 / 185, 789 |

- The same-domain natural result shows strong separation: 724/974 clean
  images and 763/974 stego images are classified correctly.
- Cross-domain performance remains weak. The detector calls 1,547/1,948 AI
  test images stego, or 79.4%, and balanced accuracy is only 0.5159.
- Training loss first fell below 0.69 at epoch 40. At epoch 300, train accuracy
  was 0.797, validation accuracy was 0.756 and validation loss was 0.4553.
- The lowest rounded validation loss occurs at epoch 289, close to the end of
  the training budget.

## Controlled algorithm comparison

Run 2 is the matching natural-domain seed-1 condition with S-UNIWARD at 0.4
bpp. Dataset, splits, payload, preprocessing, detector, seed and training
settings stay fixed; the embedding algorithm changes.

| Direction and metric | S-UNIWARD | MiPOD | Change |
|---|---:|---:|---:|
| `real_to_real` balanced accuracy | 0.7608 | 0.7633 | +0.0025 |
| `real_to_real` ROC-AUC | 0.8780 | 0.8784 | +0.0004 |
| `real_to_ai` balanced accuracy | 0.5077 | 0.5159 | +0.0082 |
| `real_to_ai` ROC-AUC | 0.5288 | 0.5430 | +0.0142 |

The natural same-domain results are practically unchanged between the two
embedding methods. MiPOD gives slightly higher cross-domain numbers, but they
remain too close to chance to count as useful transfer.

## Plain-language meaning

Changing from S-UNIWARD to MiPOD does not change the main natural-domain
result. SRNet detects both methods well in natural covers at 0.4 bpp, but the
natural-trained detector still does not generalize reliably to AI-generated
covers.

## Limitations and status

- This is one seed. The small algorithm differences are descriptive and may
  be ordinary seed variation.
- The selected checkpoint remains close to epoch 300, so the same-domain value
  may still be a lower-bound estimate.
- The cross-domain ROC-AUC of 0.5430 is above 0.5 but is not enough, from one
  seed, to support a useful-transfer or significance claim.
- The local result-folder path, W&B run ID and exact checkpoint metadata were
  not included in the supplied downloads.
- The MiPOD mixed-domain run has since completed as Run 20. Its fixed-model
  comparison shows a substantial natural-versus-AI test-domain gap.

## Evidence used

- `/Users/andrea/Downloads/config (3).yaml`
- `/Users/andrea/Downloads/output (3).log`
- `/Users/andrea/Downloads/wandb-summary (3).json`
- [[run-reviews/Run 02 - Natural seed 1 300 epochs]]
- [[run-reviews/Run 20 - Mixed seed 1 MiPOD 0.4 bpp]]
