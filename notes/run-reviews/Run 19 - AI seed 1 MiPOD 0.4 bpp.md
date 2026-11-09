---
type: run-review
run: 19
status: reviewed
updated: 2026-11-01
---

# Run 19 — AI-domain seed 1 with MiPOD at 0.4 bpp

## Identity and purpose

- Date: 2026-10-31
- Result folder: not present in the supplied W&B downloads; recover from the
  GPU host before final reporting
- W&B run ID: not present in the supplied downloads
- Scientific role: AI-domain seed 1 of the first embedding-method robustness
  condition
- Detector: SRNet
- Training domain: AI-generated covers (`ai`)
- Evaluations: `ai_to_ai` and `ai_to_real`
- Embedding: MiPOD, 0.4 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs and 974 test pairs
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f7d579f51f6c673519c651b1d8f2f3603f3714b7fc52138a07c40e2933884a2d`
- Embedding run: `data/stego/runs/conseal_mipod_p040_20261026_161657`
- Repository commit: `940109e0fcaf7b47c4eb175587fccf1956c8bcfc`
- Lowest rounded validation loss in the supplied log: 0.6156 at epoch 296;
  exact checkpoint-selection metadata was not supplied
- W&B runtime: 13,332 seconds, approximately 3 hours 42 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | EER | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `ai_to_ai` | 0.6355 | 0.7212 | 0.6158 | 0.2725 | 0.3686 | 669, 305 / 405, 569 |
| `ai_to_real` | 0.5257 | 0.5654 | 0.2048 | 0.0869 | 0.4589 | 905, 69 / 855, 119 |

- The AI same-domain result is clearly above chance, although weaker than the
  matching natural-domain result from Run 18.
- Reverse-domain transfer remains poor. The detector identifies only 119/974
  natural stego images and predicts stego for just 188/1,948 natural test
  images, or 9.7%.
- Training loss first fell below 0.69 at epoch 10. At epoch 300, train accuracy
  was 0.662, validation accuracy was 0.607 and validation loss was 0.6840.
- The lowest rounded validation loss occurs at epoch 296, again near the end
  of the training budget.

## Controlled algorithm comparison

Run 3 is the matching AI-domain seed-1 condition with S-UNIWARD at 0.4 bpp.
The embedding algorithm is the intended change.

| Direction and metric | S-UNIWARD | MiPOD | Change |
|---|---:|---:|---:|
| `ai_to_ai` balanced accuracy | 0.6242 | 0.6355 | +0.0113 |
| `ai_to_ai` ROC-AUC | 0.7034 | 0.7212 | +0.0178 |
| `ai_to_real` balanced accuracy | 0.5072 | 0.5257 | +0.0185 |
| `ai_to_real` ROC-AUC | 0.5262 | 0.5654 | +0.0392 |

MiPOD is slightly higher in every displayed metric, but the differences come
from one seed and do not establish that MiPOD transfers better. The reverse
direction remains weak and strongly biased toward the clean class.

## Natural-versus-AI MiPOD comparison

Runs 18 and 19 share the manifest, embedding method, payload, preprocessing,
detector and training settings. Their same-domain comparison changes both the
training and test cover domains.

| Same-domain metric | Natural | AI-generated | Natural minus AI |
|---|---:|---:|---:|
| Balanced accuracy | 0.7633 | 0.6355 | +0.1278 |
| ROC-AUC | 0.8784 | 0.7212 | +0.1573 |
| MCC | 0.5271 | 0.2725 | +0.2546 |

This repeats the direction seen with S-UNIWARD: the natural specialist scores
higher than the AI specialist. Run 20 has since completed the fixed-model
comparison and finds a similarly large natural-versus-AI gap.

## Plain-language meaning

The main pattern is not limited to S-UNIWARD. With MiPOD, detection is again
stronger within natural covers than within AI-generated covers, and a model
trained on one domain still transfers poorly to the other.

## Limitations and status

- This is one seed. It supports method robustness as preliminary evidence, not
  a final algorithm-level estimate with uncertainty.
- The specialist comparison changes both training and test domain. Run 20
  supplies the cleaner test-domain comparison with one fixed mixed detector.
- The `ai_to_real` ROC-AUC of 0.5654 shows weak ranking information, but the
  fixed-threshold predictions are heavily biased and balanced accuracy is only
  0.5257. No significance or useful-transfer claim is made.
- The local result-folder path, W&B run ID and exact checkpoint metadata were
  not included in the supplied downloads.

## Evidence used

- `/Users/andrea/Downloads/config (4).yaml`
- `/Users/andrea/Downloads/output (4).log`
- `/Users/andrea/Downloads/wandb-summary (4).json`
- [[run-reviews/Run 03 - AI seed 1 300 epochs]]
- [[run-reviews/Run 18 - Natural seed 1 MiPOD 0.4 bpp]]
- [[run-reviews/Run 20 - Mixed seed 1 MiPOD 0.4 bpp]]
