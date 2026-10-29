---
type: run-review
run: 16
status: reviewed
updated: 2026-10-29
---

# Run 16 — AI-domain seed 1 with S-UNIWARD at 0.2 bpp

## Identity and purpose

- Date: 2026-10-27
- Result folder: not present in the supplied W&B downloads; recover from the
  GPU host before final reporting
- W&B run: `ittjl1x0`
- Scientific role: AI-domain seed 1 of the lower-payload sensitivity condition
- Detector: SRNet
- Training domain: AI-generated covers (`ai`)
- Evaluations: `ai_to_ai` and `ai_to_real`
- Embedding: S-UNIWARD, 0.2 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs and 974 test
  pairs
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f895975ae0f58a0ac2eb49117e9337259a93ae2b639cb1d70f081d165e8465ea`
- Embedding run:
  `data/stego/runs/conseal_suniward_p020_20261025_193049`
- Repository commit: `d7f0bd2a295170140f5720c1a8cece9dd57de797`
- Lowest rounded validation loss in the supplied log: 0.6925 at epoch 15;
  the exact checkpoint-selection metadata was not supplied
- W&B runtime: 13,359 seconds, approximately 3 hours 43 minutes

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | EER | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| `ai_to_ai` | 0.5067 | 0.5102 | 0.5867 | 0.0145 | 0.4959 | 305, 669 / 292, 682 |
| `ai_to_real` | 0.5026 | 0.5025 | 0.5703 | 0.0054 | 0.4969 | 336, 638 / 331, 643 |

- Both balanced accuracy and ROC-AUC are close to the 0.5 chance reference in
  both directions.
- The same-domain model predicted stego for 1,351/1,948 AI test images, or
  69.4%. The reverse-domain evaluation predicted stego for 1,281/1,948
  natural test images, or 65.8%.
- The apparently moderate F1 scores are therefore produced by frequent stego
  predictions, not by useful class separation. The near-zero MCC and ROC-AUC
  make this visible.
- The paired same-domain-minus-transfer gap is only 0.0041 in balanced
  accuracy and 0.0077 in ROC-AUC because both results are at chance.
- Training loss first fell below 0.69 at epoch 22 and reached 0.6289 at epoch
  300. Validation loss did not follow: the lowest rounded value in the log was
  0.6925 at epoch 15, while the final value was 0.7677.
- Final training accuracy was 0.580 and final validation balanced accuracy was
  0.527. The maximum logged validation accuracy was 0.545 at epoch 287, but
  checkpoint selection uses validation loss rather than accuracy.

## Controlled payload comparison

Run 3 is the matching AI-domain seed-1 condition at 0.4 bpp. The covers,
splits, algorithm, preprocessing, detector, seed, epoch budget, batch size and
learning rate remain fixed. The intended change is the payload.

| Direction and metric | 0.4 bpp | 0.2 bpp | Change |
|---|---:|---:|---:|
| `ai_to_ai` balanced accuracy | 0.6242 | 0.5067 | **-0.1175** |
| `ai_to_ai` ROC-AUC | 0.7034 | 0.5102 | **-0.1932** |
| `ai_to_ai` F1 | 0.6269 | 0.5867 | -0.0402 |
| `ai_to_ai` MCC | 0.2485 | 0.0145 | -0.2340 |
| `ai_to_real` balanced accuracy | 0.5072 | 0.5026 | -0.0046 |
| `ai_to_real` ROC-AUC | 0.5262 | 0.5025 | -0.0237 |

Halving the payload removes almost all useful same-domain discrimination for
this AI-trained seed. Reverse-domain transfer was already near chance at 0.4
bpp and remains there at 0.2 bpp.

## Comparison with the natural 0.2 bpp specialist

| Same-domain metric | Run 15: natural | Run 16: AI-generated | AI minus natural |
|---|---:|---:|---:|
| Balanced accuracy | 0.5965 | 0.5067 | -0.0898 |
| ROC-AUC | 0.6600 | 0.5102 | -0.1498 |
| MCC | 0.1952 | 0.0145 | -0.1807 |

The natural specialist retains a weak but visible signal at 0.2 bpp, whereas
the AI specialist is at chance. This is an observed one-seed specialist gap,
not yet an isolated effect of test cover domain: each model was trained in the
same domain in which it was tested. The mixed-domain run is needed to hold the
trained detector fixed while changing only the test domain.

## Plain-language meaning

At the lower payload, the AI-trained model does not reliably distinguish clean
from stego images, even when tested on AI-generated images from its own domain.
It learns some training-set signal, but that signal does not produce useful
validation or test separation. Its result on natural images is also at chance.

The small gap between same-domain and cross-domain scores must not be described
as successful transfer. There is almost no gap because neither evaluation
works.

## Limitations and status

- This is one seed. It is a sensitivity pilot, not a final estimate with
  uncertainty.
- The supplied W&B files do not contain the local result-folder path or exact
  checkpoint-selection metadata. Those must be recovered from the GPU result
  directory before the result is treated as fully archived.
- Validation loss was best very early in the rounded training log even though
  training loss continued to improve. This is consistent with weak or unstable
  generalization, but it does not by itself identify the cause.
- The natural-versus-AI comparison changes both the training and test domains.
  The mixed-domain 0.2 bpp run remains required for the cleaner paired domain
  comparison.
- No significance claim is made from one seed.

## Evidence used

- `/Users/andrea/Downloads/config (1).yaml`
- `/Users/andrea/Downloads/output (1).log`
- `/Users/andrea/Downloads/wandb-summary (1).json`
- [[run-reviews/Run 03 - AI seed 1 300 epochs]]
- [[run-reviews/Run 15 - Natural seed 1 S-UNIWARD 0.2 bpp]]
