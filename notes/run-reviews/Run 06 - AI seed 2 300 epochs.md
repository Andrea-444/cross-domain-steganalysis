---
type: run-review
run: 6
status: reviewed
updated: 2026-10-24
---

# Run 6 — AI-domain seed 2 at 300 epochs

## Identity and purpose

- Date: 2026-10-19
- Result folder: `results/srnet_20261019_180835`
- Scientific role: AI-domain seed 2 of the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: AI-generated covers (`ai`)
- Evaluations: `ai_to_ai` and `ai_to_real`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 2, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 268 by minimum validation loss
- Detailed source record: [[Methods and Findings#10. Complete three-seed protocol]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `ai_to_ai` | 0.6391 | 0.7295 | 0.6455 | 0.2784 | 605, 369 / 334, 640 |
| `ai_to_real` | 0.5262 | 0.5541 | 0.4561 | 0.0542 | 638, 336 / 587, 387 |

- The model left its initial chance-level plateau around epoch 9.
- The selected checkpoint was epoch 268 of 300.
- Final train accuracy was 0.689 and final validation accuracy was 0.611,
  producing a larger final train-validation gap than AI seed 1.
- On AI-generated test covers, the model correctly classified 605/974 clean
  images and 640/974 stego images.
- On natural test covers, it correctly classified 638/974 clean images but
  found only 387/974 stego images. It predicted stego for 723/1,948, or 37.1%,
  of all natural test images.

## Replication of Run 3

Runs 3 and 6 share the manifest, training and test domains, embedding condition,
preprocessing, detector implementation, epoch budget, batch size, learning rate
and repository commit. Only the experiment seed changes.

| Same-domain metric | Run 3: seed 1 | Run 6: seed 2 | Seed 2 minus seed 1 |
|---|---:|---:|---:|
| Balanced accuracy | 0.6242 | 0.6391 | +0.0149 |
| ROC-AUC | 0.7034 | 0.7295 | +0.0260 |
| F1 | 0.6269 | 0.6455 | +0.0186 |
| MCC | 0.2485 | 0.2784 | +0.0299 |

Both seeds learn an above-chance AI-domain detector. Seed 2 performs somewhat
better, so the AI-domain condition varies more across these seeds than the
natural-domain condition did.

| Transfer metric | Run 3: seed 1 | Run 6: seed 2 | Seed 2 minus seed 1 |
|---|---:|---:|---:|
| Balanced accuracy | 0.5072 | 0.5262 | +0.0190 |
| ROC-AUC | 0.5262 | 0.5541 | +0.0279 |
| MCC | 0.0162 | 0.0542 | +0.0380 |

Seed 2 carries slightly more signal into natural covers, but the transfer
remains weak and far below either same-domain result.

## Paired seed-2 domain comparison

Run 5 provides the natural-domain specialist for the same experiment seed.

| Same-domain metric | Natural seed 2 | AI seed 2 | AI minus natural |
|---|---:|---:|---:|
| Balanced accuracy | 0.7613 | 0.6391 | -0.1222 |
| ROC-AUC | 0.8781 | 0.7295 | -0.1486 |
| MCC | 0.5238 | 0.2784 | -0.2454 |

The second seed repeats the direction of the seed-1 specialist comparison:
same-domain detection is weaker on AI-generated covers than on natural covers.

## Plain-language meaning

Run 6 confirms that the AI-domain result from Run 3 was not a one-seed accident.
A second independently seeded model again learns to distinguish clean from
stego AI-generated images. It performs slightly better than seed 1, but still
well below the natural detector trained with the same seed and budget.

The AI-to-natural transfer remains poor. Seed 2 is less strongly biased toward
the clean class than seed 1, but its balanced accuracy of 52.62% and ROC-AUC of
0.5541 still do not represent useful cross-domain detection.

## Interpretation

The evidence for weaker specialist detection in the AI-generated domain now
repeats across two paired seeds. The exact size varies, but the direction does
not.

The reverse-transfer failure also repeats, supporting the distribution-shift
interpretation from Run 4. The AI specialist learns useful embedding features
inside its own cover source, but those features and its decision boundary carry
little usable information into natural covers.

## Limitations and status

- This is the second AI seed, not the final cross-seed aggregate. A third seed
  is still needed for the accepted primary condition.
- The final train-validation accuracy gap is 7.7 percentage points, compared
  with 3.7 points for AI seed 1. Validation behavior is also noisy late in
  training. This suggests more overfitting sensitivity in the AI condition.
- The selected checkpoint is still late in the schedule, but the worsening
  validation metrics after it mean that simply adding epochs may not improve
  generalization without a schedule or regularization change.
- The `ai_to_real` metrics are modestly above 0.5 in this seed, but no
  significance or successful-transfer claim is supported by one run.
- Final claims should use the later three-seed protocol aggregate rather than
  averaging the first two seeds by hand.
