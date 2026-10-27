---
type: run-review
run: 3
status: reviewed
updated: 2026-10-24
---

# Run 3 — AI-domain seed 1 at 300 epochs

## Identity and purpose

- Date: 2026-10-15
- Result folder: `results/srnet_20261015_182123`
- Scientific role: AI-domain seed 1 of the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: AI-generated covers (`ai`)
- Evaluations: `ai_to_ai` and `ai_to_real`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Data per domain: 3,887 training pairs, 969 validation pairs, 974 test pairs
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 244 by minimum validation loss
- Detailed source record: [[Methods and Findings#8. The six directions: first evidence for both research questions]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `ai_to_ai` | 0.6242 | 0.7034 | 0.6269 | 0.2485 | 601, 373 / 359, 615 |
| `ai_to_real` | 0.5072 | 0.5262 | 0.3583 | 0.0162 | 720, 254 / 706, 268 |

- The model left its initial chance-level plateau around epoch 7.
- The selected checkpoint was epoch 244 of 300, in the final fifth of the
  training schedule.
- Final train accuracy was 0.660 and final validation accuracy was 0.624.
- On AI-generated test covers, the model correctly classified 601/974 clean
  images and 615/974 stego images. Detection is above chance but substantially
  weaker than the natural-domain result from Run 2.
- On natural test covers, it correctly rejected 720/974 clean images but found
  only 268/974 stego images. It predicted stego for 522/1,948, or 26.8%, of all
  natural test images.

## Comparison with Run 2

Runs 2 and 3 share the manifest, seed, embedding condition, preprocessing,
detector implementation, epoch budget, batch size, learning rate and repository
commit. The training and test cover domain change together.

| Same-domain metric | Run 2: natural | Run 3: AI-generated | AI minus natural |
|---|---:|---:|---:|
| Balanced accuracy | 0.7608 | 0.6242 | -0.1366 |
| ROC-AUC | 0.8780 | 0.7034 | -0.1746 |
| F1 | 0.7713 | 0.6269 | -0.1444 |
| MCC | 0.5238 | 0.2485 | -0.2753 |

The two cross-domain directions are nearly symmetric in overall discrimination:

| Cross-domain metric | `real_to_ai` | `ai_to_real` |
|---|---:|---:|
| Balanced accuracy | 0.5077 | 0.5072 |
| ROC-AUC | 0.5288 | 0.5262 |

Their prediction biases point in opposite directions. The natural-trained model
called 92.5% of AI images stego, whereas the AI-trained model called only 26.8%
of natural images stego.

## Plain-language meaning

This run provides the first evidence that SRNet can detect S-UNIWARD at 0.4 bpp
inside the AI-generated cover domain: 62.42% balanced accuracy is above the 50%
chance reference. The signal is much weaker than the 76.08% obtained on natural
covers under the matched specialist setup. The AI-trained detector also fails
to carry its ability back to natural photographs.

## Interpretation

The paired specialist results suggest that this embedding condition is harder
for SRNet to detect in AI-generated covers than in natural covers. However,
this comparison changes both the training domain and the test domain, so it
mixes cover-domain difficulty with differences in what each specialist learned.
The mixed-domain detector is required for a cleaner comparison in which the
trained model stays fixed and only the test domain changes.

The reverse transfer failure strengthens the domain-mismatch hypothesis. The
two specialists each work in their own domain and collapse to approximately
chance in the other, with opposite class biases. That pattern is consistent
with a decision boundary tied to cover-source statistics rather than a
domain-invariant embedding detector.

## Limitations and status

- This file describes one seed. Cross-seed uncertainty must come from the
  protocol aggregate, not the zero standard deviations in this run summary.
- The model escaped its initial plateau earlier than the natural model, so its
  lower same-domain result is not explained by a later start to learning.
- The selected checkpoint is still in the final fifth of the schedule. The
  project source record therefore treats the score as a lower bound rather
  than a measured performance ceiling.
- The 13.66-point specialist gap is an observation, not yet an isolated causal
  effect of test cover domain. The mixed-domain comparison is needed before
  making that claim.
- The `ai_to_real` ROC-AUC of 0.5262 is slightly above 0.5 but remains too weak
  to support useful transfer. No significance claim is made from this one seed.
- This run contributes the AI seed-1 record to the later three-seed aggregate.
  Final claims should use that aggregate rather than this run alone.
