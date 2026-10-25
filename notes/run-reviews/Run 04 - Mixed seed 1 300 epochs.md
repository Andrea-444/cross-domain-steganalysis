---
type: run-review
run: 4
status: reviewed
updated: 2026-10-24
---

# Run 4 — Mixed-domain seed 1 at 300 epochs

## Identity and purpose

- Date: 2026-10-16
- Result folder: `results/srnet_20261016_173640`
- Scientific role: mixed-domain control, seed 1 of the 300-epoch primary condition
- W&B configuration role: `pilot`, because this invocation contains one seed
- Detector: SRNet
- Training domain: equal mixture of natural and AI-generated covers (`mixed`)
- Evaluations: `mixed_to_real` and `mixed_to_ai`
- Embedding: S-UNIWARD, 0.4 bpp, per-image embedding seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Available data per domain: 3,887 training pairs, 969 validation pairs and
  974 test pairs
- Mixed training data: 3,886 pairs total, with 1,943 selected from each domain
- Training: seed 1, 300 epochs, batch size 32, learning rate 1e-4
- Manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Repository commit: `d5168c07cb643c51c894f1a291c91bedb23c1e4c`
- Selected checkpoint: epoch 298 by minimum validation loss
- Detailed source record: [[Methods and Findings#8. The six directions: first evidence for both research questions]]

## Recorded observations

| Direction | Balanced accuracy | ROC-AUC | F1 | MCC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| `mixed_to_real` | 0.6427 | 0.7553 | 0.5385 | 0.3199 | 846, 128 / 568, 406 |
| `mixed_to_ai` | 0.6304 | 0.7192 | 0.6146 | 0.2617 | 654, 320 / 400, 574 |

- The model left its initial chance-level plateau around epoch 18.
- The selected checkpoint was epoch 298 of 300, at the end of the training
  budget.
- Final train accuracy was 0.683 and final validation accuracy was 0.645.
- On natural test covers, the model had high specificity (0.8686) but low stego
  recall (0.4168). It correctly classified 846/974 clean images and 406/974
  stego images.
- On AI-generated test covers, its class behavior was more balanced: specificity
  was 0.6715 and stego recall was 0.5893. It correctly classified 654/974 clean
  images and 574/974 stego images.

## The controlled test-domain comparison

The same trained model is evaluated on both domains, so the test cover domain
is the main variable that changes between the two rows.

| Metric | `mixed_to_real` | `mixed_to_ai` | Natural minus AI |
|---|---:|---:|---:|
| Balanced accuracy | 0.6427 | 0.6304 | +0.0123 |
| ROC-AUC | 0.7553 | 0.7192 | +0.0361 |
| Detection error | 0.3573 | 0.3696 | -0.0123 |
| MCC | 0.3199 | 0.2617 | +0.0583 |

At seed 1, AI-generated covers are the harder side, but the balanced-accuracy
gap is only 1.23 percentage points when the trained detector stays fixed.

## Comparison with the specialist and transfer runs

| Test domain | Specialist | Mixed | Mixed minus specialist |
|---|---:|---:|---:|
| Natural covers | `real_to_real` 0.7608 | `mixed_to_real` 0.6427 | -0.1181 |
| AI-generated covers | `ai_to_ai` 0.6242 | `mixed_to_ai` 0.6304 | +0.0062 |

The mixed detector uses only 1,943 training pairs from each domain, compared
with 3,887 pairs for a specialist. Its lower natural-domain result therefore
cannot be attributed to mixed training alone; it also receives half as much
natural-domain training data. Conversely, its AI-domain result matches the AI
specialist despite receiving half as many AI training pairs.

The most important comparison is with the two cross-domain transfers:

| AI test direction | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| `real_to_ai` | 0.5077 | 0.5288 |
| `mixed_to_ai` | 0.6304 | 0.7192 |

| Natural test direction | Balanced accuracy | ROC-AUC |
|---|---:|---:|
| `ai_to_real` | 0.5072 | 0.5262 |
| `mixed_to_real` | 0.6427 | 0.7553 |

Exposure to both cover domains during training restores useful discrimination
on both test domains.

## Plain-language meaning

This is the control that makes Runs 2 and 3 interpretable. A detector trained
on only one cover source fails when moved to the other source, but one detector
trained on both sources detects steganography in both. The two cover domains
therefore do not require mutually incompatible detectors.

The cleanest seed-1 estimate of the test-domain difference is small: the mixed
detector scores 64.27% on natural covers and 63.04% on AI-generated covers.
That 1.23-point gap is much smaller than the 13.66-point gap between the two
specialists.

## Interpretation

The recovery of both test directions supports cover-source distribution shift
as the primary explanation for the failed specialist transfers. The specialists
learn decision boundaries tied to their own cover statistics. Once the model
has seen both cover sources, it can detect the embedding in both.

This does not prove that the embedding traces are identical across domains.
It shows that they are learnable within one model when both cover distributions
are represented during training.

The small mixed-domain gap complicates the initial claim that steganography is
much harder to detect in AI-generated covers. The specialist comparison and
the fixed-model comparison agree on the direction of the effect but disagree
by roughly a factor of ten on its size. Training-set size or mixed-model
capacity may explain that disagreement; a specialist trained on the same
1,943-pair domain budget is needed to separate those possibilities.

## Limitations and status

- This file describes one seed. Cross-seed uncertainty must come from the
  protocol aggregate, not the zero standard deviations in this run summary.
- The mixed detector receives half as many examples from each domain as a
  specialist. Specialist-versus-mixed differences therefore do not isolate
  the effect of mixed training.
- The selected checkpoint is epoch 298 of 300, so the run ends while useful
  learning may still be continuing. Its scores remain lower-bound estimates.
- The 1.23-point test-domain gap is an observation from one seed, not a
  significance claim.
- This run contributes the mixed seed-1 record to the later three-seed
  aggregate. Final claims should use that aggregate rather than this run alone.
