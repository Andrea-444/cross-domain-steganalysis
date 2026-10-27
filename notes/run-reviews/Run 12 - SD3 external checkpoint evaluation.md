---
type: run-review
run: 12
status: reviewed
updated: 2026-10-26
---

# Run 12 — External evaluation on Stable Diffusion 3 covers

## Identity and purpose

- Date: 2026-10-25
- Result folder:
  `results/srnet_eval_genimagepp-sd3-suniward040_20261025_155824`
- Scientific role: external-generator evaluation of saved checkpoints
- Test covers: 1,000 held-out GenImage++ Stable Diffusion 3 images
- Test samples: 1,000 clean and 1,000 stego images
- Embedding: S-UNIWARD, 0.4 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Evaluated checkpoints: three natural-trained, three FLUX-trained,
  three equal-total mixed 300-epoch checkpoints, and one mixed 500-epoch
  checkpoint
- Original training manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- SD3 evaluation manifest: `d523878e767dedbb425779d17e33ed9932188210ba07d06ab47db3beb84836c1`
- Detector: SRNet commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`
- Evaluation repository commit: `1d9288be35f0da768ae292066a22377972eed4f9`
- Device: NVIDIA GeForce RTX 5080, CUDA
- Runtime: approximately 64 seconds; the GPU was shared with an active
  training process, so this is not a timing benchmark

## Recorded observations

Population standard deviations are across the three training seeds. The
500-epoch condition has one seed and therefore has no uncertainty estimate.

| Training condition | Balanced accuracy | ROC-AUC | F1 | MCC | EER |
|---|---:|---:|---:|---:|---:|
| Natural only, 300 epochs | 0.5073 +/- 0.0017 | 0.5290 +/- 0.0076 | 0.6531 +/- 0.0112 | 0.0308 +/- 0.0055 | 0.4813 +/- 0.0082 |
| FLUX only, 300 epochs | 0.5590 +/- 0.0109 | 0.6289 +/- 0.0163 | 0.3859 +/- 0.0297 | 0.1435 +/- 0.0266 | 0.4263 +/- 0.0114 |
| Mixed, 300 epochs | **0.5797 +/- 0.0088** | **0.6516 +/- 0.0143** | **0.5046 +/- 0.0294** | **0.1675 +/- 0.0167** | **0.4120 +/- 0.0088** |
| Mixed, 500 epochs, seed 1 | 0.5585 | 0.6247 | 0.4046 | 0.1367 | 0.4390 |

Per-seed balanced accuracy and ROC-AUC:

| Training | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Natural, balanced accuracy | 0.5090 | 0.5080 | 0.5050 |
| Natural, ROC-AUC | 0.5256 | 0.5220 | 0.5395 |
| FLUX, balanced accuracy | 0.5450 | 0.5605 | 0.5715 |
| FLUX, ROC-AUC | 0.6058 | 0.6394 | 0.6414 |
| Mixed 300, balanced accuracy | 0.5905 | 0.5690 | 0.5795 |
| Mixed 300, ROC-AUC | 0.6648 | 0.6318 | 0.6582 |

Summed confusion matrices over three seeds:

| Training | Confusion matrix | Specificity | Recall |
|---|---|---:|---:|
| Natural | 257, 2,743 / 213, 2,787 | 0.0857 | 0.9290 |
| FLUX | 2,519, 481 / 2,165, 835 | 0.8397 | 0.2783 |
| Mixed 300 | 2,187, 813 / 1,709, 1,291 | 0.7290 | 0.4303 |

The natural-trained detectors mostly call every SD3 image stego. The
FLUX-trained detectors have the opposite problem: they classify clean SD3
images relatively well but miss most SD3 stego images. The mixed detectors
retain the best balance, although their stego recall is still only 0.430.

## Comparison with the original FLUX test set

The embedding algorithm, payload, preprocessing and detector checkpoints stay
fixed. The AI-image generator and the individual test images change. These are
therefore external-dataset differences, not paired-image differences.

| Training condition | FLUX balanced acc. | SD3 balanced acc. | Change | FLUX ROC-AUC | SD3 ROC-AUC | Change |
|---|---:|---:|---:|---:|---:|---:|
| Natural, 300 epochs | 0.5067 | 0.5073 | +0.0006 | 0.5297 | 0.5290 | -0.0007 |
| FLUX, 300 epochs | 0.6343 | 0.5590 | **-0.0753** | 0.7208 | 0.6289 | **-0.0919** |
| Mixed, 300 epochs | 0.6302 | 0.5797 | **-0.0505** | 0.7145 | 0.6516 | **-0.0629** |
| Mixed, 500 epochs, seed 1 | 0.6371 | 0.5585 | **-0.0786** | 0.7260 | 0.6247 | **-0.1013** |

Natural-to-AI transfer remains at chance on both AI generators. Models that
trained on FLUX retain some signal on SD3, but their discrimination drops.
The three-seed 300-epoch mixed condition loses less than the FLUX specialist
and gives the highest mean SD3 result.

## Plain-language meaning

Seeing one type of AI-generated image does not make the detector work equally
well on another generator. The FLUX specialist still detects some SD3
steganography, but much less reliably than on FLUX. Training on both natural
and FLUX covers produces the most robust detector of the tested conditions.

The 500-epoch seed-1 model improved slightly on the original FLUX AI test but
fell from 0.5905 to 0.5585 relative to its 300-epoch seed-1 checkpoint on SD3.
Longer training may have specialized that model more strongly to its original
training distributions. One seed is not enough to treat this explanation as a
finding.

## Interpretation and completed follow-up

This is evidence that the original AI-cover result is partly
generator-specific. It strengthens the paper by adding an external-generator
test and by showing that the mixed model has the highest mean external score
on SD3.

It does **not** show that embedding is intrinsically harder in SD3. No detector
was trained on SD3, so lower performance can be ordinary cover-distribution
shift.

The planned full-union follow-up is now complete. Its seed-1 checkpoint scores
0.5880 balanced accuracy and 0.6841 ROC-AUC on this SD3 set. Against the
equal-total 300-epoch seed-1 checkpoint, balanced accuracy changes by -0.0025
and ROC-AUC by +0.0193. More mixed training exposure improves ranking but does
not remove the SD3 generator gap. See
[[run-reviews/Run 14 - Full-union mixed seed 1 300 epochs]].

## Limitations and status

- This adds one external AI generator, not a representative population of AI
  generators.
- The SD3 test has 1,000 pairs; the original FLUX test has 974 pairs.
- No formal significance test or confidence interval has been selected.
- The 500-epoch comparison has one training seed.
- The repository recorded `repository_dirty: true`. The evaluator commit and
  all checkpoint hashes are present, but uncommitted files on the GPU machine
  were not enumerated in the artifact.
- Running inference beside a training job does not modify the saved
  checkpoints, but the shared-GPU runtime must not be used for timing
  comparisons.

## Evidence used

- `/Users/andrea/Downloads/config.json`
- `/Users/andrea/Downloads/summary.json`
- `results/srnet_eval_genimagepp-sd3-suniward040_20261025_155824`
- [[run-reviews/Run 11 - Mixed seed 1 500 epochs]]
