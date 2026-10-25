---
type: run-review
run: 13
status: reviewed
updated: 2026-10-25
---

# Run 13 — External evaluation on realistic-prompt FLUX covers

## Identity and purpose

- Date: 2026-10-25
- Result folder:
  `results/srnet_eval_genimagepp-flux-realistic-suniward040_20261025_184816`
- Scientific role: same-generator prompt-distribution evaluation of saved
  checkpoints
- Test covers: 1,000 held-out GenImage++ `flux_realistic` images
- Test samples: 1,000 clean and 1,000 stego images
- Embedding: S-UNIWARD, 0.4 bpp, deterministic per-image seeds
- Preprocessing: native centre crop, 256x256 grayscale PNG
- Evaluated checkpoints: three natural-trained, three FLUX-trained, three
  equal-total mixed 300-epoch checkpoints, and one mixed 500-epoch checkpoint
- Original training manifest: `f271f1352d345996e60539304280951eee7d164811637ec20e310883d3b32873`
- Evaluation manifest: `dc6814a0ef3af0c0fd060266a77b40039de071d385b646f5e953ce1fe9150017`
- Detector: SRNet commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`
- Evaluation repository commit: `bfba6081e8d09a63da5f9d0f4217695e3bf919fb`
- Device: NVIDIA GeForce RTX 5080, CUDA
- Runtime: approximately 67 seconds; this is not a timing benchmark

## Recorded observations

Population standard deviations are across three training seeds. The
500-epoch condition has one seed and therefore no uncertainty estimate.

| Training condition | Balanced accuracy | ROC-AUC | F1 | MCC | EER |
|---|---:|---:|---:|---:|---:|
| Natural only, 300 epochs | 0.5037 +/- 0.0008 | 0.5531 +/- 0.0029 | 0.6646 +/- 0.0020 | 0.0287 +/- 0.0117 | 0.4637 +/- 0.0061 |
| FLUX only, 300 epochs | 0.6997 +/- 0.0041 | **0.8111 +/- 0.0079** | 0.7004 +/- 0.0276 | 0.4053 +/- 0.0134 | 0.2923 +/- 0.0095 |
| Mixed, 300 epochs | **0.7018 +/- 0.0050** | 0.8058 +/- 0.0070 | **0.7139 +/- 0.0047** | **0.4060 +/- 0.0074** | 0.2957 +/- 0.0042 |
| Mixed, 500 epochs, seed 1 | 0.7075 | 0.8141 | 0.7178 | 0.4161 | 0.2910 |

Per-seed balanced accuracy and ROC-AUC:

| Training | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| Natural, balanced accuracy | 0.5025 | 0.5040 | 0.5045 |
| Natural, ROC-AUC | 0.5507 | 0.5513 | 0.5571 |
| FLUX, balanced accuracy | 0.6970 | 0.7055 | 0.6965 |
| FLUX, ROC-AUC | 0.8026 | 0.8216 | 0.8090 |
| Mixed 300, balanced accuracy | 0.7070 | 0.6950 | 0.7035 |
| Mixed 300, ROC-AUC | 0.8119 | 0.7960 | 0.8094 |

Summed confusion matrices over three seeds:

| Training | Confusion matrix | Specificity | Recall |
|---|---|---:|---:|
| Natural | 71, 2,929 / 49, 2,951 | 0.0237 | 0.9837 |
| FLUX | 2,069, 931 / 871, 2,129 | 0.6897 | 0.7097 |
| Mixed 300 | 1,977, 1,023 / 766, 2,234 | 0.6590 | 0.7447 |

The natural-trained detectors again label almost every AI-generated image as
stego. Their high F1 is therefore misleading: balanced accuracy and MCC show
that they do not separate the classes. FLUX-only and mixed training both give
balanced, above-chance discrimination on this subset.

## Comparison across the three AI test sets

The checkpoints, embedding algorithm, payload and preprocessing are fixed.
The generated cover subset changes.

| Training | Base FLUX BA | Realistic FLUX BA | SD3 BA | Base FLUX AUC | Realistic FLUX AUC | SD3 AUC |
|---|---:|---:|---:|---:|---:|---:|
| Natural, 300 epochs | 0.5067 | 0.5037 | 0.5073 | 0.5297 | 0.5531 | 0.5290 |
| FLUX, 300 epochs | 0.6343 | **0.6997** | 0.5590 | 0.7208 | **0.8111** | 0.6289 |
| Mixed, 300 epochs | 0.6302 | **0.7018** | 0.5797 | 0.7145 | **0.8058** | 0.6516 |
| Mixed, 500 epochs, seed 1 | 0.6371 | **0.7075** | 0.5585 | 0.7260 | **0.8141** | 0.6247 |

Relative to base FLUX, realistic-prompt FLUX raises balanced accuracy by
0.0654 for the FLUX specialist, 0.0716 for the mixed 300-epoch condition and
0.0704 for the mixed 500-epoch seed. Their ROC-AUC gains are 0.0903, 0.0913
and 0.0881 respectively.

Relative to SD3, realistic-prompt FLUX is higher by 0.1407 balanced accuracy
for the FLUX specialist and by 0.1221 for the mixed 300-epoch condition. The
corresponding ROC-AUC differences are 0.1822 and 0.1542.

## Plain-language meaning

The detector did not fail on the tested realistic prompt shift. With the same
FLUX generator and more detailed prompts, detection became easier. The large
loss on SD3 therefore looks more like sensitivity to the cover generator than
an effect reproduced by this prompt change.

The FLUX specialist and mixed detector are effectively close on this test.
The mixed detector has the highest mean balanced accuracy by 0.0022, while the
FLUX specialist has the highest mean ROC-AUC by 0.0053. No selected
significance procedure establishes a meaningful difference between them.

## Interpretation and next step

This result strengthens the generator-shift interpretation from Run 12.
Changing prompts while retaining FLUX improves detection; changing the
generator to SD3 reduces it. The evidence is about steganalysis across cover
generators, not about detecting whether an image is AI-generated.

A matching `stable_diffusion_v_3_0_realistic` evaluation would complete the
2x2 generator-by-prompt comparison. It would show whether realistic prompts
also make SD3 steganography easier, or whether the large FLUX-SD3 difference
persists under both prompt conditions.

## Limitations and status

- Base and realistic FLUX use different generated images, so changes are not
  paired at the image level.
- Prompt wording changes the generated image distribution; the measurement
  does not identify which visual property makes embedding easier to detect.
- The test covers one generator and one realistic-prompt recipe.
- No formal significance test or confidence interval has been selected.
- The 500-epoch comparison has one training seed.
- The repository recorded `repository_dirty: true`; the evaluator commit and
  checkpoint hashes remain recorded in the artifact.

## Evidence used

- `/Users/andrea/Downloads/config.json`
- `/Users/andrea/Downloads/summary.json`
- `results/srnet_eval_genimagepp-flux-realistic-suniward040_20261025_184816`
- [[run-reviews/Run 12 - SD3 external checkpoint evaluation]]
