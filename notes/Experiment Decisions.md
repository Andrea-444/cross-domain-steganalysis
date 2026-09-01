---
type: concise-decision-sheet
status: active
phase: initial-design
protocol-frozen: false
updated: 2026-09-01
---

# Experiment Decisions

> [!important] Initial design
> These are accepted working decisions for the initial experiment architecture. They are not the frozen final protocol; dataset audits, compatibility checks, and pilots may require documented revisions.

## Accepted working plan

| Step | Decision |
|---|---|
| Research target | Detect `clean` versus `stego`; compare natural and AI-generated cover domains. |
| AI source | GenImage++ `flux` archive: 6,000 FLUX images (about 7.22 GB), downloadable without the full dataset. Content audit still required. |
| Natural source | ImageNet ILSVRC validation pool, matched to the selected AI class distribution. No ready-made 6,000-image subset is confirmed; plan to select 6,000 locally from the 50,000 validation images. |
| Download scope | Do not download the full GenImage++ or 167 GB Kaggle competition dataset. Confirm that Kaggle exposes the ImageNet validation archive separately after account sign-in and rule acceptance. |
| Starting size | Prefer 6,000 covers per domain. Reduce or increase after a pilot. |
| Matching | Match class counts between domains; natural and AI images are not one-to-one pairs. |
| Splitting | Split covers before embedding. Every clean/stego twin stays in the same split. |
| Preprocessing | Use the same crop, grayscale conversion, dimensions, and lossless output format for both domains. Prefer crop-only; confirm after checking source dimensions. |
| Self-generation | Optional only if suitable external GPU access becomes available. |
| Embedding library | Keep the tested `conseal==2024.12` pin for simulated adaptive embedding. Validate deliberately before any upgrade. |
| Primary detector | Use the existing external PyTorch SRNet implementation pinned at commit `2c838cf7d26dd0b8e45eda79ebf103ca312c0355`; load it through `src/steganalysis/srnet.py` instead of rewriting SRNet. |
| Classical baseline | Plan to use `sealwatch==2025.9` for SRM/SRMQ1 features with an FLD ensemble, pending installation and compatibility validation. |

If all 1,000 ImageNet classes are used, 6,000 images means 6 images per class per domain. A simple split is:

- 4,000 train — 4 per class;
- 1,000 validation — 1 per class;
- 1,000 test — 1 per class.

## Embedding matrix

- **Primary:** S-UNIWARD at 0.2 and 0.4 bpp, with 3 seeds.
- **First robustness method:** MiPOD at 0.4 bpp, with 3 seeds.
- **Later extension:** HILL at 0.4 bpp, only if compute permits.
- **WOW:** Optional, separately labelled cross-algorithm test. A S-UNIWARD-trained detector tested on WOW measures algorithm mismatch as well as domain mismatch.
- **0.1 bpp:** Later optional condition; it may be too difficult with 6,000 covers and could produce mostly chance-level results.
- **LSB:** Excluded from the research matrix because it is too simple, and the repository's legacy LSB is not proper LSB matching.

## Still to decide

- Use all 1,000 classes or a justified subset.
- Confirm the accessible ImageNet validation archive and its download size.
- Final crop/resize rule after the dimension audit.
- Finalize the detector choice after an SRNet convergence pilot and a `sealwatch` compatibility/speed test.
- Metrics, statistics, and cross-domain test matrix.

## Required dataset audit

ImageNet images are commonly JPEG-compressed; GenImage++ images are generated and usually stored losslessly. Converting both to PNG later does **not** undo earlier JPEG compression.

JPEG quantization permanently changes high-frequency pixel relationships and can leave 8×8 block and residual patterns. Steganalyzers also examine subtle high-frequency residuals. A detector could therefore react to compression history rather than the intended natural-versus-AI cover difference.

Before accepting this dataset pair:

1. Record original formats, dimensions, and JPEG metadata.
2. Compare residual, blocking, and frequency statistics between domains.
3. Test whether clean processed images can be separated by format/residual shortcuts.
4. If the difference is too strong, change the dataset or add a justified control before final experiments.
