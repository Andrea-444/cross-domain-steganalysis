# Evidence

Results copied here by hand, because they support a recorded decision and cost
significant compute to produce.

No command writes to this directory. `data/` and `results/` hold everything a
run produces, and both are ignored by git. Tracking a path that a run also
writes makes `git pull` fail on any machine that generated the same file.

## domain-probes/

SRNet domain-separability probes. Each file asks one question: can SRNet
identify the cover domain from clean covers alone, with no stego images?

| File | Cohort | Balanced accuracy |
|---|---|---|
| `full6000-crop-balanced.json` | 5,830 per domain, crop, rebalanced | **0.9954** |
| `smoke100-fit.json` | 100 per domain, crop and resample | 0.7451 |
| `smoke100-crop.json` | 100 per domain, crop only | 0.6752 |
| `smoke100-crop-jpeg-ai.json` | 100 per domain, JPEG q96 on AI | 0.6605 |
| `smoke100-crop-jpeg-both.json` | 100 per domain, JPEG q96 on both | 0.6875 |
| `feature-screen-smoke100-fit.json` | Three-feature screen, `fit` | 0.6300 |
| `feature-screen-smoke100-crop.json` | Three-feature screen, `crop` | 0.5324 |

The full-scale file is authoritative. The 100-per-domain figures are superseded
and are a lower bound only, because 132 training covers limited the probe.

See `notes/Methods and Findings.md` for the analysis and
`notes/Decisions.md` for the decisions these files support.
