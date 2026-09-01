# Results and Writing

## Current metrics

The experiment runner currently reports:

- accuracy;
- balanced accuracy;
- precision for the stego class;
- recall for the stego class;
- F1 for the stego class;
- confusion matrix in label order `[0, 1]`;
- mean and standard deviation across supplied experiment seeds.

The confusion matrix layout is:

```text
[[correct clean, clean falsely flagged as stego],
 [missed stego,  correctly detected stego]]
```

Check sample counts and class balance before interpreting accuracy. In the balanced binary setting documented by the project, balanced accuracy near `0.5` is chance-level performance.

## Required comparisons

- Compare `ai_to_ai` with `real_to_real` when discussing same-domain detectability.
- Compare `real_to_ai` with `real_to_real` when discussing transfer from natural-image training to AI covers.
- Use `ai_to_real` as the reverse-transfer control.
- Attribute lower cross-domain performance to domain mismatch only when algorithm, payload, preprocessing, split policy, and detector training procedure are controlled.

Do not collapse the four directions into one headline number when the direction is necessary to interpret the research question.

## Evidence standards

- Cite the exact run directory and relevant `config.json`, `dataset_manifest.jsonl`, and `summary.json` when reporting a numerical result.
- State sample sizes, seed count, and whether the run was a smoke test or intended experiment.
- Separate raw observations from interpretation.
- Report uncertainty and variation; do not report only the best seed or preferred algorithm.
- Treat Aletheia screening as secondary evidence unless a later accepted protocol changes its role.
- Do not infer generator-wide or population-wide behavior from a small, convenience, or single-generator subset.
- Do not call a difference significant unless the accepted statistical method supports that statement.
- Do not interpret an existing folder name such as `mini` or `smoke` as proof of scientific validity; inspect its configuration and manifest.

## Writing conventions

- Use `natural` and `AI-generated` for cover domains in prose; use exact code values such as `real` and `ai` when describing manifests or commands.
- Use `clean` and `stego` for the two detector classes.
- Spell out an algorithm on first use when useful, then use its established short name consistently.
- State that OpenStego RandomLSB is a recoverable demonstration when it appears near primary benchmark discussion.
- Label proposed explanations as hypotheses unless supported by a controlled analysis.
- Include limitations and plausible confounders alongside conclusions.
- Prefer precise statements tied to artifacts over promotional or absolute language.

## Recommended result record

For each reported experiment, capture:

1. Research question and comparison.
2. Dataset source, preprocessing, counts, and splits.
3. Embedding algorithm, payload, and seed policy.
4. Detector and training configuration.
5. Four-direction metrics with uncertainty.
6. Failure cases or diagnostic observations.
7. Limitations and threats to validity.
8. Reproduction paths or commands.

## Pending decisions

The final table format, figure style, citation style, statistical tests, confidence intervals, primary endpoint, and thesis/paper structure are not yet established. Mark them as pending until the user approves them.
