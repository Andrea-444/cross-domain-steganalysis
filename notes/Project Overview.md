---
type: project-overview
status: active
updated: 2026-08-31
---

# Project Overview

## Purpose

Study whether hidden information is detected differently when it is embedded in AI-generated images versus natural images.

This is a steganalysis project, not an AI-image detector. The classifier predicts:

- `0` — clean image
- `1` — steganographic image

## Research questions

1. Are hidden payloads easier or harder to detect in AI-generated covers?
2. Does a steganalyzer trained on natural images generalize to AI-generated covers?

## Experiment matrix

| Training domain | Test domain | Role |
|---|---|---|
| AI | AI | Primary: AI-domain detectability |
| Natural | AI | Primary: transfer to AI covers |
| Natural | Natural | Same-domain control |
| AI | Natural | Reverse-transfer control |

## Current technical approach

- Normalize natural and AI images to balanced 256×256 grayscale PNG datasets.
- Preserve clean/stego pairing and split membership through stable image IDs.
- Generate adaptive steganography with `conseal` using WOW, S-UNIWARD, HILL, or MiPOD.
- Train SRNet separately on natural and AI domains.
- Evaluate all four cross-domain directions with repeated seeds and classification metrics.

## Key links

- [[Pipeline Summary and Experiment Plan]]
- [[Experiment Decisions]]
- [[Architecture]]
- [[Decisions]]
- [[Development Log]]
- [[Ideas]]
- [[Tasks]]

## Project status

The core dataset, embedding, and steganalysis modules exist, and the current unit tests pass. The scientific protocol, full provenance capture, result observability, and final dataset are not yet frozen. See [[Pipeline Summary and Experiment Plan]] for the detailed project plan and [[Tasks]] for the execution checklist.

## Questions to keep current

- What is the exact thesis or paper deliverable?
- Which dataset and subset will be used for the final experiments?
- Which embedding algorithms and payloads are in the final protocol?
- What compute and time constraints apply?
- What result would support or reject the main hypothesis?
