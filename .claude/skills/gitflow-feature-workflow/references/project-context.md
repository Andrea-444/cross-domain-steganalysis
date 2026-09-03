# Project context files

This repository does not use a single monolithic agent file. Context is layered across three tiers, and each tier owns something different. Read them in order; update the ones a task makes stale, in the same branch as the task.

## Tier 1 — `AGENTS.md` (repository root)

The root instruction file. It owns:

- the purpose statement and the target task (binary steganalysis, not AI-image detection);
- the **source-of-truth ordering** used when files disagree: executable code and tests → `README.md` → `notes/Decisions.md` → other notes;
- instruction routing to `agent-instructions/`;
- the non-negotiable research constraints;
- working rules and repository boundaries.

Keep it short. It is loaded for every task, so detail belongs in the tier below it. A meaningful conflict between sources must be reported, not silently resolved.

## Tier 2 — `agent-instructions/`

Deep reference, routed by the kind of work:

| File | Owns |
|---|---|
| `RESEARCH_PROTOCOL.md` | Objective, four-direction matrix, primary pipeline, current methods |
| `DEVELOPMENT_WORKFLOW.md` | Environment, commands, tests, code organization, change discipline, definition of done |
| `DATA_AND_ARTIFACTS.md` | Artifact locations, handling rules, dataset preparation, downloads, run provenance |
| `RESULTS_AND_WRITING.md` | Analysis, figures, tables, and how claims may be stated |
| `SECURITY_AND_SCOPE.md` | Demonstrations, credentials, downloads, external tools |

Update the file that owns the changed subject. When a task adds a verified command, a real gotcha, or a new artifact location, record it here rather than in the root file.

`DEVELOPMENT_WORKFLOW.md` states that no repository-wide formatter, linter, type checker, CI configuration, or release process exists. Do not add instructions that imply one does.

## Tier 3 — `notes/` (Obsidian vault)

The living research record. Wiki links (`[[Note Name]]`) connect the notes; preserve them when editing.

| Note | Owns | Update when |
|---|---|---|
| `Decisions.md` | Accepted decisions, dated, with status/context/reasoning/consequences | The user accepts or supersedes a choice |
| `Development Log.md` | Chronological lab and engineering notebook | Any substantive change or experiment run |
| `Tasks.md` | Checklist of protocol, pipeline, tracking, and execution work | Work completes or new work is discovered |
| `Experiment Decisions.md` | Concise working experiment sheet, `protocol-frozen` flag | The experiment plan changes |
| `Pipeline Summary and Experiment Plan.md` | Central planning document, gates, open questions | Planning state changes |
| `Architecture.md` | Component boundaries and data flow | Module structure changes |
| `Project Overview.md` | Short orientation | The framing changes |
| `Ideas.md` | Unaccepted possibilities | An idea is raised but not adopted |

Both `Decisions.md` and `Development Log.md` carry entry templates at the top — follow them, including the `updated:` frontmatter date.

### The distinction that matters most

`Ideas.md` holds what *might* be done. `Decisions.md` holds what the user *accepted*. Moving an entry from the first to the second is the user's call, never the agent's. A decision recorded as `proposed` is not accepted.

## What replaces a CHANGELOG

There is no `CHANGELOG.md` and no version file in this repository. The release record is:

- `notes/Development Log.md` — the dated narrative of what changed and what was verified;
- annotated Git tags on `main` — the version, and the reproducibility anchor for result directories.

Do not create a `CHANGELOG.md` as incidental cleanup. If the user wants one, that is a decision to record in `notes/Decisions.md` first.

## Rules for keeping these files true

- Update a file in the same branch as the change that made it stale.
- Record commands verbatim, as they actually succeeded, including the absolute interpreter path where a bare `python` would fail.
- Do not invent a gotcha when none occurred. Sharpen an existing entry rather than appending a near-duplicate.
- Delete guidance that no longer applies to the current codebase.
- Keep entries terse. These files are loaded into future sessions, so every line must earn its tokens.
- Never leave a note materially false as a side effect of a merged change.
