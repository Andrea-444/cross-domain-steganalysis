---
name: gitflow-feature-workflow
description: Use when making code, experiment, or versioning changes in this repository, including features, non-urgent bug fixes, urgent fixes to tagged work, milestone releases, or semver bumps. Enforces Gitflow branching, test-first development, and the project's research and artifact guardrails.
---

# Gitflow Feature Workflow

Use strict Gitflow for the complete lifecycle of repository changes. Protect `main`, integrate through `dev`, verify before merging, and leave the repository and remote in a clear final state.

This is a research repository. Alongside the usual Git discipline, every change must preserve the controlled comparison described in `AGENTS.md` — pair-safe splits, equivalent preprocessing across domains, honest labelling of smoke versus final results, and complete run provenance.

## Execution contract

1. Read `AGENTS.md` and the routed `agent-instructions/` file, then inspect Git state before changing files.
2. Select and read the exact playbook below.
3. Write the plan to `.gitflow/plan.md` using `references/plan.md`, ask every open question, and stop for the model switch.
4. Post a checklist for that playbook and execute it in order, test-first.
5. Mark a step N/A only when an observable predicate proves it absent, such as no CI config, no formatter, or no testable behavior.
6. Finish only after every applicable check passes and all authorized remote work is pushed. When work remains local or awaits review, state exactly what remains and why.

| Change | Branch | Required playbook |
|---|---|---|
| New capability, or expanding an existing one | `feat/<descriptive-name>` from `dev` | `references/feature.md` |
| Non-urgent bug fix | `bug/<descriptive-name>` from `dev` | `references/feature.md` |
| Docs, notes, instruction files, skills, specification of steps | `spec/<descriptive-name>` from `dev` | `references/feature.md` |
| Throwaway experiment that will never merge | `junk/<descriptive-name>` from `dev` | none — see Branching model |
| Urgent fix to already-tagged work | `hotfix/<version>` from `main` | `references/hotfix.md` |
| Milestone / thesis snapshot | `release/<version>` from `dev` | `references/release.md` |

Read the selected reference completely before branching.

## Load project context

This repository's instruction set is layered. Read it in this order and prefer it over rediscovering recorded context:

1. `AGENTS.md` — root instructions, source-of-truth ordering, and instruction routing.
2. The routed file under `agent-instructions/` for the work at hand — `RESEARCH_PROTOCOL.md`, `DEVELOPMENT_WORKFLOW.md`, `DATA_AND_ARTIFACTS.md`, `RESULTS_AND_WRITING.md`, or `SECURITY_AND_SCOPE.md`.
3. `notes/Decisions.md` for accepted protocol choices, then the other notes for evolving context.

`references/project-context.md` describes this structure, what each file owns, and which of them a task must update. When a recorded instruction is stale, correct it in the task branch and report the discrepancy rather than silently working around it.

## First use in this repository

Gitflow is **not yet initialized here**. `main` exists locally and on `origin`; `dev` does not exist at all, and the repository has no tags. Before the first feature branch, confirm with the user, then:

```bash
git checkout main && git pull
git checkout -b dev
git push -u origin dev
```

Do this once. Afterwards `dev` is a long-lived branch — never delete or recreate it.

Branches created before `dev` existed were cut from `main` and already follow the naming model below. When one is still active, ask the user whether to merge it into `dev` now that `dev` exists or to finish it first — do not rename, rebase, or retire it on your own.

Because there are no tags, the first release baseline is `v0.1.0` unless the user chooses otherwise.

## Plan first, implement after the model switch

Planning and implementation normally run on different models: the strong model that is loaded now writes the plan, then the user switches to a cheaper model to execute it. The plan is a standalone handoff document, not a chat message.

1. Write the plan with the current model into `.gitflow/plan.md`, following `references/plan.md`. Keep `.gitflow/` out of version control via `.gitignore`.
2. Resolve open questions before the handoff. The implementing model follows the plan literally, so ambiguity left in it becomes wrong code.
3. Report the plan path and stop. Implement in the same session only when the user says not to switch models.
4. On resume, re-read `.gitflow/plan.md` and `AGENTS.md`, confirm the branch, then execute the plan's ordered TDD steps, ticking them off as they land.
5. If reality contradicts the plan, stop and report the mismatch instead of redesigning silently.

## Ask before you build

Ask whenever an answer changes the plan: unclear requirements, undefined edge cases or error behavior, ambiguous scope, missing acceptance criteria, unclear verification expectations, or a requested library, pattern, or data model you would not choose yourself.

**Protocol ambiguity is always a question, never a default.** This project's scientific protocol is deliberately unfrozen in places. Whenever the work touches an open item in `notes/Tasks.md`, a "Still to decide" entry in `notes/Experiment Decisions.md`, an open question in `notes/Pipeline Summary and Experiment Plan.md`, or anything the recorded decisions simply do not determine, **stop and ask the person who instructed you.** State the question, the options, the trade-off and your recommendation, then implement their answer. Choosing for them and noting it in a comment is not acceptable — those choices are protocol, and they outlive the branch. Record the answer in `notes/Decisions.md` as a dated entry when it settles an accepted decision.

Never convert an unapproved idea into an accepted decision, and never treat silence or a missing answer as permission. Finish everything that does not depend on the answer, then stop at the part that does and say so.

When a better implementation exists, raise it first — the alternative, why it is better, its cost, your recommendation — then follow the user's decision. Batch questions into one round; never re-confirm explicit requirements.

## Suggest skills for the task

During planning, scan the skills available in the session and name the ones that fit this task — a data-visualization skill for figures, a document skill for written deliverables, a review skill before merging. List them in the plan under **Suggested skills**, one line each on why, invoke the clearly applicable ones, and ask when a suggestion is optional. Suggest only skills present in the session; never invent names.

## Test-driven development

TDD is the default for every feature, bug fix, hotfix, and release fix.

- Write the failing test first, run it, and show the failure. A test that never failed proves nothing.
- Work in small red → green → refactor cycles: minimal code to pass, then clean up with the suite green.
- Cover the stated behavior plus edge cases and error paths, not only the happy path.
- Commit tests together with the code that makes them pass, in the same small commit.
- The plan lists test cases before implementation steps.
- Use temporary directories and small synthetic images, per `agent-instructions/DEVELOPMENT_WORKFLOW.md`. Never make a unit test depend on a large dataset or a real training run.
- Skip TDD only where no test can observe the change — formatting, docs, notes, config with no testable behavior. Say which and why; a slow suite is not a reason.

**Experiment runs are not TDD.** Executing a smoke, pilot, or final run is evidence-gathering, not a code change: it is recorded in `notes/Development Log.md` with its command, result, and surprises. Code that *supports* a run is still developed test-first.

## Branching model

| Branch | Role | Created from | Merges into |
|---|---|---|---|
| `main` | Reviewed, citable state | — | — |
| `dev` | Integration | — | — |
| `feat/<descriptive-name>` | New capability or expansion | `dev` | `dev` |
| `bug/<descriptive-name>` | Non-urgent bug fixes | `dev` | `dev` |
| `spec/<descriptive-name>` | Docs, notes, instructions, skills, specs | `dev` | `dev` |
| `junk/<descriptive-name>` | Throwaway experiments | `dev` | nothing — never merged |
| `hotfix/<version>` | Urgent fixes to tagged work | `main` | `main` and `dev` |
| `release/<version>` | Milestone stabilization | `dev` | `main` and `dev` |

Rules:

- **Never commit directly to `main` or `dev`.**
- **Branch before touching any file.** If changes were made on a long-lived branch, stop, create the correct branch with the changes intact, then verify `git status`.
- Before creating any branch, run `git branch --show-current` and `git status`, update the base branch, and pull the latest base branch.
- All merges into `dev` and `main` use `--no-ff`.
- Every merge into `main` gets an annotated semver tag.
- Name branches with a full descriptive kebab-case phrase, not an abbreviation — for example `spec/definition-of-datasets-and-initial-steganography-techniques`. Use version numbers for hotfix/release names.
- `junk/*` branches merge into nothing. Never merge one into `dev` or `main`, never tag one, and delete it once the question it answered is settled. If throwaway work turns out to be worth keeping, open a proper `feat/`, `bug/`, or `spec/` branch for it and redo it test-first.
- Pick the prefix by what the change *is*, not by how long it will take. There is no `wip/` prefix: an unfinished `feat/` branch is simply not merged yet.
- Split unrelated changes into separate branches.

## Semver

- Feature → **minor**
- Hotfix / bug fix → **patch**
- Breaking change to a public command, manifest schema, or result layout → **major**

Read the current version from the latest `main` tag. The repository has no tags yet, so the first release is `v0.1.0`. There is no version file; the tag is the version. Call out breaking changes during planning — a manifest or result-schema change breaks existing artifacts and must be raised, not absorbed.

## Interaction, remotes, and CI

**Use local merge mode by default.** Merge into `dev` or `main` with `--no-ff` locally and push. `origin` is GitHub (`Andrea-444/cross-domain-steganalysis`). Do not open a pull request unless the user asks or a protected branch rejects a direct push.

Beyond the planning questions above, ask only when a workflow choice materially changes the result: Gitflow initialization, pre-existing uncommitted changes, destructive actions, intentionally conflicting edits, or the choice between local merge and pull request mode.

- **No remote reachable:** complete the workflow locally and state that nothing was pushed.
- **Pull request mode:** use `gh` only when the user asks for review or a protected branch rejects the push. Never bypass protection.
- **CI:** none is configured in this repository. Mark the CI check N/A and continue after local verification. If CI is added later, never merge into `dev` or `main` while it is red.
- **Uncommitted changes before switching:** ask whether to stash, commit, or abort.

## Shared implementation rules

- Plan the change before editing, and hand the plan off as described above.
- Use small Conventional Commits. Do not add a co-authorship trailer or AI attribution.
- Before every commit, inspect `git status` and the staged diff for secrets, credentials, W&B or Kaggle keys, large binaries, images, model weights, and editor/OS junk. If a secret is already committed, stop before pushing and tell the user to rotate it.
- **Never commit generated artifacts.** `data/`, `results/`, `stego_runs/`, `external/`, and `raw_data/` are ignored by design. Verify with `git status` rather than trusting the pattern.
- Every feature gets tests, written first. Every bug fix gets a regression test that fails without the fix.
- Run the full test suite with the project environment. Lint and format checks are N/A — this repository has no formatter, linter, or type checker, and `agent-instructions/DEVELOPMENT_WORKFLOW.md` forbids implying one exists.
- Update `README.md` when public behavior, commands, setup, or architecture changes.
- Update the notes vault per `references/project-context.md`: `notes/Development Log.md` for what happened, `notes/Decisions.md` for accepted choices, `notes/Tasks.md` for completed or newly discovered work.

## Definition of done

- [ ] Plan written to `.gitflow/plan.md`, open questions answered, handoff reported
- [ ] Applicable skills suggested during planning
- [ ] Correct playbook and ordered checklist used
- [ ] Correct branch created before changes
- [ ] Code complete with no unintended TODOs or dead code
- [ ] Tests written before implementation and observed failing first
- [ ] Feature/regression tests added as required
- [ ] Full suite passes; lint/format marked N/A
- [ ] Research invariants intact: pair-safe splits, equivalent preprocessing, provenance preserved, smoke output not presented as final
- [ ] `README.md` and the notes vault updated when their observable predicates apply
- [ ] Conventional Commits used; no secrets, datasets, binaries, junk, or AI co-author trailers
- [ ] Feature branch synced with latest `dev` before the final merge; hotfix/release branches synced with their required bases
- [ ] Correct targets used: feature→dev; hotfix/release→main and dev
- [ ] `--no-ff` used; main merges tagged
- [ ] CI marked N/A (none configured)
- [ ] Authorized pushes verified, or local/review-only state reported precisely

## Guardrails

- Never force-push shared branches.
- Never rebase `main` or `dev`.
- Destructive operations: confirm with the user first and explain what will be lost.
- Never delete, overwrite, or bulk-regenerate datasets, manifests, selections, stego runs, or result directories to make a workflow step easier. `agent-instructions/DATA_AND_ARTIFACTS.md` governs; create a new run instead of rewriting an old one.
- Never hand-edit result JSON or manifests to make a run look successful.
- When any git command fails mid-workflow, including a merge conflict or push rejection, stop and resolve it deliberately before continuing.
- Resolve a merge conflict when intent is obvious, but show the resolution before pushing. Ask when both sides look intentional.
