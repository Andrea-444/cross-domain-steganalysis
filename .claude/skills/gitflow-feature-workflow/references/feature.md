# Task branch playbook — `feat/`, `bug/`, `spec/`

Use this playbook for every task branch that merges back into `dev`:

| Prefix | Use for |
|---|---|
| `feat/` | A new capability, or expanding an existing one |
| `bug/` | A non-urgent bug fix |
| `spec/` | Docs, notes, instruction files, skills, specification of steps |

All three branch from `dev` and merge back into `dev`, and all three follow the steps below. Pick the prefix by what the change is, not by how long it will take. Name the branch with a full descriptive kebab-case phrase.

`junk/` branches are out of scope here: they are throwaway, merge into nothing, and are deleted when the question they answered is settled.

Below, `<branch>` stands for the full branch name, for example `feat/preserve-test-probabilities` or `spec/definition-of-datasets-and-initial-steganography-techniques`.

## 1. Plan

Read `AGENTS.md` and the routed `agent-instructions/` file, restate the requested outcome, list affected files, identify risks, and resolve genuinely ambiguous requirements. Do not pause for requirements the user already made explicit.

Write the plan to `.gitflow/plan.md` and hand it off following `references/plan.md`: ask every open question, raise any better implementation you see, list the skills that fit the task, order the work as TDD cycles, then stop so the user can switch to the implementing model. On resume, re-read the plan file before branching.

Inspect the repository before branching:

```bash
git branch --show-current
git status
git remote
```

If uncommitted work exists, ask whether to stash, commit, or abort. If unrelated changes are requested, split them into separate branches.

If `dev` does not exist yet, follow "First use in this repository" in `SKILL.md` before continuing.

## 2. Branch before changes

Update `dev`, then create and publish the task branch:

```bash
git checkout dev
git pull
git checkout -b <branch>
git push -u origin <branch>
```

When no remote is reachable, omit both push commands and complete locally. Verify the active branch before editing:

```bash
git branch --show-current
git status
```

## 3. Implement test-first and commit

Follow the plan's TDD cycles. For each behavior: write the failing test, run it and confirm it fails for the right reason, write the minimal code to pass, refactor with the suite green, then commit the test and the code together.

Use temporary directories and small synthetic images in tests. Never make a unit test depend on a large dataset, a downloaded archive, or a real training run.

Implement in small logical units with Conventional Commits:

```text
feat: select MPS device when CUDA is unavailable
fix: keep test probabilities in the experiment summary
docs: record the pilot cohort command
test: cover class-matched selection with uneven class capacity
```

Do not add a co-authorship trailer or AI attribution.

Before every commit, inspect `git status` and the staged diff. Do not commit secrets, `.env` files, Kaggle or W&B credentials, private certificates, large binaries, images, model weights, build artifacts, or editor/OS junk. `data/`, `results/`, `stego_runs/`, `external/`, and `raw_data/` are ignored by design — verify with `git status` rather than trusting the pattern. If a secret is already committed, stop before pushing and tell the user to rotate it.

## 4. Test

Every `feat/` branch gets tests, written before the implementation. Every `bug/` branch gets a regression test that fails without the fix — observed failing first, not assumed.

A `spec/` branch that touches only docs, notes, or instruction files has no testable behavior: mark TDD N/A with that reason and still run the full suite to prove nothing regressed. A `spec/` branch that changes code is not documentation-only — treat it as `feat/` or `bug/` for testing purposes.

Run, in order:

1. The focused new or regression test.
2. The full test suite.

```bash
~/miniconda3/envs/experimental_1/bin/python -m unittest discover -s tests -t . -v
```

Lint and format checks are N/A in this repository. Never merge red. Report skipped tests separately from passes. If an unrelated pre-existing failure remains, report it and ask how to proceed.

When the change affects manifests, splits, embedding, or the detector, also confirm the research invariants hold — pair-safe splits, equivalent preprocessing across domains, preserved provenance — and say how you confirmed it.

## 5. Document

Before merge:

- Update `README.md` when public behavior, setup, commands, or architecture changed.
- Update `notes/Development Log.md` with a dated entry: goal, work completed, commands, result, problems or surprises, next step, related files.
- Update `notes/Decisions.md` only when the user accepted a decision. Never promote an unapproved idea to an accepted decision.
- Tick completed items in `notes/Tasks.md`, and add newly discovered work rather than leaving it undocumented.
- Correct any `agent-instructions/` or notes content the change made materially false.

When an observable predicate does not apply, mark it N/A in the checklist instead of inventing content.

## 6. Sync and verify

Walk the definition of done in `SKILL.md`, then sync the task branch with the latest `dev`:

```bash
git checkout dev
git pull
git checkout <branch>
git merge dev
```

Resolve conflicts deliberately. Rerun the full suite after any conflict resolution.

## 7. Finish

### Local merge mode (default)

With a remote:

```bash
git push origin <branch>
git checkout dev
git merge --no-ff <branch>
git push origin dev
git branch -d <branch>
git push origin --delete <branch>
```

Without a reachable remote, perform the checkout, `--no-ff` merge, and safe local branch deletion only. State clearly that nothing was pushed.

Confirm each command succeeded before continuing. A rejected push is not completion: resolve the divergence, or switch to pull request mode when branch protection caused the rejection.

### Pull request mode (GitHub)

Use only when the user asks for review, or when a protected branch rejects a direct push. Push the branch and open a PR targeting `dev`:

```bash
gh pr create --base dev --fill
```

Report the PR link and stop for user review. Do not merge or delete the branch yourself.

## 8. Report

Summarize the outcome, changed files, commits, focused and full test results, research invariants checked, documentation and notes updates, and exactly what was pushed or left local. State that lint/format/CI were N/A.
