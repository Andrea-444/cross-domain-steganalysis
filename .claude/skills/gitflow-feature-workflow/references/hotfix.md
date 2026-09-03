# Hotfix playbook

Plan first and hand the plan off per `references/plan.md`, unless the user asks for the fix in one sitting — urgency does not remove the planning step, it shortens it.

Hotfixes fix urgent defects in already-tagged work on `main`. They branch from `main` (not `dev` — dev may contain unreleased protocol changes you must not ship) and must land in **both** long-lived branches. The dual merge is the step most often forgotten; forgetting it means the fix silently disappears at the next milestone.

In this repository a hotfix is warranted when tagged work is actively wrong in a way that matters: a result-invalidating bug in embedding, splitting, or metrics; a broken documented command; or a leaked credential. Ordinary defects found during development are features, not hotfixes — use `references/feature.md`.

```bash
# 1. Branch from up-to-date main; version = current tag + patch bump
git checkout main && git pull
git checkout -b hotfix/0.1.1
git push -u origin hotfix/0.1.1   # publish the branch before making changes

# 2. Only now touch code, test-first: write the regression test, run it and see it
#    fail, then fix the defect until it passes, then run the full suite.
#    Commit test and fix together with Conventional Commits (fix: ...)

# 3. Record the fix in notes/Development Log.md as a dated entry.
#    Commit: docs: record hotfix 0.1.1

# 4. Merge into main and tag
git checkout main && git pull
git merge --no-ff hotfix/0.1.1
git tag -a v0.1.1 -m "Hotfix 0.1.1: <one-line summary>"
git push origin main --follow-tags

# 5. Merge into dev as well — NOT optional
git checkout dev && git pull
git merge --no-ff hotfix/0.1.1
git push origin dev

# 6. Delete the branch (local and remote) and summarize
git branch -d hotfix/0.1.1
git push origin --delete hotfix/0.1.1
```

Verify both pushes (main with tags, dev) succeeded before reporting the hotfix as done — unpushed work means the task is not finished.

The definition-of-done checklist and the "Interaction, remotes, and CI" rules from `SKILL.md` apply (no reachable remote → local-only with a clear note; protected branches → pull request mode; CI is N/A here), documentation and notes updates included.

**If the defect invalidates existing results**, say so explicitly in the Development Log entry and tell the user which runs under `results/` or `stego_runs/` are affected. Do not delete, rewrite, or regenerate those runs to hide the problem — `agent-instructions/DATA_AND_ARTIFACTS.md` requires preserving them.

**Exception:** if an active `release/*` branch exists, merge the hotfix into it instead of `dev` (the release branch will carry it into dev) — mention this to the user.
