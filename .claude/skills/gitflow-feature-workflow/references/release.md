# Release playbook

Releases stabilize what is on `dev` and promote it to `main` as a citable milestone — a frozen protocol, a completed pilot, or a thesis snapshot. Only bugfixes, version metadata, and release notes are allowed on a release branch — no new features or protocol changes (they wait for the next milestone).

```bash
# 1. Decide the version per the semver rules in SKILL.md.
#    The repository has no tags yet, so the first release is v0.1.0.
git checkout dev && git pull
git checkout -b release/0.1.0
git push -u origin release/0.1.0   # publish the branch before making changes

# 2. Only now touch files: finalize the milestone entry in notes/Development Log.md
#    and confirm notes/Decisions.md reflects every accepted choice in this milestone.
#    Commit: chore: prepare release 0.1.0

# 3. Run the full test suite. Fix defects directly on this branch
#    (fix: commits only). Never merge red.

# 4. Merge into main and tag
git checkout main && git pull
git merge --no-ff release/0.1.0
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin main --follow-tags

# 5. Merge back into dev (carries release-branch fixes)
git checkout dev && git pull
git merge --no-ff release/0.1.0
git push origin dev

# 6. Delete the branch (local and remote) and summarize
git branch -d release/0.1.0
git push origin --delete release/0.1.0
```

Verify both pushes (main with tags, dev) succeeded before reporting the release as done — unpushed work means the task is not finished.

The definition-of-done checklist and the "Interaction, remotes, and CI" rules from `SKILL.md` apply (no reachable remote → local-only with a clear note; protected branches → pull request mode; CI is N/A here).

## Release notes in the notes vault

This repository does **not** use `CHANGELOG.md`. The equivalent record is the notes vault, which is already the project's lab notebook:

- `notes/Development Log.md` — a dated entry per milestone summarizing what changed, what was verified, and what remains. This is the release note.
- `notes/Decisions.md` — the authority for accepted protocol choices. A milestone must not tag work whose decisions are still recorded as `proposed`.
- `notes/Tasks.md` — completed items ticked, remaining work visible.

Before tagging, confirm the milestone's claims are honest: a run labelled `smoke` or `pilot` must not be described as a final result, and `notes/Experiment Decisions.md` must still show `protocol-frozen: false` unless the user has explicitly frozen it.

## Tagging research milestones

An annotated tag is the reproducibility anchor: it lets a result directory be tied to the exact code that produced it. When a milestone corresponds to executed experiments, name the relevant run directories under `results/` in the Development Log entry so the tag and the artifacts can be matched later.
