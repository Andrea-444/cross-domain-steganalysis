# Plan and model-handoff playbook

The plan is written by the strong model that is loaded during planning; the implementation usually runs afterwards on a cheaper model, often in a fresh session with none of the planning conversation in context. Write for that reader: explicit paths, explicit commands, explicit test names, no "as discussed above".

## 1. Investigate before writing the plan

Read `AGENTS.md`, the routed `agent-instructions/` file, and `notes/Decisions.md`; inspect Git state; and read the files the change actually touches. Never plan from filenames alone — a plan built on guessed function signatures sends the implementer into code that does not exist.

For anything touching datasets, manifests, embedding, or the detector, also check the current artifact state on disk (`data/`, `stego_runs/`, `results/`). A plan that assumes a cohort has been extracted when it has not will fail on the first command.

## 2. Ask everything now

Planning is the cheap moment to ask. Ask when the answer changes the plan:

- Requirements that admit more than one reasonable reading.
- Undefined edge cases, empty/invalid input, and error behavior.
- Scope boundaries: what is explicitly out of this branch.
- Acceptance criteria and how the user wants the result verified.
- A requested library, pattern, schema, or manifest shape you would not choose yourself.
- Anything that would change the scientific protocol rather than the code — see the "Ask before you build" section in `SKILL.md`.

When a better implementation exists, state the alternative, why it is better, what it costs, and your recommendation — then follow the user's decision and record it in the plan. Batch questions into a single round. Do not re-confirm requirements the user already made explicit.

Unanswered questions must not survive into the plan as prose hedges. Either the user answered, or the plan records an explicit assumption under **Assumptions** so the implementer knows it was a choice.

## 3. Suggest skills

Scan the skills available in the session and list the ones that fit this task, one line each on why. Suggest only skills that are actually present. If a skill will be used during implementation, name it in the plan so the cheaper model invokes it.

## 4. Write `.gitflow/plan.md`

Ensure `.gitflow/` is ignored:

```bash
grep -qxF '.gitflow/' .gitignore || printf '.gitflow/\n' >> .gitignore
```

Use this structure:

```markdown
# Plan: <short title>

Branch: feat/<descriptive-name> (from dev)   Semver: minor
Status: PLANNED

## Goal
One paragraph: the observable outcome when this is done.

## Context
Files that matter, with paths and the functions/classes involved.
Relevant recorded decisions from notes/Decisions.md.
Verified commands (see Verification below).

## Decisions
Choices made with the user, including alternatives rejected and why.

## Assumptions
Explicit assumptions the implementer must not silently change.

## Research invariants touched
Which of the non-negotiable constraints in AGENTS.md this work could affect
(pair-safe splits, equivalent preprocessing, provenance, run labelling),
and how the plan keeps each one intact. N/A with a reason when none apply.

## Suggested skills
- <skill-name> — why it applies.

## Test plan (write these first)
1. `<test file>::<test name>` — behavior asserted, expected failure before the fix.
2. ... edge cases and error paths, not only the happy path.

## Implementation steps (TDD order)
1. [ ] Red: add test 1, run `<command>`, confirm it fails for the right reason.
2. [ ] Green: minimal change in `<file>` to pass.
3. [ ] Refactor with the suite green.
4. [ ] Commit: `feat: ...`
5. [ ] ... repeat per behavior.

## Verification
Exact commands and the expected result of each.

## Docs and notes
README / notes vault entries required, or N/A with the reason.

## Out of scope
What this branch deliberately does not do.
```

## 5. Verified commands for this repository

Record these verbatim in the plan's Verification section. The project environment is the `experimental_1` conda environment.

```bash
# Full test suite
~/miniconda3/envs/experimental_1/bin/python -m unittest discover -s tests -t . -v

# One focused test module
~/miniconda3/envs/experimental_1/bin/python -m unittest tests.datasets.test_incremental_selection -v
```

Gotchas that have actually bitten a session here:

- A bare `python` is **not** on `PATH` in non-interactive shells, and `conda` itself is not either. Call the environment's interpreter by absolute path, as above. In an interactive shell `conda activate experimental_1` works.
- `head` on this machine resolves to LWP's `head` (an HTTP tool), not GNU `head`. Use `/usr/bin/head` when piping output.
- Some tests skip when optional SRNet or OpenStego dependencies are absent. Report skips separately from passes and failures.
- Lint, format, type-check, and CI commands do not exist in this repository. Mark them N/A; do not invent one.

## 6. Hand off

Report the plan path, the branch to be created (with its `feat/`, `bug/`, or `spec/` prefix), the open decisions already resolved, and stop. Tell the user they can switch to the cheaper model now. Implement in the same session only when the user explicitly says not to switch.

A useful resume prompt to give the user:

```text
Use gitflow-feature-workflow: execute .gitflow/plan.md
```

## 7. Resume as the implementing model

1. Read `.gitflow/plan.md` and `AGENTS.md` in full before touching anything.
2. Follow the playbook for the branch type; create the branch before editing.
3. Execute the implementation steps in order, test-first, ticking each `[ ]` to `[x]` in the plan file as it lands.
4. If the code contradicts the plan — the function does not exist, the test cannot be written as described, a step is already done — stop and report the mismatch. Do not redesign silently.
5. Set `Status: DONE` when the work merges, and delete `.gitflow/plan.md` or say that it was kept.
