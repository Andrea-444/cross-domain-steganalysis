#!/bin/sh
# Self-consistency check for the gitflow-feature-workflow skill.
# Verifies that the skill's files exist and that its invariants still hold.
# Usage: sh scripts/check_skill.sh [skill_path]

set -u

SKILL_PATH="${1:-$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)}"
failures=0

fail() {
    printf 'FAIL: %s\n' "$1"
    failures=$((failures + 1))
}

require_file() {
    [ -f "$SKILL_PATH/$1" ] || fail "Missing required file: $1"
}

# Reports a failure when $2 is NOT found in file $1.
require_match() {
    if ! grep -qF -- "$2" "$SKILL_PATH/$1" 2>/dev/null; then
        fail "$1 should contain: $2"
    fi
}

# Reports a failure when $2 IS found in file $1.
forbid_match() {
    if grep -qF -- "$2" "$SKILL_PATH/$1" 2>/dev/null; then
        fail "$1 must not contain: $2"
    fi
}

for f in \
    SKILL.md \
    references/plan.md \
    references/feature.md \
    references/hotfix.md \
    references/release.md \
    references/project-context.md \
    agents/openai.yaml
do
    require_file "$f"
done

[ "$failures" -eq 0 ] || { printf '\n%d check(s) failed.\n' "$failures"; exit 1; }

# Frontmatter
require_match SKILL.md 'name: gitflow-feature-workflow'
require_match SKILL.md 'description:'

# Every playbook is routed from SKILL.md
require_match SKILL.md 'references/feature.md'
require_match SKILL.md 'references/hotfix.md'
require_match SKILL.md 'references/release.md'
require_match SKILL.md 'references/plan.md'
require_match SKILL.md 'references/project-context.md'

# Core Gitflow invariants
require_match SKILL.md 'Never commit directly to'
require_match SKILL.md '--no-ff'
require_match SKILL.md '.gitflow/plan.md'
require_match references/feature.md 'git checkout -b <branch>'
require_match references/hotfix.md 'git checkout -b hotfix/'
require_match references/release.md 'git checkout -b release/'

# Hotfix must merge into BOTH long-lived branches
require_match references/hotfix.md 'git merge --no-ff hotfix/'
require_match references/hotfix.md 'NOT optional'

# The user's branch-naming convention: feat / bug / spec / junk, descriptive
# kebab-case names, and no wip/ prefix.
require_match SKILL.md '`feat/<descriptive-name>`'
require_match SKILL.md '`bug/<descriptive-name>`'
require_match SKILL.md '`spec/<descriptive-name>`'
require_match SKILL.md '`junk/<descriptive-name>`'
require_match SKILL.md 'never merged'
require_match SKILL.md 'There is no `wip/` prefix'
forbid_match SKILL.md 'feature/<short-name>'
forbid_match SKILL.md 'short kebab-case'
forbid_match references/plan.md 'feature/<short-name>'

# This repository's adaptations, which are easy to regress toward the
# upstream embroidery skill they were derived from.
require_match SKILL.md 'agent-instructions/'
require_match SKILL.md 'notes/Decisions.md'
require_match references/plan.md 'envs/experimental_1/bin/python'
require_match references/plan.md 'unittest discover -s tests'

# GitHub, not GitLab
forbid_match SKILL.md 'glab'
forbid_match references/feature.md 'glab'

# No fabricated tooling: this repo has no linter, formatter, CI, or CHANGELOG.
forbid_match SKILL.md 'CHANGELOG.md'
require_match references/project-context.md 'There is no `CHANGELOG.md`'
require_match SKILL.md 'no formatter, linter, or type checker'

# No AI attribution in commits, per project convention
require_match SKILL.md 'Do not add a co-authorship trailer'
require_match references/feature.md 'Do not add a co-authorship trailer'

if [ "$failures" -eq 0 ]; then
    printf 'All checks passed.\n'
    exit 0
fi

printf '\n%d check(s) failed.\n' "$failures"
exit 1
