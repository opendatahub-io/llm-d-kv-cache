# Sync Upstream

Sync code from the upstream llm-d org to a user's fork and open a PR to opendatahub-io.

## Configuration

```bash
ODH_BRANCH="main"  # The target branch on opendatahub-io. Change this if the branch name changes.
```

## Input

Ask the user: **"Sync to upstream/main HEAD or a specific commit SHA?"**

- If the user provides a **commit SHA**, use that as the target commit
- If no SHA is given, default to `upstream/main` HEAD

## Workflow

1. **Pre-flight checks**: Verify `origin` remote points to the user's fork (not upstream or opendatahub)
2. **Fetch remotes**: Add/update `upstream` and `opendatahub` remotes, fetch `upstream/main` and `opendatahub/${ODH_BRANCH}`
3. **Resolve target commit**: Use the user-provided SHA, or default to `upstream/main` HEAD. Verify it exists on `upstream/main`
4. **Show what will be synced**: Display a formatted summary to the user before proceeding. Use this exact format:
   ```
   Target: upstream/main HEAD (<short_sha>)
   Into: opendatahub-io/<repo-name> ${ODH_BRANCH} branch

   <N> commits including:
   - <sha> — <commit message>
   - ...

   <N> files changed, +<additions> / -<deletions> lines

   Creating the sync branch and merging now.
   ```
5. **Check for duplicates**: If a sync branch or PR for this SHA already exists, inform the user and stop
6. **Create sync branch**: Create `sync/upstream-<short_sha>` based on `opendatahub/${ODH_BRANCH}`
7. **Merge upstream**: Merge the target upstream commit into the sync branch. Resolve conflicts if needed
8. **Verify merge**: Confirm merge completed successfully
9. **Push branch**: Push to `origin`
10. **Verify push**: Confirm branch was pushed successfully
11. **Show PR summary**: Display what will be included in the PR
12. **Confirm PR creation**: Ask the user whether to open a PR or stop with the branch pushed only
13. **Open PR** (if confirmed): Open a PR to `opendatahub-io/llm-d-kv-cache` targeting `${ODH_BRANCH}`

## Commands

**IMPORTANT**: Use `git -C <repo_path>` for all git commands to avoid working directory issues. Detect the repository root dynamically.

```bash
# 0. Detect repository path and save current branch
REPO_PATH=$(git rev-parse --show-toplevel)
ORIGINAL_BRANCH=$(git -C "${REPO_PATH}" rev-parse --abbrev-ref HEAD)

# 1. Pre-flight: verify origin is the user's fork
ORIGIN_URL=$(git -C "${REPO_PATH}" remote get-url origin)
if echo "${ORIGIN_URL}" | grep -qE '(llm-d/llm-d-kv-cache|opendatahub-io/llm-d-kv-cache)'; then
  echo "Error: origin remote points to upstream or opendatahub, not your fork"
  exit 1
fi

# 2. Set up remotes and fetch
git -C "${REPO_PATH}" remote add upstream https://github.com/llm-d/llm-d-kv-cache.git 2>/dev/null || \
  git -C "${REPO_PATH}" remote set-url upstream https://github.com/llm-d/llm-d-kv-cache.git
git -C "${REPO_PATH}" remote add opendatahub https://github.com/opendatahub-io/llm-d-kv-cache.git 2>/dev/null || \
  git -C "${REPO_PATH}" remote set-url opendatahub https://github.com/opendatahub-io/llm-d-kv-cache.git
git -C "${REPO_PATH}" fetch upstream main
git -C "${REPO_PATH}" fetch opendatahub "${ODH_BRANCH}"

# 3. Resolve target commit
TARGET_COMMIT="${USER_SHA:-upstream/main}"
FULL_SHA=$(git -C "${REPO_PATH}" rev-parse "${TARGET_COMMIT}")
SHORT_SHA=$(git -C "${REPO_PATH}" rev-parse --short "${TARGET_COMMIT}")
git -C "${REPO_PATH}" merge-base --is-ancestor "${FULL_SHA}" upstream/main || {
  echo "Error: commit not on upstream/main"; exit 1;
}

# 4. Show what will be synced
echo "=== Commits to be synced ==="
git -C "${REPO_PATH}" log opendatahub/${ODH_BRANCH}..${FULL_SHA} --oneline
echo ""
echo "=== File changes summary ==="
git -C "${REPO_PATH}" diff --stat opendatahub/${ODH_BRANCH}...${FULL_SHA}

# 5. Check for duplicates
BRANCH="sync/upstream-${SHORT_SHA}"
if git -C "${REPO_PATH}" show-ref --verify --quiet "refs/heads/${BRANCH}" || \
   git -C "${REPO_PATH}" show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
  echo "Warning: branch ${BRANCH} already exists"
  # Ask user whether to force-update or skip
fi

# 6. Create branch from opendatahub/${ODH_BRANCH}
git -C "${REPO_PATH}" checkout -b "${BRANCH}" opendatahub/${ODH_BRANCH}

# 7. Merge upstream commit into the sync branch
git -C "${REPO_PATH}" merge --no-ff "${FULL_SHA}" --no-edit \
  -m "Sync upstream llm-d/llm-d-kv-cache ${SHORT_SHA}"

# 8. Verify merge succeeded
if [ $? -eq 0 ]; then
  echo "✓ Merge completed successfully"
  git -C "${REPO_PATH}" log -1 --stat
else
  echo "✗ Merge failed"
  exit 1
fi
```

## Conflict Resolution

If step 7 produces merge conflicts:

1. List conflicted files with `git -C "${REPO_PATH}" diff --name-only --diff-filter=U`
2. Show the conflicts to the user
3. Attempt to resolve trivial conflicts automatically (whitespace, import order)
4. For non-trivial conflicts, show the diff and ask the user how to resolve each file
5. After all conflicts are resolved:
   ```bash
   git -C "${REPO_PATH}" add -u
   git -C "${REPO_PATH}" commit --no-edit
   ```

## Push and Open PR

```bash
# 9. Push to origin (user's fork)
git -C "${REPO_PATH}" push -u origin "${BRANCH}"

# 10. Verify push succeeded
if git -C "${REPO_PATH}" ls-remote --heads origin "${BRANCH}" | grep -q "${BRANCH}"; then
  echo "✓ Branch pushed successfully to origin/${BRANCH}"
else
  echo "✗ Push verification failed"
  exit 1
fi
```

**Before creating the PR, show what will be included and ask the user whether to open a PR:**

```bash
# 11. Show PR summary
echo "=== PR Summary ==="
echo "From: upstream/main (${FULL_SHA})"
echo "To: opendatahub-io/llm-d-kv-cache ${ODH_BRANCH}"
echo ""
echo "Commits to be included:"
git -C "${REPO_PATH}" log opendatahub/${ODH_BRANCH}..HEAD --oneline
echo ""
echo "Files changed:"
git -C "${REPO_PATH}" diff --stat opendatahub/${ODH_BRANCH}..HEAD
```

**Ask the user: "Do you want to open a PR to opendatahub-io or skip?"**

If the user chooses to open the PR:

```bash
# 12. Get fork owner from origin remote URL
# NOTE: Do not use `gh repo view --json owner` — on forks it resolves to the parent repo owner.
FORK_OWNER=$(git -C "${REPO_PATH}" remote get-url origin | sed -E 's|.*[:/]([^/]+)/[^/]+(.git)?$|\1|')

# 13. Open PR to opendatahub-io
gh pr create \
  --repo opendatahub-io/llm-d-kv-cache \
  --base "${ODH_BRANCH}" \
  --head "${FORK_OWNER}:${BRANCH}" \
  --title "[sync] upstream llm-d main branch ${SHORT_SHA} [$(date -u +%Y-%m-%d)]" \
  --body "$(cat <<'EOF'
Syncs llm-d/llm-d-kv-cache main branch into ODH ${ODH_BRANCH} branch.

Upstream commit: https://github.com/llm-d/llm-d-kv-cache/commit/${FULL_SHA}
EOF
)"
```

After PR creation (or skip), immediately return to the original branch without asking:

```bash
# 14. Return to the original branch
git -C "${REPO_PATH}" checkout "${ORIGINAL_BRANCH}"
```

If `gh pr create` fails, inform the user that the branch has been pushed to `origin/${BRANCH}` and ask them to create the PR manually at `https://github.com/opendatahub-io/llm-d-kv-cache/compare/${ODH_BRANCH}...${FORK_OWNER}:${BRANCH}`.

## Completion Summary

After all steps are done, display a formatted completion summary to the user. Use this exact format:

```
Sync completed successfully

- Branch: sync/upstream-<short_sha> pushed to origin
- PR: opendatahub-io/llm-d-kv-cache#<pr_number>
- URL: <pr_url>
- Syncs: <N> upstream commits from llm-d/llm-d-kv-cache main (<short_sha>) into opendatahub-io ${ODH_BRANCH}
- Conflict resolved: <file> (<resolution details>)   ← only if conflicts were resolved
- Returned to: <original_branch> branch
```

If the user chose to skip the PR, omit the PR and URL lines.

## Error Handling

- If the user-provided SHA does not exist on `upstream/main`, report the error and ask for a valid SHA
- If conflicts cannot be resolved, abort the merge and clean up:
  ```bash
  git -C "${REPO_PATH}" merge --abort
  git -C "${REPO_PATH}" checkout "${ORIGINAL_BRANCH}"
  git -C "${REPO_PATH}" branch -D "${BRANCH}"
  ```
- If the branch already exists, ask the user whether to force-update or skip
- On any failure after branch creation, clean up with:
  ```bash
  git -C "${REPO_PATH}" checkout "${ORIGINAL_BRANCH}"
  git -C "${REPO_PATH}" branch -D "${BRANCH}"
  ```
- Always return the PR URL on success

## Best Practices

1. **Use `git -C` consistently**: Never rely on `cd` for git operations - always use `git -C "${REPO_PATH}"` to avoid shell working directory issues

2. **Run commands in background when needed**: For long-running operations that might have shell issues, use background bash with `run_in_background: true` and check output with `BashOutput` tool

3. **Verify critical operations**: Always verify that merge and push succeeded before proceeding

4. **Show before asking**: Display what will be synced/included before asking user to confirm PR creation

5. **Handle errors gracefully**: Provide clear error messages and cleanup instructions if operations fail
