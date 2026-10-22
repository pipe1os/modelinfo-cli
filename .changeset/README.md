# Changesets

Each pull request that changes user-facing behavior adds a changeset: a small markdown file in this folder that names the bump type and describes the change in one sentence.

```bash
npx changeset
```

Pick `patch` for fixes, `minor` for new features or flags, and `major` for breaking changes. Write the summary the way it should read in `CHANGELOG.md`.

To cut a release, run `npm run version`. It consumes the changesets, bumps `package.json`, writes the new section to `CHANGELOG.md`, and copies the version into `pyproject.toml` and `src/modelinfo/__init__.py`. Commit the result, then publish a GitHub release tagged `vX.Y.Z` to trigger the PyPI publish workflow.
