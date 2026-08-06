# Upload checklist

Do not upload until `PUBLICATION_BLOCKERS.md` is resolved.

## 1. Finalise metadata

1. Replace bracketed fields in `CITATION.cff.template` and rename it to
   `CITATION.cff`.
2. Add the approved top-level licence and preserve file-specific third-party
   notices.
3. Replace `[GITHUB URL]`, `[VERSION]`, and repository/DOI placeholders in the
   availability text.
4. Freeze the exact code version used to produce the final manuscript tables.

## 2. Create the GitHub repository

Create an empty repository, then run from this directory:

```bash
git init
git add .
git commit -m "Release DeniAD research artifact"
git branch -M main
git remote add origin https://github.com/OWNER/DeniAD.git
git push -u origin main
```

The repository should initially be private while the licence, blind-review
policy, and final test reproducibility are checked.

## 3. Create an immutable release

After validation, create a tagged release (for example, `v1.0.0`). Archive the
release with Zenodo or another repository that assigns a DOI. Put the release
tag, code DOI, data DOI, and SHA-256 manifest in the manuscript.

For double-blind review, follow the venue's anonymity rules. Do not expose
author names in the repository, commit history, archive metadata, or URLs if
the venue requires an anonymous artifact.

