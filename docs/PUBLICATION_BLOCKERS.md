# Publication blockers

Resolve these items before pushing the repository publicly or putting its URL
in the final manuscript.

- [ ] Confirm the final paper title, author list, affiliations, and repository
      owner; complete `CITATION.cff`.
- [ ] Choose a licence for author-owned code.
- [ ] Resolve redistribution permission for the Transformer Hawkes Process
      derived files and the Flow Matching derived file.
- [ ] Freeze the exact DeniAD version used for all final tables. The current
      working tree contains later exploratory development, so final results
      must be regenerated or an exact historical snapshot must be selected.
- [ ] Record the upstream Train-Ticket repository URL and exact commit.
- [ ] Verify that the processed Train-Ticket archive contains no credentials,
      private endpoints, personal identifiers, or third-party restricted data.
- [ ] Add final GitHub URL, release tag, archive DOI, and SHA-256 values to the
      README, Data Availability statement, and manuscript.
- [ ] Test all commands in a clean Linux environment.
- [ ] Create an immutable release and archive that release in Zenodo or another
      repository that issues a persistent identifier.

## Acceptable minimisation

It is appropriate to omit raw public datasets, checkpoints, caches, failed
runs, internal deployment details, and exploratory code. It is not appropriate
to omit the core joint type--time model, formal decision mechanism, split
construction, threshold-selection protocol, or scripts needed to regenerate
reported tables while claiming full reproducibility.

