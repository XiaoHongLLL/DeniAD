# Version provenance

This review artifact separates the manuscript reproduction snapshot from later
exploratory development.

- Core `main.py`: frozen legacy-absence/RQ3 alignment snapshot dated
  2026-07-15.
- `transformer/Models.py` and RQ4 cloud runner: context-absence compatible
  snapshot dated 2026-07-13.
- RQ4 threshold-selection CSV fix and dev-only objective support: 2026-07-19.
- Processed Train-Ticket dataset archive: expanded100 v0.4, SHA-256 recorded in
  `dataset/trainticket_processed/SHA256SUMS.txt`.

The final public tag should be created only after rerunning the exact final
paper commands and confirming that all reported tables point to this code
snapshot. Later local-conformal, DACD, matched-supervision, and FlexLog adapter
experiments are intentionally outside this artifact.

