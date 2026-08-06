# Third-party provenance and licensing review

This artifact contains modified components whose provenance must be retained.

1. The temporal encoder follows the Transformer Hawkes Process implementation:
   SimiaoZuo/Transformer-Hawkes-Process. The upstream repository does not
   clearly expose a software licence in the current project metadata. Obtain
   permission or replace the copied implementation with a clean, independently
   licensed implementation before assigning a permissive licence to these
   files.
2. `flow_matching/solver.py` states that it is based on Meta/Facebook Research
   Flow Matching. The current upstream project reports a CC BY-NC licence.
   Preserve attribution and confirm that the intended public/research use is
   compatible with the exact upstream version used.
3. `torchdiffeq` is an external runtime dependency and is not redistributed.
4. HDFS, BGL, and Thunderbird are obtained from Loghub. Spirit and Liberty are
   obtained from the USENIX Computer Failure Data Repository. Their raw logs
   are not redistributed in this repository.
5. FlexLog, Multi-CAD, DeepLog, LogAnomaly, LogBERT, and LogSD source trees are
   not redistributed. The manuscript and reproduction notes should cite their
   original repositories and record exact commits.

The project authors must choose a licence only for code they own and must not
relicense third-party files contrary to upstream terms.

