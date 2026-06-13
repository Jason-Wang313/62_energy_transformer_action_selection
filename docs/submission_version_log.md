# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed synthetic metrics, stronger synthetic baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats were not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Real MuJoCo/PyTorch Rebuild

- Added concrete rebuild plan.
- Replaced synthetic scaffold with real MuJoCo action-set rollout labels and PyTorch energy scorers.
- Added six stress splits, implemented baselines, ablations, paired statistics, and figures.
- Rewrote paper and docs around the actual evidence.
- Terminal decision: STRONG_REVISE.
