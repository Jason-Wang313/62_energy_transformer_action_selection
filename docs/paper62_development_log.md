# Paper 62 Development Log

Date: 2026-06-19

## Pre-Freeze Development

- Read the existing v4 repo state, manuscript, final audit, terminal evidence, and CSV summaries.
- Confirmed starting terminal state: `STRONG_REVISE`, with the transformer failing to consistently beat MLP energy, nominal MPC, or robust MPC.
- Wrote `docs/paper62_expanded_submission_plan_20260619.md` before editing code.
- Rebuilt `src/run_experiment.py` into a v5 RC-RET runner:
  - expanded candidate action library from 14 to 63 actions;
  - added branch-risk, clearance, rank, and robust-energy features;
  - added Deep Sets, shielded MLP, branch-CVaR, old-transformer, and RC-RET variants;
  - added stronger stress splits and raw ablation outputs;
  - added paired bootstrap/sign-flip statistics and additional plots.
- Added dev-output subdirectories so smoke runs do not overwrite canonical final results.
- Smoke run: `--train-tasks 4 --epochs 1 --seeds 1 --episodes 1 --splits nominal combined_shift`.
- Medium dev run: `--train-tasks 80 --epochs 4 --seeds 2 --episodes 4 --splits nominal low_friction narrow_clearance combined_shift`.
- Medium dev result exposed a legitimate pre-freeze method defect: pure neural RC-RET scoring was too fragile relative to branch-risk baselines.
- Pre-freeze fix: anchored RC-RET inference to the predefined robust-risk/geometry frontier and used the transformer as residual calibration.
- Medium anchored dev run showed better regret stability but a hard padded shield was too conservative.
- Pre-freeze fix: reduced the safety shield from an extra-padded clearance rule to the modeled contact margin plus a small buffer.
- Bumped the cache version after changing shield-dependent risk labels.
- Final post-fix smoke run passed.

## Freeze Boundary

After `docs/paper62_protocol_freeze_20260619.md` is written, no further method/protocol changes are allowed unless needed to fix recoverable infrastructure failures. Final results must be reported honestly even if they keep the paper in `STRONG_REVISE`.

