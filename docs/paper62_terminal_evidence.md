# Paper 62 Terminal Evidence

Decision: STRONG_REVISE.

ICLR main ready: no.

Submission-hardening version: v4 real MuJoCo/PyTorch rebuild.

## What Changed

- Replaced synthetic probability tables with a real MuJoCo action-selection benchmark.
- Generated MuJoCo rollout labels for candidate push-action sets.
- Trained lightweight PyTorch MLP and set-transformer energy scorers.
- Evaluated against random, geometric, nominal MPC, robust MPC, and oracle rollout baselines.
- Added six stress splits, five seeds, confidence intervals, ablations, paired deltas, figures, and reproducible CSVs.

## Main Evidence

Transformer energy success rates:
- nominal: 0.188 +/- 0.086
- low_friction: 0.062 +/- 0.053
- high_friction: 0.212 +/- 0.090
- heavy_object: 0.125 +/- 0.073
- obstacle_shift: 0.463 +/- 0.110
- combined_shift: 0.412 +/- 0.109

The transformer beats random and geometric selection, but it does not consistently beat MLP energy, nominal rollout MPC, or robust worst-case MPC. It also raises violation rate relative to robust MPC on several splits.

## Ablation Evidence

Combined-shift ablations do not isolate self-attention as the decisive component. MLP energy and small-data transformer match or slightly beat the full transformer on success/regret, while no-feasibility increases violation/regret.

## Terminal Reason

The paper now has real learned-model and high-fidelity simulation evidence, but the claimed transformer mechanism is not validated. The correct terminal state is STRONG_REVISE, not ICLR_MAIN_READY.
