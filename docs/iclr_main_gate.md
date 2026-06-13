# ICLR Main Gate

Paper: 62 energy_transformer_action_selection

Previous v3 decision: KILL_ARCHIVE.

v4 gate verdict: STRONG_REVISE.

ICLR main ready: no.

Evidence digest: real MuJoCo/PyTorch action-selection benchmark with 360 MuJoCo training tasks, 5 seeds, 6 stress splits, 7 main methods, ablations, confidence intervals, and figures.

Remaining blockers:
- Transformer energy scorer does not clearly beat MLP energy.
- Transformer energy scorer does not consistently beat nominal rollout MPC or robust worst-case MPC.
- Violation rate is higher than robust MPC on several splits.
- No hardware or external public benchmark.
- Related-work retrieval is noisy and requires manual robotics synthesis.

The only honest main-conference-safe decision is STRONG_REVISE, not submission ready.
