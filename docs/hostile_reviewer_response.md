# Hostile Reviewer Response

Paper: 62 Energy Transformer Action Selection

## Strongest Technical Threats

The local retrieval set is noisy and contains many non-robotics energy/transformer papers. That is itself a weakness. The real technical threats for a revived version are energy-based action selection, sampling-based MPC, diffusion/transformer robot policies, and constraint-aware model-predictive control.

## ICLR Main Response

A hostile ICLR reviewer would no longer be correct to reject the paper for having only synthetic evidence. The v4 rebuild contains MuJoCo rollout labels, a trained PyTorch transformer energy scorer, baselines, stress splits, ablations, and confidence intervals.

The reviewer would still be correct to reject the paper as an ICLR-main submission because the claimed transformer mechanism is not decisively validated. MLP energy and robust MPC remain competitive or better, and the transformer increases violation rate on several splits.

## Honest Action

The paper is marked `STRONG_REVISE`. This keeps the real benchmark and code, but prevents overclaiming submission readiness.

## What Would Be Needed To Revive

- Manual robotics related-work synthesis.
- Clear transformer advantage over MLP energy.
- Clear advantage over robust/nominal MPC.
- Larger and more diverse training distribution.
- Public benchmark or hardware validation.
- Stronger feasibility guarantee.
