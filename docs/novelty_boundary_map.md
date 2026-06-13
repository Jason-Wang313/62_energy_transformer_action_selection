# Novelty Boundary Map

## Crowded Territory
- Bigger data/model scaling.
- New benchmark only.
- Generic active learning or uncertainty.
- Combining a planner with a learned policy without a new state/action object.

## Claimed Boundary
Energy transformer action selection keeps action-critical alternatives explicit until a physical observation collapses them.

## What Would Falsify The Claim
If observed-only baselines match the adverse-mode coverage and closed-loop success of the proposed branch-aware mechanism, the paper should be revised or killed.

## v4 Outcome
The v4 MuJoCo/PyTorch rebuild triggers a strong-revise condition. The transformer energy scorer beats random/geometric selection, but MLP energy and robust MPC remain competitive or better. The "transformer" part of the mechanism is not yet supported as necessary for action selection.
