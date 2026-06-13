# Submission Attack Log

Paper: 62 energy_transformer_action_selection

This v4 pass rebuilds the paper with real MuJoCo/PyTorch evidence. The result is strong revise, not ICLR-main readiness.

## Rebuild Round 1

Attack: The previous evidence was synthetic/template-generated.

Verdict: Recovered.

Action: Replaced `src/run_experiment.py` with MuJoCo data generation and PyTorch training.

## Rebuild Round 2

Attack: No learned model.

Verdict: Recovered.

Action: Added MLP and set-transformer energy scorers trained from MuJoCo rollout labels.

## Rebuild Round 3

Attack: No implemented baselines.

Verdict: Recovered.

Action: Added random, geometric, nominal rollout MPC, robust worst-case MPC, MLP energy, and oracle rollout selectors.

## Rebuild Round 4

Attack: No stress tests.

Verdict: Recovered.

Action: Added nominal, low-friction, high-friction, heavy-object, obstacle-shift, and combined-shift splits.

## Rebuild Round 5

Attack: No ablations.

Verdict: Recovered, but unfavorable.

Action: Combined-shift ablations show MLP and small-data transformer match or exceed the full transformer on success/regret.

## Rebuild Round 6

Attack: Transformer mechanism must beat MLP.

Verdict: Not recovered.

Action: Mark as a remaining blocker.

## Rebuild Round 7

Attack: Method must beat strong MPC baselines.

Verdict: Not recovered.

Action: Transformer does not consistently beat nominal rollout MPC or robust worst-case MPC.

## Rebuild Round 8

Attack: Feasibility should improve, not degrade.

Verdict: Not recovered.

Action: Transformer has higher violation rate than robust MPC on several splits.

## Rebuild Round 9

Attack: Related work is noisy and not manually synthesized.

Verdict: Not recovered.

Action: Keep as a blocker.

## Terminal Decision

Decision: STRONG_REVISE.

The paper is no longer a synthetic archive, but it is not submission ready.
