# Paper 62 Final v5 Evidence Summary

Decision: `STRONG_REVISE`.

ICLR main readiness: no.

## Frozen Evidence Size

- Main raw rows: 11520.
- Ablation raw rows: 2112.
- Metric rows: 120.
- Ablation metric rows: 22.
- Pairwise rows: 99.
- Frozen run: 480 training action sets, 24 epochs, 8 seeds, 12 episodes, 10 stress splits, 12 main methods.

## Gate Results

- Strong-baseline success wins: 1/8.
- Non-oracle ties-or-wins under the success gate: 3/7.
- Non-oracle safety ties-or-improvements: 6/7.
- Aggregate versus robust MPC: success delta -0.0042, regret improvement -0.0004, violation delta -0.0052.
- Aggregate versus MLP: success delta -0.0240, regret improvement -0.0013, violation delta -0.0333.
- Aggregate versus Deep Sets: success delta -0.0437, regret improvement 0.0070, violation delta -0.0813.
- Combined-shift RC-RET: success 0.1354, regret 0.0617, violation 0.0417.
- Narrow-clearance RC-RET: success 0.0000, regret 0.0456, violation 0.0417.

## Honest Interpretation

The v5 rebuild is substantially stronger than the v4 scaffold: it uses a larger action manifold, branch-risk features, a calibrated safety shield, Deep Sets and shielded baselines, ten stress splits, and paired statistics. The evidence is real and useful.

It is still not ICLR-main ready. RC-RET is close to robust/CVaR branch scoring and safer than several unshielded learned baselines, but it does not clear the success gate against MLP, Deep Sets, nominal MPC, or the oracle. The narrow-clearance split is a particularly hard negative case.

The correct terminal state is `STRONG_REVISE`, not `ICLR_MAIN_TARGET_READY`.
