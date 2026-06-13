# Experiment Rigor Checklist

## v4 Real Rigor

- [x] High-fidelity MuJoCo action-selection benchmark.
- [x] MuJoCo-generated training labels.
- [x] Trained PyTorch MLP and transformer energy scorers.
- [x] Paired evaluation tasks across methods.
- [x] Multiple seeds: 5.
- [x] Episodes per split/method: 80.
- [x] Six stress splits.
- [x] Implemented baselines.
- [x] Ablations.
- [x] Confidence intervals and paired deltas.
- [x] Reproducible CSVs and figures.

## ICLR Main Bar

- [ ] Real-robot validation.
- [ ] External manipulation benchmark.
- [ ] Clear transformer gain over MLP energy.
- [ ] Clear gain over robust/nominal MPC.
- [ ] Stronger feasibility guarantee.
- [ ] Manual full-paper related-work synthesis.

Decision: STRONG_REVISE. The evidence is real, but the transformer mechanism is not decisive enough for ICLR main.
