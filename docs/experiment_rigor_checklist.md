# Experiment Rigor Checklist

## v5 Real Rigor

- [x] High-fidelity MuJoCo action-selection benchmark.
- [x] MuJoCo-generated training labels.
- [x] CPU-only PyTorch MLP, Deep Sets, transformer, and RC-RET models.
- [x] Expanded 63-action candidate manifold.
- [x] Branch-risk, clearance, and set-relative features.
- [x] Paired evaluation tasks across all methods.
- [x] Multiple seeds: 8.
- [x] Episodes per split: 96 paired episodes.
- [x] Ten stress splits.
- [x] Strong baselines: robust MPC, branch-CVaR, shielded MLP, Deep Sets, oracle.
- [x] Ablations on combined shift and narrow clearance.
- [x] Confidence intervals, paired bootstrap intervals, and sign-flip p-values.
- [x] Raw CSVs, summary CSVs, figures, generated LaTeX tables, and validation script.
- [x] 25-page ICLR-style PDF with bright boxed clickable citations.

## ICLR Main Bar

- [ ] Real-robot validation.
- [ ] External public manipulation benchmark.
- [ ] Clear success gain over MLP energy.
- [ ] Clear success gain over Deep Sets.
- [ ] Clear success/regret gain over robust MPC or branch-CVaR.
- [ ] Attention isolated as necessary.
- [ ] Richer primitive generator for narrow-clearance cases.

Decision: STRONG_REVISE. The evidence is real and substantially stronger than v4, but it does not justify an ICLR-main-ready positive claim.
