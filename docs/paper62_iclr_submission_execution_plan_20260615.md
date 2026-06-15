# Paper 62 ICLR-Main Execution Plan

Date: 2026-06-15

Paper: `62_energy_transformer_action_selection`

Goal: determine whether the current MuJoCo/PyTorch evidence can honestly upgrade the paper from `STRONG_REVISE` to an ICLR-main-target submission, or reaffirm a terminal non-ready decision with exact evidence.

## RAM-Light Execution Policy

Use existing checked-in MuJoCo/PyTorch outputs as the primary evidence source unless integrity checks fail. Do not rerun full training by default because it is heavier than needed for a continuation audit. Re-run lightweight checks, CSV validation, plots/PDF rebuilds, and code compilation first.

## Execution Gates

1. Reproducibility gate:
   - Compile `src/run_experiment.py`.
   - Confirm the stored results include main, seed, paired, ablation, stress, and negative-case outputs.
   - Rebuild the PDF from `paper/main.tex`.

2. Evidence gate:
   - Confirm the benchmark uses real MuJoCo rollout labels, not synthetic probability tables.
   - Confirm PyTorch MLP and set-transformer energy scorers are implemented.
   - Confirm multiple seeds, six stress splits, uncertainty estimates, paired comparisons, and ablations exist.
   - Confirm baselines include random candidate, geometric greedy, nominal rollout MPC, robust worst-case MPC, MLP energy scorer, and oracle MuJoCo rollout selector.

3. ICLR-main claim gate:
   - Require transformer energy to clearly beat MLP energy.
   - Require transformer energy to clearly beat nominal/robust MPC on success or regret without increasing violation rate.
   - Require ablations to isolate self-attention, feasibility, and action-set context as necessary.
   - Require hostile related-work pressure and honest limitations.

4. Artifact gate:
   - Rebuild `paper/main.pdf`.
   - Copy only `C:/Users/wangz/Downloads/62.pdf`.
   - Confirm `C:/Users/wangz/Desktop/62.pdf` is absent.
   - Confirm the GitHub repository is public and pushed.

## Decision Rule

Upgrade only if every ICLR-main claim gate is supported by current evidence. If transformer energy still ties or loses to MLP/robust/nominal MPC, if violation rate is worse, or if ablations do not isolate the mechanism, keep the terminal decision as `STRONG_REVISE`.
