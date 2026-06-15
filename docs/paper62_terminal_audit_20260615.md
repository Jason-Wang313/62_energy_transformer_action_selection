# Paper 62 Terminal Audit

Date: 2026-06-15

Paper: `62_energy_transformer_action_selection`

Decision: `STRONG_REVISE`

ICLR-main ready: no

## Commands Executed

- `python -m py_compile src\run_experiment.py`
- CSV finite/schema audit over `results/energy_action_raw.csv`, `results/energy_action_metrics.csv`, `results/energy_action_pairwise.csv`, `results/energy_action_ablation.csv`, `results/energy_action_seed_metrics.csv`, `results/negative_cases.csv`, and compatibility CSVs.
- `pdflatex`, `pdflatex` in `paper`
- `Copy-Item paper\main.pdf C:\Users\wangz\Downloads\62.pdf -Force`

## Verified Evidence

- Real MuJoCo/PyTorch action-selection benchmark is implemented in `src/run_experiment.py`.
- Main evidence contains 3,360 paired evaluation rows: 6 stress splits, 5 seeds, 16 episodes per seed/split/method, and 7 methods.
- Ablation evidence contains 480 rows on the combined-shift split.
- The benchmark includes MuJoCo rollout labels, PyTorch MLP and set-transformer energy scorers, confidence intervals, paired deltas, stress splits, ablations, negative cases, and generated figures.
- Baselines include random candidate, geometric greedy, nominal rollout MPC, robust worst-case MPC, MLP energy scorer, and oracle MuJoCo rollout selector.
- The rebuilt PDF is `C:/Users/wangz/Downloads/62.pdf`.
- `C:/Users/wangz/Desktop/62.pdf` is absent.

## Blocking Results

The current evidence does not support ICLR-main readiness:

- Transformer energy does not clearly beat MLP energy: combined shift is transformer `0.412 +/- 0.109` versus MLP `0.425 +/- 0.109`, and obstacle shift is tied.
- Transformer energy does not clearly beat robust MPC: combined and obstacle shifts are tied; nominal, low-friction, high-friction, and heavy-object splits are worse on success.
- Transformer energy often increases violation rate relative to robust MPC; for combined shift, transformer violation is `0.075` versus robust `0.037`.
- Ablations do not isolate self-attention: MLP energy and the small-data transformer match or slightly exceed the full transformer on combined-shift success/regret.
- The related-work retrieval is noisy and not a manual robotics prior-work synthesis.
- There is no public benchmark or hardware validation.

## Gate Decision

This paper satisfies the local evidence-package requirements for `STRONG_REVISE`: real simulator labels, a trained-model pipeline, strong baselines, ablations, stress tests, uncertainty, reproducible code, rebuilt PDF, and a public repository.

It does not satisfy the ICLR-main-ready bar because the claimed transformer mechanism is not decisively better than MLP energy or robust/nominal MPC, and the feasibility behavior is not strong enough.

Required revival work:

- larger and more diverse training data;
- clear transformer advantage over MLP energy;
- clear advantage over robust/nominal MPC without extra violations;
- public manipulation benchmark or robot validation;
- manual robotics related-work synthesis;
- stronger feasibility or constraint objective.
