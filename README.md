# 62 Energy Transformer Action Selection

Submission-hardening version: v4 real MuJoCo/PyTorch rebuild.

Terminal decision: STRONG_REVISE for ICLR main conference.

This repository now contains a real MuJoCo action-selection benchmark and a lightweight PyTorch energy-scorer training/evaluation pipeline. The result is not ICLR-main ready: the transformer energy scorer improves over random and geometric selection, but it does not consistently beat MLP energy, nominal rollout MPC, or robust worst-case MPC.

## Evidence Summary

- MuJoCo training labels: 360 action-set tasks.
- Main evaluation: 5 seeds, 16 episodes per seed/split/method.
- Splits: nominal, low friction, high friction, heavy object, obstacle shift, combined shift.
- Baselines: random candidate, geometric greedy, nominal rollout MPC, robust worst-case MPC, MLP energy scorer, oracle MuJoCo rollout selector.
- Ablations: no feasibility objective, no obstacle features, top-3 geometric filter, small-data transformer, MLP/no self-attention.
- Terminal state: strong revise, not submission ready.

## Reproduce

```powershell
python src\run_experiment.py --train-tasks 360 --epochs 24 --seeds 5 --episodes 16 --torch-threads 4
```

## Build PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/62.pdf`

GitHub: https://github.com/Jason-Wang313/62_energy_transformer_action_selection
