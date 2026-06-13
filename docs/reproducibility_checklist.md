# Reproducibility Checklist

## What Reproduces

- [x] `python src/run_experiment.py --train-tasks 360 --epochs 24 --seeds 5 --episodes 16 --torch-threads 4`
- [x] `results/energy_action_raw.csv`
- [x] `results/energy_action_metrics.csv`
- [x] `results/energy_action_seed_metrics.csv`
- [x] `results/energy_action_ablation.csv`
- [x] `results/energy_action_pairwise.csv`
- [x] `figures/energy_success_by_split.png`
- [x] `figures/energy_regret_by_split.png`
- [x] `figures/energy_ablation_regret.png`
- [x] `paper/main.tex`
- [x] Canonical PDF: `C:/Users/wangz/Downloads/62.pdf`

## What Does Not Yet Reproduce

- [ ] Real robot results.
- [ ] External public benchmark results.
- [ ] Large-scale transformer policy training.
- [ ] Manual full-paper related-work notes.
- [ ] Hardware feasibility/failure videos.

This is reproducible as a real MuJoCo/PyTorch strong-revise paper, not as an ICLR-main-ready robotics system.
