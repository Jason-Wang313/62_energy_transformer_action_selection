# Reproducibility Checklist

## What Reproduces

- [x] `python src\run_experiment.py --train-tasks 480 --epochs 24 --seeds 8 --episodes 12 --torch-threads 4`
- [x] `results/energy_action_raw.csv`
- [x] `results/energy_action_ablation_raw.csv`
- [x] `results/energy_action_metrics.csv`
- [x] `results/energy_action_seed_metrics.csv`
- [x] `results/energy_action_ablation.csv`
- [x] `results/energy_action_pairwise.csv`
- [x] `figures/energy_success_by_split.png`
- [x] `figures/energy_regret_by_split.png`
- [x] `figures/energy_violation_by_split.png`
- [x] `figures/energy_ablation_regret.png`
- [x] `paper/results_tables.tex`
- [x] `paper/appendix_results_tables.tex`
- [x] `paper/main.tex`
- [x] Canonical PDF: `C:/Users/wangz/Downloads/62.pdf`

## What Does Not Yet Reproduce

- [ ] Real robot results.
- [ ] External public benchmark results.
- [ ] Online latent/contact inference.
- [ ] Rich multi-stage primitive generation.
- [ ] Hardware feasibility/failure videos.

This is reproducible as a real MuJoCo/PyTorch strong-revise paper, not as an ICLR-main-ready robotics system.
