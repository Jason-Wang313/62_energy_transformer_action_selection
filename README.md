# 62 Risk-Calibrated Relational Energy Transformers

Submission-hardening version: v5 frozen RC-RET rebuild.

Terminal decision: STRONG_REVISE for ICLR main conference.

This repository contains a CPU-only MuJoCo/PyTorch action-set selection benchmark for contact-rich pushing. The v5 rebuild expands the older energy-transformer scaffold into a risk-calibrated relational energy transformer (RC-RET), stronger baselines, ten stress splits, full ablations, formal theory, and a 25-page ICLR-style manuscript with bright boxed clickable citations.

The result is honest: RC-RET is safer than several unshielded learned baselines and close to robust/CVaR branch scoring, but it does not clear the success gate against MLP, Deep Sets, robust MPC, or the oracle. Terminal state remains STRONG_REVISE, not ICLR-main-ready.

## Frozen Evidence Summary

- Training labels: 480 MuJoCo action sets, 63 candidates per set.
- Main evaluation: 11,520 rows.
- Ablation evaluation: 2,112 rows.
- Splits: nominal, low friction, high friction, light object, heavy object, obstacle shift, narrow clearance, actuation noise, far target, combined shift.
- Main methods: random, geometric, nominal MPC, robust MPC, branch-CVaR, MLP, shielded MLP, Deep Sets, old v4 transformer, RC-RET no-shield, RC-RET v5, oracle.
- Terminal gate: STRONG_REVISE because RC-RET does not beat MLP/Deep Sets/robust MPC on the frozen success gate.
- Canonical PDF: `C:/Users/wangz/Downloads/62.pdf`.
- Desktop PDF: intentionally absent.

## Reproduce Final Run

```powershell
python src\run_experiment.py --train-tasks 480 --epochs 24 --seeds 8 --episodes 12 --torch-threads 4
```

## Build And Validate PDF

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
python scripts\validate_submission_artifacts.py
```

GitHub: https://github.com/Jason-Wang313/62_energy_transformer_action_selection
