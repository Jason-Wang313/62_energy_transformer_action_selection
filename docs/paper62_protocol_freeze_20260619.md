# Paper 62 Frozen Protocol

Date: 2026-06-19

Paper: `62_energy_transformer_action_selection`

Protocol status: frozen before terminal v5 run.

## Frozen Command

```powershell
python src\run_experiment.py --train-tasks 480 --epochs 24 --seeds 8 --episodes 12 --torch-threads 4
```

Default frozen splits:

- nominal
- low_friction
- high_friction
- light_object
- heavy_object
- obstacle_shift
- narrow_clearance
- actuation_noise
- far_target
- combined_shift

Default frozen ablation splits:

- combined_shift
- narrow_clearance

## Expected Evidence Size

- Candidate actions per task: 63.
- Training action sets: 480.
- Main evaluation episodes: 10 splits x 8 seeds x 12 episodes = 960 episodes.
- Main methods: 12.
- Expected main rows: 11,520.
- Ablation episodes: 2 splits x 8 seeds x 12 episodes = 192 episodes.
- Ablation methods: 11.
- Expected ablation rows: 2,112.

## Frozen Main Methods

- random_candidate
- geometric_greedy
- nominal_rollout_mpc
- robust_worst_case_mpc
- cvar_branch_score
- mlp_energy_scorer
- shielded_mlp_energy_scorer
- deep_sets_energy_scorer
- set_transformer_energy_scorer_v4
- rc_ret_no_shield
- rc_ret_energy_scorer_v5
- oracle_mujoco_rollout_selector

## Frozen Ablations

- rc_ret_energy_scorer_v5
- rc_ret_no_shield
- shielded_mlp_energy_scorer
- deep_sets_energy_scorer
- mlp_energy_scorer
- set_transformer_energy_scorer_v4
- rc_ret_no_risk_features
- rc_ret_no_cvar_loss
- rc_ret_no_feasibility
- rc_ret_small_data
- cvar_branch_score

## Frozen Gate

`ICLR_MAIN_TARGET_READY` requires all of:

- RC-RET beats or statistically ties while improving safety over MLP, shielded MLP, Deep Sets, nominal MPC, and robust MPC on aggregate paired comparisons.
- RC-RET does not increase violation rate against robust MPC on combined shift or narrow-clearance stress.
- Ablations show that set/risk/calibration machinery matters; a plain MLP, Deep Sets, or v4 transformer must not match or dominate the full method.
- The final paper is at least 25 pages, citation-clean, reproducible, and honest.

If these gates fail, the terminal decision is `STRONG_REVISE` or `KILL_ARCHIVE`; the manuscript must report the failure rather than optimize for pretty results.

