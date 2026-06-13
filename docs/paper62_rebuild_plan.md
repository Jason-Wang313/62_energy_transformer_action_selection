# Paper 62 Rebuild Plan

Paper: Energy Transformer Action Selection.

Goal: rebuild the v3 archive into a real ICLR-main-target evidence package, or terminate honestly as STRONG_REVISE / KILL_ARCHIVE.

## Salvageable Thesis

Replace direct next-action decoding with energy-ranked feasible action manifolds. The core claim is only interesting if a learned energy scorer improves action selection under physical constraints and shift, not merely if it rephrases MPC or uncertainty ranking.

## Real Evidence Target

Build a MuJoCo contact-pushing action-selection benchmark:

- State: puck pose, target pose, obstacle location, candidate push primitive set.
- Hidden dynamics: object mass, surface friction, actuation noise.
- Candidate actions: parameterized push primitives and short primitive sequences.
- Label/evaluation: MuJoCo rollout final distance, obstacle/contact violation, effort, and success within target tolerance.
- Splits: nominal, low friction, high friction, heavy object, obstacle shift, combined dynamics/obstacle shift.

## Proposed Method

Train a lightweight PyTorch energy scorer over candidate action sets:

- Candidate token features: action angle, offset, push distance, nominal geometric progress, estimated effort, state-to-target geometry, obstacle geometry.
- Set/transformer module: small self-attention encoder over candidate actions so scores are context-dependent within the feasible action manifold.
- Output: scalar energy for each candidate; select the lowest-energy feasible action.
- Loss: supervised ranking/regression against MuJoCo rollout energy with pairwise ranking loss and feasibility penalty.

Keep RAM light:

- CPU-friendly model, small hidden width, no large image tensors.
- Generate compact tabular MuJoCo rollout datasets.
- Checkpoint results after every split/seed.
- Use worker count <= 4 unless runtime forces otherwise.

## Baselines

- random_candidate
- geometric_greedy
- nominal_rollout_mpc
- robust_worst_case_mpc
- mlp_energy_scorer
- transformer_energy_scorer (proposed)
- oracle_mujoco_rollout_selector

## Ablations

- no_self_attention_mlp
- no_feasibility_penalty
- no_pairwise_ranking_loss
- no_obstacle_features
- small_data_transformer
- top1_geometric_then_energy_filter

## Metrics

- Success rate with 95 percent CI.
- Final distance to target.
- Obstacle/contact violation rate.
- Energy regret versus oracle.
- Feasibility rate.
- Paired deltas against MLP, nominal MPC, and robust MPC.

## Terminal Criteria

ICLR_MAIN_TARGET_READY requires:

- Real MuJoCo data generation and held-out evaluation.
- Multiple seeds and stress splits.
- Proposed transformer energy scorer clearly beats MLP energy, nominal MPC, and robust MPC on success/regret without raising violation rate.
- Ablations isolate self-attention/ranking/feasibility as necessary.
- Manual hostile prior-work synthesis supports novelty.

STRONG_REVISE if:

- The benchmark and learned scorer are real and reproducible, but gains are mixed or custom-only.

KILL_ARCHIVE if:

- The learned method fails to beat MLP/robust baselines, cannot train reliably, or the evidence remains effectively synthetic.
