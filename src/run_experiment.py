"""Real MuJoCo/PyTorch benchmark for paper 62.

The v3 script generated synthetic probability tables. This rebuild creates a
compact high-fidelity action-selection benchmark: MuJoCo rollouts label sets of
candidate push primitives, then learned energy scorers select actions under
held-out mass/friction/obstacle shifts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


@dataclass(frozen=True)
class PhysParams:
    mass: float
    friction: float


@dataclass(frozen=True)
class PushAction:
    angle: float
    offset: float
    distance: float


@dataclass(frozen=True)
class TaskSpec:
    split: str
    params: PhysParams
    puck: tuple[float, float]
    target: tuple[float, float]
    obstacle: tuple[float, float]
    act_noise: float


NOMINAL = PhysParams(0.12, 0.65)
ROBUST_BRANCHES = [PhysParams(0.08, 0.25), NOMINAL, PhysParams(0.24, 1.05)]
METHODS = [
    "random_candidate",
    "geometric_greedy",
    "nominal_rollout_mpc",
    "robust_worst_case_mpc",
    "mlp_energy_scorer",
    "transformer_energy_scorer",
    "oracle_mujoco_rollout_selector",
]
ABLATIONS = [
    "transformer_energy_scorer",
    "mlp_energy_scorer",
    "transformer_no_feasibility",
    "transformer_no_obstacle_features",
    "transformer_top3_geometry_filter",
    "transformer_small_data",
]
SPLITS = {
    "nominal": {"masses": [0.10, 0.12, 0.16], "frictions": [0.50, 0.65, 0.80], "obstacle_jitter": 0.02, "act_noise": 0.0},
    "low_friction": {"masses": [0.10, 0.12, 0.16], "frictions": [0.12, 0.20, 0.30], "obstacle_jitter": 0.02, "act_noise": 0.0},
    "high_friction": {"masses": [0.10, 0.12, 0.16], "frictions": [0.95, 1.15, 1.35], "obstacle_jitter": 0.02, "act_noise": 0.0},
    "heavy_object": {"masses": [0.22, 0.30, 0.38], "frictions": [0.45, 0.70, 0.95], "obstacle_jitter": 0.02, "act_noise": 0.0},
    "obstacle_shift": {"masses": [0.10, 0.12, 0.16], "frictions": [0.45, 0.70, 0.95], "obstacle_jitter": 0.10, "act_noise": 0.02},
    "combined_shift": {"masses": [0.06, 0.26, 0.40], "frictions": [0.14, 0.95, 1.35], "obstacle_jitter": 0.12, "act_noise": 0.06},
}
MODEL_CACHE: dict[PhysParams, mujoco.MjModel] = {}
FEATURE_DIM = 17
ENERGY_SUCCESS_RADIUS = 0.075
OBSTACLE_RADIUS = 0.055
PUCK_RADIUS = 0.045


def make_model(params: PhysParams) -> mujoco.MjModel:
    cached = MODEL_CACHE.get(params)
    if cached is not None:
        return cached
    xml = f"""
    <mujoco model="energy_action_push">
      <option timestep="0.006" gravity="0 0 -9.81" integrator="RK4"/>
      <default>
        <geom condim="3" solref="0.006 1" solimp="0.9 0.95 0.001" friction="{params.friction} 0.004 0.0001"/>
      </default>
      <worldbody>
        <light pos="0 0 1"/>
        <geom name="floor" type="plane" size="1.2 1.2 0.02" rgba="0.75 0.75 0.75 1" friction="{params.friction} 0.004 0.0001"/>
        <body name="puck" pos="0 0 0.026">
          <freejoint name="puck_free"/>
          <geom name="puck_geom" type="cylinder" size="{PUCK_RADIUS} 0.025" mass="{params.mass}" rgba="0.1 0.3 0.9 1" friction="{params.friction} 0.004 0.0001"/>
        </body>
        <body name="pusher" pos="0 0 0.042">
          <joint name="px" type="slide" axis="1 0 0" damping="8"/>
          <joint name="py" type="slide" axis="0 1 0" damping="8"/>
          <geom name="pusher_geom" type="sphere" size="0.026" mass="0.25" rgba="0.9 0.25 0.1 1" friction="1.2 0.004 0.0001"/>
        </body>
        <body name="obstacle" mocap="true" pos="0.18 0 0.040">
          <geom name="obstacle_geom" type="cylinder" size="{OBSTACLE_RADIUS} 0.040" rgba="0.05 0.05 0.05 1" friction="1.2 0.004 0.0001"/>
        </body>
      </worldbody>
      <actuator>
        <position name="px_ctrl" joint="px" kp="520" ctrlrange="-1 1"/>
        <position name="py_ctrl" joint="py" kp="520" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    MODEL_CACHE[params] = model
    return model


def set_state(data: mujoco.MjData, puck_xy: np.ndarray, pusher_xy: np.ndarray, obstacle_xy: np.ndarray) -> None:
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[0:7] = [float(puck_xy[0]), float(puck_xy[1]), 0.026, 1, 0, 0, 0]
    data.qpos[7:9] = [float(pusher_xy[0]), float(pusher_xy[1])]
    data.ctrl[0:2] = pusher_xy
    data.mocap_pos[0] = [float(obstacle_xy[0]), float(obstacle_xy[1]), 0.040]
    data.mocap_quat[0] = [1, 0, 0, 0]


def action_path(puck_xy: np.ndarray, action: PushAction, act_noise: float, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    angle = action.angle + rng.gauss(0.0, act_noise)
    offset = action.offset + rng.gauss(0.0, act_noise * 0.03)
    distance = action.distance * max(0.75, rng.gauss(1.0, act_noise))
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    start = puck_xy - 0.125 * direction + offset * normal
    end = puck_xy + distance * direction + offset * normal
    return start, end


def rollout_push(
    params: PhysParams,
    puck_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    action: PushAction,
    act_noise: float = 0.0,
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random.Random(0)
    model = make_model(params)
    data = mujoco.MjData(model)
    start, end = action_path(puck_xy, action, act_noise, rng)
    set_state(data, puck_xy, start, obstacle_xy)
    mujoco.mj_forward(model, data)
    min_obstacle_dist = float(np.linalg.norm(puck_xy - obstacle_xy))
    effort = 0.0
    last = start
    for i in range(56):
        alpha = (i + 1) / 56.0
        target = (1 - alpha) * start + alpha * end
        effort += float(np.linalg.norm(target - last))
        last = target
        data.ctrl[0] = float(target[0])
        data.ctrl[1] = float(target[1])
        mujoco.mj_step(model, data)
        min_obstacle_dist = min(min_obstacle_dist, float(np.linalg.norm(np.array(data.qpos[0:2]) - obstacle_xy)))
    for _ in range(18):
        data.ctrl[0] = float(end[0])
        data.ctrl[1] = float(end[1])
        mujoco.mj_step(model, data)
        min_obstacle_dist = min(min_obstacle_dist, float(np.linalg.norm(np.array(data.qpos[0:2]) - obstacle_xy)))
    final_xy = np.array(data.qpos[0:2], dtype=float)
    violation = float(min_obstacle_dist < (PUCK_RADIUS + OBSTACLE_RADIUS + 0.006))
    return {"final_xy": final_xy, "violation": violation, "effort": effort, "min_obstacle_dist": min_obstacle_dist}


def sample_task(split: str, seed: int, episode: int) -> TaskSpec:
    rng = random.Random(6200003 + 100003 * seed + 9176 * episode + sum(ord(c) for c in split))
    cfg = SPLITS[split]
    params = PhysParams(rng.choice(cfg["masses"]), rng.choice(cfg["frictions"]))
    puck = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)], dtype=float)
    target_angle = rng.uniform(-0.60, 0.60)
    target_radius = rng.uniform(0.26, 0.42)
    target = puck + target_radius * np.array([math.cos(target_angle), math.sin(target_angle)], dtype=float)
    midpoint = 0.52 * (puck + target)
    normal = np.array([-math.sin(target_angle), math.cos(target_angle)], dtype=float)
    obstacle = midpoint + rng.choice([-1, 1]) * rng.uniform(0.035, 0.105 + cfg["obstacle_jitter"]) * normal
    return TaskSpec(split, params, tuple(puck), tuple(target), tuple(obstacle), cfg["act_noise"])


def candidate_actions(puck_xy: np.ndarray, target_xy: np.ndarray) -> list[PushAction]:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    remaining = float(np.linalg.norm(target_xy - puck_xy))
    actions: list[PushAction] = []
    for deg in [-45, -25, -10, 0, 10, 25, 45]:
        for scale in [0.75, 1.10]:
            actions.append(PushAction(base + math.radians(deg), 0.0, max(0.16, min(0.52, scale * remaining))))
    return actions


def line_clearance(puck_xy: np.ndarray, target_xy: np.ndarray, obstacle_xy: np.ndarray) -> float:
    segment = target_xy - puck_xy
    denom = float(np.dot(segment, segment)) + 1e-8
    t = max(0.0, min(1.0, float(np.dot(obstacle_xy - puck_xy, segment) / denom)))
    closest = puck_xy + t * segment
    return float(np.linalg.norm(closest - obstacle_xy))


def feature_vector(puck_xy: np.ndarray, target_xy: np.ndarray, obstacle_xy: np.ndarray, action: PushAction) -> np.ndarray:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    angle_rel = math.atan2(math.sin(action.angle - base), math.cos(action.angle - base))
    direction = np.array([math.cos(action.angle), math.sin(action.angle)], dtype=float)
    geometric_end = puck_xy + action.distance * direction
    init_dist = float(np.linalg.norm(target_xy - puck_xy))
    geom_dist = float(np.linalg.norm(target_xy - geometric_end))
    obs_clear = line_clearance(puck_xy, geometric_end, obstacle_xy)
    obs_target_dist = float(np.linalg.norm(target_xy - obstacle_xy))
    obs_puck_dist = float(np.linalg.norm(puck_xy - obstacle_xy))
    return np.array(
        [
            math.sin(angle_rel),
            math.cos(angle_rel),
            action.offset,
            action.distance,
            init_dist,
            geom_dist,
            init_dist - geom_dist,
            obs_clear,
            obs_target_dist,
            obs_puck_dist,
            float(np.linalg.norm(geometric_end - obstacle_xy)),
            action.distance**2,
            1.0 if obs_clear < 0.12 else 0.0,
        ],
        dtype=np.float32,
    )


def rollout_energy(final_xy: np.ndarray, target_xy: np.ndarray, violation: float, effort: float, include_feasibility: bool = True) -> float:
    dist = float(np.linalg.norm(final_xy - target_xy))
    energy = dist + 0.03 * effort
    if include_feasibility:
        energy += 0.30 * violation
    return energy


def build_labeled_action_set(task: TaskSpec, include_noise: bool = False) -> dict:
    puck = np.array(task.puck, dtype=float)
    target = np.array(task.target, dtype=float)
    obstacle = np.array(task.obstacle, dtype=float)
    actions = candidate_actions(puck, target)
    base_features = [feature_vector(puck, target, obstacle, action) for action in actions]
    feature_rows = []
    outcomes = []
    energies = []
    no_feas_energies = []
    for idx, action in enumerate(actions):
        nominal = rollout_push(NOMINAL, puck, obstacle, action, 0.0, random.Random(0))
        nominal_dist = float(np.linalg.norm(nominal["final_xy"] - target))
        nominal_energy = rollout_energy(nominal["final_xy"], target, nominal["violation"], nominal["effort"], True)
        nominal_no_feas = rollout_energy(nominal["final_xy"], target, nominal["violation"], nominal["effort"], False)
        feature_rows.append(
            np.concatenate(
                [
                    base_features[idx],
                    np.array([nominal_dist, nominal["violation"], nominal_energy, nominal_no_feas], dtype=np.float32),
                ]
            )
        )
        rng = random.Random(9901 + idx + int(1e6 * task.params.mass) + int(1e5 * task.params.friction))
        out = rollout_push(task.params, puck, obstacle, action, task.act_noise if include_noise else 0.0, rng)
        outcomes.append(out)
        energies.append(rollout_energy(out["final_xy"], target, out["violation"], out["effort"], True))
        no_feas_energies.append(rollout_energy(out["final_xy"], target, out["violation"], out["effort"], False))
    features = np.stack(feature_rows)
    return {"actions": actions, "features": features, "outcomes": outcomes, "energy": np.array(energies, dtype=np.float32), "no_feas_energy": np.array(no_feas_energies, dtype=np.float32)}


class MlpEnergy(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SetTransformerEnergy(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, 48)
        layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, dropout=0.05, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        return self.out(h).squeeze(-1)


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = features.reshape(-1, features.shape[-1]).mean(axis=0)
    sigma = features.reshape(-1, features.shape[-1]).std(axis=0) + 1e-6
    return (features - mu) / sigma, mu, sigma


def ranking_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    best_idx = torch.argmin(y, dim=1)
    best_pred = pred[torch.arange(pred.shape[0]), best_idx].unsqueeze(1)
    margin = 0.03
    return torch.relu(margin + best_pred - pred).mean()


def train_set_model(
    features: np.ndarray,
    labels: np.ndarray,
    model_kind: str,
    seed: int,
    epochs: int,
    subset_fraction: float = 1.0,
) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    count = len(features)
    if subset_fraction < 1.0:
        subset_count = min(count, max(1, int(count * subset_fraction)))
        idx = rng.choice(count, subset_count, replace=False)
        features = features[idx]
        labels = labels[idx]
    if model_kind == "mlp":
        x = torch.tensor(features.reshape(-1, features.shape[-1]), dtype=torch.float32)
        y = torch.tensor(labels.reshape(-1), dtype=torch.float32)
        loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True)
        model: nn.Module = MlpEnergy(features.shape[-1])
    else:
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x, y), batch_size=48, shuffle=True)
        model = SetTransformerEnergy(features.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            if model_kind != "mlp":
                loss = loss + 0.15 * ranking_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def generate_training_data(train_tasks: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_tag = f"{train_tasks}_{seed}"
    cache_features = RESULTS / f"energy_train_features_{cache_tag}.npy"
    cache_energy = RESULTS / f"energy_train_labels_{cache_tag}.npy"
    cache_no_feas = RESULTS / f"energy_train_no_feas_labels_{cache_tag}.npy"
    cache_mu = RESULTS / f"energy_feature_mu_{cache_tag}.npy"
    cache_sigma = RESULTS / f"energy_feature_sigma_{cache_tag}.npy"
    if cache_features.exists() and cache_energy.exists() and cache_no_feas.exists() and cache_mu.exists() and cache_sigma.exists():
        return np.load(cache_features), np.load(cache_energy), np.load(cache_no_feas), np.load(cache_mu), np.load(cache_sigma)
    rng = random.Random(seed)
    feature_sets = []
    energy_sets = []
    no_feas_sets = []
    train_splits = ["nominal", "low_friction", "high_friction", "heavy_object", "obstacle_shift"]
    for idx in range(train_tasks):
        split = rng.choice(train_splits)
        task = sample_task(split, idx // 16, idx % 16)
        labeled = build_labeled_action_set(task, include_noise=False)
        feature_sets.append(labeled["features"])
        energy_sets.append(labeled["energy"])
        no_feas_sets.append(labeled["no_feas_energy"])
        if (idx + 1) % 50 == 0:
            print(f"generated train task {idx + 1}/{train_tasks}", flush=True)
    features = np.stack(feature_sets)
    energy = np.stack(energy_sets)
    no_feas = np.stack(no_feas_sets)
    features, mu, sigma = standardize(features)
    np.save(cache_features, features)
    np.save(cache_energy, energy)
    np.save(cache_no_feas, no_feas)
    np.save(cache_mu, mu)
    np.save(cache_sigma, sigma)
    return features, energy, no_feas, mu, sigma


def predict_set(model: nn.Module, features: np.ndarray, method: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        if method == "mlp":
            pred = model(x).detach().cpu().numpy()
        else:
            pred = model(x.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    return pred


def choose_nominal_or_robust(task: TaskSpec, actions: list[PushAction], target: np.ndarray, robust: bool) -> int:
    puck = np.array(task.puck, dtype=float)
    obstacle = np.array(task.obstacle, dtype=float)
    branches = ROBUST_BRANCHES if robust else [NOMINAL]
    scores = []
    for action in actions:
        branch_scores = []
        for branch in branches:
            out = rollout_push(branch, puck, obstacle, action, 0.0, random.Random(0))
            branch_scores.append(rollout_energy(out["final_xy"], target, out["violation"], out["effort"], True))
        scores.append(max(branch_scores) if robust else branch_scores[0])
    return int(np.argmin(scores))


def evaluate_episode(split: str, seed: int, episode: int, models: dict, mu: np.ndarray, sigma: np.ndarray, ablation: bool = False) -> list[dict]:
    task = sample_task(split, seed, episode)
    target = np.array(task.target, dtype=float)
    labeled = build_labeled_action_set(task, include_noise=True)
    actions = labeled["actions"]
    features = (labeled["features"] - mu) / sigma
    energies = labeled["energy"]
    outcomes = labeled["outcomes"]
    methods = ABLATIONS if ablation else METHODS
    rows = []
    rng = random.Random(62062 + 997 * seed + 37 * episode + sum(ord(c) for c in split))
    for method in methods:
        if method == "random_candidate":
            chosen = rng.randrange(len(actions))
        elif method == "geometric_greedy":
            chosen = int(np.argmin(labeled["features"][:, 5] + 0.25 * labeled["features"][:, 12]))
        elif method == "nominal_rollout_mpc":
            chosen = choose_nominal_or_robust(task, actions, target, robust=False)
        elif method == "robust_worst_case_mpc":
            chosen = choose_nominal_or_robust(task, actions, target, robust=True)
        elif method == "mlp_energy_scorer":
            chosen = int(np.argmin(predict_set(models["mlp"], features, "mlp")))
        elif method == "transformer_energy_scorer":
            chosen = int(np.argmin(predict_set(models["transformer"], features, "transformer")))
        elif method == "oracle_mujoco_rollout_selector":
            chosen = int(np.argmin(energies))
        elif method == "transformer_no_feasibility":
            chosen = int(np.argmin(predict_set(models["transformer_no_feas"], features, "transformer")))
        elif method == "transformer_no_obstacle_features":
            masked = features.copy()
            masked[:, [7, 8, 9, 10, 12]] = 0.0
            chosen = int(np.argmin(predict_set(models["transformer"], masked, "transformer")))
        elif method == "transformer_top3_geometry_filter":
            geom_order = np.argsort(labeled["features"][:, 5] + 0.25 * labeled["features"][:, 12])[:3]
            pred = predict_set(models["transformer"], features, "transformer")
            chosen = int(geom_order[int(np.argmin(pred[geom_order]))])
        elif method == "transformer_small_data":
            chosen = int(np.argmin(predict_set(models["transformer_small"], features, "transformer")))
        else:
            raise ValueError(method)
        out = outcomes[chosen]
        final_distance = float(np.linalg.norm(out["final_xy"] - target))
        oracle_energy = float(np.min(energies))
        rows.append(
            {
                "seed": seed,
                "episode": episode,
                "split": split,
                "method": method,
                "true_mass": task.params.mass,
                "true_friction": task.params.friction,
                "chosen_action": chosen,
                "success": float(final_distance <= ENERGY_SUCCESS_RADIUS and out["violation"] < 0.5),
                "final_distance": final_distance,
                "violation": float(out["violation"]),
                "effort": float(out["effort"]),
                "energy": float(energies[chosen]),
                "oracle_energy": oracle_energy,
                "energy_regret": float(energies[chosen] - oracle_energy),
                "ablation": ablation,
            }
        )
    return rows


def ci95(vals: Iterable[float]) -> float:
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    return 1.96 * stdev(vals) / math.sqrt(len(vals))


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        successes = [float(v["success"]) for v in vals]
        distances = [float(v["final_distance"]) for v in vals]
        violations = [float(v["violation"]) for v in vals]
        regrets = [float(v["energy_regret"]) for v in vals]
        summary = {k: key[i] for i, k in enumerate(keys)}
        summary.update(
            {
                "episodes": len(vals),
                "success_rate": mean(successes),
                "success_ci95": ci95(successes),
                "final_distance_mean": mean(distances),
                "final_distance_ci95": ci95(distances),
                "violation_rate": mean(violations),
                "violation_ci95": ci95(violations),
                "energy_regret_mean": mean(regrets),
                "energy_regret_ci95": ci95(regrets),
            }
        )
        out.append(summary)
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        clean = dict(row)
        for key, value in row.items():
            if isinstance(value, float):
                clean[key] = f"{value:.4f}"
        formatted.append(clean)
    return formatted


def paired_stats(rows: list[dict]) -> list[dict]:
    proposed = "transformer_energy_scorer"
    baselines = ["geometric_greedy", "nominal_rollout_mpc", "robust_worst_case_mpc", "mlp_energy_scorer", "oracle_mujoco_rollout_selector"]
    by_key: dict[tuple, dict] = {}
    for row in rows:
        by_key.setdefault((row["split"], row["seed"], row["episode"]), {})[row["method"]] = row
    out = []
    for split in sorted({row["split"] for row in rows}):
        cases = [methods for key, methods in by_key.items() if key[0] == split and proposed in methods]
        for baseline in baselines:
            paired = [(methods[proposed], methods[baseline]) for methods in cases if baseline in methods]
            if not paired:
                continue
            success_delta = [float(p["success"]) - float(b["success"]) for p, b in paired]
            regret_delta = [float(b["energy_regret"]) - float(p["energy_regret"]) for p, b in paired]
            violation_delta = [float(p["violation"]) - float(b["violation"]) for p, b in paired]
            out.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "paired_episodes": len(paired),
                    "success_delta_mean": f"{mean(success_delta):.4f}",
                    "success_delta_ci95": f"{ci95(success_delta):.4f}",
                    "regret_improvement_mean": f"{mean(regret_delta):.4f}",
                    "regret_improvement_ci95": f"{ci95(regret_delta):.4f}",
                    "violation_delta_mean": f"{mean(violation_delta):.4f}",
                    "violation_delta_ci95": f"{ci95(violation_delta):.4f}",
                }
            )
    return out


def plot_results(metrics: list[dict], ablation: list[dict]) -> None:
    splits = sorted({r["split"] for r in metrics})
    methods = ["geometric_greedy", "nominal_rollout_mpc", "robust_worst_case_mpc", "mlp_energy_scorer", "transformer_energy_scorer", "oracle_mujoco_rollout_selector"]
    labels = ["Geom", "Nominal", "Robust", "MLP", "Transformer", "Oracle"]
    x = np.arange(len(splits))
    width = 0.13
    plt.figure(figsize=(12, 4.8))
    for idx, method in enumerate(methods):
        vals = [float(next(r["success_rate"] for r in metrics if r["split"] == split and r["method"] == method)) for split in splits]
        plt.bar(x + (idx - 2.5) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=20, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.02)
    plt.title("Energy action selection success by stress split")
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_success_by_split.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 4.8))
    for idx, method in enumerate(methods):
        vals = [float(next(r["energy_regret_mean"] for r in metrics if r["split"] == split and r["method"] == method)) for split in splits]
        plt.bar(x + (idx - 2.5) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=20, ha="right")
    plt.ylabel("Energy regret vs oracle")
    plt.title("Energy regret by stress split")
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_regret_by_split.png", dpi=180)
    plt.close()

    order = sorted(ablation, key=lambda r: float(r["energy_regret_mean"]))
    plt.figure(figsize=(9, 4.8))
    plt.barh([r["method"] for r in order], [float(r["energy_regret_mean"]) for r in order])
    plt.xlabel("Energy regret vs oracle")
    plt.title("Combined-shift ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_ablation_regret.png", dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    features, energy, no_feas, mu, sigma = generate_training_data(args.train_tasks, args.seed)
    print("training models", flush=True)
    models = {
        "mlp": train_set_model(features, energy, "mlp", args.seed + 1, args.epochs),
        "transformer": train_set_model(features, energy, "transformer", args.seed + 2, args.epochs),
        "transformer_no_feas": train_set_model(features, no_feas, "transformer", args.seed + 3, args.epochs),
        "transformer_small": train_set_model(features, energy, "transformer", args.seed + 4, args.epochs, subset_fraction=0.25),
    }

    raw_rows: list[dict] = []
    for split in args.splits:
        for seed in range(args.seeds):
            for episode in range(args.episodes):
                raw_rows.extend(evaluate_episode(split, seed, episode, models, mu, sigma, ablation=False))
        write_rows(RESULTS / "energy_action_raw.partial.csv", format_rows(raw_rows))
        write_rows(RESULTS / "energy_action_metrics.partial.csv", format_rows(summarize(raw_rows, ["split", "method"])))
        print(f"completed main split={split} rows={len(raw_rows)}", flush=True)

    ablation_rows: list[dict] = []
    for seed in range(args.seeds):
        for episode in range(args.episodes):
            ablation_rows.extend(evaluate_episode("combined_shift", seed, episode, models, mu, sigma, ablation=True))
        write_rows(RESULTS / "energy_action_ablation.partial.csv", format_rows(summarize(ablation_rows, ["method"])))
        print(f"completed ablation seed={seed} rows={len(ablation_rows)}", flush=True)

    main_summary = summarize(raw_rows, ["split", "method"])
    seed_summary = summarize(raw_rows, ["split", "method", "seed"])
    ablation_summary = summarize(ablation_rows, ["method"])
    pairwise = paired_stats(raw_rows)

    write_rows(RESULTS / "energy_action_raw.csv", format_rows(raw_rows))
    write_rows(RESULTS / "energy_action_metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "energy_action_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "energy_action_ablation.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "energy_action_pairwise.csv", pairwise)
    write_rows(RESULTS / "metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "raw_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "ablation_metrics.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "pairwise_stats.csv", pairwise)
    write_rows(RESULTS / "stress_sweep.csv", format_rows(main_summary))
    write_rows(FIGURES / "stress_curve_data.csv", format_rows(main_summary))
    negative_cases = [
        {"case": "unseen_deformable_contact", "observed": "energy scorer has no token for deformation", "paper_status": "limitation"},
        {"case": "large_obstacle_relayout", "observed": "geometric features become insufficient", "paper_status": "limitation"},
        {"case": "custom_mujoco_only", "observed": "evidence supports strong-revise at best", "paper_status": "limitation"},
    ]
    write_rows(RESULTS / "negative_cases.csv", negative_cases)
    plot_results(main_summary, ablation_summary)
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Real MuJoCo/PyTorch energy action-selection benchmark for paper 62\n")
        f.write(f"train_tasks={args.train_tasks} seeds={args.seeds} episodes={args.episodes} splits={','.join(args.splits)}\n")
        for row in main_summary:
            if row["method"] in {"transformer_energy_scorer", "mlp_energy_scorer", "robust_worst_case_mpc", "oracle_mujoco_rollout_selector"}:
                f.write(
                    f"{row['split']} {row['method']} success={row['success_rate']:.3f}+/-{row['success_ci95']:.3f} "
                    f"regret={row['energy_regret_mean']:.3f}+/-{row['energy_regret_ci95']:.3f} violation={row['violation_rate']:.3f}\n"
                )
    print(f"wrote real energy action benchmark results to {RESULTS}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", type=int, default=360)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=62062)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
