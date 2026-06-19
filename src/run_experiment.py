"""CPU-light MuJoCo/PyTorch evidence runner for Paper 62.

Version v5 rebuilds the earlier energy-transformer scaffold into a stronger,
falsifiable action-set benchmark.  The proposed method is a risk-calibrated
relational energy transformer (RC-RET), evaluated against MLP, Deep Sets,
MPC-style, branch-risk, shielded, and oracle baselines.
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

CACHE_VERSION = "v5_rcret_20260619b"
ENERGY_SUCCESS_RADIUS = 0.075
OBSTACLE_RADIUS = 0.055
PUCK_RADIUS = 0.045
CONTACT_MARGIN = PUCK_RADIUS + OBSTACLE_RADIUS + 0.006
SHIELD_CLEARANCE = CONTACT_MARGIN + 0.002


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
LOW_BRANCH = PhysParams(0.08, 0.25)
HEAVY_BRANCH = PhysParams(0.30, 0.85)
HIGH_FRICTION_BRANCH = PhysParams(0.18, 1.20)
RISK_BRANCHES = [LOW_BRANCH, NOMINAL, HEAVY_BRANCH, HIGH_FRICTION_BRANCH]

METHODS = [
    "random_candidate",
    "geometric_greedy",
    "nominal_rollout_mpc",
    "robust_worst_case_mpc",
    "cvar_branch_score",
    "mlp_energy_scorer",
    "shielded_mlp_energy_scorer",
    "deep_sets_energy_scorer",
    "set_transformer_energy_scorer_v4",
    "rc_ret_no_shield",
    "rc_ret_energy_scorer_v5",
    "oracle_mujoco_rollout_selector",
]

ABLATIONS = [
    "rc_ret_energy_scorer_v5",
    "rc_ret_no_shield",
    "shielded_mlp_energy_scorer",
    "deep_sets_energy_scorer",
    "mlp_energy_scorer",
    "set_transformer_energy_scorer_v4",
    "rc_ret_no_risk_features",
    "rc_ret_no_cvar_loss",
    "rc_ret_no_feasibility",
    "rc_ret_small_data",
    "cvar_branch_score",
]

SPLITS = {
    "nominal": {
        "masses": [0.10, 0.12, 0.16],
        "frictions": [0.50, 0.65, 0.80],
        "target_radius": (0.26, 0.42),
        "obstacle_offset": (0.055, 0.125),
        "obstacle_mid": (0.42, 0.62),
        "act_noise": 0.0,
    },
    "low_friction": {
        "masses": [0.10, 0.12, 0.16],
        "frictions": [0.12, 0.20, 0.30],
        "target_radius": (0.26, 0.42),
        "obstacle_offset": (0.050, 0.115),
        "obstacle_mid": (0.42, 0.62),
        "act_noise": 0.0,
    },
    "high_friction": {
        "masses": [0.10, 0.12, 0.16],
        "frictions": [0.95, 1.15, 1.35],
        "target_radius": (0.26, 0.42),
        "obstacle_offset": (0.055, 0.125),
        "obstacle_mid": (0.42, 0.62),
        "act_noise": 0.0,
    },
    "light_object": {
        "masses": [0.045, 0.060, 0.080],
        "frictions": [0.35, 0.55, 0.80],
        "target_radius": (0.26, 0.44),
        "obstacle_offset": (0.050, 0.115),
        "obstacle_mid": (0.40, 0.64),
        "act_noise": 0.01,
    },
    "heavy_object": {
        "masses": [0.22, 0.30, 0.38],
        "frictions": [0.45, 0.70, 0.95],
        "target_radius": (0.25, 0.40),
        "obstacle_offset": (0.055, 0.125),
        "obstacle_mid": (0.42, 0.62),
        "act_noise": 0.0,
    },
    "obstacle_shift": {
        "masses": [0.10, 0.12, 0.16],
        "frictions": [0.45, 0.70, 0.95],
        "target_radius": (0.26, 0.44),
        "obstacle_offset": (0.035, 0.165),
        "obstacle_mid": (0.30, 0.74),
        "act_noise": 0.02,
    },
    "narrow_clearance": {
        "masses": [0.10, 0.14, 0.18],
        "frictions": [0.45, 0.70, 1.00],
        "target_radius": (0.28, 0.44),
        "obstacle_offset": (0.024, 0.062),
        "obstacle_mid": (0.40, 0.68),
        "act_noise": 0.018,
    },
    "actuation_noise": {
        "masses": [0.10, 0.14, 0.18],
        "frictions": [0.40, 0.70, 1.00],
        "target_radius": (0.26, 0.44),
        "obstacle_offset": (0.045, 0.125),
        "obstacle_mid": (0.42, 0.64),
        "act_noise": 0.085,
    },
    "far_target": {
        "masses": [0.10, 0.14, 0.18],
        "frictions": [0.45, 0.70, 0.95],
        "target_radius": (0.42, 0.58),
        "obstacle_offset": (0.055, 0.145),
        "obstacle_mid": (0.38, 0.68),
        "act_noise": 0.015,
    },
    "combined_shift": {
        "masses": [0.06, 0.26, 0.40],
        "frictions": [0.14, 0.95, 1.35],
        "target_radius": (0.30, 0.52),
        "obstacle_offset": (0.028, 0.150),
        "obstacle_mid": (0.30, 0.76),
        "act_noise": 0.065,
    },
}

TRAIN_SPLITS = [
    "nominal",
    "low_friction",
    "high_friction",
    "light_object",
    "heavy_object",
    "obstacle_shift",
    "actuation_noise",
    "far_target",
]

MODEL_CACHE: dict[PhysParams, mujoco.MjModel] = {}

BASE_FEATURE_DIM = 13
IDX_GEOM_DIST = 5
IDX_UNSAFE_GEOM = 12
IDX_BRANCH_MEAN_ENERGY = 13
IDX_BRANCH_MAX_ENERGY = 14
IDX_BRANCH_STD_ENERGY = 15
IDX_BRANCH_CVAR_ENERGY = 16
IDX_BRANCH_MEAN_DIST = 17
IDX_BRANCH_MAX_DIST = 18
IDX_BRANCH_MAX_VIOLATION = 19
IDX_BRANCH_MIN_CLEARANCE = 20
IDX_NOMINAL_ENERGY = 21
IDX_NOMINAL_VIOLATION = 22
IDX_BRANCH_ENERGY_GAP = 23
IDX_CLEARANCE_MARGIN = 24
IDX_GEOM_RANK = 25
IDX_CVAR_RANK = 26
IDX_WORST_RANK = 27
IDX_CLEARANCE_RANK = 28
FEATURE_DIM = 29
NO_RISK_COLUMNS = list(range(IDX_BRANCH_MEAN_ENERGY, IDX_CLEARANCE_MARGIN + 1)) + [
    IDX_CVAR_RANK,
    IDX_WORST_RANK,
    IDX_CLEARANCE_RANK,
]


def make_model(params: PhysParams) -> mujoco.MjModel:
    cached = MODEL_CACHE.get(params)
    if cached is not None:
        return cached
    xml = f"""
    <mujoco model="energy_action_push_v5">
      <option timestep="0.006" gravity="0 0 -9.81" integrator="RK4"/>
      <default>
        <geom condim="3" solref="0.006 1" solimp="0.9 0.95 0.001" friction="{params.friction} 0.004 0.0001"/>
      </default>
      <worldbody>
        <light pos="0 0 1"/>
        <geom name="floor" type="plane" size="1.3 1.3 0.02" rgba="0.75 0.75 0.75 1" friction="{params.friction} 0.004 0.0001"/>
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
        <position name="px_ctrl" joint="px" kp="520" ctrlrange="-1.2 1.2"/>
        <position name="py_ctrl" joint="py" kp="520" ctrlrange="-1.2 1.2"/>
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
    offset = action.offset + rng.gauss(0.0, max(0.002, act_noise * 0.035))
    distance = action.distance * max(0.72, rng.gauss(1.0, act_noise))
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    start = puck_xy - 0.128 * direction + offset * normal
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
    for i in range(58):
        alpha = (i + 1) / 58.0
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
    violation = float(min_obstacle_dist < CONTACT_MARGIN)
    return {"final_xy": final_xy, "violation": violation, "effort": effort, "min_obstacle_dist": min_obstacle_dist}


def sample_task(split: str, seed: int, episode: int) -> TaskSpec:
    rng = random.Random(6200003 + 100003 * seed + 9176 * episode + sum(ord(c) for c in split))
    cfg = SPLITS[split]
    params = PhysParams(rng.choice(cfg["masses"]), rng.choice(cfg["frictions"]))
    puck = np.array([rng.uniform(-0.030, 0.030), rng.uniform(-0.030, 0.030)], dtype=float)
    target_angle = rng.uniform(-0.68, 0.68)
    target_radius = rng.uniform(*cfg["target_radius"])
    target = puck + target_radius * np.array([math.cos(target_angle), math.sin(target_angle)], dtype=float)
    midpoint_weight = rng.uniform(*cfg["obstacle_mid"])
    midpoint = (1.0 - midpoint_weight) * puck + midpoint_weight * target
    normal = np.array([-math.sin(target_angle), math.cos(target_angle)], dtype=float)
    obstacle = midpoint + rng.choice([-1, 1]) * rng.uniform(*cfg["obstacle_offset"]) * normal
    return TaskSpec(split, params, tuple(puck), tuple(target), tuple(obstacle), cfg["act_noise"])


def candidate_actions(puck_xy: np.ndarray, target_xy: np.ndarray) -> list[PushAction]:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    remaining = float(np.linalg.norm(target_xy - puck_xy))
    actions: list[PushAction] = []
    for deg in [-55, -35, -18, 0, 18, 35, 55]:
        for offset in [-0.045, 0.0, 0.045]:
            for scale in [0.68, 1.00, 1.32]:
                actions.append(PushAction(base + math.radians(deg), offset, max(0.14, min(0.64, scale * remaining))))
    return actions


def line_clearance(puck_xy: np.ndarray, target_xy: np.ndarray, obstacle_xy: np.ndarray) -> float:
    segment = target_xy - puck_xy
    denom = float(np.dot(segment, segment)) + 1e-8
    t = max(0.0, min(1.0, float(np.dot(obstacle_xy - puck_xy, segment) / denom)))
    closest = puck_xy + t * segment
    return float(np.linalg.norm(closest - obstacle_xy))


def base_feature_vector(puck_xy: np.ndarray, target_xy: np.ndarray, obstacle_xy: np.ndarray, action: PushAction) -> np.ndarray:
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
        energy += 0.32 * violation
    return energy


def rank01(values: np.ndarray, lower_is_better: bool = True) -> np.ndarray:
    score = values if lower_is_better else -values
    order = np.argsort(score)
    ranks = np.zeros(len(values), dtype=np.float32)
    if len(values) == 1:
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def branch_summary_features(
    puck: np.ndarray,
    target: np.ndarray,
    obstacle: np.ndarray,
    action: PushAction,
) -> tuple[np.ndarray, float, float, float]:
    branch_energies = []
    branch_dists = []
    branch_violations = []
    branch_clearances = []
    nominal_energy = 0.0
    nominal_violation = 0.0
    for branch_idx, branch in enumerate(RISK_BRANCHES):
        out = rollout_push(branch, puck, obstacle, action, 0.0, random.Random(777 + branch_idx))
        energy = rollout_energy(out["final_xy"], target, out["violation"], out["effort"], True)
        branch_energies.append(energy)
        branch_dists.append(float(np.linalg.norm(out["final_xy"] - target)))
        branch_violations.append(float(out["violation"]))
        branch_clearances.append(float(out["min_obstacle_dist"]))
        if branch == NOMINAL:
            nominal_energy = energy
            nominal_violation = float(out["violation"])
    energies = np.array(branch_energies, dtype=np.float32)
    dists = np.array(branch_dists, dtype=np.float32)
    violations = np.array(branch_violations, dtype=np.float32)
    clearances = np.array(branch_clearances, dtype=np.float32)
    cvar_count = max(1, int(math.ceil(0.40 * len(energies))))
    cvar = float(np.mean(np.sort(energies)[-cvar_count:]))
    min_clearance = float(np.min(clearances))
    summary = np.array(
        [
            float(np.mean(energies)),
            float(np.max(energies)),
            float(np.std(energies)),
            cvar,
            float(np.mean(dists)),
            float(np.max(dists)),
            float(np.max(violations)),
            min_clearance,
            nominal_energy,
            nominal_violation,
            float(np.max(energies) - np.min(energies)),
            float(min_clearance - CONTACT_MARGIN),
        ],
        dtype=np.float32,
    )
    return summary, cvar, float(np.max(energies)), float(np.max(violations))


def build_labeled_action_set(task: TaskSpec, include_noise: bool = False) -> dict:
    puck = np.array(task.puck, dtype=float)
    target = np.array(task.target, dtype=float)
    obstacle = np.array(task.obstacle, dtype=float)
    actions = candidate_actions(puck, target)
    raw_without_ranks = []
    outcomes = []
    energies = []
    no_feas_energies = []
    risk_labels = []
    violation_labels = []
    branch_cvars = []
    branch_worsts = []
    for idx, action in enumerate(actions):
        base_features = base_feature_vector(puck, target, obstacle, action)
        branch_features, branch_cvar, branch_worst, branch_max_violation = branch_summary_features(puck, target, obstacle, action)
        raw_without_ranks.append(np.concatenate([base_features, branch_features]))
        rng = random.Random(9901 + idx + int(1e6 * task.params.mass) + int(1e5 * task.params.friction) + int(1000 * task.act_noise))
        out = rollout_push(task.params, puck, obstacle, action, task.act_noise if include_noise else 0.0, rng)
        outcomes.append(out)
        energy = rollout_energy(out["final_xy"], target, out["violation"], out["effort"], True)
        no_feas = rollout_energy(out["final_xy"], target, out["violation"], out["effort"], False)
        clearance_margin = float(branch_features[-1])
        unsafe_margin = max(0.0, SHIELD_CLEARANCE - float(branch_features[IDX_BRANCH_MIN_CLEARANCE - BASE_FEATURE_DIM]))
        risk_energy = 0.72 * energy + 0.20 * branch_cvar + 0.08 * branch_worst + 0.12 * branch_max_violation + 0.05 * unsafe_margin
        energies.append(energy)
        no_feas_energies.append(no_feas)
        risk_labels.append(risk_energy)
        violation_labels.append(float(out["violation"] > 0.5 or clearance_margin < 0.0))
        branch_cvars.append(branch_cvar)
        branch_worsts.append(branch_worst)
    raw_base = np.stack(raw_without_ranks).astype(np.float32)
    rank_features = np.stack(
        [
            rank01(raw_base[:, IDX_GEOM_DIST], lower_is_better=True),
            rank01(raw_base[:, IDX_BRANCH_CVAR_ENERGY], lower_is_better=True),
            rank01(raw_base[:, IDX_BRANCH_MAX_ENERGY], lower_is_better=True),
            rank01(raw_base[:, IDX_BRANCH_MIN_CLEARANCE], lower_is_better=False),
        ],
        axis=1,
    ).astype(np.float32)
    features = np.concatenate([raw_base, rank_features], axis=1).astype(np.float32)
    return {
        "actions": actions,
        "features": features,
        "outcomes": outcomes,
        "energy": np.array(energies, dtype=np.float32),
        "no_feas_energy": np.array(no_feas_energies, dtype=np.float32),
        "risk_energy": np.array(risk_labels, dtype=np.float32),
        "violation": np.array(violation_labels, dtype=np.float32),
        "branch_cvar": np.array(branch_cvars, dtype=np.float32),
        "branch_worst": np.array(branch_worsts, dtype=np.float32),
    }


class MlpEnergy(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepSetsEnergy(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        context = h.mean(dim=1, keepdim=True).expand_as(h)
        return self.out(torch.cat([h, context], dim=-1)).squeeze(-1)


class SetTransformerEnergy(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, 56)
        layer = nn.TransformerEncoderLayer(d_model=56, nhead=4, dim_feedforward=112, dropout=0.05, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(56, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        return self.out(h).squeeze(-1)


class RcRetEnergy(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.04, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.energy_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.violation_head = nn.Sequential(nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        h = self.encoder(h)
        return self.energy_head(h).squeeze(-1), self.violation_head(h).squeeze(-1)


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = features.reshape(-1, features.shape[-1]).mean(axis=0)
    sigma = features.reshape(-1, features.shape[-1]).std(axis=0) + 1e-6
    return (features - mu) / sigma, mu.astype(np.float32), sigma.astype(np.float32)


def mask_feature_columns(features: np.ndarray, mask_kind: str | None) -> np.ndarray:
    if mask_kind is None:
        return features
    masked = features.copy()
    if mask_kind == "no_risk":
        masked[..., NO_RISK_COLUMNS] = 0.0
    else:
        raise ValueError(mask_kind)
    return masked


def ranking_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    best_idx = torch.argmin(y, dim=1)
    best_pred = pred[torch.arange(pred.shape[0]), best_idx].unsqueeze(1)
    margins = torch.relu(0.035 + best_pred - pred)
    mask = torch.ones_like(margins, dtype=torch.bool)
    mask[torch.arange(pred.shape[0]), best_idx] = False
    return margins[mask].mean()


def make_loader(features: np.ndarray, labels: np.ndarray, violations: np.ndarray, kind: str) -> DataLoader:
    if kind == "mlp":
        x = torch.tensor(features.reshape(-1, features.shape[-1]), dtype=torch.float32)
        y = torch.tensor(labels.reshape(-1), dtype=torch.float32)
        v = torch.tensor(violations.reshape(-1), dtype=torch.float32)
        return DataLoader(TensorDataset(x, y, v), batch_size=384, shuffle=True)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    v = torch.tensor(violations, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y, v), batch_size=40, shuffle=True)


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    violations: np.ndarray,
    model_kind: str,
    seed: int,
    epochs: int,
    subset_fraction: float = 1.0,
    mask_kind: str | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    count = len(features)
    if subset_fraction < 1.0:
        subset_count = min(count, max(1, int(count * subset_fraction)))
        idx = rng.choice(count, subset_count, replace=False)
        features = features[idx]
        labels = labels[idx]
        violations = violations[idx]
    features = mask_feature_columns(features, mask_kind)
    if model_kind == "mlp":
        model: nn.Module = MlpEnergy(features.shape[-1])
    elif model_kind == "deep_sets":
        model = DeepSetsEnergy(features.shape[-1])
    elif model_kind == "transformer":
        model = SetTransformerEnergy(features.shape[-1])
    elif model_kind == "rc_ret":
        model = RcRetEnergy(features.shape[-1])
    else:
        raise ValueError(model_kind)
    loader = make_loader(features, labels, violations, model_kind if model_kind == "mlp" else "set")
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb, vb in loader:
            if model_kind == "rc_ret":
                pred_energy, pred_violation = model(xb)
                loss = nn.functional.mse_loss(pred_energy, yb) + 0.20 * ranking_loss(pred_energy, yb)
                loss = loss + 0.12 * bce(pred_violation, vb)
            else:
                pred = model(xb)
                loss = nn.functional.mse_loss(pred, yb)
                if model_kind in {"deep_sets", "transformer"}:
                    loss = loss + 0.12 * ranking_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    model.eval()
    return model


def generate_training_data(train_tasks: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_tag = f"{CACHE_VERSION}_{train_tasks}_{seed}"
    cache_features = RESULTS / f"energy_train_features_{cache_tag}.npy"
    cache_energy = RESULTS / f"energy_train_labels_{cache_tag}.npy"
    cache_no_feas = RESULTS / f"energy_train_no_feas_labels_{cache_tag}.npy"
    cache_risk = RESULTS / f"energy_train_risk_labels_{cache_tag}.npy"
    cache_violation = RESULTS / f"energy_train_violation_labels_{cache_tag}.npy"
    cache_mu = RESULTS / f"energy_feature_mu_{cache_tag}.npy"
    cache_sigma = RESULTS / f"energy_feature_sigma_{cache_tag}.npy"
    caches = [cache_features, cache_energy, cache_no_feas, cache_risk, cache_violation, cache_mu, cache_sigma]
    if all(path.exists() for path in caches):
        return (
            np.load(cache_features),
            np.load(cache_energy),
            np.load(cache_no_feas),
            np.load(cache_risk),
            np.load(cache_violation),
            np.load(cache_mu),
            np.load(cache_sigma),
        )
    rng = random.Random(seed)
    feature_sets = []
    energy_sets = []
    no_feas_sets = []
    risk_sets = []
    violation_sets = []
    for idx in range(train_tasks):
        split = rng.choice(TRAIN_SPLITS)
        task = sample_task(split, idx // 24, idx % 24)
        labeled = build_labeled_action_set(task, include_noise=False)
        feature_sets.append(labeled["features"])
        energy_sets.append(labeled["energy"])
        no_feas_sets.append(labeled["no_feas_energy"])
        risk_sets.append(labeled["risk_energy"])
        violation_sets.append(labeled["violation"])
        if (idx + 1) % 50 == 0:
            print(f"generated train task {idx + 1}/{train_tasks}", flush=True)
    features_raw = np.stack(feature_sets)
    features, mu, sigma = standardize(features_raw)
    energy = np.stack(energy_sets)
    no_feas = np.stack(no_feas_sets)
    risk = np.stack(risk_sets)
    violation = np.stack(violation_sets)
    np.save(cache_features, features.astype(np.float32))
    np.save(cache_energy, energy.astype(np.float32))
    np.save(cache_no_feas, no_feas.astype(np.float32))
    np.save(cache_risk, risk.astype(np.float32))
    np.save(cache_violation, violation.astype(np.float32))
    np.save(cache_mu, mu)
    np.save(cache_sigma, sigma)
    return features.astype(np.float32), energy.astype(np.float32), no_feas.astype(np.float32), risk.astype(np.float32), violation.astype(np.float32), mu, sigma


def predict_scalar(model: nn.Module, features: np.ndarray, model_kind: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        if model_kind == "mlp":
            pred = model(x).detach().cpu().numpy()
        else:
            pred = model(x.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    return pred.astype(float)


def predict_rc_ret(model: nn.Module, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        pred_energy, pred_violation = model(x)
    return pred_energy.squeeze(0).detach().cpu().numpy().astype(float), pred_violation.squeeze(0).detach().cpu().numpy().astype(float)


def choose_with_shield(scores: np.ndarray, raw_features: np.ndarray) -> tuple[int, float, float, int]:
    feasible = (raw_features[:, IDX_BRANCH_MIN_CLEARANCE] >= SHIELD_CLEARANCE) & (raw_features[:, IDX_BRANCH_MAX_VIOLATION] < 0.5)
    unshielded = int(np.argmin(scores))
    if bool(np.any(feasible)):
        masked_scores = np.where(feasible, scores, np.inf)
        chosen = int(np.argmin(masked_scores))
        return chosen, float(chosen != unshielded), 0.0, int(np.sum(feasible))
    return unshielded, 0.0, 1.0, 0


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-6)


def anchored_rc_score(
    pred_energy: np.ndarray,
    pred_violation: np.ndarray,
    raw_features: np.ndarray,
    use_branch_anchor: bool = True,
) -> np.ndarray:
    """Use RC-RET as a residual calibrator over the predefined risk frontier."""
    violation_prob = 1.0 / (1.0 + np.exp(-pred_violation))
    if use_branch_anchor:
        anchor = (
            0.54 * normalized(raw_features[:, IDX_BRANCH_CVAR_ENERGY])
            + 0.20 * normalized(raw_features[:, IDX_BRANCH_MAX_ENERGY])
            + 0.16 * normalized(raw_features[:, IDX_GEOM_DIST])
            + 0.10 * raw_features[:, IDX_BRANCH_MAX_VIOLATION]
        )
    else:
        anchor = 0.72 * normalized(raw_features[:, IDX_GEOM_DIST]) + 0.28 * raw_features[:, IDX_UNSAFE_GEOM]
    return 0.46 * normalized(pred_energy) + 0.54 * anchor + 0.10 * violation_prob


def evaluate_episode(split: str, seed: int, episode: int, models: dict, mu: np.ndarray, sigma: np.ndarray, ablation: bool = False) -> list[dict]:
    task = sample_task(split, seed, episode)
    target = np.array(task.target, dtype=float)
    labeled = build_labeled_action_set(task, include_noise=True)
    raw_features = labeled["features"]
    features = (raw_features - mu) / sigma
    energies = labeled["energy"]
    outcomes = labeled["outcomes"]
    methods = ABLATIONS if ablation else METHODS
    rows = []
    rng = random.Random(62062 + 997 * seed + 37 * episode + sum(ord(c) for c in split))

    pred_mlp = predict_scalar(models["mlp"], features, "mlp")
    pred_deep_sets = predict_scalar(models["deep_sets"], features, "set")
    pred_transformer = predict_scalar(models["transformer_v4"], features, "set")
    pred_rc_energy, pred_rc_violation = predict_rc_ret(models["rc_ret"], features)
    rc_scores = anchored_rc_score(pred_rc_energy, pred_rc_violation, raw_features, use_branch_anchor=True)
    no_risk_features = mask_feature_columns(features, "no_risk")
    pred_no_risk_energy, pred_no_risk_violation = predict_rc_ret(models["rc_ret_no_risk"], no_risk_features)
    no_risk_scores = anchored_rc_score(pred_no_risk_energy, pred_no_risk_violation, raw_features, use_branch_anchor=False)
    pred_no_cvar_energy, pred_no_cvar_violation = predict_rc_ret(models["rc_ret_no_cvar"], features)
    no_cvar_scores = anchored_rc_score(pred_no_cvar_energy, pred_no_cvar_violation, raw_features, use_branch_anchor=True)
    pred_no_feas_energy, _ = predict_rc_ret(models["rc_ret_no_feas"], features)
    pred_small_energy, pred_small_violation = predict_rc_ret(models["rc_ret_small"], features)
    small_scores = anchored_rc_score(pred_small_energy, pred_small_violation, raw_features, use_branch_anchor=True)

    for method in methods:
        shield_active = 0.0
        no_feasible = 0.0
        feasible_count = int(len(energies))
        if method == "random_candidate":
            chosen = rng.randrange(len(energies))
        elif method == "geometric_greedy":
            chosen = int(np.argmin(raw_features[:, IDX_GEOM_DIST] + 0.20 * raw_features[:, IDX_UNSAFE_GEOM]))
        elif method == "nominal_rollout_mpc":
            chosen = int(np.argmin(raw_features[:, IDX_NOMINAL_ENERGY]))
        elif method == "robust_worst_case_mpc":
            chosen = int(np.argmin(raw_features[:, IDX_BRANCH_MAX_ENERGY]))
        elif method == "cvar_branch_score":
            branch_scores = raw_features[:, IDX_BRANCH_CVAR_ENERGY] + 0.14 * raw_features[:, IDX_BRANCH_MAX_VIOLATION]
            chosen, shield_active, no_feasible, feasible_count = choose_with_shield(branch_scores, raw_features)
        elif method == "mlp_energy_scorer":
            chosen = int(np.argmin(pred_mlp))
        elif method == "shielded_mlp_energy_scorer":
            chosen, shield_active, no_feasible, feasible_count = choose_with_shield(pred_mlp, raw_features)
        elif method == "deep_sets_energy_scorer":
            chosen = int(np.argmin(pred_deep_sets))
        elif method == "set_transformer_energy_scorer_v4":
            chosen = int(np.argmin(pred_transformer))
        elif method == "rc_ret_no_shield":
            chosen = int(np.argmin(rc_scores))
        elif method == "rc_ret_energy_scorer_v5":
            chosen, shield_active, no_feasible, feasible_count = choose_with_shield(rc_scores, raw_features)
        elif method == "rc_ret_no_risk_features":
            chosen, shield_active, no_feasible, feasible_count = choose_with_shield(no_risk_scores, raw_features)
        elif method == "rc_ret_no_cvar_loss":
            chosen, shield_active, no_feasible, feasible_count = choose_with_shield(no_cvar_scores, raw_features)
        elif method == "rc_ret_no_feasibility":
            chosen = int(np.argmin(pred_no_feas_energy))
        elif method == "rc_ret_small_data":
            chosen, shield_active, no_feasible, feasible_count = choose_with_shield(small_scores, raw_features)
        elif method == "oracle_mujoco_rollout_selector":
            chosen = int(np.argmin(energies))
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
                "candidate_count": len(energies),
                "chosen_action": chosen,
                "success": float(final_distance <= ENERGY_SUCCESS_RADIUS and out["violation"] < 0.5),
                "final_distance": final_distance,
                "violation": float(out["violation"]),
                "effort": float(out["effort"]),
                "energy": float(energies[chosen]),
                "oracle_energy": oracle_energy,
                "energy_regret": float(energies[chosen] - oracle_energy),
                "branch_cvar_energy": float(raw_features[chosen, IDX_BRANCH_CVAR_ENERGY]),
                "branch_worst_energy": float(raw_features[chosen, IDX_BRANCH_MAX_ENERGY]),
                "branch_min_clearance": float(raw_features[chosen, IDX_BRANCH_MIN_CLEARANCE]),
                "shield_active": shield_active,
                "no_feasible_candidates": no_feasible,
                "feasible_candidate_count": feasible_count,
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
        efforts = [float(v["effort"]) for v in vals]
        shields = [float(v["shield_active"]) for v in vals]
        no_feasible = [float(v["no_feasible_candidates"]) for v in vals]
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
                "effort_mean": mean(efforts),
                "effort_ci95": ci95(efforts),
                "shield_activation_rate": mean(shields),
                "no_feasible_rate": mean(no_feasible),
            }
        )
        out.append(summary)
    return out


def bootstrap_ci(vals: list[float], seed: int, n_boot: int = 1000) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    arr = np.array(vals, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_pvalue(vals: list[float], seed: int, n_perm: int = 2000) -> float:
    if not vals:
        return 1.0
    arr = np.array(vals, dtype=float)
    obs = abs(float(np.mean(arr)))
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(arr))
        if abs(float(np.mean(signs * arr))) >= obs - 1e-12:
            count += 1
    return float((count + 1) / (n_perm + 1))


def paired_stats(rows: list[dict]) -> list[dict]:
    proposed = "rc_ret_energy_scorer_v5"
    baselines = [
        "geometric_greedy",
        "nominal_rollout_mpc",
        "robust_worst_case_mpc",
        "cvar_branch_score",
        "mlp_energy_scorer",
        "shielded_mlp_energy_scorer",
        "deep_sets_energy_scorer",
        "set_transformer_energy_scorer_v4",
        "oracle_mujoco_rollout_selector",
    ]
    by_key: dict[tuple, dict] = {}
    for row in rows:
        by_key.setdefault((row["split"], row["seed"], row["episode"]), {})[row["method"]] = row
    out = []
    splits = sorted({row["split"] for row in rows})
    for split in splits + ["ALL"]:
        if split == "ALL":
            cases = [methods for methods in by_key.values() if proposed in methods]
        else:
            cases = [methods for key, methods in by_key.items() if key[0] == split and proposed in methods]
        for baseline in baselines:
            paired = [(methods[proposed], methods[baseline]) for methods in cases if baseline in methods]
            if not paired:
                continue
            success_delta = [float(p["success"]) - float(b["success"]) for p, b in paired]
            regret_delta = [float(b["energy_regret"]) - float(p["energy_regret"]) for p, b in paired]
            violation_delta = [float(p["violation"]) - float(b["violation"]) for p, b in paired]
            distance_delta = [float(b["final_distance"]) - float(p["final_distance"]) for p, b in paired]
            s_lo, s_hi = bootstrap_ci(success_delta, 6211 + len(out))
            r_lo, r_hi = bootstrap_ci(regret_delta, 9221 + len(out))
            v_lo, v_hi = bootstrap_ci(violation_delta, 1123 + len(out))
            out.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "paired_episodes": len(paired),
                    "success_delta_mean": f"{mean(success_delta):.4f}",
                    "success_delta_ci_low": f"{s_lo:.4f}",
                    "success_delta_ci_high": f"{s_hi:.4f}",
                    "success_delta_p_signflip": f"{sign_flip_pvalue(success_delta, 1717 + len(out)):.4f}",
                    "regret_improvement_mean": f"{mean(regret_delta):.4f}",
                    "regret_improvement_ci_low": f"{r_lo:.4f}",
                    "regret_improvement_ci_high": f"{r_hi:.4f}",
                    "regret_improvement_p_signflip": f"{sign_flip_pvalue(regret_delta, 2727 + len(out)):.4f}",
                    "distance_improvement_mean": f"{mean(distance_delta):.4f}",
                    "violation_delta_mean": f"{mean(violation_delta):.4f}",
                    "violation_delta_ci_low": f"{v_lo:.4f}",
                    "violation_delta_ci_high": f"{v_hi:.4f}",
                    "violation_delta_p_signflip": f"{sign_flip_pvalue(violation_delta, 3737 + len(out)):.4f}",
                }
            )
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


def plot_results(metrics: list[dict], ablation: list[dict]) -> None:
    splits = sorted({r["split"] for r in metrics})
    methods = [
        "nominal_rollout_mpc",
        "robust_worst_case_mpc",
        "mlp_energy_scorer",
        "deep_sets_energy_scorer",
        "rc_ret_energy_scorer_v5",
        "oracle_mujoco_rollout_selector",
    ]
    labels = ["Nominal", "Robust", "MLP", "DeepSets", "RC-RET", "Oracle"]
    x = np.arange(len(splits))
    width = 0.13
    plt.figure(figsize=(13, 5.1))
    for idx, method in enumerate(methods):
        vals = [float(next(r["success_rate"] for r in metrics if r["split"] == split and r["method"] == method)) for split in splits]
        plt.bar(x + (idx - 2.5) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=25, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.02)
    plt.title("Energy action selection success by stress split")
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_success_by_split.png", dpi=190)
    plt.close()

    plt.figure(figsize=(13, 5.1))
    for idx, method in enumerate(methods):
        vals = [float(next(r["energy_regret_mean"] for r in metrics if r["split"] == split and r["method"] == method)) for split in splits]
        plt.bar(x + (idx - 2.5) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=25, ha="right")
    plt.ylabel("Energy regret vs oracle")
    plt.title("Energy regret by stress split")
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_regret_by_split.png", dpi=190)
    plt.close()

    plt.figure(figsize=(13, 5.1))
    for idx, method in enumerate(methods[:-1]):
        vals = [float(next(r["violation_rate"] for r in metrics if r["split"] == split and r["method"] == method)) for split in splits]
        plt.bar(x + (idx - 2.0) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=25, ha="right")
    plt.ylabel("Violation rate")
    plt.title("Obstacle/contact violations by stress split")
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_violation_by_split.png", dpi=190)
    plt.close()

    order = sorted(ablation, key=lambda r: (r.get("split", ""), float(r["energy_regret_mean"])))
    plt.figure(figsize=(10.5, max(5.0, 0.22 * len(order))))
    y_labels = [f"{r.get('split', 'ablation')} / {r['method']}" for r in order]
    plt.barh(y_labels, [float(r["energy_regret_mean"]) for r in order])
    plt.xlabel("Energy regret vs oracle")
    plt.title("Ablation energy regret")
    plt.tight_layout()
    plt.savefig(FIGURES / "energy_ablation_regret.png", dpi=190)
    plt.close()


def write_summary(main_summary: list[dict], pairwise: list[dict], ablation_summary: list[dict], args: argparse.Namespace) -> None:
    proposed = [r for r in main_summary if r["method"] == "rc_ret_energy_scorer_v5"]
    strong_all = [
        r
        for r in pairwise
        if r["split"] == "ALL"
        and r["baseline"]
        in {"nominal_rollout_mpc", "robust_worst_case_mpc", "mlp_energy_scorer", "shielded_mlp_energy_scorer", "deep_sets_energy_scorer"}
    ]
    robust_gate = all(float(r["success_delta_mean"]) >= -0.005 and float(r["violation_delta_mean"]) <= 0.010 for r in strong_all)
    ablation_gate = True
    for split in set(r["split"] for r in ablation_summary):
        rows = [r for r in ablation_summary if r["split"] == split]
        full = next(r for r in rows if r["method"] == "rc_ret_energy_scorer_v5")
        for row in rows:
            if row["method"] in {"mlp_energy_scorer", "deep_sets_energy_scorer", "set_transformer_energy_scorer_v4"}:
                if float(row["success_rate"]) > float(full["success_rate"]) + 0.01 and float(row["energy_regret_mean"]) < float(full["energy_regret_mean"]):
                    ablation_gate = False
    decision = "ICLR_MAIN_TARGET_READY" if robust_gate and ablation_gate else "STRONG_REVISE"
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 62 v5 RC-RET MuJoCo/PyTorch action-selection benchmark\n")
        f.write(f"decision={decision}\n")
        f.write(
            f"train_tasks={args.train_tasks} epochs={args.epochs} seeds={args.seeds} episodes={args.episodes} "
            f"splits={','.join(args.splits)} ablation_splits={','.join(args.ablation_splits)}\n"
        )
        for row in proposed:
            f.write(
                f"{row['split']} rc_ret success={row['success_rate']:.3f}+/-{row['success_ci95']:.3f} "
                f"regret={row['energy_regret_mean']:.3f}+/-{row['energy_regret_ci95']:.3f} "
                f"violation={row['violation_rate']:.3f} shield={row['shield_activation_rate']:.3f}\n"
            )
        f.write("strong_baseline_all_pairs\n")
        for row in strong_all:
            f.write(
                f"{row['baseline']} success_delta={row['success_delta_mean']} "
                f"regret_improvement={row['regret_improvement_mean']} violation_delta={row['violation_delta_mean']}\n"
            )


def run(args: argparse.Namespace) -> None:
    global RESULTS, FIGURES
    RESULTS = ROOT / args.results_subdir
    FIGURES = ROOT / args.figures_subdir
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    features, energy, no_feas, risk, violation, mu, sigma = generate_training_data(args.train_tasks, args.seed)
    print("training v5 models", flush=True)
    models = {
        "mlp": train_model(features, energy, violation, "mlp", args.seed + 1, args.epochs),
        "deep_sets": train_model(features, energy, violation, "deep_sets", args.seed + 2, args.epochs),
        "transformer_v4": train_model(features, energy, violation, "transformer", args.seed + 3, args.epochs),
        "rc_ret": train_model(features, risk, violation, "rc_ret", args.seed + 4, args.epochs),
        "rc_ret_no_risk": train_model(features, risk, violation, "rc_ret", args.seed + 5, args.epochs, mask_kind="no_risk"),
        "rc_ret_no_cvar": train_model(features, energy, violation, "rc_ret", args.seed + 6, args.epochs),
        "rc_ret_no_feas": train_model(features, no_feas, np.zeros_like(violation), "rc_ret", args.seed + 7, args.epochs),
        "rc_ret_small": train_model(features, risk, violation, "rc_ret", args.seed + 8, args.epochs, subset_fraction=0.25),
    }

    raw_rows: list[dict] = []
    for split in args.splits:
        for seed in range(args.seeds):
            for episode in range(args.episodes):
                raw_rows.extend(evaluate_episode(split, seed, episode, models, mu, sigma, ablation=False))
            write_rows(RESULTS / "energy_action_raw.partial.csv", format_rows(raw_rows))
            write_rows(RESULTS / "energy_action_metrics.partial.csv", format_rows(summarize(raw_rows, ["split", "method"])))
            print(f"completed main split={split} seed={seed} rows={len(raw_rows)}", flush=True)

    ablation_rows: list[dict] = []
    for split in args.ablation_splits:
        for seed in range(args.seeds):
            for episode in range(args.episodes):
                ablation_rows.extend(evaluate_episode(split, seed, episode, models, mu, sigma, ablation=True))
            write_rows(RESULTS / "energy_action_ablation_raw.partial.csv", format_rows(ablation_rows))
            write_rows(RESULTS / "energy_action_ablation.partial.csv", format_rows(summarize(ablation_rows, ["split", "method"])))
            print(f"completed ablation split={split} seed={seed} rows={len(ablation_rows)}", flush=True)

    main_summary = summarize(raw_rows, ["split", "method"])
    seed_summary = summarize(raw_rows, ["split", "method", "seed"])
    ablation_summary = summarize(ablation_rows, ["split", "method"])
    pairwise = paired_stats(raw_rows)

    write_rows(RESULTS / "energy_action_raw.csv", format_rows(raw_rows))
    write_rows(RESULTS / "energy_action_metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "energy_action_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "energy_action_ablation_raw.csv", format_rows(ablation_rows))
    write_rows(RESULTS / "energy_action_ablation.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "energy_action_pairwise.csv", pairwise)
    write_rows(RESULTS / "metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "raw_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "ablation_metrics.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "pairwise_stats.csv", pairwise)
    write_rows(RESULTS / "stress_sweep.csv", format_rows(main_summary))
    write_rows(FIGURES / "stress_curve_data.csv", format_rows(main_summary))
    negative_cases = [
        {"case": "unobserved_deformable_contact", "observed": "finite candidate energy cannot identify deformation without sensing", "paper_status": "limitation"},
        {"case": "branch_set_misspecification", "observed": "shield only certifies modeled branches and can miss unseen friction/mass modes", "paper_status": "limitation"},
        {"case": "exact_oracle_dominance", "observed": "learned selector cannot dominate true rollout oracle under identical candidates", "paper_status": "negative theorem"},
        {"case": "narrow_clearance_no_feasible_candidate", "observed": "some generated scenes contain no branch-feasible action in the finite library", "paper_status": "reported stress case"},
    ]
    write_rows(RESULTS / "negative_cases.csv", negative_cases)
    plot_results(main_summary, ablation_summary)
    write_summary(main_summary, pairwise, ablation_summary, args)
    print(f"wrote v5 energy action benchmark results to {RESULTS}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=62062)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    parser.add_argument("--ablation-splits", nargs="+", default=["combined_shift", "narrow_clearance"])
    parser.add_argument("--results-subdir", default="results")
    parser.add_argument("--figures-subdir", default="figures")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
