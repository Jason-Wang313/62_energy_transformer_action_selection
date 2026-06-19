from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find(rows: list[dict[str, str]], **kwargs: str) -> dict[str, str]:
    for row in rows:
        if all(row[k] == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)


def main() -> None:
    raw = read_csv("energy_action_raw.csv")
    ablation_raw = read_csv("energy_action_ablation_raw.csv")
    metrics = read_csv("energy_action_metrics.csv")
    pairwise = read_csv("energy_action_pairwise.csv")
    ablation = read_csv("energy_action_ablation.csv")
    strong = [
        "nominal_rollout_mpc",
        "robust_worst_case_mpc",
        "cvar_branch_score",
        "mlp_energy_scorer",
        "shielded_mlp_energy_scorer",
        "deep_sets_energy_scorer",
        "set_transformer_energy_scorer_v4",
        "oracle_mujoco_rollout_selector",
    ]
    strong_rows = [find(pairwise, split="ALL", baseline=baseline) for baseline in strong]
    wins = sum(float(row["success_delta_mean"]) > 0 for row in strong_rows)
    ties_or_wins = sum(float(row["success_delta_mean"]) >= -0.005 for row in strong_rows if row["baseline"] != "oracle_mujoco_rollout_selector")
    safer = sum(float(row["violation_delta_mean"]) <= 0 for row in strong_rows if row["baseline"] != "oracle_mujoco_rollout_selector")
    combined = find(metrics, split="combined_shift", method="rc_ret_energy_scorer_v5")
    narrow = find(metrics, split="narrow_clearance", method="rc_ret_energy_scorer_v5")
    robust_all = find(pairwise, split="ALL", baseline="robust_worst_case_mpc")
    mlp_all = find(pairwise, split="ALL", baseline="mlp_energy_scorer")
    deepsets_all = find(pairwise, split="ALL", baseline="deep_sets_energy_scorer")
    decision = "STRONG_REVISE"
    lines = [
        "# Paper 62 Final v5 Evidence Summary",
        "",
        f"Decision: `{decision}`.",
        "",
        "ICLR main readiness: no.",
        "",
        "## Frozen Evidence Size",
        "",
        f"- Main raw rows: {len(raw)}.",
        f"- Ablation raw rows: {len(ablation_raw)}.",
        f"- Metric rows: {len(metrics)}.",
        f"- Ablation metric rows: {len(ablation)}.",
        f"- Pairwise rows: {len(pairwise)}.",
        "- Frozen run: 480 training action sets, 24 epochs, 8 seeds, 12 episodes, 10 stress splits, 12 main methods.",
        "",
        "## Gate Results",
        "",
        f"- Strong-baseline success wins: {wins}/{len(strong_rows)}.",
        f"- Non-oracle ties-or-wins under the success gate: {ties_or_wins}/7.",
        f"- Non-oracle safety ties-or-improvements: {safer}/7.",
        f"- Aggregate versus robust MPC: success delta {robust_all['success_delta_mean']}, regret improvement {robust_all['regret_improvement_mean']}, violation delta {robust_all['violation_delta_mean']}.",
        f"- Aggregate versus MLP: success delta {mlp_all['success_delta_mean']}, regret improvement {mlp_all['regret_improvement_mean']}, violation delta {mlp_all['violation_delta_mean']}.",
        f"- Aggregate versus Deep Sets: success delta {deepsets_all['success_delta_mean']}, regret improvement {deepsets_all['regret_improvement_mean']}, violation delta {deepsets_all['violation_delta_mean']}.",
        f"- Combined-shift RC-RET: success {combined['success_rate']}, regret {combined['energy_regret_mean']}, violation {combined['violation_rate']}.",
        f"- Narrow-clearance RC-RET: success {narrow['success_rate']}, regret {narrow['energy_regret_mean']}, violation {narrow['violation_rate']}.",
        "",
        "## Honest Interpretation",
        "",
        "The v5 rebuild is substantially stronger than the v4 scaffold: it uses a larger action manifold, branch-risk features, a calibrated safety shield, Deep Sets and shielded baselines, ten stress splits, and paired statistics. The evidence is real and useful.",
        "",
        "It is still not ICLR-main ready. RC-RET is close to robust/CVaR branch scoring and safer than several unshielded learned baselines, but it does not clear the success gate against MLP, Deep Sets, nominal MPC, or the oracle. The narrow-clearance split is a particularly hard negative case.",
        "",
        "The correct terminal state is `STRONG_REVISE`, not `ICLR_MAIN_TARGET_READY`.",
        "",
    ]
    out = DOCS / "paper62_final_v5_evidence_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

