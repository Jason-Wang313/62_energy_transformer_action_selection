from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(text: str) -> str:
    return text.replace("_", r"\_")


def pct(value: str) -> str:
    return f"{100.0 * float(value):.1f}"


def num(value: str) -> str:
    return f"{float(value):.3f}"


def metric(metrics: list[dict[str, str]], split: str, method: str, key: str) -> str:
    for row in metrics:
        if row["split"] == split and row["method"] == method:
            return row[key]
    raise KeyError((split, method, key))


def table_success(metrics: list[dict[str, str]]) -> str:
    splits = [
        "nominal",
        "low_friction",
        "high_friction",
        "light_object",
        "heavy_object",
        "obstacle_shift",
        "narrow_clearance",
        "actuation_noise",
        "far_target",
        "combined_shift",
    ]
    methods = [
        ("rc_ret_energy_scorer_v5", "RC-RET"),
        ("mlp_energy_scorer", "MLP"),
        ("shielded_mlp_energy_scorer", "Shielded MLP"),
        ("deep_sets_energy_scorer", "Deep Sets"),
        ("robust_worst_case_mpc", "Robust MPC"),
        ("cvar_branch_score", "CVaR branch"),
        ("oracle_mujoco_rollout_selector", "Oracle"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen v5 success rates. Entries are percentages over 96 paired episodes per split.}",
        r"\label{tab:v5-success}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Split & " + " & ".join(label for _, label in methods) + r" \\",
        r"\midrule",
    ]
    for split in splits:
        vals = [pct(metric(metrics, split, method, "success_rate")) for method, _ in methods]
        lines.append(f"{esc(split)} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_regret(metrics: list[dict[str, str]]) -> str:
    splits = [
        "nominal",
        "low_friction",
        "high_friction",
        "light_object",
        "heavy_object",
        "obstacle_shift",
        "narrow_clearance",
        "actuation_noise",
        "far_target",
        "combined_shift",
    ]
    methods = [
        ("rc_ret_energy_scorer_v5", "RC-RET"),
        ("mlp_energy_scorer", "MLP"),
        ("shielded_mlp_energy_scorer", "Shielded MLP"),
        ("deep_sets_energy_scorer", "Deep Sets"),
        ("robust_worst_case_mpc", "Robust MPC"),
        ("cvar_branch_score", "CVaR branch"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen v5 energy regret relative to the true rollout oracle. Lower is better.}",
        r"\label{tab:v5-regret}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Split & " + " & ".join(label for _, label in methods) + r" \\",
        r"\midrule",
    ]
    for split in splits:
        vals = [num(metric(metrics, split, method, "energy_regret_mean")) for method, _ in methods]
        lines.append(f"{esc(split)} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_pairwise(pairwise: list[dict[str, str]]) -> str:
    baselines = [
        "nominal_rollout_mpc",
        "robust_worst_case_mpc",
        "cvar_branch_score",
        "mlp_energy_scorer",
        "shielded_mlp_energy_scorer",
        "deep_sets_energy_scorer",
        "set_transformer_energy_scorer_v4",
        "oracle_mujoco_rollout_selector",
    ]
    by_base = {row["baseline"]: row for row in pairwise if row["split"] == "ALL"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Aggregate paired deltas for RC-RET v5 against strong baselines. Positive success and regret-improvement values favor RC-RET; negative violation deltas mean RC-RET is safer.}",
        r"\label{tab:v5-pairwise}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Baseline & Success $\Delta$ & 95\% CI & Regret improv. & Violation $\Delta$ \\",
        r"\midrule",
    ]
    for baseline in baselines:
        row = by_base[baseline]
        ci = f"[{row['success_delta_ci_low']}, {row['success_delta_ci_high']}]"
        lines.append(
            f"{esc(baseline)} & {row['success_delta_mean']} & {ci} & "
            f"{row['regret_improvement_mean']} & {row['violation_delta_mean']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_ablation(ablation: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen v5 ablations on combined-shift and narrow-clearance stress.}",
        r"\label{tab:v5-ablation}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Split & Method & Success & Regret & Violation & Shield \\",
        r"\midrule",
    ]
    for row in ablation:
        lines.append(
            f"{esc(row['split'])} & {esc(row['method'])} & {pct(row['success_rate'])} & "
            f"{num(row['energy_regret_mean'])} & {pct(row['violation_rate'])} & "
            f"{pct(row['shield_activation_rate'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def appendix_metric_tables(metrics: list[dict[str, str]]) -> str:
    splits = [
        "nominal",
        "low_friction",
        "high_friction",
        "light_object",
        "heavy_object",
        "obstacle_shift",
        "narrow_clearance",
        "actuation_noise",
        "far_target",
        "combined_shift",
    ]
    lines: list[str] = ["% Auto-generated detailed metric tables"]
    for split in splits:
        rows = [row for row in metrics if row["split"] == split]
        lines.extend(
            [
                r"\begin{table}[p]",
                r"\centering",
                rf"\caption{{Detailed frozen metrics for {esc(split)}.}}",
                rf"\label{{tab:detail-{split.replace('_', '-')}}}",
                r"\small",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"Method & Success & Regret & Distance & Violation & Shield \\",
                r"\midrule",
            ]
        )
        for row in rows:
            lines.append(
                f"{esc(row['method'])} & {pct(row['success_rate'])} & {num(row['energy_regret_mean'])} & "
                f"{num(row['final_distance_mean'])} & {pct(row['violation_rate'])} & "
                f"{pct(row['shield_activation_rate'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def appendix_pairwise_tables(pairwise: list[dict[str, str]]) -> str:
    splits = [
        "nominal",
        "low_friction",
        "high_friction",
        "light_object",
        "heavy_object",
        "obstacle_shift",
        "narrow_clearance",
        "actuation_noise",
        "far_target",
        "combined_shift",
    ]
    keep = {
        "nominal_rollout_mpc",
        "robust_worst_case_mpc",
        "cvar_branch_score",
        "mlp_energy_scorer",
        "shielded_mlp_energy_scorer",
        "deep_sets_energy_scorer",
        "set_transformer_energy_scorer_v4",
        "oracle_mujoco_rollout_selector",
    }
    lines: list[str] = ["% Auto-generated detailed pairwise tables"]
    for split in splits:
        rows = [row for row in pairwise if row["split"] == split and row["baseline"] in keep]
        lines.extend(
            [
                r"\begin{table}[p]",
                r"\centering",
                rf"\caption{{Paired deltas for {esc(split)}. Positive success and regret-improvement favor RC-RET; negative violation deltas favor RC-RET.}}",
                rf"\label{{tab:pair-{split.replace('_', '-')}}}",
                r"\small",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                r"Baseline & Success $\Delta$ & Regret improv. & Distance improv. & Violation $\Delta$ \\",
                r"\midrule",
            ]
        )
        for row in rows:
            lines.append(
                f"{esc(row['baseline'])} & {row['success_delta_mean']} & {row['regret_improvement_mean']} & "
                f"{row['distance_improvement_mean']} & {row['violation_delta_mean']} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    metrics = read_csv("energy_action_metrics.csv")
    pairwise = read_csv("energy_action_pairwise.csv")
    ablation = read_csv("energy_action_ablation.csv")
    content = "\n".join(
        [
            "% Auto-generated by scripts/render_latex_tables.py",
            table_success(metrics),
            table_regret(metrics),
            table_pairwise(pairwise),
            table_ablation(ablation),
        ]
    )
    (PAPER / "results_tables.tex").write_text(content, encoding="utf-8")
    appendix_content = "\n\n".join([appendix_metric_tables(metrics), appendix_pairwise_tables(pairwise)])
    (PAPER / "appendix_results_tables.tex").write_text(appendix_content, encoding="utf-8")
    print(PAPER / "results_tables.tex")
    print(PAPER / "appendix_results_tables.tex")


if __name__ == "__main__":
    main()
