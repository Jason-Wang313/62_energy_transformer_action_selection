# Paper 62 Expanded Submission Plan

Date: 2026-06-19

Paper: `62_energy_transformer_action_selection`

Target venue posture: ICLR main-conference submission candidate if, and only if, the frozen evidence survives strong baselines, stress tests, ablations, and hostile prior-work positioning.

Operating constraints:

- CPU only.
- Keep RAM light by using compact tabular tensors, streaming CSV writes, cached rollout labels, small PyTorch models, and bounded torch threads.
- Do not compromise paper quality for RAM. If a higher-quality test is CPU-expensive but memory-light, run it.
- Do not add filler to reach 25 pages. Every added page must carry theory, protocol, evidence, negative analysis, related-work positioning, or reproducibility detail.
- Keep the numbered PDF in `C:/Users/wangz/Downloads/62.pdf` only. Never copy it to the visible Desktop.
- Freeze the final protocol before the terminal run. Development smoke tests may guide fixes, but final claims must report all predefined results honestly.

## 1. Starting Diagnosis

The current v4 repository is clean and reproducible, but it is not submission-ready.

Verified starting facts:

- Current PDF is 4 pages.
- Existing main evidence has 3,360 rows: 6 splits, 5 seeds, 16 episodes, 7 methods.
- Existing ablation evidence has 480 rows.
- Existing terminal decision is `STRONG_REVISE`.
- The transformer scorer beats weak random/geometric selectors but does not consistently beat MLP energy, nominal MPC, or robust MPC.
- Current ablations do not isolate self-attention as necessary.
- Current bibliography is noisy and contains mostly irrelevant keyword matches.

Core failure to fix:

The current transformer operates over candidate action tokens, but the features already give each action strong local rollout/geometric information. With only shallow set context and no explicit relational, robust-risk, or calibrated safety structure, self-attention has no convincing reason to beat an MLP or exact robust MPC.

## 2. Revised Thesis

The revised paper should not claim that "a transformer is better" by architectural branding. It should claim a narrower, testable thesis:

> Contact-rich action selection benefits when candidate actions are scored as a permutation-equivariant action set under calibrated robust risk, because feasibility and physical regret are relative properties of the entire candidate manifold rather than independent per-action labels.

This thesis requires evidence for three separable mechanisms:

- Set context: decisions change when competing feasible actions define different risk frontiers.
- Robust risk: a calibrated CVaR/minimax-style energy reduces distribution-shift regret without becoming overly conservative.
- Safety calibration: obstacle and contact violations are controlled without hiding behind oracle rollouts.

If any of these mechanisms fail, the terminal decision remains `STRONG_REVISE` or becomes `KILL_ARCHIVE`.

## 3. Method Rebuild: RC-RET v5

Implement a new proposed method: Risk-Calibrated Relational Energy Transformer (RC-RET).

Required components:

- Candidate library expansion:
  - Use multiple push angles, push distances, and lateral offsets.
  - Include a small number of two-stage or recovery-style primitives only if they remain CPU-light.
  - Keep the same candidate library for all methods.

- Token features:
  - Existing geometric and nominal rollout features.
  - Branch-risk summaries under mass/friction hypotheses: mean, worst, variance, and lower-tail clearance.
  - Feasibility indicators: obstacle clearance, predicted collision margin, effort, path length, and near-grazing flags.
  - Relative-rank features inside the candidate set: normalized progress rank, clearance rank, and robust-energy rank.

- Relational action-set encoding:
  - Add pairwise or set-normalized features that make the attention mechanism substantively useful.
  - Preserve permutation equivariance over candidate actions.
  - Add a Deep Sets baseline so attention is tested against a non-attention set model, not only against an MLP.

- Risk-calibrated objective:
  - Train on rollout energy plus feasibility penalty.
  - Add ranking loss against the best feasible candidate.
  - Add CVaR-style tail-risk loss over dynamics branches.
  - Add violation-calibration loss so predicted low energy does not silently select unsafe actions.

- Inference:
  - Primary score is learned calibrated robust energy.
  - Apply a predefined safety shield: candidates below a minimum clearance threshold are not selected unless no feasible candidate exists.
  - Report shield activation rate as an outcome, not as a hidden implementation trick.

Fairness constraints:

- Robust MPC remains a strong exact-simulation baseline.
- If RC-RET uses branch-risk features, include a baseline that uses the same branch summaries without attention.
- If the safety shield improves RC-RET, include shielded MLP and shielded robust-score baselines where appropriate.
- Do not tune the final method after seeing frozen final results.

## 4. Baseline Suite

Main methods:

- `random_candidate`
- `geometric_greedy`
- `nominal_rollout_mpc`
- `robust_worst_case_mpc`
- `mlp_energy_scorer`
- `deep_sets_energy_scorer`
- `set_transformer_energy_scorer_v4`
- `rc_ret_energy_scorer_v5`
- `rc_ret_no_shield`
- `shielded_mlp_energy_scorer`
- `oracle_mujoco_rollout_selector`

Optional CPU-light baselines if implementation remains clean:

- `cvar_branch_score`
- `ensemble_mlp_energy_scorer`
- `topk_mpc_then_energy`

The paper may only claim a learned set-model contribution if RC-RET beats or ties the strongest non-oracle baselines on predefined gates.

## 5. Stress Splits

Main frozen splits:

- nominal
- low friction
- high friction
- light object
- heavy object
- obstacle shift
- narrow-clearance obstacle
- actuation noise
- far target
- combined dynamics + obstacle + noise shift

Each split must report:

- Success rate.
- Final distance.
- Energy regret versus true rollout oracle.
- Violation rate.
- Effort.
- Safety shield activation.
- Paired deltas against MLP, Deep Sets, nominal MPC, robust MPC, and oracle.

## 6. Ablations

Run ablations on combined shift and narrow-clearance stress:

- Full RC-RET.
- No self-attention / Deep Sets replacement.
- No pairwise-relative features.
- No branch-risk features.
- No CVaR/tail-risk loss.
- No feasibility objective.
- No safety shield.
- No rank loss.
- Small-data training.
- Old v4 transformer architecture.

Gate: the full method must not simply be matched by MLP, Deep Sets, or old v4 transformer. If those match it, the manuscript must say so.

## 7. Statistical Protocol

Development phase:

- Run smoke tests with tiny train/eval settings only for correctness.
- Use smoke results only to find crashes, schema bugs, pathological runtime, and obvious modeling defects.
- Log any method changes in `docs/paper62_development_log.md`.

Final freeze:

- Write `docs/paper62_protocol_freeze_20260619.md` before the terminal run.
- Include exact hyperparameters, train-task count, seeds, episodes, splits, method list, ablation list, and gates.
- After the freeze, only fix recoverable infrastructure failures. Do not change the protocol to chase prettier results.

Final run target:

- Use at least 8 seeds and 24 episodes per split if runtime remains tractable.
- Prefer more seeds/episodes over larger neural models.
- Use `--torch-threads` no greater than 4 by default.
- Stream partial CSVs after each split and seed.
- Support resume from partial outputs.

Statistics:

- Paired bootstrap confidence intervals.
- Sign-flip or paired permutation p-values for key deltas.
- Holm correction over the main strong-baseline comparisons.
- Report all predefined split/method pairs, including losses.
- Include oracle regret and gap-to-oracle tables.

## 8. Theory Additions

The paper must include real theory, not decorative notation.

Required theory sections:

- Formal problem setup for finite action-set selection under latent contact parameters.
- Energy decomposition into progress, effort, violation, and tail-risk terms.
- Permutation-equivariance proposition for the relational set scorer.
- Regret decomposition showing approximation error, calibration error, branch-mismatch error, and candidate-set discretization error.
- Safety-shield lemma: if predicted clearance is calibrated with error bound epsilon and threshold tau exceeds epsilon plus physical margin, selected actions satisfy the clearance margin under the modeled branch set.
- Conservatism analysis comparing worst-case robust MPC with calibrated CVaR selection.
- Negative identifiability result: no learned selector can guarantee oracle dominance under unobserved contact parameters without either additional observations or assumptions on the latent-parameter distribution.

## 9. Related Work and Citations

Replace the noisy bibliography with a manual, primary-source bibliography.

Must cover:

- Energy-based policies and implicit behavioral cloning.
- Diffusion and transformer robot policies as adjacent action-distribution approaches.
- Set functions, Deep Sets, and Set Transformer.
- Model-based control, MPPI, PETS, and robust MPC.
- MuJoCo and contact-rich simulation.
- Safety/constrained action selection.

Citation UX requirement:

- Use `hyperref` with `colorlinks=false`.
- Set bright boxed citation borders, for example `citebordercolor={1 0.48 0}`.
- Ensure in-text citations route to bibliography entries.
- Verify with a final LaTeX build and warning scan.

Initial primary-source anchors gathered during planning:

- Implicit Behavioral Cloning, arXiv:2109.00137.
- Set Transformer, arXiv:1810.00825.
- Deep Sets, arXiv:1703.06114.
- Diffusion Policy, arXiv:2303.04137 / IJRR version.
- Action Chunking with Transformers, arXiv:2304.13705.
- MPPI, arXiv:1509.01149.
- PETS, arXiv:1805.12114.
- MuJoCo, Todorov, Erez, and Tassa, IROS 2012.

## 10. Manuscript Rebuild

Replace the current 4-page report with a full ICLR-style paper.

Minimum contents:

- Abstract that states the terminal decision honestly.
- Introduction with hostile-review framing.
- Formal problem and notation.
- Method with architecture, loss, shield, and computational profile.
- Theory section.
- Experimental protocol.
- Main results.
- Strong-baseline paired comparisons.
- Ablations.
- Stress tests.
- Failure cases and negative results.
- Related work.
- Limitations.
- Reproducibility checklist.
- Appendix with hyperparameters, branch definitions, additional tables, and per-split plots.

Length requirement:

- Build to at least 25 pages without filler.
- If the final evidence remains mixed, the 25 pages should still be valuable as a rigorous negative/strong-revise submission package.

## 11. Validation and Artifact Gates

Before committing:

- `python -m py_compile src/run_experiment.py`
- Run final frozen experiment or resume to completion.
- Run analysis scripts and generate all tables/figures.
- Build PDF with enough LaTeX passes to resolve citations and references.
- Verify `paper/main.pdf` has at least 25 pages.
- Copy only to `C:/Users/wangz/Downloads/62.pdf`.
- Verify `C:/Users/wangz/Desktop/62.pdf` is absent.
- Verify Downloads PDF hash or byte size matches `paper/main.pdf`.
- Run a LaTeX log scan for undefined citations, undefined references, fatal errors, and label-change warnings.
- Run `git diff --check`.
- Update `README.md`, `child_status.md`, `docs/final_audit.md`, `docs/submission_readiness_decision.md`, `docs/experiment_rigor_checklist.md`, and root ledgers.
- Commit and push to the public GitHub repository.

## 12. Terminal Decision Gates

`ICLR_MAIN_TARGET_READY` requires all of:

- RC-RET beats or statistically ties while improving safety over MLP, Deep Sets, nominal MPC, and robust MPC on the aggregate robust gate.
- RC-RET does not increase violation rate against robust MPC on the combined/narrow-clearance stress gates.
- Ablations show self-attention, robust-risk features, and safety calibration matter.
- Related work supports a credible novelty boundary.
- Final PDF is 25+ pages, reproducible, and citation-clean.

`STRONG_REVISE` if:

- Evidence is real and rigorous but the proposed method has mixed gains, unclear ablations, insufficient novelty, or custom-only validation.

`KILL_ARCHIVE` if:

- The learned method cannot beat strong baselines in any defensible way, violates safety more often, or requires unfair oracle information.

