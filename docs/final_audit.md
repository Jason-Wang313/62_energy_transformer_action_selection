# Final Audit

Paper: `62_energy_transformer_action_selection`

Date: 2026-06-19

## Terminal Decision

Decision: STRONG_REVISE.

ICLR main readiness: no.

Submission-hardening version: v5 frozen RC-RET rebuild.

## Evidence

- Main raw rows: 11,520.
- Ablation raw rows: 2,112.
- Metric rows: 120.
- Ablation metric rows: 22.
- Pairwise rows: 99.
- Training action sets: 480.
- Candidate actions per episode: 63.
- Stress splits: 10.
- Main methods: 12.
- Ablation methods: 11.

## Main Result

RC-RET is close to robust/CVaR branch scoring and improves safety relative to several unshielded learned baselines. It does not clear the success gate against MLP energy, Deep Sets, robust MPC, nominal MPC, or the oracle. Narrow-clearance cases expose a finite-candidate limitation.

## Verification

- `python -m py_compile src\run_experiment.py` passed.
- Final frozen run completed with expected row counts.
- `python scripts\render_latex_tables.py` passed.
- `python scripts\summarize_final_decision.py` passed.
- `powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1` produced `paper/main.pdf` and `C:/Users/wangz/Downloads/62.pdf`.
- `python scripts\validate_submission_artifacts.py` passed.
- LaTeX scan found no undefined citations/references or fatal warnings.
- `C:/Users/wangz/Desktop/62.pdf` is absent.

## Artifact Paths

- Repo PDF: `paper/main.pdf`
- Canonical Downloads PDF: `C:/Users/wangz/Downloads/62.pdf`
- Final evidence summary: `docs/paper62_final_v5_evidence_summary.md`
- Protocol freeze: `docs/paper62_protocol_freeze_20260619.md`
- Development log: `docs/paper62_development_log.md`
