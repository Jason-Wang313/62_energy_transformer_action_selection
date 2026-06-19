# Submission Readiness Decision

Decision: STRONG_REVISE.

ICLR main-conference readiness: NO.

Submission-hardening version: v5 frozen RC-RET rebuild.

Reason: the paper now has a real 25-page ICLR-style manuscript, formal theory, manual primary-source bibliography, bright boxed clickable citations, a CPU-only MuJoCo/PyTorch frozen benchmark, stronger baselines, ten stress splits, 11,520 main rows, 2,112 ablation rows, paired statistics, stress figures, and reproducibility scripts. However, RC-RET does not clear the success gate against MLP, Deep Sets, robust MPC, nominal MPC, or the oracle. Ablations support risk and feasibility but do not isolate attention as decisive.

Honest terminal action: do not submit to ICLR main in this form. Preserve as a rigorous strong-revise evidence package.

Revival condition: add richer primitive generation, online latent/contact inference, public benchmark or hardware validation, and a decisive win over shielded MLP, Deep Sets, and robust MPC without increasing violations.
