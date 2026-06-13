# Submission Readiness Decision

Decision: STRONG_REVISE.

ICLR main-conference readiness: NO.

Submission-hardening version: v4 real MuJoCo/PyTorch rebuild.

Reason: the paper now has real MuJoCo rollout labels, a trained PyTorch energy scorer, implemented baselines, ablations, stress tests, and uncertainty summaries. However, the transformer energy scorer does not consistently outperform MLP energy, nominal rollout MPC, or robust worst-case MPC. The self-attention mechanism is not isolated as necessary, and no hardware/public-benchmark evidence exists.

Honest terminal action: do not submit to ICLR main in this form. Keep as a strong-revise empirical scaffold.

Revival condition: show statistically clear gains over MLP/robust MPC, reduce violation rate, add external benchmarks or hardware, and perform manual robotics related-work synthesis.
