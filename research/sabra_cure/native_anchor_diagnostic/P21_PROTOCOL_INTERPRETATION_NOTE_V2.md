# P21 Protocol Interpretation Note V2

This pre-execution correction supersedes only the solver sentence in
`P21_PROTOCOL_INTERPRETATION_NOTE.md`. The P21 execution environment has no
SciPy installation, so SciPy `L-BFGS-B` cannot be a reproducible frozen
implementation. No P21 scientific outcome, attempt marker, or execution-base
artifact exists at this correction.

P1/P2 use CPU-only PyTorch `torch.optim.LBFGS` on float64 tensors, with
all-zero initialization, `lr=1.0`, `max_iter=500`, `max_eval=1000`,
`history_size=10`, `tolerance_grad=1e-10`, `tolerance_change=1e-15`, and
`line_search_fn="strong_wolfe"`. `torch.set_num_threads(1)` is set before
fitting. The already frozen all-pair summed logistic objective and L2
coefficient `1.0` are unchanged. CUDA RankNet remains disabled.

No other part of the original preregistration or interpretation note changes.
