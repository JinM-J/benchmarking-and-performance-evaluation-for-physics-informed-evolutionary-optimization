# evaluation/metrics/surrogate.py
"""Surrogate quality: grid MSE and mean squared PDE residuals.

R_PDE = mean(abs(N[u_theta](Q))**2). PINN uses autograd; PINO differentiates
its regular output grid and interpolates to valid query points. Operator PIGP
reports full and linear residuals from closed-form kernel derivatives.
Models without a supported derivative path return NaN in this evaluator;
the separate offline finite-difference evaluator supports saved field models."""
import numpy as np


def pde_residual_mse(pinn_surrogate, Q: np.ndarray, batch_size: int = 65536) -> float:
    """Mean squared PDE residual using surrogate derivatives at query points.

    The problem defines physics.residual; this routine only batches evaluation."""
    import torch  # Lazy import: importing metrics does not require torch.

    s = pinn_surrogate
    if not hasattr(s, "_pde_residual") or getattr(s, "model", None) is None:
        return float("nan")

    Q = np.asarray(Q, dtype=float).reshape(-1, s._dim)
    sse, cnt = 0.0, 0
    s.model.eval()
    for i in range(0, Q.shape[0], batch_size):
        j = min(Q.shape[0], i + batch_size)
        Qb = torch.tensor(Q[i:j], dtype=torch.float32, device=s.device)
        r = s._pde_residual(s.model, Qb).reshape(-1)
        finite = torch.isfinite(r)
        if finite.any():
            values = r[finite]
            sse += float(torch.sum(values * values).detach().cpu())
            cnt += int(values.numel())
    return sse / cnt if cnt else float("nan")


def pigp_operator_residuals(surrogate, problem, Q: np.ndarray,
                            batch_size: int = 8192):
    """Return (full_mse, linear_mse) for an operator-informed GP.

    The full residual includes nonlinear terms via physics.residual. The linear
    residual is the covariance-conditioning operator minus its right-hand side.
    Requires predict_with_derivatives; kernel derivatives are evaluated analytically."""
    import torch  # Some problem residuals use torch internally.

    names = problem.physics.coordinate_names
    lin_op = problem.physics.linear_operator()
    Q = np.asarray(Q, dtype=float).reshape(-1, len(names))
    sse_f, sse_l, cnt = 0.0, 0.0, 0
    for i in range(0, Q.shape[0], batch_size):
        Qb = Q[i:i + batch_size]
        u, d = surrogate.predict_with_derivatives(Qb)

        # Full problem residual on torch columns of shape (N,1).
        Qt = torch.tensor(Qb, dtype=torch.float64)
        ut = torch.tensor(u.reshape(-1, 1), dtype=torch.float64)
        dt = {k: torch.tensor(np.asarray(v).reshape(-1, 1),
                              dtype=torch.float64)
              for k, v in d.items()}
        r_full = problem.physics.residual(Qt, ut, dt)
        r_full = np.asarray(r_full.detach().cpu(), dtype=float).reshape(-1)

        # Assemble linear terms; order zero denotes u and absent rhs denotes zero.
        r_lin = np.zeros(Qb.shape[0])
        for term in lin_op.terms:
            key = tuple(nm for nm, o in zip(names, term.deriv)
                        for _ in range(int(o)))
            val = u if not key else np.asarray(d[key], dtype=float)
            c = term.coef(Qb) if callable(term.coef) else term.coef
            r_lin = (r_lin + np.asarray(c, dtype=float).reshape(-1)
                     * np.asarray(val, dtype=float).reshape(-1))
        if lin_op.rhs is not None:
            r_lin = r_lin - np.asarray(lin_op.rhs(Qb), dtype=float).reshape(-1)

        sse_f += float(r_full @ r_full)
        sse_l += float(r_lin @ r_lin)
        cnt += Qb.shape[0]
    if cnt == 0:
        return float("nan"), float("nan")
    return sse_f / cnt, sse_l / cnt
