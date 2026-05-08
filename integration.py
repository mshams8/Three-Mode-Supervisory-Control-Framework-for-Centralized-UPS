import math
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp


def rk2_step(f: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + dt, y + dt * k1)
    return y + 0.5 * dt * (k1 + k2)


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_case(
    f: Callable[[float, np.ndarray], list[float]],
    y0: list[float],
    t_end_s: float,
    points: int,
    *,
    dt: float | None = None,
    rk: str = "rk4",
    ds_rate: int = 1,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_step: float = 1e-3,
):
    """
    Integrate the provided ODE either using SciPy's solve_ivp or a deterministic fixed-step RK method.

    Returns:
      - solve_ivp result if dt is None
      - SimpleNamespace(t, y, success, message) if dt is set
    """
    if dt is None:
        t_eval = np.linspace(0.0, t_end_s, points)
        return solve_ivp(
            lambda t, y: f(t, y),
            t_span=(0.0, t_end_s),
            y0=np.asarray(y0, dtype=float),
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

    if dt <= 0.0:
        raise ValueError("dt must be > 0 when using fixed-step integration")
    if ds_rate < 1:
        raise ValueError("ds_rate must be >= 1")
    stepper = rk4_step if rk == "rk4" else rk2_step

    y = np.asarray(y0, dtype=float)
    n_states = y.size
    n_steps = int(math.floor(t_end_s / dt)) + 1
    keep = (n_steps + ds_rate - 1) // ds_rate
    t_hist = np.zeros(keep)
    y_hist = np.zeros((n_states, keep))

    def rhs(t_now: float, y_now: np.ndarray) -> np.ndarray:
        return np.asarray(f(t_now, y_now), dtype=float)

    idx = 0
    for k in range(n_steps):
        t = k * dt
        if k % ds_rate == 0:
            t_hist[idx] = t
            y_hist[:, idx] = y
            idx += 1
        y = stepper(rhs, t, y, dt)
        if not np.all(np.isfinite(y)):
            return SimpleNamespace(t=t_hist[:idx], y=y_hist[:, :idx], success=False, message="State became non-finite")

    return SimpleNamespace(t=t_hist[:idx], y=y_hist[:, :idx], success=True, message="OK")

