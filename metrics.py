import numpy as np

from common import SimParams, pq_from_v_i, v_th_dq_grid_frame
from dq_plant import v_pcc_from_grid_impedance


def _time_mask_for_fault(t: np.ndarray, p: SimParams) -> np.ndarray:
    return (t >= p.fault_start_s) & (t <= p.fault_end_s)


def compute_stage1_unserved_energy_mwh(sol, p: SimParams, *, is_gfm: bool) -> float:
    """
    Proxy for continuity to IT load during the fault window:
      - For GFL benchmark: P_supplied_to_IT = P_draw_from_grid (no coordinated BESS supply).
      - For GFM: P_supplied_to_IT = P_draw + P_bess.

    Unserved energy integrates max(0, P_load - P_supplied_to_IT).
    """
    t = sol.t
    y = sol.y
    id_grid, iq_grid = y[0], y[1]
    p_bess = y[5] if is_gfm else np.zeros_like(id_grid)

    d = p.derived()
    p_load = np.array([1.0 if ti >= p.load_step_t_s else 0.0 for ti in t], dtype=float)
    p_supplied = np.zeros_like(t, dtype=float)
    for k in range(t.size):
        v_th_d, v_th_q = v_th_dq_grid_frame(float(t[k]), p)
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
            v_th_d, v_th_q, float(id_grid[k]), float(iq_grid[k]), d["r_th"], d["x_th"]
        )
        p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, float(id_grid[k]), float(iq_grid[k]))
        p_draw = -p_grid
        p_supplied[k] = p_draw + float(p_bess[k])

    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    unserved_pu_s = np.trapz(np.maximum(0.0, p_load[mask] - p_supplied[mask]), t[mask])
    unserved_mwh = (unserved_pu_s / 3600.0) * p.base_mw
    return float(unserved_mwh)


def compute_fault_min_vdc(sol, p: SimParams, vdc_index: int) -> float:
    t = sol.t
    y = sol.y
    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    return float(np.min(y[vdc_index][mask]))


def compute_fault_min_vpcc(sol, p: SimParams) -> float:
    t = sol.t
    y = sol.y
    id_grid, iq_grid = y[0], y[1]
    d = p.derived()
    vals = np.zeros_like(t, dtype=float)
    for k in range(t.size):
        v_th_d, v_th_q = v_th_dq_grid_frame(float(t[k]), p)
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
            v_th_d, v_th_q, float(id_grid[k]), float(iq_grid[k]), d["r_th"], d["x_th"]
        )
        vals[k] = float(np.sqrt(v_pcc_d * v_pcc_d + v_pcc_q * v_pcc_q))
    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    return float(np.min(vals[mask]))


def compute_fault_min_vpcc_settled(sol, p: SimParams, *, settle_s: float) -> float:
    """
    Minimum |Vpcc| during the fault window, excluding an initial settling interval after fault start.

    settle_s: time to exclude after p.fault_start_s (seconds).
    """
    t = sol.t
    y = sol.y
    id_grid, iq_grid = y[0], y[1]
    d = p.derived()
    vals = np.zeros_like(t, dtype=float)
    for k in range(t.size):
        v_th_d, v_th_q = v_th_dq_grid_frame(float(t[k]), p)
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
            v_th_d, v_th_q, float(id_grid[k]), float(iq_grid[k]), d["r_th"], d["x_th"]
        )
        vals[k] = float(np.sqrt(v_pcc_d * v_pcc_d + v_pcc_q * v_pcc_q))
    t0 = p.fault_start_s + max(0.0, float(settle_s))
    mask = (t >= t0) & (t <= p.fault_end_s)
    if not np.any(mask):
        return float("nan")
    return float(np.min(vals[mask]))


def compute_fault_max_current(sol, p: SimParams) -> float:
    t = sol.t
    y = sol.y
    id_grid, iq_grid = y[0], y[1]
    i_mag = np.sqrt(id_grid * id_grid + iq_grid * iq_grid)
    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    return float(np.max(i_mag[mask]))


def compute_fault_mean_pdraw(sol, p: SimParams) -> float:
    t = sol.t
    y = sol.y
    id_grid, iq_grid = y[0], y[1]
    d = p.derived()
    p_draw = np.zeros_like(t, dtype=float)
    for k in range(t.size):
        v_th_d, v_th_q = v_th_dq_grid_frame(float(t[k]), p)
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
            v_th_d, v_th_q, float(id_grid[k]), float(iq_grid[k]), d["r_th"], d["x_th"]
        )
        p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, float(id_grid[k]), float(iq_grid[k]))
        p_draw[k] = -p_grid
    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(p_draw[mask]))


def compute_fault_metrics(sol_gfl, sol_gfm, p: SimParams) -> dict[str, dict[str, float]]:
    return {
        "gfl": {
            "min_vpcc": compute_fault_min_vpcc(sol_gfl, p),
            "max_i": compute_fault_max_current(sol_gfl, p),
            "min_vdc": compute_fault_min_vdc(sol_gfl, p, vdc_index=4),
            "unserved_mwh": compute_stage1_unserved_energy_mwh(sol_gfl, p, is_gfm=False),
        },
        "gfm": {
            "min_vpcc": compute_fault_min_vpcc(sol_gfm, p),
            "max_i": compute_fault_max_current(sol_gfm, p),
            "min_vdc": compute_fault_min_vdc(sol_gfm, p, vdc_index=7),
            "unserved_mwh": compute_stage1_unserved_energy_mwh(sol_gfm, p, is_gfm=True),
        },
    }


def compute_fault_metrics_multi(solutions: dict[str, tuple[object, int]], p: SimParams) -> dict[str, dict[str, float]]:
    """
    Compute a common fault-window metric set for multiple solutions.

    solutions: mapping name -> (sol, vdc_index)
      - vdc_index indicates which state index corresponds to Vdc in that solution.
    """
    out: dict[str, dict[str, float]] = {}
    for name, (sol, vdc_index) in solutions.items():
        out[name] = {
            "min_vpcc": compute_fault_min_vpcc(sol, p),
            "max_i": compute_fault_max_current(sol, p),
            "min_vdc": compute_fault_min_vdc(sol, p, vdc_index=vdc_index),
            "mean_pdraw": compute_fault_mean_pdraw(sol, p),
        }
    return out


def compute_postfault_max_dpdraw_ref(sol, p: SimParams, *, pdraw_ref_index: int, window_s: float = 0.25) -> float:
    """
    Post-fault ramp stress metric using the controller reference state Pdraw,ref:
    maximum absolute d(Pdraw,ref)/dt (pu/s) in a window after fault clearing.
    """
    t = sol.t
    if t.size < 3:
        return float("nan")

    p_ref = sol.y[pdraw_ref_index]
    dt = np.diff(t)
    dp = np.diff(p_ref)
    dpdt = np.abs(dp / np.maximum(dt, 1e-9))

    t_mid = t[:-1] + 0.5 * dt
    t0 = p.fault_end_s
    t1 = p.fault_end_s + window_s
    mask = (t_mid >= t0) & (t_mid <= t1)
    if not np.any(mask):
        return float("nan")
    return float(np.max(dpdt[mask]))


def compute_fault_max_abs_pbess(sol, p: SimParams, *, pbess_index: int, absolute: bool = True) -> float:
    """
    Maximum (absolute) UPS-BESS power during the fault window.

    pbess_index: state index of P_bess in the solution.
    """
    t = sol.t
    y = sol.y
    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    pbess = y[pbess_index][mask]
    if pbess.size == 0:
        return float("nan")
    if absolute:
        return float(np.max(np.abs(pbess)))
    return float(np.max(pbess))


def compute_fault_min_soc(sol, p: SimParams, *, soc_index: int) -> float:
    """Minimum SoC proxy during the fault window."""
    t = sol.t
    y = sol.y
    mask = _time_mask_for_fault(t, p)
    if not np.any(mask):
        return float("nan")
    soc = y[soc_index][mask]
    if soc.size == 0:
        return float("nan")
    return float(np.min(soc))


def compute_postfault_recovery_time_pdraw(
    sol,
    p: SimParams,
    *,
    tol_pu: float = 0.05,
    min_hold_s: float = 0.05,
) -> float:
    """
    Time after fault clearing for P_draw (computed from Vpcc and currents) to enter and stay within
    tol_pu of the load level (1.0 pu after load step) for at least min_hold_s.

    Returns NaN if it never recovers within the simulation horizon.
    """
    t, p_draw = compute_time_series_pdraw(sol, p)
    p_load = np.array([1.0 if ti >= p.load_step_t_s else 0.0 for ti in t], dtype=float)

    t0 = p.fault_end_s
    start = int(np.searchsorted(t, t0))
    if start >= t.size:
        return float("nan")

    dt = np.diff(t)
    within = np.abs(p_draw - p_load) <= tol_pu

    hold_steps = 1
    if t.size > 1:
        hold_steps = int(np.ceil(min_hold_s / max(1e-9, float(np.median(dt)))))
        hold_steps = max(1, hold_steps)

    for k in range(start, t.size):
        if not within[k]:
            continue
        k_end = min(t.size, k + hold_steps)
        if np.all(within[k:k_end]):
            return float(t[k] - t0)
    return float("nan")


def compute_time_series_pdraw(sol, p: SimParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time series of Pdraw (pu on data-center base) from solution currents and PCC voltage.
    """
    t = sol.t
    y = sol.y
    id_grid, iq_grid = y[0], y[1]
    d = p.derived()
    p_draw = np.zeros_like(t, dtype=float)
    for k in range(t.size):
        v_th_d, v_th_q = v_th_dq_grid_frame(float(t[k]), p)
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
            v_th_d, v_th_q, float(id_grid[k]), float(iq_grid[k]), d["r_th"], d["x_th"]
        )
        p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, float(id_grid[k]), float(iq_grid[k]))
        p_draw[k] = -p_grid
    return t, p_draw
