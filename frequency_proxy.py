import numpy as np

from common import SimParams, clamp


def run_frequency_proxy(
    t: np.ndarray, p_draw_total_pu_on_dc_base: np.ndarray, p: SimParams, *, n_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Swing-equation frequency proxy:
      2H dx/dt = (P_m - P_e) - D x
    with x=(f-f0)/f0 and P_e driven by aggregated data-center draw mapped onto system base.
    """
    f0 = 60.0
    p_e_pu_sys = (n_blocks * p.base_mw / p.sys_base_mw) * p_draw_total_pu_on_dc_base
    # Pre-fault: assume the bulk system is balanced (Pm tracks Pe), then hold Pm constant
    # through the fault window to visualize frequency deviation caused by load behavior.
    k_hold = int(max(0, np.searchsorted(t, p.fault_start_s) - 1))
    p_m_hold = float(p_e_pu_sys[k_hold])
    x = 0.0
    f_hist = np.zeros_like(t)
    for k in range(t.size):
        dt = 0.0 if k == 0 else float(t[k] - t[k - 1])
        p_e = float(p_e_pu_sys[k])
        p_m = float(p_e) if k < k_hold else p_m_hold
        dx = ((p_m - p_e) - p.sys_damping_pu * x) / (2.0 * p.sys_inertia_h_s)
        x += dx * dt
        f_hist[k] = f0 * (1.0 + x)
    return t, f_hist


def run_frequency_event_with_droop(
    p: SimParams,
    *,
    t: np.ndarray,
    n_blocks: int,
    enable_droop: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal Stage-3 illustration: apply a system-base Pm step and optionally add droop-based load relief.

    Returns (t, f_hz, p_support_pu_sys)
    """
    f0 = 60.0
    dt = float(t[1] - t[0])
    x = 0.0
    f_hist = np.zeros_like(t)
    p_support = np.zeros_like(t)

    p_e0_pu_sys = (n_blocks * p.base_mw / p.sys_base_mw) * 1.0
    p_m0 = p_e0_pu_sys

    for k in range(t.size):
        t_now = float(t[k])
        p_m = p_m0 + (p.sys_event_step_pu if t_now >= p.sys_event_t_s else 0.0)
        p_supp = 0.0
        if enable_droop:
            p_supp = clamp(p.vsm_droop_gain_pu_per_pu * (-x), 0.0, p.sys_ffr_p_max_pu)
        p_e = p_e0_pu_sys - p_supp
        dx = ((p_m - p_e) - p.sys_damping_pu * x) / (2.0 * p.sys_inertia_h_s)
        x += dx * dt
        f_hist[k] = f0 * (1.0 + x)
        p_support[k] = p_supp
    return t, f_hist, p_support
