import math
from dataclasses import dataclass

import numpy as np

from common import SimParams, _in_any_fault_window, clamp, pq_from_v_i, rl_from_scr_xr, v_mag, v_th_abc_asymmetric
from controllers import case_gfl_pll_lvrt, case_gfl_standard, case_gfm_proposed
from emt_utils import (
    EMTPlantParams,
    abc_from_vmag_theta,
    clarke_abc_to_alpha_beta,
    inv_clarke_alpha_beta_to_abc,
    inv_park_dq_to_alpha_beta,
    park_alpha_beta_to_dq,
    pwm_switch_abc,
    solve_series_rl_step,
)


@dataclass(frozen=True)
class EMTResult:
    t: np.ndarray
    i_abc: np.ndarray  # shape (3, N)
    v_pcc_abc: np.ndarray  # shape (3, N)
    v_src_abc: np.ndarray  # shape (3, N)
    v_inv_abc: np.ndarray  # shape (3, N)
    # controller state history (optional; depends on case)
    y_ctrl: np.ndarray  # shape (n_state, N)


def _dq_from_abc(v_abc: np.ndarray, theta: float) -> tuple[float, float]:
    v_alpha, v_beta = clarke_abc_to_alpha_beta(float(v_abc[0]), float(v_abc[1]), float(v_abc[2]))
    return park_alpha_beta_to_dq(v_alpha, v_beta, theta)


def _abc_from_dq(v_d: float, v_q: float, theta: float) -> np.ndarray:
    v_alpha, v_beta = inv_park_dq_to_alpha_beta(v_d, v_q, theta)
    va, vb, vc = inv_clarke_alpha_beta_to_abc(v_alpha, v_beta)
    return np.array([va, vb, vc], dtype=float)


def run_emt_case(
    case: str,
    p: SimParams,
    *,
    dt: float = 50e-6,
    t_end_s: float = 1.2,
    ds_rate: int = 10,
    max_inner_iters: int = 1,
) -> EMTResult:
    """
    abc-frame (fixed-step) simulation of the same high-level controller cases.

    Important scope:
      - Without enable_switching: the inverter is a commanded voltage source behind a series RL plant.
      - With enable_switching: carrier-based SPWM is applied to the voltage command, producing
        switched ±V_dc/2 per phase. An anti-aliasing measurement filter (500 Hz cutoff) is applied
        to abc measurements before the dq transformation, matching real digital controller practice.
      - The grid Thevenin is modeled as an RL source (derived from SCR/XR); no LC filter is included.
      - Integration is fixed-step forward Euler.

    The controller dynamics reuse the averaged controller ODEs (GFL/GFM) evaluated in a local dq frame,
    while the plant is stepped in abc. This provides an intermediate fidelity step between fundamental dq
    phasor models and full switching EMT with LC filter dynamics.
    """
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if ds_rate < 1:
        raise ValueError("ds_rate must be >= 1")

    r_th, x_th = rl_from_scr_xr(p.scr, p.xr_ratio)
    plant = EMTPlantParams(r_th=r_th, x_th=x_th, r_f=p.r_f_pu, x_f=p.x_f_pu)

    n_steps = int(math.floor(t_end_s / dt)) + 1
    keep = (n_steps + ds_rate - 1) // ds_rate
    t_hist = np.zeros(keep)
    i_hist = np.zeros((3, keep))
    v_pcc_hist = np.zeros((3, keep))
    v_src_hist = np.zeros((3, keep))
    v_inv_hist = np.zeros((3, keep))

    if case == "GFL-MC":
        y = np.array([0.0, 0.0, 0.0, 0.0, p.v_dc_init_pu], dtype=float)
    elif case == "GFL-PLL":
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p.v_dc_init_pu], dtype=float)
    elif case == "GFM":
        # Match the averaged-dq case initialization: start from zero current and let the grid-draw
        # shaping/ramp limits bring the operating point up before the fault window.
        base_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p.soc_init, p.v_dc_init_pu, 0.0]
        if p.neg_seq_suppress and p.fault_type != "balanced":
            base_states.extend([1.0, 0.0])  # vpcc_d_avg=1.0 (nominal), vpcc_q_avg=0.0
        y = np.array(base_states, dtype=float)
    else:
        raise ValueError("case must be one of: GFL-MC, GFL-PLL, GFM")

    y_hist = np.zeros((y.size, keep))

    # abc current state (defined positive inverter -> grid)
    i_abc = np.zeros(3, dtype=float)
    # initialize from dq current if provided
    theta0 = 0.0
    i_alpha0, i_beta0 = inv_park_dq_to_alpha_beta(float(y[0]), float(y[1]), theta0)
    ia0, ib0, ic0 = inv_clarke_alpha_beta_to_abc(i_alpha0, i_beta0)
    i_abc[:] = np.array([ia0, ib0, ic0], dtype=float)

    v_pcc_abc = np.array(abc_from_vmag_theta(1.0, theta0), dtype=float)

    # Anti-aliasing measurement filter (first-order LPF on abc before dq transformation).
    # Every real digital controller has this; cutoff well below f_sw but above fundamental.
    # alpha_meas = dt / (dt + 1/(2*pi*f_cutoff)); f_cutoff ~ 500 Hz for f_sw = 3060 Hz.
    meas_filter_cutoff_hz = 500.0
    tau_meas = 1.0 / (2.0 * math.pi * meas_filter_cutoff_hz)
    alpha_meas = dt / (dt + tau_meas)
    v_pcc_filt = v_pcc_abc.copy()
    i_filt = i_abc.copy()

    idx = 0
    for k in range(n_steps):
        t = k * dt
        theta_grid = 2.0 * math.pi * 60.0 * t

        # Source voltage scheduling: supports balanced and asymmetric faults
        in_fault = _in_any_fault_window(t, p)
        if in_fault and p.fault_type != "balanced":
            va, vb, vc = v_th_abc_asymmetric(theta_grid, p.fault_type, p.fault_v_pu)
            v_src_abc = np.array([va, vb, vc], dtype=float)
        else:
            v_src_mag = p.fault_v_pu if in_fault else 1.0
            v_src_abc = np.array(abc_from_vmag_theta(v_src_mag, theta_grid), dtype=float)

        # Fixed-point refinement: v_inv depends on measured v_pcc, which depends on v_inv.
        # IMPORTANT: do not advance controller states inside the fixed-point loop, otherwise the
        # controller dynamics (e.g., ramp limits) are integrated multiple times per time step.
        v_inv_abc = np.zeros(3, dtype=float)
        di_dt = np.zeros(3, dtype=float)
        dy = None
        for _ in range(max_inner_iters + 1):
            # Update anti-aliasing measurement filters (LPF on abc before dq transformation).
            # Use filtered signals for controller, raw signals for plant dynamics.
            if p.enable_switching:
                v_pcc_filt += alpha_meas * (v_pcc_abc - v_pcc_filt)
                i_filt += alpha_meas * (i_abc - i_filt)
                v_meas = v_pcc_filt
                i_meas = i_filt
            else:
                v_meas = v_pcc_abc
                i_meas = i_abc

            # compute dq measurements (grid-aligned by default; PLL case uses internal theta state)
            if case == "GFL-PLL":
                theta_pll = float(y[4])
                vd_pcc, vq_pcc = _dq_from_abc(v_meas, theta_pll)
                id_grid, iq_grid = _dq_from_abc(i_meas, theta_pll)
            else:
                vd_pcc, vq_pcc = _dq_from_abc(v_meas, theta_grid)
                id_grid, iq_grid = _dq_from_abc(i_meas, theta_grid)

            # map dq measurements into the controller state vector (id/iq are the first two states)
            y_local = y.copy()
            y_local[0] = id_grid
            y_local[1] = iq_grid

            # Call the corresponding averaged controller ODE to compute v_inv_dq "as if" the plant were dq.
            # We emulate this by performing a short Euler step to update controller states, then recovering
            # the voltage command via the same internal algebra used in the controller functions.
            #
            # To keep this lightweight and consistent, we reuse the existing ODE functions and treat the
            # inverter voltage command as: v_inv = v_pcc + feedforward + PI(i*-i), then convert to abc.
            #
            # For this EMT wrapper, we reconstruct v_pcc in dq directly from measured abc, and use the
            # same filter parameters (r_f, x_f) and current controller gains.

            # Evaluate the case ODE derivative (updates xi, vdc, etc.).
            if case == "GFL-MC":
                dy = np.asarray(case_gfl_standard(t, y_local, p), dtype=float)
            elif case == "GFL-PLL":
                dy = np.asarray(case_gfl_pll_lvrt(t, y_local, p), dtype=float)
            else:
                dy = np.asarray(case_gfm_proposed(t, y_local, p), dtype=float)

            # Recompute the voltage command by re-evaluating the controller ODE but extracting the
            # same algebraic v_inv computation via a small re-implementation of the inner PI.
            #
            # Rather than duplicating all controller logic, approximate the voltage command as the
            # inner-loop feedforward + PI based on the references implied by the ODE evaluation.
            # Here we approximate i* from the instantaneous derivative of the PI integrators:
            #   d_xi = i* - i  =>  i* ≈ d_xi + i.
            # This keeps reference selection consistent with the existing code paths.
            if case == "GFL-PLL":
                d_xi_id = float(dy[2])
                d_xi_iq = float(dy[3])
                id_ref = d_xi_id + id_grid
                iq_ref = d_xi_iq + iq_grid
            elif case == "GFL-MC":
                d_xi_id = float(dy[2])
                d_xi_iq = float(dy[3])
                id_ref = d_xi_id + id_grid
                iq_ref = d_xi_iq + iq_grid
            else:
                d_xi_id = float(dy[2])
                d_xi_iq = float(dy[3])
                id_ref = d_xi_id + id_grid
                iq_ref = d_xi_iq + iq_grid

            # PI with integrator clamp (match averaged model)
            xi_id_eff = clamp(float(y[2]), -p.xi_lim, p.xi_lim)
            xi_iq_eff = clamp(float(y[3]), -p.xi_lim, p.xi_lim)
            v_pi_d = p.cc_kp * (id_ref - id_grid) + p.cc_ki * xi_id_eff
            v_pi_q = p.cc_kp * (iq_ref - iq_grid) + p.cc_ki * xi_iq_eff

            # dq feedforward for filter in the same dq frame
            v_inv_d = vd_pcc + p.r_f_pu * id_grid - p.x_f_pu * iq_grid + v_pi_d
            v_inv_q = vq_pcc + p.r_f_pu * iq_grid + p.x_f_pu * id_grid + v_pi_q

            # voltage magnitude limit
            v_dc = float(y[7]) if case == "GFM" else float(y[-1])
            v_lim = p.e_max_pu * v_dc if case == "GFM" else p.e_max_pu * v_dc
            v_mag_cmd = v_mag(v_inv_d, v_inv_q)
            if v_mag_cmd > v_lim and v_mag_cmd > 1e-9:
                scale = v_lim / v_mag_cmd
                v_inv_d *= scale
                v_inv_q *= scale

            # convert to abc in the same frame used for measurement
            if case == "GFL-PLL":
                v_inv_abc = _abc_from_dq(v_inv_d, v_inv_q, float(y[4]))
            else:
                v_inv_abc = _abc_from_dq(v_inv_d, v_inv_q, theta_grid)

            # Optional PWM switching model: apply carrier-based SPWM.
            # The state v_dc is a normalized energy proxy (~1.0 pu on AC base).
            # A 2-level VSC has V_dc_actual = 2 * V_ac_base for unity modulation index
            # at rated voltage, so scale accordingly.
            if p.enable_switching:
                v_dc_sw = 2.0 * v_dc  # actual DC bus in AC per-unit
                v_inv_abc = pwm_switch_abc(v_inv_abc, v_dc_sw, t, p.switching_freq_hz)

            # update di/dt and v_pcc estimate for the next fixed-point pass
            _, di_dt, v_pcc_new = solve_series_rl_step(i_abc, v_inv_abc, v_src_abc, plant, dt)
            v_pcc_abc = v_pcc_new

        # advance plant currents using the last command
        i_next, di_dt, v_pcc_abc = solve_series_rl_step(i_abc, v_inv_abc, v_src_abc, plant, dt)
        i_abc = i_next
        if dy is None:
            raise RuntimeError("Internal error: dy not computed in EMT loop")

        # Advance controller states once per fixed step (Euler), leaving id/iq to the plant.
        for j in range(2, y.size):
            y[j] = y[j] + dt * float(dy[j])

        if k % ds_rate == 0:
            t_hist[idx] = t
            i_hist[:, idx] = i_abc
            v_pcc_hist[:, idx] = v_pcc_abc
            v_src_hist[:, idx] = v_src_abc
            v_inv_hist[:, idx] = v_inv_abc
            y_hist[:, idx] = y
            idx += 1

    return EMTResult(
        t=t_hist[:idx],
        i_abc=i_hist[:, :idx],
        v_pcc_abc=v_pcc_hist[:, :idx],
        v_src_abc=v_src_hist[:, :idx],
        v_inv_abc=v_inv_hist[:, :idx],
        y_ctrl=y_hist[:, :idx],
    )
