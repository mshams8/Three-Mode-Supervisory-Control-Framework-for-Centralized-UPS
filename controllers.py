import math

import numpy as np

from common import (
    W_BASE,
    SimParams,
    _last_fault_end_before,
    clamp,
    p_load_profile,
    pq_from_v_i,
    v_mag,
    v_th_dq_grid_frame,
    v_th_sequence_magnitudes,
)
from dq_plant import plant_dynamics_grid_frame, v_pcc_from_grid_impedance


# ---------------------------------------------------------------------------
# Negative-sequence extraction / suppression helpers
# ---------------------------------------------------------------------------

def _extract_pos_neg_dq(vd: float, vq: float, t: float,
                        vd_avg: float, vq_avg: float,
                        alpha: float = 50.0) -> tuple[float, float, float, float]:
    """
    Approximate positive/negative sequence decomposition in dq frame.

    In the synchronous dq frame, the positive sequence is the DC component
    and the negative sequence oscillates at 2*omega. A moving-average or
    low-pass filter extracts the DC (positive seq); the residual is the
    negative sequence.

    This function uses an exponential moving-average state (vd_avg, vq_avg)
    with bandwidth alpha (rad/s) to track the positive-sequence component.

    Returns:
        (vd_pos, vq_pos, vd_neg, vq_neg)
    """
    vd_pos = vd_avg
    vq_pos = vq_avg
    vd_neg = vd - vd_pos
    vq_neg = vq - vq_pos
    return vd_pos, vq_pos, vd_neg, vq_neg


def _neg_seq_filter_derivatives(vd: float, vq: float,
                                vd_avg: float, vq_avg: float,
                                alpha: float = 50.0) -> tuple[float, float]:
    """
    Time derivatives for the positive-sequence extraction filter states.
    Simple first-order LPF: d(v_avg)/dt = alpha * (v - v_avg)
    """
    return alpha * (vd - vd_avg), alpha * (vq - vq_avg)


def case_gfl_standard(t: float, y: np.ndarray, p: SimParams) -> list[float]:
    """
    Grid-following benchmark (simplified, averaged):
      - Grid-aligned dq current control with a constant-power outer loop,
      - "Momentary cessation"/protection behavior in deep voltage dips.

    Notes:
      - This is not a detailed PLL interaction model. It is intended as a simple benchmark
        for the disconnection/momentary-cessation behavior highlighted in the paper.

    State: [id_grid, iq_grid, xi_id, xi_iq, v_dc]
    """
    id_grid, iq_grid, xi_id, xi_iq, v_dc = y
    d = p.derived()
    v_th_d, v_th_q = v_th_dq_grid_frame(t, p)
    v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(v_th_d, v_th_q, id_grid, iq_grid, d["r_th"], d["x_th"])
    v_pcc_mag = v_mag(v_pcc_d, v_pcc_q)

    # Outer loop: scheduled active power draw (load), i.e., P_ref is negative (inverter absorbs).
    p_load = p_load_profile(t, p)
    p_ref = -p_load
    q_ref = 0.0

    if v_pcc_mag < p.v_mc_pu:
        # Benchmark "momentary cessation" / protection behavior: cease current injection.
        id_ref = 0.0
        iq_ref = 0.0
    else:
        # Use a voltage-magnitude proxy for P<->I conversion to avoid sensitivity
        # to the dq reference angle under weak-grid phase shifts.
        v_for_p = max(0.05, v_pcc_mag)
        id_ref = p_ref / (1.5 * v_for_p)
        iq_ref = -q_ref / (1.5 * v_for_p)

        i_ref_mag = math.sqrt(id_ref * id_ref + iq_ref * iq_ref)
        if i_ref_mag > p.i_max_pu:
            scale = p.i_max_pu / i_ref_mag
            id_ref *= scale
            iq_ref *= scale

    xi_id_eff = clamp(xi_id, -p.xi_lim, p.xi_lim)
    xi_iq_eff = clamp(xi_iq, -p.xi_lim, p.xi_lim)

    # dq current PI in grid frame, with decoupling and PCC voltage feed-forward
    v_pi_d = p.cc_kp * (id_ref - id_grid) + p.cc_ki * xi_id_eff
    v_pi_q = p.cc_kp * (iq_ref - iq_grid) + p.cc_ki * xi_iq_eff
    v_inv_d = v_pcc_d + d["r_f"] * id_grid - d["x_f"] * iq_grid + v_pi_d
    v_inv_q = v_pcc_q + d["r_f"] * iq_grid + d["x_f"] * id_grid + v_pi_q

    # Voltage limiting (simple circular clamp)
    # AC voltage capability is bounded by DC-link voltage (proxy): |v_inv| <= E_max * V_dc.
    # Use the same E_max across all cases for apples-to-apples comparison.
    v_lim = p.e_max_pu * v_dc
    v_inv_mag = v_mag(v_inv_d, v_inv_q)
    if v_inv_mag > v_lim:
        scale = v_lim / v_inv_mag
        v_inv_d *= scale
        v_inv_q *= scale

    did_dt, diq_dt = plant_dynamics_grid_frame(
        id_grid, iq_grid, v_inv_d, v_inv_q, v_pcc_d, v_pcc_q, d["r_f"], d["x_f"]
    )

    d_xi_id = id_ref - id_grid
    d_xi_iq = iq_ref - iq_grid

    # Simple integrator clamping (anti-windup proxy)
    if xi_id > p.xi_lim and d_xi_id > 0:
        d_xi_id = 0.0
    if xi_id < -p.xi_lim and d_xi_id < 0:
        d_xi_id = 0.0
    if xi_iq > p.xi_lim and d_xi_iq > 0:
        d_xi_iq = 0.0
    if xi_iq < -p.xi_lim and d_xi_iq < 0:
        d_xi_iq = 0.0
    d_xi_id += p.xi_recover_k * (xi_id_eff - xi_id)
    d_xi_iq += p.xi_recover_k * (xi_iq_eff - xi_iq)

    # DC-link energy proxy: d(0.5*Vdc^2)/dt = (P_in - P_load)/T_dc
    p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, id_grid, iq_grid)
    p_draw = -p_grid  # positive draw from grid
    p_in = p_draw  # no coordinated BESS in benchmark
    e_dot = (p_in - p_load) / max(1e-6, p.dc_link_energy_s)
    dv_dc = e_dot / max(p.v_dc_min_pu, v_dc)
    dv_dc = clamp(dv_dc, -50.0, 50.0)
    if v_dc <= p.v_dc_min_pu and dv_dc < 0.0:
        dv_dc = 0.0
    if v_dc >= p.v_dc_max_pu and dv_dc > 0.0:
        dv_dc = 0.0

    return [did_dt, diq_dt, d_xi_id, d_xi_iq, dv_dc]


def _rot(vd: float, vq: float, delta: float) -> tuple[float, float]:
    """Rotate a dq vector by +delta (new = R(delta)*old)."""
    c = math.cos(delta)
    s = math.sin(delta)
    return vd * c - vq * s, vd * s + vq * c


def case_gfl_pll_lvrt(t: float, y: np.ndarray, p: SimParams) -> list[float]:
    """
    Stronger GFL baseline (still averaged):
      - SRF-PLL angle tracking (theta_pll),
      - P control as constant-power outer loop (scheduled load draw),
      - IEEE-2800-like reactive current priority during voltage dips (LVRT support),
      - Vector current limit, no "momentary cessation" forcing currents to zero.

    State: [id_grid, iq_grid, xi_id, xi_iq, theta_pll, xi_pll, v_dc]
    """
    id_grid, iq_grid, xi_id, xi_iq, theta_pll, xi_pll, v_dc = y
    d = p.derived()

    v_th_d, v_th_q = v_th_dq_grid_frame(t, p)
    v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(v_th_d, v_th_q, id_grid, iq_grid, d["r_th"], d["x_th"])
    v_pcc_mag = v_mag(v_pcc_d, v_pcc_q)

    # PLL: operate on PCC voltage in PLL frame.
    v_d_pll, v_q_pll = _rot(v_pcc_d, v_pcc_q, -theta_pll)
    # Standard SRF-PLL: drive v_q -> 0. Use small-signal PI on v_q.
    d_xi_pll = v_q_pll
    omega_pll = W_BASE * (1.0 + p.pll_kp * v_q_pll + p.pll_ki * xi_pll)
    d_theta = omega_pll - W_BASE

    # Power references (load draw): P_ref negative from inverter->grid perspective.
    p_load = p_load_profile(t, p)
    p_ref = -p_load

    # Reactive current injection during dips (capacitive support: i_q < 0 when v_q=0).
    iq_support = 0.0
    if v_pcc_mag < p.v_thresh_pu:
        iq_support = -clamp(p.k_v * (1.0 - v_pcc_mag), 0.0, p.i_max_pu)

    v_for_p = max(0.05, abs(v_d_pll))
    id_ref_pll = p_ref / (1.5 * v_for_p)
    iq_ref_pll = iq_support

    # Vector current limit with reactive priority.
    i_mag = math.sqrt(id_ref_pll * id_ref_pll + iq_ref_pll * iq_ref_pll)
    if i_mag > p.i_max_pu:
        # keep iq_ref_pll as-is, limit id_ref_pll to remaining headroom
        iq_abs = min(abs(iq_ref_pll), p.i_max_pu)
        iq_ref_pll = math.copysign(iq_abs, iq_ref_pll)
        id_lim = math.sqrt(max(0.0, p.i_max_pu * p.i_max_pu - iq_ref_pll * iq_ref_pll))
        id_ref_pll = clamp(id_ref_pll, -id_lim, id_lim)

    # Map PLL-frame current refs to grid frame.
    id_ref, iq_ref = _rot(id_ref_pll, iq_ref_pll, theta_pll)

    xi_id_eff = clamp(xi_id, -p.xi_lim, p.xi_lim)
    xi_iq_eff = clamp(xi_iq, -p.xi_lim, p.xi_lim)

    v_pi_d = p.cc_kp * (id_ref - id_grid) + p.cc_ki * xi_id_eff
    v_pi_q = p.cc_kp * (iq_ref - iq_grid) + p.cc_ki * xi_iq_eff
    v_inv_d = v_pcc_d + d["r_f"] * id_grid - d["x_f"] * iq_grid + v_pi_d
    v_inv_q = v_pcc_q + d["r_f"] * iq_grid + d["x_f"] * id_grid + v_pi_q

    # Voltage limiting (use the same E_max across all cases for apples-to-apples comparison)
    v_lim = p.e_max_pu * v_dc
    v_inv_mag = v_mag(v_inv_d, v_inv_q)
    if v_inv_mag > v_lim:
        scale = v_lim / v_inv_mag
        v_inv_d *= scale
        v_inv_q *= scale

    did_dt, diq_dt = plant_dynamics_grid_frame(
        id_grid, iq_grid, v_inv_d, v_inv_q, v_pcc_d, v_pcc_q, d["r_f"], d["x_f"]
    )

    d_xi_id = id_ref - id_grid
    d_xi_iq = iq_ref - iq_grid
    if xi_id > p.xi_lim and d_xi_id > 0:
        d_xi_id = 0.0
    if xi_id < -p.xi_lim and d_xi_id < 0:
        d_xi_id = 0.0
    if xi_iq > p.xi_lim and d_xi_iq > 0:
        d_xi_iq = 0.0
    if xi_iq < -p.xi_lim and d_xi_iq < 0:
        d_xi_iq = 0.0
    d_xi_id += p.xi_recover_k * (xi_id_eff - xi_id)
    d_xi_iq += p.xi_recover_k * (xi_iq_eff - xi_iq)

    p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, id_grid, iq_grid)
    p_draw = -p_grid
    p_in = p_draw
    e_dot = (p_in - p_load) / max(1e-6, p.dc_link_energy_s)
    dv_dc = e_dot / max(p.v_dc_min_pu, v_dc)
    dv_dc = clamp(dv_dc, -50.0, 50.0)
    if v_dc <= p.v_dc_min_pu and dv_dc < 0.0:
        dv_dc = 0.0
    if v_dc >= p.v_dc_max_pu and dv_dc > 0.0:
        dv_dc = 0.0

    return [did_dt, diq_dt, d_xi_id, d_xi_iq, d_theta, d_xi_pll, dv_dc]


def stage3_support_pu_dc(x_sys: float, soc: float, p: SimParams, *, n_blocks: int, p_load_pu: float) -> float:
    """
    Stage-3 droop/FFR support mapping from system-base frequency deviation x=(f-f0)/f0
    to a data-center-base support power offset (pu on data-center base).

    Convention:
      - returned p_support_pu_dc > 0 reduces grid draw (load relief), buffering with the UPS-BESS.
      - returned p_support_pu_dc < 0 increases grid draw (charge BESS), within limits.
    """
    # Deadband on frequency error (in Hz) to avoid chattering.
    deadband_x = float(p.stage3_deadband_hz) / 60.0
    x_eff = 0.0 if abs(x_sys) <= deadband_x else x_sys

    p_support_pu_sys = clamp(p.vsm_droop_gain_pu_per_pu * (-x_eff), -p.sys_ffr_p_max_pu, p.sys_ffr_p_max_pu)
    p_support_pu_dc = p_support_pu_sys * p.sys_base_mw / (max(1, n_blocks) * p.base_mw)

    # Enforce BESS power and SoC feasibility on the data-center base.
    p_support_dis_max = min(float(p_load_pu), p.bess_p_dis_max_pu)
    p_support_chg_max = p.bess_p_chg_max_pu
    p_support_pu_dc = clamp(p_support_pu_dc, -p_support_chg_max, p_support_dis_max)
    if soc <= p.soc_min and p_support_pu_dc > 0.0:
        p_support_pu_dc = 0.0
    if soc >= p.soc_max and p_support_pu_dc < 0.0:
        p_support_pu_dc = 0.0
    return float(p_support_pu_dc)


def _case_gfm_proposed_core(t: float, y: np.ndarray, p: SimParams, *, p_support_pu: float = 0.0) -> tuple[list[float], float]:
    """
    Core dynamics for the proposed controller (Stage 2), with an optional Stage-3 support offset.

    p_support_pu convention:
      - positive: reduce grid draw (load relief) by p_support_pu, buffering with the UPS-BESS
      - negative: increase grid draw (charge BESS), within limits

    Returns (derivatives, p_draw) where p_draw is the instantaneous grid draw (pu on data-center base).

    State vector (9 base + 2 optional negative-seq filter states):
      [id_grid, iq_grid, xi_id, xi_iq, p_draw_ref, p_bess, soc, v_dc, xi_vdc,
       vpcc_d_avg, vpcc_q_avg]  (indices 9,10 only present when neg_seq_suppress=True)
    """
    # Unpack base states
    id_grid, iq_grid, xi_id, xi_iq, p_draw_ref, p_bess, soc, v_dc, xi_vdc = y[:9]

    # Optional negative-sequence filter states
    has_neg_seq = p.neg_seq_suppress and p.fault_type != "balanced" and len(y) >= 11
    vpcc_d_avg = float(y[9]) if has_neg_seq else 0.0
    vpcc_q_avg = float(y[10]) if has_neg_seq else 0.0
    d = p.derived()
    v_th_d, v_th_q = v_th_dq_grid_frame(t, p)
    v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(v_th_d, v_th_q, id_grid, iq_grid, d["r_th"], d["x_th"])

    # Negative-sequence decomposition: extract positive-seq component for control
    d_vpcc_d_avg = 0.0
    d_vpcc_q_avg = 0.0
    if has_neg_seq:
        alpha_ns = p.neg_seq_filter_bw
        d_vpcc_d_avg, d_vpcc_q_avg = _neg_seq_filter_derivatives(
            v_pcc_d, v_pcc_q, vpcc_d_avg, vpcc_q_avg, alpha_ns
        )
        # Use positive-sequence voltage magnitude for control decisions
        v_pcc_pos_d, v_pcc_pos_q = vpcc_d_avg, vpcc_q_avg
        v_pcc_mag = v_mag(v_pcc_pos_d, v_pcc_pos_q)
    else:
        v_pcc_mag = v_mag(v_pcc_d, v_pcc_q)

    # Fault detection delay: model measurement + logic + gating latency before Stage-2 logic engages.
    # Use the scheduled fault window as the primary "dip" indicator (after the delay) to avoid
    # numerical chattering when PCC voltage briefly rebounds under severe current limiting.
    t_fault_detect = p.fault_start_s + max(0.0, p.fault_detect_delay_s)
    in_fault_mode = (t_fault_detect <= t <= p.fault_end_s) or ((t >= t_fault_detect) and (v_pcc_mag < p.v_thresh_pu))
    p_load = p_load_profile(t, p)
    # Use a voltage-magnitude proxy for P<->I conversion in the PCC-voltage-aligned basis.
    v_for_p = max(0.05, v_pcc_mag)

    # Reactive priority during dips, but not at the expense of infeasible active power supply.
    # Use a V_pcc-aligned basis so that i_parallel controls P and i_perp controls Q even when v_q != 0.
    i_perp_des = 0.0
    if in_fault_mode:
        i_perp_des = clamp(p.k_v * (1.0 - v_pcc_mag), 0.0, p.i_max_pu)

    # If the BESS is power-limited, ensure enough current remains for the minimum grid draw.
    p_draw_min_required = max(0.0, p_load - p.bess_p_dis_max_pu)
    # Additionally, reserve some active power (grid draw) during faults to reduce aggregate load-drop risk.
    p_draw_min_policy = p.p_fault_min_grid_draw_pu if in_fault_mode else 0.0
    p_draw_min_policy = min(p_load, max(0.0, p_draw_min_policy))

    i_parallel_required = max(p_draw_min_required, p_draw_min_policy) / (1.5 * v_for_p)
    i_perp_max_feasible = math.sqrt(max(0.0, p.i_max_pu * p.i_max_pu - i_parallel_required * i_parallel_required))
    i_perp_ref = min(i_perp_des, i_perp_max_feasible)

    i_parallel_max = math.sqrt(max(0.0, p.i_max_pu * p.i_max_pu - i_perp_ref * i_perp_ref))
    p_draw_max_current = 1.5 * v_for_p * i_parallel_max

    # Stage 1 (internal reliability): DC stiff-bus regulation and slow SoC recovery in normal operation.
    # Stage 2 has priority: Stage 1 is only active outside fault/low-voltage operation.
    stage1_active = bool(p.stage1_enable) and (not in_fault_mode)

    p_soc_bias = 0.0
    p_vdc = 0.0
    d_xi_vdc = 0.0
    if stage1_active:
        soc_err = p.stage1_soc_ref - soc
        last_fault_end = _last_fault_end_before(t, p)
        soc_bias_delay_elapsed = (
            last_fault_end is None
            or t >= last_fault_end + max(0.0, p.stage1_soc_bias_delay_s)
        )
        if soc < p.soc_max and soc_bias_delay_elapsed:
            p_soc_bias = clamp(p.stage1_soc_kp * soc_err, 0.0, p.stage1_soc_bias_chg_max_pu)
        vdc_err = p.v_dc_ref_pu - v_dc
        xi_vdc_eff = clamp(float(xi_vdc), -p.stage1_vdc_xi_lim, p.stage1_vdc_xi_lim)
        p_vdc = p.stage1_vdc_kp * vdc_err + p.stage1_vdc_ki * xi_vdc_eff
        # Keep the Stage-1 stiffness term bounded; final feasibility is enforced by the BESS saturation below.
        p_vdc = clamp(p_vdc, -p.bess_p_chg_max_pu, p.bess_p_dis_max_pu)
        d_xi_vdc = vdc_err

    # Grid draw reference:
    #  - Stage 2: filter the load in normal operation; enforce ramp limits for soft return.
    #  - Stage 3 (optional): add a droop-based support offset by modifying the target draw.
    p_draw_target = p_load + p_soc_bias
    if (not in_fault_mode) and (p_support_pu != 0.0):
        p_draw_target = clamp(p_draw_target - p_support_pu, 0.0, p_load + p.bess_p_chg_max_pu)

    if in_fault_mode:
        # Enforce the minimum-draw policy where feasible, but never exceed current-limited capability.
        p_draw_target = p_draw_max_current if p_draw_max_current < p_draw_min_policy else clamp(
            p_load, p_draw_min_policy, p_draw_max_current
        )

    tau = max(1e-3, p.grid_tau_s)
    dp_draw_unclamped = (p_draw_target - p_draw_ref) / tau
    dp_draw_ref = clamp(dp_draw_unclamped, -p.grid_ramp_down_pu_s, d["r_limit_pu_s"])

    # Keep the reference within BESS feasibility bounds (prevents commanding >P_bess,max)
    p_draw_ref_lower = max(0.0, p_load - p.bess_p_dis_max_pu)
    if in_fault_mode:
        p_draw_ref_lower = max(p_draw_ref_lower, p_draw_min_policy)
    # Respect the inverter current limit even in non-fault operation (e.g., Stage-3 over-frequency charging).
    p_draw_ref_upper = min(p_load + p.bess_p_chg_max_pu, p_draw_max_current)
    # If limits are mutually infeasible (e.g., no BESS plus deep dip), prioritize physical current limit.
    if p_draw_ref_lower > p_draw_ref_upper:
        p_draw_ref_lower = p_draw_ref_upper
    if p_draw_ref < p_draw_ref_lower and dp_draw_ref < 0.0:
        dp_draw_ref = 0.0
    if p_draw_ref > p_draw_ref_upper and dp_draw_ref > 0.0:
        dp_draw_ref = 0.0

    # Use a clamped grid-draw command so the minimum-draw policy is enforced even if the BESS
    # dynamics transiently oversupply the load.
    p_draw_cmd = clamp(p_draw_ref, p_draw_ref_lower, p_draw_ref_upper)

    # UPS-BESS power command and dynamics (power/ramp/SoC limited)
    p_bess_cmd = p_load - p_draw_cmd + p_vdc
    if soc <= p.soc_min and p_bess_cmd > 0.0:
        p_bess_cmd = 0.0
    if soc >= p.soc_max and p_bess_cmd < 0.0:
        p_bess_cmd = 0.0
    p_bess_cmd = clamp(p_bess_cmd, -p.bess_p_chg_max_pu, p.bess_p_dis_max_pu)
    dp_bess_unclamped = (p_bess_cmd - p_bess) / max(1e-3, p.bess_tau_s)
    dp_bess = clamp(dp_bess_unclamped, -p.bess_ramp_pu_s, p.bess_ramp_pu_s)
    d_soc = -p_bess / max(1.0, p.bess_autonomy_s)

    # Convert desired grid draw (P) and voltage support (Q) into dq current references.
    i_parallel_unclamped = -p_draw_cmd / (1.5 * v_for_p)  # negative => load (power absorbed from grid)
    i_parallel_ref = clamp(i_parallel_unclamped, -i_parallel_max, i_parallel_max)

    if has_neg_seq and v_pcc_mag > 0.05:
        # Use positive-sequence voltage for reference alignment to suppress neg-seq current
        v_ref_d = vpcc_d_avg
        v_ref_q = vpcc_q_avg
        v_ref_mag = max(0.05, v_mag(v_ref_d, v_ref_q))
        id_ref = (v_ref_d * i_parallel_ref + v_ref_q * i_perp_ref) / v_ref_mag
        iq_ref = (v_ref_q * i_parallel_ref - v_ref_d * i_perp_ref) / v_ref_mag
    else:
        id_ref = (v_pcc_d * i_parallel_ref + v_pcc_q * i_perp_ref) / v_for_p
        iq_ref = (v_pcc_q * i_parallel_ref - v_pcc_d * i_perp_ref) / v_for_p

    xi_id_eff = clamp(xi_id, -p.xi_lim, p.xi_lim)
    xi_iq_eff = clamp(xi_iq, -p.xi_lim, p.xi_lim)

    # Inner current control (grid-aligned frame)
    v_pi_d = p.cc_kp * (id_ref - id_grid) + p.cc_ki * xi_id_eff
    v_pi_q = p.cc_kp * (iq_ref - iq_grid) + p.cc_ki * xi_iq_eff
    v_inv_d = v_pcc_d + d["r_f"] * id_grid - d["x_f"] * iq_grid + v_pi_d
    v_inv_q = v_pcc_q + d["r_f"] * iq_grid + d["x_f"] * id_grid + v_pi_q

    # Voltage limiting (simple circular clamp)
    # AC voltage capability is bounded by DC-link voltage (proxy): |v_inv| <= E_max * V_dc.
    v_lim = p.e_max_pu * v_dc
    v_inv_mag = v_mag(v_inv_d, v_inv_q)
    if v_inv_mag > v_lim:
        scale = v_lim / v_inv_mag
        v_inv_d *= scale
        v_inv_q *= scale

    did_dt, diq_dt = plant_dynamics_grid_frame(
        id_grid, iq_grid, v_inv_d, v_inv_q, v_pcc_d, v_pcc_q, d["r_f"], d["x_f"]
    )

    d_xi_id = id_ref - id_grid
    d_xi_iq = iq_ref - iq_grid

    # Simple integrator clamping (anti-windup proxy)
    if xi_id > p.xi_lim and d_xi_id > 0:
        d_xi_id = 0.0
    if xi_id < -p.xi_lim and d_xi_id < 0:
        d_xi_id = 0.0
    if xi_iq > p.xi_lim and d_xi_iq > 0:
        d_xi_iq = 0.0
    if xi_iq < -p.xi_lim and d_xi_iq < 0:
        d_xi_iq = 0.0
    d_xi_id += p.xi_recover_k * (xi_id_eff - xi_id)
    d_xi_iq += p.xi_recover_k * (xi_iq_eff - xi_iq)

    # DC-link energy proxy: d(0.5*Vdc^2)/dt = (P_in - P_load)/T_dc
    # P_in = P_draw + P_bess  (both positive into IT bus energy balance).
    p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, id_grid, iq_grid)
    p_draw = -p_grid
    p_in = p_draw + p_bess
    e_dot = (p_in - p_load) / max(1e-6, p.dc_link_energy_s)
    dv_dc = e_dot / max(p.v_dc_min_pu, v_dc)
    dv_dc = clamp(dv_dc, -50.0, 50.0)
    if v_dc <= p.v_dc_min_pu and dv_dc < 0.0:
        dv_dc = 0.0
    if v_dc >= p.v_dc_max_pu and dv_dc > 0.0:
        dv_dc = 0.0

    base_dy = [did_dt, diq_dt, d_xi_id, d_xi_iq, dp_draw_ref, dp_bess, d_soc, dv_dc, d_xi_vdc]
    if has_neg_seq:
        base_dy.extend([d_vpcc_d_avg, d_vpcc_q_avg])
    return base_dy, p_draw


def gfm_in_fault_mode(t: float, y: np.ndarray, p: SimParams) -> bool:
    """
    Return True if the GFM controller would be in fault mode at (t, y, p).

    Replicates the fault detection logic from _case_gfm_proposed_core without
    running the full ODE, so it can be called post-hoc on saved trajectories.
    """
    has_neg_seq = p.neg_seq_suppress and p.fault_type != "balanced" and len(y) >= 11
    d = p.derived()
    v_th_d, v_th_q = v_th_dq_grid_frame(t, p)
    v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(v_th_d, v_th_q, float(y[0]), float(y[1]), d["r_th"], d["x_th"])
    if has_neg_seq:
        v_pcc_mag = v_mag(float(y[9]), float(y[10]))
    else:
        v_pcc_mag = v_mag(v_pcc_d, v_pcc_q)
    t_fault_detect = p.fault_start_s + max(0.0, p.fault_detect_delay_s)
    return (t_fault_detect <= t <= p.fault_end_s) or ((t >= t_fault_detect) and (v_pcc_mag < p.v_thresh_pu))


def case_gfm_proposed(t: float, y: np.ndarray, p: SimParams) -> list[float]:
    """
    Proposed grid-forming control (paper-aligned, averaged and current-limited):
      - Voltage-source inverter behind impedance (plant),
      - P-Q split with P retention: during faults prioritize Q, but retain P within remaining current headroom,
      - UPS-BESS flexibility: a limited BESS buffers the difference between IT load and grid power,
      - Soft return: ramp the grid power back after fault clearing.
      - Negative-sequence current suppression for asymmetric faults (when enabled).

    This simplified model assumes an internal oscillator aligned to the grid angle (no PLL),
    and uses inner dq current control to realize the P-Q split behavior described in the paper.

    State: [id_grid, iq_grid, xi_id, xi_iq, p_draw_ref, p_bess, soc, v_dc, xi_vdc]
      (+ [vpcc_d_avg, vpcc_q_avg] when neg_seq_suppress=True and fault_type != balanced)
      - p_draw_ref is the desired active power drawn from the grid (pu, positive = consume)
      - p_bess is the UPS-BESS active power (pu, positive = discharge to IT bus)
    """
    dy, _ = _case_gfm_proposed_core(t, y, p, p_support_pu=0.0)
    return dy


def case_gfm_proposed_with_support(t: float, y: np.ndarray, p: SimParams, *, p_support_pu: float) -> list[float]:
    """Same as case_gfm_proposed, but allows an external Stage-3 support offset (pu on data-center base)."""
    dy, _ = _case_gfm_proposed_core(t, y, p, p_support_pu=float(p_support_pu))
    return dy


def case_gfm_proposed_with_stage3_x(t: float, y: np.ndarray, p: SimParams, *, x_sys: float, n_blocks: int) -> list[float]:
    """
    Proposed controller with Stage 3 enabled via an external frequency deviation input x=(f-f0)/f0.
    Stage 2 has priority: the support offset is only applied outside fault/low-voltage conditions.
    """
    if not p.stage3_enable:
        return case_gfm_proposed(t, y, p)
    p_load = p_load_profile(t, p)
    p_support = stage3_support_pu_dc(float(x_sys), float(y[6]), p, n_blocks=n_blocks, p_load_pu=p_load)
    return case_gfm_proposed_with_support(t, y, p, p_support_pu=p_support)


def case_gfm_proposed_stage3_frequency_event(
    t: float,
    y: np.ndarray,
    p: SimParams,
    *,
    n_blocks: int,
    pm0_pu_sys: float,
    enable_droop: bool,
) -> list[float]:
    """
    Closed-loop Stage-3 frequency-event co-simulation.

    Adds a bulk-system swing-equation proxy state x=(f-f0)/f0 driven by the aggregated data-center draw.
    When enabled, Stage 3 applies droop-based load relief by modifying the grid-draw target of the same
    supervisory controller used in Stage 2. Stage 2 (fault-mode logic) has priority: the support offset
    is only applied outside fault/low-voltage conditions.

    State: [...base_states..., x_sys]
    where base_states is 9 (balanced) or 11 (with neg-seq filter).
    """
    # x_sys is the LAST state
    x_sys = float(y[-1])
    soc = float(y[6])

    p_support_pu_dc = 0.0
    if enable_droop:
        p_load = p_load_profile(t, p)
        p_support_pu_dc = stage3_support_pu_dc(x_sys, soc, p, n_blocks=n_blocks, p_load_pu=p_load)

    # Base states are everything except the last (x_sys)
    n_base = len(y) - 1
    dy_dc, p_draw = _case_gfm_proposed_core(t, y[:n_base], p, p_support_pu=p_support_pu_dc)

    # Swing equation on system base: 2H dx/dt = (P_m - P_e) - D x
    p_e_pu_sys = (n_blocks * p.base_mw / p.sys_base_mw) * float(p_draw)
    p_m = pm0_pu_sys + (p.sys_event_step_pu if t >= p.sys_event_t_s else 0.0)
    dx = ((p_m - p_e_pu_sys) - p.sys_damping_pu * x_sys) / (2.0 * p.sys_inertia_h_s)
    return dy_dc + [dx]
