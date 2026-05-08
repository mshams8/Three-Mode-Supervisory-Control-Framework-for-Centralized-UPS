import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

W_BASE = 2 * np.pi * 60.0


class FaultType(Enum):
    BALANCED = "balanced"
    SLG = "slg"       # Single-line-to-ground (phase A)
    LL = "ll"          # Line-to-line (B-C)
    LLG = "llg"        # Double-line-to-ground (B-C-G)


class WorkloadType(Enum):
    CONSTANT = "constant"
    SINGLE_FREQ = "single_freq"
    BROADBAND = "broadband"
    TRAINING = "training"
    INFERENCE = "inference"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def v_mag(vd: float, vq: float) -> float:
    return math.sqrt(vd * vd + vq * vq)


def pq_from_v_i(vd: float, vq: float, id_: float, iq_: float) -> tuple[float, float]:
    """
    Three-phase active/reactive power in per-unit using dq quantities.

    Convention: positive P means power flows from inverter -> grid.
    With a data-center load (power drawn from grid), expect P < 0.

    Q convention: positive Q corresponds to capacitive support (i_q < 0 when v_q = 0).
    """
    p = 1.5 * (vd * id_ + vq * iq_)
    q = 1.5 * (vq * id_ - vd * iq_)
    return p, q


def rl_from_scr_xr(scr: float, xr_ratio: float) -> tuple[float, float]:
    """
    Return (R_pu, X_pu) of a Thevenin impedance with magnitude 1/SCR and X/R ratio.
    """
    if scr <= 0:
        raise ValueError("SCR must be > 0")
    if xr_ratio <= 0:
        raise ValueError("X/R ratio must be > 0")
    z_mag = 1.0 / scr
    r = z_mag / math.sqrt(1.0 + xr_ratio * xr_ratio)
    x = r * xr_ratio
    return r, x


@dataclass(frozen=True)
class SimParams:
    base_mw: float = 50.0

    fault_start_s: float = 0.50
    fault_end_s: float = 0.65
    fault_v_pu: float = 0.50

    p_load_pu: float = 1.0
    i_max_pu: float = 1.0

    # Grid strength
    scr: float = 1.5
    xr_ratio: float = 5.0

    # Filter (pu reactance at 60 Hz)
    x_f_pu: float = 0.15
    r_f_pu: float = 0.01

    # R1 addition: LCL filter parameters for single-converter impedance screening.
    # output_impedance.py uses these values for the LCL terminal-impedance model:
    # inverter-side L1 (= x_f_pu / W_BASE), capacitor C with damping resistor R_d,
    # and grid-side L2 (= x_f2_pu / W_BASE), all on the same per-unit base.
    # The time-domain EMT runner remains a series-RL/SPWM cross-check unless it is
    # explicitly extended to dispatch to solve_lcl_step().
    x_f2_pu: float = 0.0          # grid-side reactance at 60 Hz (pu); 0 disables LCL
    r_f2_pu: float = 0.005        # grid-side series resistance (pu)
    cf_pu: float = 0.0            # filter capacitance, pu susceptance at 60 Hz; 0 disables LCL
    r_d_pu: float = 0.10          # capacitor damping resistor (pu, in series with C)

    # DC-link / internal bus proxy (energy state)
    dc_link_energy_s: float = 0.50  # energy base in seconds (E_base = S_base * dc_link_energy_s)
    v_dc_init_pu: float = 1.00
    v_dc_min_pu: float = 0.70
    v_dc_max_pu: float = 1.20
    v_dc_ref_pu: float = 1.00

    # GFL control
    pll_kp: float = 20.0
    pll_ki: float = 200.0
    cc_kp: float = 0.4
    cc_ki: float = 50.0
    xi_lim: float = 5.0  # PI integrator clamp (anti-windup proxy)
    xi_recover_k: float = 50.0  # 1/s, pulls integrators back inside xi_lim if they overshoot

    # Voltage thresholds (paper-aligned)
    v_thresh_pu: float = 0.85  # triggers P-Q split (GFM) / ride-through logic
    v_mc_pu: float = 0.70  # "momentary cessation"/protection for GFL benchmark

    # Fault detection / commutation delay (DC-side hold-up requirement driver)
    # Models measurement + logic + gating/commutation latency before Stage-2 fault logic takes effect.
    fault_detect_delay_s: float = 0.002

    # GFM control (paper behavior)
    k_v: float = 2.5  # voltage support gain (maps V error -> Iq_ref during fault)
    e_max_pu: float = 1.30

    # UPS-BESS (simple average model)
    bess_autonomy_s: float = 600.0  # 10 minutes at 1.0 pu
    bess_p_dis_max_pu: float = 1.0  # max discharge power (pu)
    bess_p_chg_max_pu: float = 0.5  # max charge power (pu)
    bess_ramp_pu_s: float = 5.0  # |dP_bess/dt| limit (pu/s)
    bess_tau_s: float = 0.02  # first-order lag for BESS power command
    soc_init: float = 0.80
    soc_min: float = 0.20
    soc_max: float = 1.00

    # Stage 1 (internal reliability): DC stiff-bus regulation in normal operation
    stage1_enable: bool = False
    stage1_freeze_in_fault: bool = True
    stage1_vdc_kp: float = 0.50  # pu power / pu Vdc error (through BESS command)
    stage1_vdc_ki: float = 10.0  # 1/s, integrator on Vdc error
    stage1_vdc_xi_lim: float = 0.20  # clamp for Vdc PI integrator (pu)
    stage1_soc_ref: float = 0.80
    stage1_soc_kp: float = 0.30  # pu grid-draw bias / pu SoC error
    stage1_soc_bias_chg_max_pu: float = 0.20  # max extra grid draw for SoC recovery (pu)
    stage1_soc_bias_delay_s: float = 0.0  # post-fault delay before SoC-recovery grid-draw bias re-engages

    # Grid power shaping / smoothing (Stage 2: oscillation filtering + soft return)
    grid_tau_s: float = 0.50  # low-pass time constant from P_load -> P_grid_ref
    grid_ramp_down_pu_s: float = 5.0  # allow fast reduction if needed

    # Fault-mode trade-off: reserve some active power to reduce aggregate load-drop risk
    # (at the cost of less reactive current during the dip).
    p_fault_min_grid_draw_pu: float = 0.20  # minimum grid draw during fault (pu on data-center base)

    # Bulk-system frequency proxy (simple swing equation on an external system base)
    sys_base_mw: float = 30000.0  # bulk power system base for frequency proxy
    sys_inertia_h_s: float = 5.0  # inertia constant H (seconds)
    sys_damping_pu: float = 1.0  # damping coefficient D (pu/pu)
    sys_n_blocks: int = 20  # number of identical 50 MW blocks (aggregation sensitivity)

    # Stage 3 (droop/FFR): controlled load modulation (maps frequency deviation to P_support)
    stage3_enable: bool = False
    stage3_deadband_hz: float = 0.02  # ignore small frequency errors (Hz)
    # Stage 3 (simple droop/FFR demonstration on the bulk frequency proxy)
    sys_event_step_pu: float = -0.01  # step change in P_m (pu on system base), e.g., gen loss
    sys_event_t_s: float = 1.0
    vsm_droop_gain_pu_per_pu: float = 2.0  # P_support = K * (-x) (load relief), pu_sys
    sys_ffr_p_max_pu: float = 0.02  # max Stage-3 support (|ΔP|) in pu on system base

    # Soft return
    r_limit_mw_s: float = 10.0  # MW/s

    # Asymmetric fault type
    fault_type: str = "balanced"  # balanced, slg, ll, llg

    # Broadband workload model
    workload_type: str = "single_freq"  # constant, single_freq, broadband, training, inference
    broadband_freqs_hz: tuple = (0.08, 0.25, 1.0, 3.5, 7.0)
    broadband_amps_pu: tuple = (0.12, 0.10, 0.08, 0.05, 0.03)
    # Training-specific: large batch cycles
    training_batch_period_s: float = 10.0   # ~0.1 Hz batch cycle
    training_ramp_fraction: float = 0.3     # fraction of period spent ramping
    training_amp_pu: float = 0.20
    # Inference-specific: bursty high-frequency
    inference_burst_freq_hz: float = 5.0
    inference_amp_pu: float = 0.15
    inference_noise_std_pu: float = 0.03

    # Negative-sequence current suppression (for asymmetric faults)
    neg_seq_suppress: bool = True          # enable negative-seq current suppression
    neg_seq_filter_wn: float = 753.98      # notch filter freq (2*w_base) rad/s
    neg_seq_filter_bw: float = 50.0        # notch bandwidth rad/s

    # Switching EMT model
    switching_freq_hz: float = 3060.0      # PWM carrier frequency
    enable_switching: bool = False          # use switched vs averaged inverter model

    # Multi-fault windows: list of (start_s, end_s) tuples for sequential fault events.
    # When non-empty, overrides fault_start_s / fault_end_s for Thevenin voltage scheduling.
    fault_windows: tuple = ()  # e.g., ((0.5, 0.65), (1.5, 1.65))

    # Simulation horizon
    t_end_s: float = 6.0
    points: int = 12001

    # Load schedule
    load_step_t_s: float = 0.10
    load_step_pu: float = 1.0

    # Pulsed load (forced oscillations demonstration)
    pulse_enable: bool = False
    pulse_start_s: float = 1.5
    pulse_end_s: float = 5.5
    pulse_amp_pu: float = 0.25
    pulse_freq_hz: float = 1.0

    def derived(self) -> dict[str, float]:
        r_th, x_th = rl_from_scr_xr(self.scr, self.xr_ratio)
        r_limit_pu_s = self.r_limit_mw_s / self.base_mw
        return {
            "r_th": r_th,
            "x_th": x_th,
            "r_f": self.r_f_pu,
            "x_f": self.x_f_pu,
            "r_limit_pu_s": r_limit_pu_s,
        }


def p_load_profile(t: float, p: SimParams) -> float:
    base = 0.0 if t < p.load_step_t_s else p.load_step_pu

    # Legacy single-frequency pulsed load (backward compatible)
    if p.pulse_enable and p.workload_type == "single_freq":
        if p.pulse_start_s <= t <= p.pulse_end_s:
            base += p.pulse_amp_pu * math.sin(2.0 * math.pi * p.pulse_freq_hz * (t - p.pulse_start_s))
        return base

    if not p.pulse_enable:
        return base
    if t < p.pulse_start_s or t > p.pulse_end_s:
        return base

    t_rel = t - p.pulse_start_s

    if p.workload_type == "broadband":
        # Multi-frequency superposition modeling diverse AI workload dynamics
        for freq, amp in zip(p.broadband_freqs_hz, p.broadband_amps_pu):
            base += amp * math.sin(2.0 * math.pi * freq * t_rel)
    elif p.workload_type == "training":
        # Training batch cycles: periodic ramp-up/hold/ramp-down pattern
        period = p.training_batch_period_s
        phase = (t_rel % period) / period
        ramp = p.training_ramp_fraction
        if phase < ramp:
            base += p.training_amp_pu * (phase / ramp)
        elif phase < (1.0 - ramp):
            base += p.training_amp_pu
        else:
            base += p.training_amp_pu * ((1.0 - phase) / ramp)
    elif p.workload_type == "inference":
        # Bursty inference: sine + higher harmonics + stochastic component
        base += p.inference_amp_pu * math.sin(2.0 * math.pi * p.inference_burst_freq_hz * t_rel)
        base += 0.4 * p.inference_amp_pu * math.sin(2.0 * math.pi * 2.3 * p.inference_burst_freq_hz * t_rel)
        base += 0.2 * p.inference_amp_pu * math.sin(2.0 * math.pi * 5.1 * p.inference_burst_freq_hz * t_rel)
        # Deterministic pseudo-noise (reproducible)
        base += p.inference_noise_std_pu * math.sin(137.0 * t_rel + 0.7 * math.sin(43.0 * t_rel))
    else:
        # single_freq fallback
        base += p.pulse_amp_pu * math.sin(2.0 * math.pi * p.pulse_freq_hz * t_rel)

    return base


def v_th_sequence_magnitudes(fault_type: str, v_ret: float) -> tuple[float, float]:
    """
    Positive and negative sequence voltage magnitudes for different fault types.

    Args:
        fault_type: "balanced", "slg", "ll", or "llg"
        v_ret: retained voltage magnitude during fault (p.u.)

    Returns:
        (V_pos, V_neg) in per-unit.
    """
    if fault_type == "slg":
        # SLG on phase A: V1 = (V_ret + 2)/3, V2 = (1 - V_ret)/3
        return (v_ret + 2.0) / 3.0, (1.0 - v_ret) / 3.0
    elif fault_type == "ll":
        # LL between B-C: V1 = (1 + V_ret)/2, V2 = (1 - V_ret)/2
        return (1.0 + v_ret) / 2.0, (1.0 - v_ret) / 2.0
    elif fault_type == "llg":
        # LLG on B-C: V1 = (1 + 2*V_ret)/3, V2 = (1 - V_ret)/3
        return (1.0 + 2.0 * v_ret) / 3.0, (1.0 - v_ret) / 3.0
    else:
        # Balanced: all positive sequence
        return v_ret, 0.0


def _in_any_fault_window(t: float, p: SimParams) -> bool:
    """Check if t falls inside any fault window (multi-fault or single)."""
    if p.fault_windows:
        return any(s <= t <= e for s, e in p.fault_windows)
    return p.fault_start_s <= t <= p.fault_end_s


def _last_fault_end_before(t: float, p: SimParams) -> float | None:
    """Most recent fault end time not later than t, or None if no fault has ended yet."""
    if p.fault_windows:
        ended = [float(e) for _, e in p.fault_windows if e <= t]
        return max(ended) if ended else None
    if p.fault_end_s <= t:
        return float(p.fault_end_s)
    return None


def v_th_dq_grid_frame(t: float, p: SimParams) -> tuple[float, float]:
    """
    Thevenin source voltage in the grid dq frame.

    For balanced faults: vq = 0 (all positive sequence).
    For asymmetric faults: negative sequence appears as 2*omega oscillation.
    """
    in_fault = _in_any_fault_window(t, p)
    if not in_fault:
        return 1.0, 0.0

    v_pos, v_neg = v_th_sequence_magnitudes(p.fault_type, p.fault_v_pu)
    if v_neg < 1e-9:
        return v_pos, 0.0

    # Negative sequence rotates at -omega in stationary frame,
    # appearing as -2*omega in the synchronous dq frame.
    theta_2w = 2.0 * W_BASE * t
    v_d = v_pos + v_neg * math.cos(theta_2w)
    v_q = v_neg * math.sin(theta_2w)
    return v_d, v_q


def v_th_abc_asymmetric(theta: float, fault_type: str, v_ret: float) -> tuple[float, float, float]:
    """
    Per-phase Thevenin source voltages for asymmetric faults in abc frame.

    Args:
        theta: electrical angle (rad)
        fault_type: "balanced", "slg", "ll", or "llg"
        v_ret: retained voltage magnitude during fault (p.u.)

    Returns:
        (va, vb, vc) instantaneous phase voltages
    """
    TWO_PI_3 = 2.0 * math.pi / 3.0
    if fault_type == "slg":
        # Phase A dips, B and C normal
        return (v_ret * math.cos(theta),
                1.0 * math.cos(theta - TWO_PI_3),
                1.0 * math.cos(theta + TWO_PI_3))
    elif fault_type == "ll":
        # B-C fault: phases B and C dip
        return (1.0 * math.cos(theta),
                v_ret * math.cos(theta - TWO_PI_3),
                v_ret * math.cos(theta + TWO_PI_3))
    elif fault_type == "llg":
        # B and C to ground
        return (1.0 * math.cos(theta),
                v_ret * math.cos(theta - TWO_PI_3),
                v_ret * math.cos(theta + TWO_PI_3))
    else:
        # Balanced three-phase
        return (v_ret * math.cos(theta),
                v_ret * math.cos(theta - TWO_PI_3),
                v_ret * math.cos(theta + TWO_PI_3))
