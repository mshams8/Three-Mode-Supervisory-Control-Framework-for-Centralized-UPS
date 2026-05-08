import math
from dataclasses import dataclass

import numpy as np

from common import W_BASE


def abc_from_vmag_theta(v_mag: float, theta: float) -> tuple[float, float, float]:
    """
    Balanced three-phase phase-to-neutral voltages from a dq-aligned magnitude.

    With v_mag as the dq-frame d-axis magnitude, this uses:
      va = v_mag * sin(theta)
      vb = v_mag * sin(theta - 2π/3)
      vc = v_mag * sin(theta + 2π/3)
    """
    va = v_mag * math.cos(theta)
    vb = v_mag * math.cos(theta - 2.0 * math.pi / 3.0)
    vc = v_mag * math.cos(theta + 2.0 * math.pi / 3.0)
    return va, vb, vc


def clarke_abc_to_alpha_beta(va: float, vb: float, vc: float) -> tuple[float, float]:
    # Power-invariant Clarke (consistent with 1.5*dq power convention)
    v_alpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
    v_beta = (2.0 / 3.0) * ((math.sqrt(3.0) / 2.0) * (vb - vc))
    return v_alpha, v_beta


def park_alpha_beta_to_dq(v_alpha: float, v_beta: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    v_d = v_alpha * c + v_beta * s
    v_q = -v_alpha * s + v_beta * c
    return v_d, v_q


def inv_park_dq_to_alpha_beta(v_d: float, v_q: float, theta: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    v_alpha = v_d * c - v_q * s
    v_beta = v_d * s + v_q * c
    return v_alpha, v_beta


def inv_clarke_alpha_beta_to_abc(v_alpha: float, v_beta: float) -> tuple[float, float, float]:
    va = v_alpha
    vb = -0.5 * v_alpha + (math.sqrt(3.0) / 2.0) * v_beta
    vc = -0.5 * v_alpha - (math.sqrt(3.0) / 2.0) * v_beta
    return va, vb, vc


@dataclass(frozen=True)
class EMTPlantParams:
    """
    EMT-style plant parameters for a series Thevenin RL + filter RL, per-phase, on the same per-unit base.

    r_th, x_th are derived from SCR/XR (pu). r_f, x_f are the filter parameters (pu).
    The reactances are interpreted at W_BASE, i.e., L_pu = X_pu / W_BASE.
    """

    r_th: float
    x_th: float
    r_f: float
    x_f: float

    @property
    def r_total(self) -> float:
        return self.r_th + self.r_f

    @property
    def l_th(self) -> float:
        return self.x_th / W_BASE

    @property
    def l_f(self) -> float:
        return self.x_f / W_BASE

    @property
    def l_total(self) -> float:
        return self.l_th + self.l_f


def solve_series_rl_step(
    i_abc: np.ndarray,
    v_inv_abc: np.ndarray,
    v_src_abc: np.ndarray,
    plant: EMTPlantParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One explicit step for series RL dynamics:
      L_total di/dt = v_inv - v_src - R_total i

    Returns (i_next, di_dt, v_pcc) where v_pcc is the node voltage between grid RL and filter RL:
      v_pcc = v_inv - R_f i - L_f di/dt
    """
    r = plant.r_total
    l = plant.l_total
    di_dt = (v_inv_abc - v_src_abc - r * i_abc) / max(1e-12, l)
    i_next = i_abc + dt * di_dt
    v_pcc = v_inv_abc - plant.r_f * i_abc - plant.l_f * di_dt
    return i_next, di_dt, v_pcc


@dataclass(frozen=True)
class LCLPlantParams:
    """
    R1 addition: LCL plant for single-converter impedance screening.
    All quantities in per-unit on the same AC base. Reactances are at W_BASE.
    """

    r_th: float
    x_th: float
    r_f1: float    # inverter-side resistance
    x_f1: float    # inverter-side reactance
    r_f2: float    # grid-side resistance
    x_f2: float    # grid-side reactance
    cf_pu: float   # capacitor susceptance (W * C) in pu at W_BASE
    r_d: float     # capacitor damping resistor

    @property
    def l_th(self) -> float:
        return self.x_th / W_BASE

    @property
    def l_f1(self) -> float:
        return self.x_f1 / W_BASE

    @property
    def l_f2(self) -> float:
        return self.x_f2 / W_BASE

    @property
    def cap(self) -> float:
        # cf_pu is W*C in pu, so C in pu = cf_pu / W_BASE
        return self.cf_pu / W_BASE


def solve_lcl_step(
    i1_abc: np.ndarray,
    i2_abc: np.ndarray,
    vc_abc: np.ndarray,
    v_inv_abc: np.ndarray,
    v_src_abc: np.ndarray,
    plant: LCLPlantParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    R1 addition: one explicit Euler step for LCL plant dynamics in abc frame.

    State:  i1 (inverter-side current), i2 (grid-side current), vc (capacitor voltage).
    Returns (i1_next, i2_next, vc_next, di1_dt, v_pcc) with
      v_pcc = v_src + (R_th + jX_th_eq) * i2  (interpreted in abc; the grid-side leg
      provides the immediate PCC as v_node - R_f2*i2 - L_f2*di2/dt).
    For the EMT cross-check we report v_pcc as the voltage on the grid side of the
    grid-side filter inductor: v_pcc = vc + R_d * (i1 - i2) - R_f2 * i2 - L_f2 * di2/dt.
    """
    r_total_grid = plant.r_th + plant.r_f2
    l_grid = plant.l_th + plant.l_f2

    # Capacitor branch: i_c = i1 - i2; v_node = vc + R_d * i_c
    ic = i1_abc - i2_abc
    v_node = vc_abc + plant.r_d * ic

    # Inverter-side leg
    di1_dt = (v_inv_abc - v_node - plant.r_f1 * i1_abc) / max(1e-12, plant.l_f1)
    i1_next = i1_abc + dt * di1_dt

    # Grid-side leg + Thevenin
    di2_dt = (v_node - v_src_abc - r_total_grid * i2_abc) / max(1e-12, l_grid)
    i2_next = i2_abc + dt * di2_dt

    # Capacitor voltage update
    dvc_dt = ic / max(1e-12, plant.cap)
    vc_next = vc_abc + dt * dvc_dt

    # PCC voltage for reporting
    v_pcc = v_node - plant.r_f2 * i2_abc - plant.l_f2 * di2_dt

    return i1_next, i2_next, vc_next, di1_dt, v_pcc


# ---------------------------------------------------------------------------
# PWM Switching model utilities
# ---------------------------------------------------------------------------

def pwm_carrier(t: float, f_sw: float) -> float:
    """
    Symmetric triangular carrier for carrier-based PWM.
    Returns value in [-1, 1].
    """
    period = 1.0 / f_sw
    phase = (t % period) / period  # [0, 1)
    if phase < 0.5:
        return -1.0 + 4.0 * phase
    else:
        return 3.0 - 4.0 * phase


def pwm_switch_abc(v_mod_abc: np.ndarray, v_dc: float, t: float, f_sw: float) -> np.ndarray:
    """
    Three-phase carrier-based SPWM switching.

    Args:
        v_mod_abc: modulation signals (3,) in per-unit of V_dc/2
        v_dc: DC-link voltage (per-unit)
        t: current time (s)
        f_sw: switching/carrier frequency (Hz)

    Returns:
        v_inv_abc: switched inverter voltages (3,)
    """
    carrier = pwm_carrier(t, f_sw)
    v_inv = np.zeros(3, dtype=float)
    half_vdc = 0.5 * v_dc
    for i in range(3):
        mod = v_mod_abc[i] / max(1e-9, half_vdc)
        if mod > carrier:
            v_inv[i] = half_vdc
        else:
            v_inv[i] = -half_vdc
    return v_inv


def sequence_components_from_abc(va: float, vb: float, vc: float) -> tuple[complex, complex, complex]:
    """
    Compute symmetrical sequence components (V0, V1, V2) from abc quantities.

    Returns: (V_zero, V_positive, V_negative) as complex phasors.
    """
    a = complex(math.cos(2*math.pi/3), math.sin(2*math.pi/3))
    a2 = a * a
    v0 = (va + vb + vc) / 3.0
    v1 = (va + a * vb + a2 * vc) / 3.0
    v2 = (va + a2 * vb + a * vc) / 3.0
    return complex(v0), complex(v1), complex(v2)
