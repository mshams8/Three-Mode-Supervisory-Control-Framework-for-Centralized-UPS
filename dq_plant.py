from common import W_BASE

# ---------------------------------------------------------------------------
# R1 addition: LCL filter dynamics for single-converter impedance screening.
# Reviewer 2 (point 2) asked for LC/LCL filter dynamics and frequency-domain
# impedance evidence. This module adds LCL plant equations alongside the
# existing series-RL plant. The series-RL helpers are unchanged so the entire
# averaged-dq pipeline keeps working when LCL parameters are zero.
# ---------------------------------------------------------------------------


def v_pcc_from_grid_impedance(
    v_th_d: float, v_th_q: float, id_grid: float, iq_grid: float, r_th: float, x_th: float
) -> tuple[float, float]:
    """
    Fundamental-frequency approximation in the grid dq frame.

    Convention: current is defined positive from inverter -> grid (PCC -> Thevenin).
    Therefore, the PCC voltage is:
      V_pcc = V_th + Z_th * I
    """
    v_pcc_d = v_th_d + r_th * id_grid - x_th * iq_grid
    v_pcc_q = v_th_q + r_th * iq_grid + x_th * id_grid
    return v_pcc_d, v_pcc_q


def plant_dynamics_grid_frame(
    id_grid: float,
    iq_grid: float,
    v_inv_d: float,
    v_inv_q: float,
    v_pcc_d: float,
    v_pcc_q: float,
    r_f: float,
    x_f: float,
) -> tuple[float, float]:
    """
    Averaged filter RL plant in the grid dq frame.

    x_f is the per-unit reactance at W_BASE, not an inductance.
    """
    did_dt = (W_BASE / x_f) * (v_inv_d - v_pcc_d - r_f * id_grid) + W_BASE * iq_grid
    diq_dt = (W_BASE / x_f) * (v_inv_q - v_pcc_q - r_f * iq_grid) - W_BASE * id_grid
    return did_dt, diq_dt


def plant_dynamics_with_lc(
    i1_d: float, i1_q: float,
    i2_d: float, i2_q: float,
    vc_d: float, vc_q: float,
    v_inv_d: float, v_inv_q: float,
    v_pcc_d: float, v_pcc_q: float,
    r_f1: float, x_f1: float,
    r_f2: float, x_f2: float,
    cf_pu: float, r_d: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Averaged LCL plant in the grid dq frame for single-converter impedance screening.

    State variables:
      i1 = inverter-side filter current (pu)
      i2 = grid-side filter current (pu)
      vc = capacitor voltage (pu)

    Topology:  v_inv -[R_f1, L_f1]- (capacitor node) -[R_f2, L_f2]- v_pcc
               capacitor C in series with damping resistor R_d to ground.

    Per-unit reactances are at W_BASE (rad/s).  cf_pu is interpreted as the
    capacitor susceptance at W_BASE in pu (B_pu = W_BASE * C_pu), so the
    actual capacitor admittance current is i_c = cf_pu * dvc/dt / W_BASE.

    The capacitor branch carries a series damping resistor R_d:
      v_inv_node = vc + R_d * i_c        (node voltage on the cap branch)
      i_c        = i1 - i2               (KCL at the capacitor node)

    Returns derivatives (di1_d, di1_q, di2_d, di2_q, dvc_d, dvc_q).
    """
    # Capacitor current (KCL at filter mid-node)
    ic_d = i1_d - i2_d
    ic_q = i1_q - i2_q

    # Voltage at the capacitor branch node = vc + R_d * i_c
    v_node_d = vc_d + r_d * ic_d
    v_node_q = vc_q + r_d * ic_q

    # Inverter-side leg (R_f1 + j X_f1):  v_inv - v_node = R_f1 * i1 + L_f1 * di1/dt + j*W*L_f1*i1
    di1_d_dt = (W_BASE / x_f1) * (v_inv_d - v_node_d - r_f1 * i1_d) + W_BASE * i1_q
    di1_q_dt = (W_BASE / x_f1) * (v_inv_q - v_node_q - r_f1 * i1_q) - W_BASE * i1_d

    # Grid-side leg (R_f2 + j X_f2):  v_node - v_pcc = R_f2 * i2 + L_f2 * di2/dt + j*W*L_f2*i2
    di2_d_dt = (W_BASE / x_f2) * (v_node_d - v_pcc_d - r_f2 * i2_d) + W_BASE * i2_q
    di2_q_dt = (W_BASE / x_f2) * (v_node_q - v_pcc_q - r_f2 * i2_q) - W_BASE * i2_d

    # Capacitor:  C * dvc/dt = i_c (in dq with cross-coupling for the rotating frame)
    # Using the susceptance-based per-unit:  dvc/dt = (W_BASE / cf_pu_susceptance) * i_c
    # cf_pu here is W*C in pu, so the gain is W_BASE / cf_pu.
    dvc_d_dt = (W_BASE / cf_pu) * ic_d + W_BASE * vc_q
    dvc_q_dt = (W_BASE / cf_pu) * ic_q - W_BASE * vc_d

    return di1_d_dt, di1_q_dt, di2_d_dt, di2_q_dt, dvc_d_dt, dvc_q_dt

