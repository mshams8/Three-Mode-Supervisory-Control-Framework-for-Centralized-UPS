import matplotlib.pyplot as plt
import numpy as np

from common import SimParams, p_load_profile, pq_from_v_i, v_mag, v_th_dq_grid_frame
from controllers import case_gfl_pll_lvrt, case_gfl_standard, case_gfm_proposed, case_gfm_proposed_stage3_frequency_event
from dq_plant import v_pcc_from_grid_impedance
from frequency_proxy import run_frequency_proxy
from integration import run_case
from metrics import compute_fault_metrics, compute_fault_metrics_multi


def _extract_v_pcc(sol, p: SimParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = sol.t
    id_grid, iq_grid = sol.y[0], sol.y[1]
    d = p.derived()
    v_d = np.zeros_like(t, dtype=float)
    v_q = np.zeros_like(t, dtype=float)
    v_abs = np.zeros_like(t, dtype=float)
    for k in range(t.size):
        v_th_d, v_th_q = v_th_dq_grid_frame(float(t[k]), p)
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
            v_th_d, v_th_q, float(id_grid[k]), float(iq_grid[k]), d["r_th"], d["x_th"]
        )
        v_d[k] = v_pcc_d
        v_q[k] = v_pcc_q
        v_abs[k] = v_mag(v_pcc_d, v_pcc_q)
    return t, v_d, v_q, v_abs


def _extract_p_draw(sol, p: SimParams) -> tuple[np.ndarray, np.ndarray]:
    t = sol.t
    id_grid, iq_grid = sol.y[0], sol.y[1]
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


def build_stage1_stiffbus_plot(p: SimParams, out_path: str) -> None:
    """
    Stage-1 demonstration: DC stiff-bus regulation in normal (no-fault) operation.

    Compares Stage 1 disabled vs enabled under a pulsed-load scenario where the supervisory shaping forces
    the UPS-BESS to buffer fast power variations.
    """
    p_base = SimParams(
        **{
            **p.__dict__,
            "fault_v_pu": 1.0,
            "fault_start_s": 10.0,
            "fault_end_s": 10.1,
            "load_step_t_s": 0.0,
            "load_step_pu": 1.0,
            "pulse_enable": True,
            "pulse_amp_pu": 0.25,
            "pulse_freq_hz": 1.0,
            "pulse_start_s": 1.0,
            "pulse_end_s": 5.0,
            "t_end_s": 6.0,
            "points": 6001,
            # Initialize with a DC-link energy deficit so Stage 1 provides clear error rejection
            # (Stage 1 off cannot recover Vdc in this averaged energy proxy when Pin≈Pload).
            "v_dc_init_pu": 0.92,
            # Disable SoC recovery bias for this figure to isolate the DC stiff-bus regulation effect.
            "stage1_soc_kp": 0.0,
            "stage1_soc_bias_chg_max_pu": 0.0,
        }
    )

    p_off = SimParams(**{**p_base.__dict__, "stage1_enable": False})
    p_on = SimParams(**{**p_base.__dict__, "stage1_enable": True})

    # Start near steady draw at 1 pu (grid supplies load, BESS idle).
    y0 = [-2.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_base.soc_init, p_base.v_dc_init_pu, 0.0]

    sol_off = run_case(
        lambda t, y: case_gfm_proposed(t, y, p_off),
        y0,
        p_off.t_end_s,
        p_off.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    sol_on = run_case(
        lambda t, y: case_gfm_proposed(t, y, p_on),
        y0,
        p_on.t_end_s,
        p_on.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    if not sol_off.success:
        raise RuntimeError(f"Stage-1 off run failed: {sol_off.message}")
    if not sol_on.success:
        raise RuntimeError(f"Stage-1 on run failed: {sol_on.message}")

    t = sol_off.t
    vdc_off = sol_off.y[7]
    vdc_on = sol_on.y[7]
    pbess_off = sol_off.y[5]
    pbess_on = sol_on.y[5]
    soc_off = sol_off.y[6]
    soc_on = sol_on.y[6]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 5.6), sharex=True)

    ax = axes[0]
    ax.plot(t, vdc_off, "k--", linewidth=1.2, label="Stage 1 off")
    ax.plot(t, vdc_on, "b-", linewidth=1.2, label="Stage 1 on (DC stiff bus)")
    ax.axhline(p_base.v_dc_ref_pu, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvspan(p_base.pulse_start_s, p_base.pulse_end_s, color="yellow", alpha=0.15)
    ax.set_ylabel("$V_{dc}$ (p.u.)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)

    ax = axes[1]
    ax.plot(t, pbess_off, "k--", linewidth=1.1, label="Stage 1 off")
    ax.plot(t, pbess_on, "b-", linewidth=1.1, label="Stage 1 on")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvspan(p_base.pulse_start_s, p_base.pulse_end_s, color="yellow", alpha=0.15)
    ax.set_ylabel("$P_{bess}$ (p.u.)")
    ax.grid(True, linestyle=":", alpha=0.6)

    ax = axes[2]
    ax.plot(t, soc_off, "k--", linewidth=1.1, label="Stage 1 off")
    ax.plot(t, soc_on, "b-", linewidth=1.1, label="Stage 1 on")
    ax.axvspan(p_base.pulse_start_s, p_base.pulse_end_s, color="yellow", alpha=0.15)
    ax.set_ylabel("SoC (p.u.)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98, hspace=0.15)
    plt.savefig(out_path, dpi=500)


def build_plots(sol_gfl_mc, sol_gfl_pll, sol_gfm, p: SimParams, out_path: str) -> None:
    metrics = compute_fault_metrics_multi(
        {
            "GFL_MC": (sol_gfl_mc, 4),
            "GFL_PLL": (sol_gfl_pll, 6),
            "GFM": (sol_gfm, 7),
        },
        p,
    )

    t, _, _, v_abs_gfl_mc = _extract_v_pcc(sol_gfl_mc, p)
    _, _, _, v_abs_gfl_pll = _extract_v_pcc(sol_gfl_pll, p)
    _, _, _, v_abs_gfm = _extract_v_pcc(sol_gfm, p)
    _, p_draw_gfl_mc = _extract_p_draw(sol_gfl_mc, p)
    _, p_draw_gfl_pll = _extract_p_draw(sol_gfl_pll, p)
    _, p_draw_gfm = _extract_p_draw(sol_gfm, p)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.9), sharex=True)

    ax = axes[0, 0]
    ax.plot(t, p_draw_gfl_mc, "r--", label="GFL (momentary cessation)")
    ax.plot(t, p_draw_gfl_pll, color="m", linestyle="--", label="GFL (PLL + LVRT)")
    ax.plot(t, p_draw_gfm, "b-", label="Proposed GFM")
    ax.plot(t, np.array([1.0 if ti >= p.load_step_t_s else 0.0 for ti in t], dtype=float), "k:", label="IT Load")
    ax.axvspan(p.fault_start_s, p.fault_end_s, color="yellow", alpha=0.2)
    ax.axvline(p.fault_end_s, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    # Show the Mode-2 minimum grid-draw policy during the dip to avoid misreading the proposed
    # trace as "momentary cessation" (the policy is enforced where feasible under current limit).
    ax.hlines(
        p.p_fault_min_grid_draw_pu,
        p.fault_start_s,
        p.fault_end_s,
        colors="gray",
        linestyles=":",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_ylabel("$P_{draw}$ (p.u.)")
    ax.set_title("(a) Active Power Drawn from Grid")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    k_anno = int(np.searchsorted(t, p.fault_end_s + 0.20))
    k_anno = min(max(k_anno, 0), t.size - 1)
    ax.annotate(
        "Soft return\n(ramp-limited)",
        xy=(float(t[k_anno]), float(p_draw_gfm[k_anno])),
        xytext=(p.fault_end_s + 0.15, 0.62),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
    )
    ax.annotate(
        "Min draw policy\n($P_{draw,min}^{fault}$)",
        xy=(p.fault_start_s + 0.5 * (p.fault_end_s - p.fault_start_s), float(p.p_fault_min_grid_draw_pu)),
        xytext=(p.fault_start_s + 0.02, float(p.p_fault_min_grid_draw_pu) + 0.20),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
    )

    ax = axes[0, 1]
    ax.plot(t, v_abs_gfl_mc, "r--", label="GFL (momentary cessation)")
    ax.plot(t, v_abs_gfl_pll, color="m", linestyle="--", label="GFL (PLL + LVRT)")
    ax.plot(t, v_abs_gfm, "b-", label="Proposed GFM")
    ax.axhline(1.0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvspan(p.fault_start_s, p.fault_end_s, color="yellow", alpha=0.2)
    ax.axvline(p.fault_end_s, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("$|V_{pcc}|$ (p.u.)")
    ax.set_title("(b) PCC Voltage Magnitude")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    ax = axes[1, 0]
    ax.plot(t, sol_gfl_mc.y[0], "r--", label="$i_d$ (GFL-MC)")
    ax.plot(t, sol_gfl_mc.y[1], "r:", label="$i_q$ (GFL-MC)")
    ax.plot(t, sol_gfl_pll.y[0], color="m", linestyle="--", label="$i_d$ (GFL-PLL)")
    ax.plot(t, sol_gfl_pll.y[1], color="m", linestyle=":", label="$i_q$ (GFL-PLL)")
    ax.plot(t, sol_gfm.y[0], "b-", label="$i_d$ (GFM)")
    ax.plot(t, sol_gfm.y[1], "b:", label="$i_q$ (GFM)")
    ax.axvspan(p.fault_start_s, p.fault_end_s, color="yellow", alpha=0.2)
    ax.axvline(p.fault_end_s, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Current (p.u.; $i_d<0$ means draw)")
    ax.set_title("(c) $dq$ Currents (Reactive Priority under Limit)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(ncol=3, fontsize=7.5, framealpha=0.9, loc="lower right")

    ax = axes[1, 1]
    vdc_gfl_mc = sol_gfl_mc.y[4]
    vdc_gfl = sol_gfl_pll.y[6]
    vdc_gfm = sol_gfm.y[7]
    ax.plot(t, vdc_gfl_mc, "r--", label="$V_{dc}$ (GFL-MC)")
    ax.plot(t, vdc_gfl, color="m", linestyle="--", label="$V_{dc}$ (GFL-PLL)")
    ax.plot(t, vdc_gfm, "b-", label="$V_{dc}$ (GFM)")
    ax.axvspan(p.fault_start_s, p.fault_end_s, color="yellow", alpha=0.2)
    ax.axvline(p.fault_end_s, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("$V_{dc}$ (p.u.)")
    ax.set_title("(d) DC-Link Proxy Voltage")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes.flat:
        ax.tick_params(direction="in")

    # Use the paper caption instead of an in-figure title to avoid cramped layout in IEEE columns.
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.96, wspace=0.24, hspace=0.40)
    plt.savefig(out_path, dpi=500)


def build_pulse_plot(sol_fast, sol_smooth, p: SimParams, out_path: str) -> None:
    t, p_draw_fast = _extract_p_draw(sol_fast, p)
    _, p_draw_smooth = _extract_p_draw(sol_smooth, p)
    p_bess_fast = sol_fast.y[5]
    p_bess_smooth = sol_smooth.y[5]
    p_load = np.array([p_load_profile(float(tt), p) for tt in t], dtype=float)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.4), sharex=True)

    ax = axes[0]
    ax.plot(t, p_load, color="0.55", linewidth=1.1, alpha=0.9, label="IT load ($P_{load}$)")
    ax.plot(t, p_draw_fast, "k--", label="Fast grid reference ($\\tau_{grid}=0.01$ s)")
    ax.plot(t, p_draw_smooth, "b-", label=f"Smoothed grid reference ($\\tau_{{grid}}={p.grid_tau_s}$ s)")
    ax.set_ylabel("$P_{draw}$ (p.u.)")
    ax.set_title("Grid Power Seen by Bulk System (Smoothing)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    ax = axes[1]
    ax.plot(t, p_bess_fast, "k--", label="$P_{bess}$ (fast)")
    ax.plot(t, p_bess_smooth, "b-", label="$P_{bess}$ (smoothed)")
    ax.set_ylabel("$P_{bess}$ (p.u.)")
    ax.set_xlabel("Time (s)")
    ax.set_title("UPS-BESS Buffer Power (High-Frequency Component)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


def build_ramp_plot(sol_fast, sol_smooth, p: SimParams, out_path: str) -> None:
    """
    Grid-draw shaping demonstration (ramp/LPF behavior) under a load step.

    This plot is intentionally free of oscillatory forcing so the reader can interpret the slow
    grid-draw trajectory as a shaping choice rather than an oscillation artifact.
    """
    t, p_draw_fast = _extract_p_draw(sol_fast, p)
    _, p_draw_smooth = _extract_p_draw(sol_smooth, p)
    p_bess_fast = sol_fast.y[5]
    p_bess_smooth = sol_smooth.y[5]
    p_load = np.array([p_load_profile(float(tt), p) for tt in t], dtype=float)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.0), sharex=True)

    ax = axes[0]
    ax.plot(t, p_load, color="0.55", linewidth=1.1, alpha=0.9, label="IT load ($P_{load}$)")
    ax.plot(t, p_draw_fast, "k--", label="Fast grid reference ($\\tau_{grid}=0.01$ s)")
    ax.plot(t, p_draw_smooth, "b-", label=f"Shaped grid reference ($\\tau_{{grid}}={p.grid_tau_s}$ s)")
    ax.set_ylabel("$P_{draw}$ (p.u.)")
    ax.set_title("Grid-Draw Shaping Under a Load Step (Ramping/LPF)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    ax = axes[1]
    ax.plot(t, p_bess_fast, "k--", label="$P_{bess}$ (fast)")
    ax.plot(t, p_bess_smooth, "b-", label="$P_{bess}$ (shaped)")
    ax.set_ylabel("$P_{bess}$ (p.u.)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


def build_freq_plot(sol_gfl_mc, sol_gfl_pll, sol_gfm, p: SimParams, out_path: str, *, n_blocks: int) -> None:
    # Fault aggregation proxy: use steady-state pre-fault initialization so the frequency proxy
    # reflects fault-time load behavior (not initial load-step transients).
    p_agg = SimParams(
        **{
            **p.__dict__,
            "load_step_t_s": 0.0,
            "load_step_pu": 1.0,
            "pulse_enable": False,
        }
    )
    sol_gfl_mc_agg = run_case(
        lambda t, y: case_gfl_standard(t, y, p_agg),
        [-2.0 / 3.0, 0.0, 0.0, 0.0, p_agg.v_dc_init_pu],
        p_agg.t_end_s,
        p_agg.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    sol_gfl_pll_agg = run_case(
        lambda t, y: case_gfl_pll_lvrt(t, y, p_agg),
        [-2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_agg.v_dc_init_pu],
        p_agg.t_end_s,
        p_agg.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    sol_gfm_agg = run_case(
        lambda t, y: case_gfm_proposed(t, y, p_agg),
        [-2.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_agg.soc_init, p_agg.v_dc_init_pu, 0.0],
        p_agg.t_end_s,
        p_agg.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    if not sol_gfl_mc_agg.success:
        raise RuntimeError(f"Frequency proxy (fault aggregation) GFL-MC run failed: {sol_gfl_mc_agg.message}")
    if not sol_gfl_pll_agg.success:
        raise RuntimeError(f"Frequency proxy (fault aggregation) GFL-PLL run failed: {sol_gfl_pll_agg.message}")
    if not sol_gfm_agg.success:
        raise RuntimeError(f"Frequency proxy (fault aggregation) GFM run failed: {sol_gfm_agg.message}")

    t, p_draw_gfl_mc = _extract_p_draw(sol_gfl_mc_agg, p_agg)
    _, p_draw_gfl_pll = _extract_p_draw(sol_gfl_pll_agg, p_agg)
    _, p_draw_gfm = _extract_p_draw(sol_gfm_agg, p_agg)

    _, f_gfl_mc = run_frequency_proxy(t, p_draw_gfl_mc, p_agg, n_blocks=n_blocks)
    _, f_gfl_pll = run_frequency_proxy(t, p_draw_gfl_pll, p_agg, n_blocks=n_blocks)
    _, f_gfm = run_frequency_proxy(t, p_draw_gfm, p_agg, n_blocks=n_blocks)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    # Stage-3 frequency event (no fault): closed-loop co-simulation with droop/FFR enabled.
    p_evt = SimParams(
        **{
            **p.__dict__,
            "fault_v_pu": 1.0,
            "fault_start_s": 10.0,
            "fault_end_s": 10.1,
            "pulse_enable": False,
            "load_step_t_s": 0.0,
            "load_step_pu": 1.0,
        }
    )

    # Pre-settle the averaged UPS controller to avoid a spurious frequency transient from initial conditions.
    y0_guess = [-2.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_evt.soc_init, p_evt.v_dc_init_pu, 0.0]
    sol_settle = run_case(
        lambda t, y: case_gfm_proposed_stage3_frequency_event(t, y, p_evt, n_blocks=1, pm0_pu_sys=0.0, enable_droop=False),
        y0_guess + [0.0],
        2.0,
        2001,
    )
    if not sol_settle.success:
        raise RuntimeError(f"Stage-3 settle run failed: {sol_settle.message}")
    y0_evt = [float(v) for v in sol_settle.y[:, -1]]
    y0_evt[9] = 0.0  # start from nominal frequency

    def _p_draw_at_state(id_grid: float, iq_grid: float) -> float:
        v_th_d, v_th_q = v_th_dq_grid_frame(0.0, p_evt)
        d = p_evt.derived()
        v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(v_th_d, v_th_q, id_grid, iq_grid, d["r_th"], d["x_th"])
        p_grid, _ = pq_from_v_i(v_pcc_d, v_pcc_q, id_grid, iq_grid)
        return -float(p_grid)

    p_draw0 = _p_draw_at_state(float(y0_evt[0]), float(y0_evt[1]))
    pm0_pu_sys = (n_blocks * p_evt.base_mw / p_evt.sys_base_mw) * p_draw0

    sol_evt_base = run_case(
        lambda t, y: case_gfm_proposed_stage3_frequency_event(
            t, y, p_evt, n_blocks=n_blocks, pm0_pu_sys=pm0_pu_sys, enable_droop=False
        ),
        y0_evt,
        p_evt.t_end_s,
        p_evt.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    sol_evt_droop = run_case(
        lambda t, y: case_gfm_proposed_stage3_frequency_event(
            t, y, p_evt, n_blocks=n_blocks, pm0_pu_sys=pm0_pu_sys, enable_droop=True
        ),
        y0_evt,
        p_evt.t_end_s,
        p_evt.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    if not sol_evt_base.success:
        raise RuntimeError(f"Stage-3 baseline run failed: {sol_evt_base.message}")
    if not sol_evt_droop.success:
        raise RuntimeError(f"Stage-3 droop run failed: {sol_evt_droop.message}")

    t_evt = sol_evt_base.t
    f_evt_base = 60.0 * (1.0 + sol_evt_base.y[9])
    f_evt_droop = 60.0 * (1.0 + sol_evt_droop.y[9])

    # Do not share the y-axis: the left panel includes the worst-case GFL-MC excursion,
    # which compresses the (smaller) Stage-3 benefit in the right panel.
    # R1 fix (Reviewer 3.5a): increase figure height and top margin so the panel
    # subtitles "Fault Aggregation Proxy" / "Frequency Event (No Fault)" render
    # without clipping when scaled to IEEE \columnwidth.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), sharey=False)

    ax = axes[0]
    ax.plot(t, f_gfl_mc, "r--", label=f"GFL-MC (N={n_blocks})")
    ax.plot(t, f_gfl_pll, color="m", linestyle="--", label=f"GFL-PLL (N={n_blocks})")
    ax.plot(t, f_gfm, "b-", label=f"Proposed GFM (N={n_blocks})")
    ax.axhline(60.0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvspan(p.fault_start_s, p.fault_end_s, color="yellow", alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f (Hz)")
    ax.set_title("Fault Aggregation Proxy")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)
    # Panel-specific y-limits (pad by a small margin for readability)
    f_left = np.concatenate([f_gfl_mc, f_gfl_pll, f_gfm])
    ypad = 0.02
    ax.set_ylim(float(f_left.min() - ypad), float(f_left.max() + ypad))

    ax2 = axes[1]
    ax2.plot(t_evt, f_evt_base, "k--", label="No FFR")
    ax2.plot(t_evt, f_evt_droop, "g-", label="Stage 3 Droop/FFR")
    ax2.axhline(60.0, color="k", linewidth=0.8, alpha=0.5)
    ax2.axvline(p.sys_event_t_s, color="gray", linestyle=":", alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Frequency Event (No Fault)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", framealpha=0.9)
    f_right = np.concatenate([f_evt_base, f_evt_droop])
    ypad_r = 0.01
    ax2.set_ylim(float(f_right.min() - ypad_r), float(f_right.max() + ypad_r))
    # Use the paper caption instead of an in-figure title to avoid cramped layout in IEEE columns.
    # R1 fix (Reviewer 3.5a): top=0.88 leaves headroom for "Fault Aggregation Proxy" /
    # "Frequency Event (No Fault)" subtitles when the figure is scaled to single-column width.
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.88, wspace=0.22)
    plt.savefig(out_path, dpi=500)
