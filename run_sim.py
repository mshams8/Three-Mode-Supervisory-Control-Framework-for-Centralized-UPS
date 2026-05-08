import argparse

from common import SimParams
from controllers import case_gfl_pll_lvrt, case_gfl_standard, case_gfm_proposed, case_gfm_proposed_with_stage3_x, gfm_in_fault_mode
from integration import run_case
from emt_sim import run_emt_case
from plots import build_freq_plot, build_plots, build_pulse_plot, build_ramp_plot, build_stage1_stiffbus_plot


def main() -> int:
    parser = argparse.ArgumentParser(description="Averaged dq simulation for GFL vs proposed GFM control.")
    parser.add_argument("--scr", type=float, default=1.5)
    parser.add_argument("--xr", type=float, default=5.0)
    parser.add_argument("--fault-v", type=float, default=0.5)
    parser.add_argument("--fault-start", type=float, default=0.50)
    parser.add_argument("--fault-end", type=float, default=0.65)
    parser.add_argument("--t-end", type=float, default=6.0)
    parser.add_argument("--load-step-t", type=float, default=0.10)
    parser.add_argument("--load-pu", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="simulation_results.png")
    parser.add_argument("--pulse-out", type=str, default="")
    parser.add_argument("--freq-out", type=str, default="simulation_results_freq.png")
    parser.add_argument("--stage1-plot-out", type=str, default="", help="Write a Stage-1 stiff-bus plot to this path.")
    parser.add_argument("--emt-out", type=str, default="", help="Write a dq-vs-EMT comparison plot to this path.")
    parser.add_argument("--emt-dt", type=float, default=50e-6, help="EMT fixed-step size (s) used with --emt-out.")
    parser.add_argument(
        "--comparison-out",
        type=str,
        default="",
        help="Write Case 1/2/3 comparison metrics (CSV) to this path.",
    )
    parser.add_argument("--stress-out", type=str, default="")
    parser.add_argument("--ablations-out", type=str, default="", help="Write Stage-2 ablation metrics (CSV) to this path.")
    parser.add_argument("--sweep-out", type=str, default="", help="Write robustness sweeps (CSV) to this path.")
    parser.add_argument("--dt", type=float, default=0.0, help="Fixed-step size (s). If >0, uses RK4 by default.")
    parser.add_argument(
        "--rk", type=str, default="rk4", choices=["rk2", "rk4"], help="Fixed-step integrator (requires --dt)."
    )
    parser.add_argument(
        "--ds-rate", type=int, default=1, help="Downsample saving: keep every Nth step (requires --dt)."
    )
    parser.add_argument(
        "--dump-params",
        type=str,
        default="",
        help="Write the resolved simulation parameter set (JSON) to this path.",
    )
    parser.add_argument("--stage1", action="store_true", help="Enable Stage 1 DC stiff-bus regulation (normal operation).")
    parser.add_argument("--stage3", action="store_true", help="Enable Stage 3 droop/FFR load modulation (requires frequency input).")
    # New scenarios for reviewer response
    parser.add_argument("--fault-type", type=str, default="balanced", choices=["balanced", "slg", "ll", "llg"],
                        help="Fault type: balanced, slg (single-line-to-ground), ll (line-to-line), llg (double-line-to-ground).")
    parser.add_argument("--asymmetric-out", type=str, default="", help="Write asymmetric fault comparison plot to this path.")
    parser.add_argument("--broadband-out", type=str, default="", help="Write broadband workload attenuation plot to this path.")
    parser.add_argument("--switching-out", type=str, default="", help="Write switching vs averaged EMT comparison plot to this path.")
    parser.add_argument("--reverse-out", type=str, default="", help="Write GFM-to-normal reverse transition plot to this path.")
    parser.add_argument("--workload-type", type=str, default="single_freq",
                        choices=["constant", "single_freq", "broadband", "training", "inference"],
                        help="AI workload model type.")
    parser.add_argument(
        "--stage3-freq-csv",
        type=str,
        default="",
        help="CSV with columns (t_s,f_hz) to drive Stage 3 (no extra proxy state).",
    )
    parser.add_argument(
        "--stage3-n-blocks",
        type=int,
        default=0,
        help="Aggregation count N used for Stage-3 system-base mapping (0 uses SimParams.sys_n_blocks).",
    )
    args = parser.parse_args()

    p = SimParams(
        scr=args.scr,
        xr_ratio=args.xr,
        fault_v_pu=args.fault_v,
        fault_start_s=args.fault_start,
        fault_end_s=args.fault_end,
        t_end_s=args.t_end,
        load_step_t_s=args.load_step_t,
        load_step_pu=args.load_pu,
        stage1_enable=bool(args.stage1),
        stage3_enable=bool(args.stage3),
        fault_type=args.fault_type,
        workload_type=args.workload_type,
    )
    if args.dump_params:
        import json

        with open(args.dump_params, "w", encoding="utf-8") as f:
            json.dump(p.__dict__, f, indent=2, sort_keys=True)

    print("Running Case 1: Standard GFL benchmark...")
    # Approximate steady state at unity voltage, Q=0: P = -1 pu => id ≈ P/(1.5*V) = -0.666..
    # Approximate steady state at unity voltage, Q=0: P = -1 pu => i_d ≈ P/(1.5*V) = -0.666..
    y0_gfl = [-2.0 / 3.0, 0.0, 0.0, 0.0, p.v_dc_init_pu]
    sol_gfl = run_case(
        lambda t, y: case_gfl_standard(t, y, p),
        y0_gfl,
        p.t_end_s,
        p.points,
        dt=(args.dt if args.dt > 0 else None),
        rk=args.rk,
        ds_rate=args.ds_rate,
    )

    print("Running Case 2: GFL with PLL + LVRT baseline...")
    y0_gfl_pll = [-2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, p.v_dc_init_pu]
    sol_gfl_pll = run_case(
        lambda t, y: case_gfl_pll_lvrt(t, y, p),
        y0_gfl_pll,
        p.t_end_s,
        p.points,
        dt=(args.dt if args.dt > 0 else None),
        rk=args.rk,
        ds_rate=args.ds_rate,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )

    print("Running Case 3: Proposed GFM control...")
    # Start near steady state: grid supplies load, BESS idle.
    y0_gfm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p.soc_init, p.v_dc_init_pu, 0.0]

    n_blocks_stage3 = p.sys_n_blocks if args.stage3_n_blocks <= 0 else int(args.stage3_n_blocks)
    if args.stage3 and args.stage3_freq_csv:
        import csv
        import numpy as np

        t_s: list[float] = []
        f_hz: list[float] = []
        with open(args.stage3_freq_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("Stage-3 freq CSV must contain headers (t_s,f_hz)")
            for row in reader:
                if row is None:
                    continue
                try:
                    t_s.append(float(row["t_s"]))
                    f_hz.append(float(row["f_hz"]))
                except KeyError as e:
                    raise ValueError("Stage-3 freq CSV must contain headers t_s and f_hz") from e
                except ValueError:
                    continue
        if len(t_s) < 2:
            raise ValueError("Stage-3 freq CSV must have at least 2 numeric rows")
        t_s_arr = np.asarray(t_s, dtype=float)
        f_hz_arr = np.asarray(f_hz, dtype=float)

        def x_of_t(t_now: float) -> float:
            f_now = float(np.interp(t_now, t_s_arr, f_hz_arr, left=float(f_hz_arr[0]), right=float(f_hz_arr[-1])))
            return (f_now - 60.0) / 60.0

        rhs_gfm = lambda t, y: case_gfm_proposed_with_stage3_x(t, y, p, x_sys=x_of_t(t), n_blocks=n_blocks_stage3)
    else:
        rhs_gfm = lambda t, y: case_gfm_proposed(t, y, p)
    sol_gfm = run_case(
        rhs_gfm,
        y0_gfm,
        p.t_end_s,
        p.points,
        dt=(args.dt if args.dt > 0 else None),
        rk=args.rk,
        ds_rate=args.ds_rate,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )

    build_plots(sol_gfl, sol_gfl_pll, sol_gfm, p, args.out)
    print(f"Simulation complete. Plot saved to {args.out}")

    build_freq_plot(sol_gfl, sol_gfl_pll, sol_gfm, p, args.freq_out, n_blocks=p.sys_n_blocks)
    print(f"Frequency proxy plot saved to {args.freq_out}")

    if args.emt_out:
        import matplotlib.pyplot as plt
        import numpy as np

        print("Running EMT-style (abc) validation case for Proposed controller...")
        # Align the EMT-style abc run with the averaged-dq scenario history (startup + load step),
        # then apply the fault later so both traces are settled and directly comparable.
        # Keep the same Mode-2 ramp limit used elsewhere. To ensure both models reach the same
        # pre-fault operating point (Pdraw ~= 1.0 pu), place the dip after the ramp has settled.
        # With R_limit = 0.2 pu/s, reaching 1.0 pu takes ~5 s.
        p_emt = SimParams(
            **{
                **p.__dict__,
                "t_end_s": 7.0,
                "points": 14001,
                "load_step_t_s": p.load_step_t_s,
                "load_step_pu": p.load_step_pu,
                "fault_start_s": 6.0,
                "fault_end_s": 6.15,
            }
        )
        emt = run_emt_case("GFM", p_emt, dt=args.emt_dt, t_end_s=p_emt.t_end_s, ds_rate=10, max_inner_iters=1)

        # Compute dq and power from EMT histories in the grid frame (theta = w t).
        t = emt.t
        w = 2.0 * np.pi * 60.0
        theta = w * t
        # Clarke + Park (vectorized)
        va, vb, vc = emt.v_pcc_abc[0], emt.v_pcc_abc[1], emt.v_pcc_abc[2]
        ia, ib, ic = emt.i_abc[0], emt.i_abc[1], emt.i_abc[2]
        v_alpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
        v_beta = (2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (vb - vc))
        i_alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
        i_beta = (2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (ib - ic))
        c = np.cos(theta)
        s = np.sin(theta)
        v_d = v_alpha * c + v_beta * s
        v_q = -v_alpha * s + v_beta * c
        i_d = i_alpha * c + i_beta * s
        i_q = -i_alpha * s + i_beta * c

        # For an apples-to-apples comparison with the averaged-dq model, compute the *phasor-equivalent*
        # PCC voltage implied by the same Thevenin impedance model used in the averaged equations:
        #   V_pcc = V_th + Z_th I  (in the grid dq frame, v_th_q = 0 by definition).
        # This avoids interpreting EMT inductor transients as a "stronger grid" when plotting |V_pcc|.
        d = p_emt.derived()
        v_th_d = np.where((t >= p_emt.fault_start_s) & (t <= p_emt.fault_end_s), p_emt.fault_v_pu, 1.0)
        v_th_q = 0.0
        v_pcc_d_eq = v_th_d + d["r_th"] * i_d - d["x_th"] * i_q
        v_pcc_q_eq = v_th_q + d["r_th"] * i_q + d["x_th"] * i_d
        v_abs = np.sqrt(v_pcc_d_eq * v_pcc_d_eq + v_pcc_q_eq * v_pcc_q_eq)
        p_grid = 1.5 * (v_pcc_d_eq * i_d + v_pcc_q_eq * i_q)
        p_draw = -p_grid
        i_abs = np.sqrt(i_d * i_d + i_q * i_q)

        # DQ proposed trace (same scenario as EMT, averaged-dq plant)
        p_short = p_emt
        y0_gfm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_short.soc_init, p_short.v_dc_init_pu, 0.0]
        sol_gfm_short = run_case(
            lambda tt, yy: case_gfm_proposed(tt, yy, p_short),
            y0_gfm,
            p_short.t_end_s,
            p_short.points,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            max_step=1e-3,
        )

        # Compute dq-derived Pdraw and |Vpcc| for the dq run
        t_dq = sol_gfm_short.t
        id_dq, iq_dq = sol_gfm_short.y[0], sol_gfm_short.y[1]
        # reuse existing v_pcc computation via dq phasor equivalent
        from dq_plant import v_pcc_from_grid_impedance
        from common import v_th_dq_grid_frame
        d = p_short.derived()
        vpcc_abs_dq = np.zeros_like(t_dq, dtype=float)
        pdraw_dq = np.zeros_like(t_dq, dtype=float)
        for k in range(t_dq.size):
            v_th_d, v_th_q = v_th_dq_grid_frame(float(t_dq[k]), p_short)
            v_pcc_d, v_pcc_q = v_pcc_from_grid_impedance(
                v_th_d, v_th_q, float(id_dq[k]), float(iq_dq[k]), d["r_th"], d["x_th"]
            )
            vpcc_abs_dq[k] = float(np.sqrt(v_pcc_d * v_pcc_d + v_pcc_q * v_pcc_q))
            p_grid_k = 1.5 * (v_pcc_d * float(id_dq[k]) + v_pcc_q * float(iq_dq[k]))
            pdraw_dq[k] = -p_grid_k

        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"font.size": 11})
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.6), sharex=True)

        ax = axes[0]
        ax.plot(t_dq, pdraw_dq, "k--", linewidth=1.2, label="Averaged $dq$ (Proposed)")
        ax.plot(t, p_draw, "b-", linewidth=1.0, label="EMT-style $abc$ (avg. VSC)")
        ax.axvspan(p_emt.fault_start_s, p_emt.fault_end_s, color="yellow", alpha=0.2)
        ax.set_ylabel("$P_{draw}$ (p.u.)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.9)

        ax = axes[1]
        ax.plot(t_dq, vpcc_abs_dq, "k--", linewidth=1.2, label="Averaged $dq$")
        ax.plot(t, v_abs, "b-", linewidth=1.0, label="EMT-style $abc$")
        ax.axvspan(p_emt.fault_start_s, p_emt.fault_end_s, color="yellow", alpha=0.2)
        ax.set_ylabel("$|V_{pcc}|$ (p.u.)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.9)

        ax = axes[2]
        ax.plot(t_dq, np.sqrt(sol_gfm_short.y[0] ** 2 + sol_gfm_short.y[1] ** 2), "k--", linewidth=1.2, label="Averaged $dq$")
        ax.plot(t, i_abs, "b-", linewidth=1.0, label="EMT-style $abc$")
        ax.axvspan(p_emt.fault_start_s, p_emt.fault_end_s, color="yellow", alpha=0.2)
        ax.set_ylabel("$|I|$ (p.u.)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.9)

        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.98, hspace=0.25)
        plt.savefig(args.emt_out, dpi=450)
        print(f"EMT comparison plot saved to {args.emt_out}")

    if args.comparison_out:
        from metrics import (
            compute_fault_max_current,
            compute_fault_min_vdc,
            compute_fault_min_vpcc,
            compute_fault_min_vpcc_settled,
            compute_stage1_unserved_energy_mwh,
        )

        rows: list[str] = []
        settle_s = 1.0 / 60.0  # one 60 Hz cycle
        header = "case,min_vpcc,min_vpcc_settle_1cyc,max_i,min_vdc,unserved_mwh"
        rows.append(header)
        rows.append(
            "GFL-MC,"
            f"{compute_fault_min_vpcc(sol_gfl, p):.6f},"
            f"{compute_fault_min_vpcc_settled(sol_gfl, p, settle_s=settle_s):.6f},"
            f"{compute_fault_max_current(sol_gfl, p):.6f},"
            f"{compute_fault_min_vdc(sol_gfl, p, vdc_index=4):.6f},"
            f"{compute_stage1_unserved_energy_mwh(sol_gfl, p, is_gfm=False):.6f}"
        )
        rows.append(
            "GFL-PLL,"
            f"{compute_fault_min_vpcc(sol_gfl_pll, p):.6f},"
            f"{compute_fault_min_vpcc_settled(sol_gfl_pll, p, settle_s=settle_s):.6f},"
            f"{compute_fault_max_current(sol_gfl_pll, p):.6f},"
            f"{compute_fault_min_vdc(sol_gfl_pll, p, vdc_index=6):.6f},"
            f"{compute_stage1_unserved_energy_mwh(sol_gfl_pll, p, is_gfm=False):.6f}"
        )
        rows.append(
            "Proposed,"
            f"{compute_fault_min_vpcc(sol_gfm, p):.6f},"
            f"{compute_fault_min_vpcc_settled(sol_gfm, p, settle_s=settle_s):.6f},"
            f"{compute_fault_max_current(sol_gfm, p):.6f},"
            f"{compute_fault_min_vdc(sol_gfm, p, vdc_index=7):.6f},"
            f"{compute_stage1_unserved_energy_mwh(sol_gfm, p, is_gfm=True):.6f}"
        )
        with open(args.comparison_out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Comparison metrics saved to {args.comparison_out}")

    if args.stress_out:
        from metrics import (
            compute_fault_max_current,
            compute_fault_mean_pdraw,
            compute_fault_min_vdc,
            compute_fault_min_vpcc,
            compute_fault_min_vpcc_settled,
            compute_stage1_unserved_energy_mwh,
        )

        print("Running Case 4: Stress tests (graceful degradation)...")
        stress_cases: list[tuple[str, dict[str, float]]] = [
            ("Baseline", {}),
            ("Low DC energy", {"dc_link_energy_s": 0.10}),
            ("Detection delay (10 ms)", {"fault_detect_delay_s": 0.010}),
            ("Slow BESS response (tau_bess=0.10 s)", {"bess_tau_s": 0.10}),
            ("Low BESS power", {"bess_p_dis_max_pu": 0.30}),
            ("Low BESS ramp", {"bess_ramp_pu_s": 1.00}),
            ("No BESS", {"bess_p_dis_max_pu": 0.0, "bess_p_chg_max_pu": 0.0}),
        ]

        rows: list[str] = []
        header = "case,min_vpcc,min_vpcc_settle_1cyc,max_i,mean_pdraw,min_vdc,unserved_mwh"
        rows.append(header)
        for name, overrides in stress_cases:
            p_case = SimParams(**{**p.__dict__, **overrides})
            sol_case = run_case(
                lambda t, y: case_gfm_proposed(t, y, p_case),
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_case.soc_init, p_case.v_dc_init_pu, 0.0],
                p_case.t_end_s,
                p_case.points,
                dt=(args.dt if args.dt > 0 else None),
                rk=args.rk,
                ds_rate=args.ds_rate,
                method="RK45",
                rtol=1e-4,
                atol=1e-6,
                max_step=1e-3,
            )
            if not sol_case.success:
                raise RuntimeError(f"Stress case '{name}' failed: {sol_case.message}")
            settle_s = 1.0 / 60.0  # one 60 Hz cycle
            min_vpcc = compute_fault_min_vpcc(sol_case, p_case)
            min_vpcc_settle = compute_fault_min_vpcc_settled(sol_case, p_case, settle_s=settle_s)
            max_i = compute_fault_max_current(sol_case, p_case)
            mean_pdraw = compute_fault_mean_pdraw(sol_case, p_case)
            min_vdc = compute_fault_min_vdc(sol_case, p_case, 7)
            unserved_mwh = compute_stage1_unserved_energy_mwh(sol_case, p_case, is_gfm=True)
            rows.append(
                f"{name},{min_vpcc:.6f},{min_vpcc_settle:.6f},{max_i:.6f},{mean_pdraw:.6f},{min_vdc:.6f},{unserved_mwh:.6f}"
            )

        with open(args.stress_out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Stress-test metrics saved to {args.stress_out}")

    if args.ablations_out:
        from metrics import (
            compute_fault_max_current,
            compute_fault_mean_pdraw,
            compute_fault_min_vdc,
            compute_fault_min_vpcc,
            compute_fault_min_vpcc_settled,
            compute_postfault_max_dpdraw_ref,
            compute_stage1_unserved_energy_mwh,
        )

        print("Running Stage-2 ablations...")
        # Variants intended to isolate contributions:
        #  - No Q-support: disable reactive priority (Kv=0)
        #  - No min-draw policy: Pdraw,min^fault = 0
        #  - No soft return: remove ramp constraint (very large Rlimit) and very small grid_tau
        #  - No BESS: remove BESS power capability
        ablation_cases: list[tuple[str, dict[str, float]]] = [
            ("Baseline", {}),
            ("No Q-support (Kv=0)", {"k_v": 0.0}),
            ("No min draw policy", {"p_fault_min_grid_draw_pu": 0.0}),
            ("No soft return", {"r_limit_mw_s": 1e6, "grid_tau_s": 0.01}),
            ("No BESS", {"bess_p_dis_max_pu": 0.0, "bess_p_chg_max_pu": 0.0}),
        ]

        rows: list[str] = []
        header = "case,min_vpcc,min_vpcc_settle_1cyc,max_i,mean_pdraw,min_vdc,unserved_mwh,max_dpdraw_pu_s"
        rows.append(header)
        for name, overrides in ablation_cases:
            p_case = SimParams(**{**p.__dict__, **overrides})
            sol_case = run_case(
                lambda t, y: case_gfm_proposed(t, y, p_case),
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_case.soc_init, p_case.v_dc_init_pu, 0.0],
                p_case.t_end_s,
                p_case.points,
                dt=(args.dt if args.dt > 0 else None),
                rk=args.rk,
                ds_rate=args.ds_rate,
                method="RK45",
                rtol=1e-4,
                atol=1e-6,
                max_step=1e-3,
            )
            if not sol_case.success:
                raise RuntimeError(f"Ablation case '{name}' failed: {sol_case.message}")
            settle_s = 1.0 / 60.0  # one 60 Hz cycle
            min_vpcc = compute_fault_min_vpcc(sol_case, p_case)
            min_vpcc_settle = compute_fault_min_vpcc_settled(sol_case, p_case, settle_s=settle_s)
            max_i = compute_fault_max_current(sol_case, p_case)
            mean_pdraw = compute_fault_mean_pdraw(sol_case, p_case)
            min_vdc = compute_fault_min_vdc(sol_case, p_case, 7)
            unserved_mwh = compute_stage1_unserved_energy_mwh(sol_case, p_case, is_gfm=True)
            max_dpdraw = compute_postfault_max_dpdraw_ref(sol_case, p_case, pdraw_ref_index=4)
            rows.append(
                f"{name},{min_vpcc:.6f},{min_vpcc_settle:.6f},{max_i:.6f},{mean_pdraw:.6f},{min_vdc:.6f},{unserved_mwh:.6f},{max_dpdraw:.3f}"
            )

        with open(args.ablations_out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Ablation metrics saved to {args.ablations_out}")

    if args.sweep_out:
        from metrics import (
            compute_fault_max_abs_pbess,
            compute_fault_max_current,
            compute_fault_mean_pdraw,
            compute_fault_min_soc,
            compute_fault_min_vdc,
            compute_fault_min_vpcc,
            compute_fault_min_vpcc_settled,
            compute_postfault_max_dpdraw_ref,
            compute_stage1_unserved_energy_mwh,
        )

        print("Running Stage-2 robustness sweeps (SCR, dip depth, key parameters)...")

        def _run_gfm(p_case: SimParams):
            # Faster horizon for sweeps: include dip + short recovery window.
            y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_case.soc_init, p_case.v_dc_init_pu, 0.0]
            sol = run_case(
                lambda t, y: case_gfm_proposed(t, y, p_case),
                y0,
                p_case.t_end_s,
                p_case.points,
                dt=(args.dt if args.dt > 0 else None),
                rk=args.rk,
                ds_rate=args.ds_rate,
                method="RK45",
                rtol=1e-4,
                atol=1e-6,
                max_step=1e-3,
            )
            if not sol.success:
                raise RuntimeError(f"Sweep run failed: {sol.message}")
            settle_s = 1.0 / 60.0
            return {
                "min_vpcc": compute_fault_min_vpcc(sol, p_case),
                "min_vpcc_settle_1cyc": compute_fault_min_vpcc_settled(sol, p_case, settle_s=settle_s),
                "max_i": compute_fault_max_current(sol, p_case),
                "mean_pdraw": compute_fault_mean_pdraw(sol, p_case),
                "min_vdc": compute_fault_min_vdc(sol, p_case, 7),
                "unserved_mwh": compute_stage1_unserved_energy_mwh(sol, p_case, is_gfm=True),
                "max_dpdraw_pu_s": compute_postfault_max_dpdraw_ref(sol, p_case, pdraw_ref_index=4),
                "max_abs_pbess": compute_fault_max_abs_pbess(sol, p_case, pbess_index=5),
                "min_soc": compute_fault_min_soc(sol, p_case, soc_index=6),
            }

        # Base for sweeps: match the main-case time resolution while limiting the horizon to what is
        # required for fault-window metrics and the post-fault ramp metric (window_s=0.25 s).
        # Main default is 6 s with 12001 points (~0.5 ms sampling). Here we keep ~0.5 ms sampling.
        p_fast = SimParams(**{**p.__dict__, "t_end_s": 1.0, "points": 2001})

        rows: list[str] = []
        header = (
            "sweep,case,label,scr,fault_v,fault_dur_s,k_v,p_fault_min_grid_draw_pu,r_limit_mw_s,fault_detect_delay_s,"
            "min_vpcc,min_vpcc_settle_1cyc,max_i,mean_pdraw,min_vdc,unserved_mwh,max_dpdraw_pu_s,max_abs_pbess,min_soc"
        )
        rows.append(header)

        # 1) SCR sweep (fixed dip)
        for scr in [1.2, 1.5, 2.0, 3.0]:
            p_case = SimParams(**{**p_fast.__dict__, "scr": scr})
            m = _run_gfm(p_case)
            rows.append(
                "SCR,"
                "Proposed,"
                f"scr={scr:.1f},"
                f"{p_case.scr:.3f},{p_case.fault_v_pu:.3f},{(p_case.fault_end_s-p_case.fault_start_s):.3f},"
                f"{p_case.k_v:.3f},{p_case.p_fault_min_grid_draw_pu:.3f},{p_case.r_limit_mw_s:.3f},{p_case.fault_detect_delay_s:.6f},"
                f"{m['min_vpcc']:.6f},{m['min_vpcc_settle_1cyc']:.6f},{m['max_i']:.6f},{m['mean_pdraw']:.6f},{m['min_vdc']:.6f},"
                f"{m['unserved_mwh']:.6f},{m['max_dpdraw_pu_s']:.3f},{m['max_abs_pbess']:.6f},{m['min_soc']:.6f}"
            )

        # 2) Dip depth sweep (fixed SCR)
        for vdip in [0.4, 0.5, 0.7]:
            p_case = SimParams(**{**p_fast.__dict__, "fault_v_pu": vdip})
            m = _run_gfm(p_case)
            rows.append(
                "Dip,"
                "Proposed,"
                f"vdip={vdip:.1f},"
                f"{p_case.scr:.3f},{p_case.fault_v_pu:.3f},{(p_case.fault_end_s-p_case.fault_start_s):.3f},"
                f"{p_case.k_v:.3f},{p_case.p_fault_min_grid_draw_pu:.3f},{p_case.r_limit_mw_s:.3f},{p_case.fault_detect_delay_s:.6f},"
                f"{m['min_vpcc']:.6f},{m['min_vpcc_settle_1cyc']:.6f},{m['max_i']:.6f},{m['mean_pdraw']:.6f},{m['min_vdc']:.6f},"
                f"{m['unserved_mwh']:.6f},{m['max_dpdraw_pu_s']:.3f},{m['max_abs_pbess']:.6f},{m['min_soc']:.6f}"
            )

        # 3) Policy/gain sweeps (fixed SCR and dip)
        for pfault in [0.0, 0.1, 0.2, 0.3]:
            p_case = SimParams(**{**p_fast.__dict__, "p_fault_min_grid_draw_pu": pfault})
            m = _run_gfm(p_case)
            rows.append(
                "Pmin,"
                "Proposed,"
                f"pmin={pfault:.1f},"
                f"{p_case.scr:.3f},{p_case.fault_v_pu:.3f},{(p_case.fault_end_s-p_case.fault_start_s):.3f},"
                f"{p_case.k_v:.3f},{p_case.p_fault_min_grid_draw_pu:.3f},{p_case.r_limit_mw_s:.3f},{p_case.fault_detect_delay_s:.6f},"
                f"{m['min_vpcc']:.6f},{m['min_vpcc_settle_1cyc']:.6f},{m['max_i']:.6f},{m['mean_pdraw']:.6f},{m['min_vdc']:.6f},"
                f"{m['unserved_mwh']:.6f},{m['max_dpdraw_pu_s']:.3f},{m['max_abs_pbess']:.6f},{m['min_soc']:.6f}"
            )

        for kv in [1.5, 2.5, 3.5]:
            p_case = SimParams(**{**p_fast.__dict__, "k_v": kv})
            m = _run_gfm(p_case)
            rows.append(
                "Kv,"
                "Proposed,"
                f"kv={kv:.1f},"
                f"{p_case.scr:.3f},{p_case.fault_v_pu:.3f},{(p_case.fault_end_s-p_case.fault_start_s):.3f},"
                f"{p_case.k_v:.3f},{p_case.p_fault_min_grid_draw_pu:.3f},{p_case.r_limit_mw_s:.3f},{p_case.fault_detect_delay_s:.6f},"
                f"{m['min_vpcc']:.6f},{m['min_vpcc_settle_1cyc']:.6f},{m['max_i']:.6f},{m['mean_pdraw']:.6f},{m['min_vdc']:.6f},"
                f"{m['unserved_mwh']:.6f},{m['max_dpdraw_pu_s']:.3f},{m['max_abs_pbess']:.6f},{m['min_soc']:.6f}"
            )

        for rlim in [5.0, 10.0, 20.0]:
            p_case = SimParams(**{**p_fast.__dict__, "r_limit_mw_s": rlim})
            m = _run_gfm(p_case)
            rows.append(
                "Rlimit,"
                "Proposed,"
                f"rlim={rlim:.0f},"
                f"{p_case.scr:.3f},{p_case.fault_v_pu:.3f},{(p_case.fault_end_s-p_case.fault_start_s):.3f},"
                f"{p_case.k_v:.3f},{p_case.p_fault_min_grid_draw_pu:.3f},{p_case.r_limit_mw_s:.3f},{p_case.fault_detect_delay_s:.6f},"
                f"{m['min_vpcc']:.6f},{m['min_vpcc_settle_1cyc']:.6f},{m['max_i']:.6f},{m['mean_pdraw']:.6f},{m['min_vdc']:.6f},"
                f"{m['unserved_mwh']:.6f},{m['max_dpdraw_pu_s']:.3f},{m['max_abs_pbess']:.6f},{m['min_soc']:.6f}"
            )

        for tdet in [0.002, 0.005, 0.010]:
            p_case = SimParams(**{**p_fast.__dict__, "fault_detect_delay_s": tdet})
            m = _run_gfm(p_case)
            rows.append(
                "Tdet,"
                "Proposed,"
                f"tdet={tdet:.3f},"
                f"{p_case.scr:.3f},{p_case.fault_v_pu:.3f},{(p_case.fault_end_s-p_case.fault_start_s):.3f},"
                f"{p_case.k_v:.3f},{p_case.p_fault_min_grid_draw_pu:.3f},{p_case.r_limit_mw_s:.3f},{p_case.fault_detect_delay_s:.6f},"
                f"{m['min_vpcc']:.6f},{m['min_vpcc_settle_1cyc']:.6f},{m['max_i']:.6f},{m['mean_pdraw']:.6f},{m['min_vdc']:.6f},"
                f"{m['unserved_mwh']:.6f},{m['max_dpdraw_pu_s']:.3f},{m['max_abs_pbess']:.6f},{m['min_soc']:.6f}"
            )

        with open(args.sweep_out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Sweep metrics saved to {args.sweep_out}")

    if args.pulse_out:
        print("Running Case 3: Pulsed load smoothing (forced oscillation filtering)...")
        # Pulse case: start from steady 1.0 pu load to avoid conflating the startup ramp with oscillation filtering.
        p_pulse = SimParams(
            scr=p.scr,
            xr_ratio=p.xr_ratio,
            fault_v_pu=1.0,
            fault_start_s=10.0,
            fault_end_s=10.1,
            t_end_s=6.0,
            points=6001,
            load_step_t_s=0.0,
            load_step_pu=1.0,
            pulse_enable=True,
            pulse_amp_pu=p.pulse_amp_pu,
            pulse_freq_hz=p.pulse_freq_hz,
            pulse_start_s=1.0,
            pulse_end_s=5.0,
            grid_tau_s=p.grid_tau_s,
        )

        # Compare "no smoothing" vs "with smoothing" using the same BESS model.
        p_fast = SimParams(**{**p_pulse.__dict__, "grid_tau_s": 0.01})
        # Initialize at steady 1.0 pu grid draw with BESS idle.
        y0_fast = [-2.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_fast.soc_init, p_fast.v_dc_init_pu, 0.0]
        y0_smooth = [-2.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_pulse.soc_init, p_pulse.v_dc_init_pu, 0.0]
        sol_fast = run_case(
            lambda t, y: case_gfm_proposed(t, y, p_fast),
            y0_fast,
            p_fast.t_end_s,
            p_fast.points,
            dt=(args.dt if args.dt > 0 else None),
            rk=args.rk,
            ds_rate=args.ds_rate,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            max_step=1e-3,
        )
        sol_smooth = run_case(
            lambda t, y: case_gfm_proposed(t, y, p_pulse),
            y0_smooth,
            p_pulse.t_end_s,
            p_pulse.points,
            dt=(args.dt if args.dt > 0 else None),
            rk=args.rk,
            ds_rate=args.ds_rate,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            max_step=1e-3,
        )
        build_pulse_plot(sol_fast, sol_smooth, p_pulse, args.pulse_out)
        print(f"Pulsed-load plot saved to {args.pulse_out}")

        # Also write a separate ramping/shaping plot (load step, no pulsing) alongside the pulse plot.
        ramp_out = args.pulse_out
        if ramp_out.lower().endswith(".png"):
            ramp_out = ramp_out[:-4] + "_ramp.png"
        else:
            ramp_out = ramp_out + "_ramp.png"

        print("Running Case 3: Grid-draw shaping under a load step (ramping)...")
        p_ramp = SimParams(
            scr=p.scr,
            xr_ratio=p.xr_ratio,
            fault_v_pu=1.0,
            fault_start_s=10.0,
            fault_end_s=10.1,
            t_end_s=2.0,
            points=2001,
            load_step_t_s=0.10,
            load_step_pu=1.0,
            pulse_enable=False,
            grid_tau_s=p.grid_tau_s,
        )
        p_ramp_fast = SimParams(**{**p_ramp.__dict__, "grid_tau_s": 0.01})
        y0_ramp_fast = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_ramp_fast.soc_init, p_ramp_fast.v_dc_init_pu, 0.0]
        y0_ramp_smooth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_ramp.soc_init, p_ramp.v_dc_init_pu, 0.0]
        sol_ramp_fast = run_case(
            lambda t, y: case_gfm_proposed(t, y, p_ramp_fast),
            y0_ramp_fast,
            p_ramp_fast.t_end_s,
            p_ramp_fast.points,
            dt=(args.dt if args.dt > 0 else None),
            rk=args.rk,
            ds_rate=args.ds_rate,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            max_step=1e-3,
        )
        sol_ramp_smooth = run_case(
            lambda t, y: case_gfm_proposed(t, y, p_ramp),
            y0_ramp_smooth,
            p_ramp.t_end_s,
            p_ramp.points,
            dt=(args.dt if args.dt > 0 else None),
            rk=args.rk,
            ds_rate=args.ds_rate,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            max_step=1e-3,
        )
        build_ramp_plot(sol_ramp_fast, sol_ramp_smooth, p_ramp, ramp_out)
        print(f"Ramp/shaping plot saved to {ramp_out}")

    if args.stage1_plot_out:
        build_stage1_stiffbus_plot(p, args.stage1_plot_out)
        print(f"Stage-1 stiff-bus plot saved to {args.stage1_plot_out}")

    # ===================================================================
    # NEW SCENARIOS (addressing reviewer comments)
    # ===================================================================

    if args.asymmetric_out:
        import matplotlib.pyplot as plt
        import numpy as np

        print("Running asymmetric fault comparison (SLG, LL, LLG vs balanced)...")
        fault_types = ["balanced", "slg", "ll", "llg"]
        fault_labels = ["Balanced 3-ph", "SLG (phase A)", "LL (B-C)", "LLG (B-C-G)"]
        fault_colors = ["blue", "red", "green", "orange"]

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), sharex=True)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"font.size": 11})

        from dq_plant import v_pcc_from_grid_impedance
        from common import v_th_dq_grid_frame as vthdq

        for idx, (ft, label, color) in enumerate(zip(fault_types, fault_labels, fault_colors)):
            p_asym = SimParams(**{
                **p.__dict__,
                "fault_type": ft,
                "neg_seq_suppress": (ft != "balanced"),
            })
            # Build initial state vector with neg-seq filter states if needed
            y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_asym.soc_init, p_asym.v_dc_init_pu, 0.0]
            if p_asym.neg_seq_suppress and ft != "balanced":
                y0.extend([1.0, 0.0])  # vpcc_d_avg, vpcc_q_avg

            sol = run_case(
                lambda t, y, _p=p_asym: case_gfm_proposed(t, y, _p),
                y0,
                p_asym.t_end_s,
                p_asym.points,
                method="RK45", rtol=1e-4, atol=1e-6, max_step=1e-3,
            )

            t = sol.t
            id_g, iq_g = sol.y[0], sol.y[1]
            d = p_asym.derived()
            vpcc_abs = np.zeros_like(t)
            pdraw = np.zeros_like(t)
            for k in range(t.size):
                vthd, vthq = vthdq(float(t[k]), p_asym)
                vpd, vpq = v_pcc_from_grid_impedance(vthd, vthq, float(id_g[k]), float(iq_g[k]), d["r_th"], d["x_th"])
                vpcc_abs[k] = float(np.sqrt(vpd**2 + vpq**2))
                pg = 1.5 * (vpd * float(id_g[k]) + vpq * float(iq_g[k]))
                pdraw[k] = -pg
            i_mag = np.sqrt(id_g**2 + iq_g**2)

            axes[0, 0].plot(t, pdraw, color=color, linewidth=1.1, label=label)
            axes[0, 1].plot(t, vpcc_abs, color=color, linewidth=1.1, label=label)
            axes[1, 0].plot(t, i_mag, color=color, linewidth=1.1, label=label)
            axes[1, 1].plot(t, sol.y[7], color=color, linewidth=1.1, label=label)

        for ax in axes.flat:
            ax.axvspan(p.fault_start_s, p.fault_end_s, color="yellow", alpha=0.2)
            ax.grid(True, linestyle=":", alpha=0.6)

        axes[0, 0].set_ylabel("$P_{draw}$ (p.u.)")
        axes[0, 0].set_title("(a) Active Power")
        axes[0, 0].legend(fontsize=8, framealpha=0.9)
        axes[0, 1].set_ylabel("$|V_{pcc}|$ (p.u.)")
        axes[0, 1].set_title("(b) PCC Voltage")
        axes[0, 1].legend(fontsize=8, framealpha=0.9)
        axes[1, 0].set_ylabel("$|I|$ (p.u.)")
        axes[1, 0].set_title("(c) Current Magnitude")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("$V_{dc}$ (p.u.)")
        axes[1, 1].set_title("(d) DC-Link Proxy")
        axes[1, 1].set_xlabel("Time (s)")

        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.96, wspace=0.24, hspace=0.30)
        plt.savefig(args.asymmetric_out, dpi=500)
        print(f"Asymmetric fault comparison saved to {args.asymmetric_out}")

    if args.broadband_out:
        import matplotlib.pyplot as plt
        import numpy as np

        print("Running broadband workload attenuation comparison...")
        workload_types = ["single_freq", "broadband", "training", "inference"]
        workload_labels = ["Single 1 Hz", "Broadband (5 freq)", "Training batch", "Inference burst"]
        workload_colors = ["black", "blue", "red", "green"]

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), sharex=True)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"font.size": 11})

        from common import p_load_profile as plp

        for idx, (wt, label, color) in enumerate(zip(workload_types, workload_labels, workload_colors)):
            p_wl = SimParams(**{
                **p.__dict__,
                "fault_v_pu": 1.0, "fault_start_s": 100.0, "fault_end_s": 100.1,  # no fault
                "pulse_enable": True, "pulse_start_s": 1.0, "pulse_end_s": 5.0,
                "load_step_t_s": 0.0, "load_step_pu": 1.0,
                "workload_type": wt, "t_end_s": 6.0, "points": 12001,
            })
            y0 = [-2.0/3.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_wl.soc_init, p_wl.v_dc_init_pu, 0.0]
            sol = run_case(
                lambda t, y, _p=p_wl: case_gfm_proposed(t, y, _p),
                y0, p_wl.t_end_s, p_wl.points,
                method="RK45", rtol=1e-4, atol=1e-6, max_step=1e-3,
            )

            t = sol.t
            p_load = np.array([plp(float(tt), p_wl) for tt in t])

            from dq_plant import v_pcc_from_grid_impedance
            from common import v_th_dq_grid_frame as vthdq2
            d = p_wl.derived()
            pdraw = np.zeros_like(t)
            for k in range(t.size):
                vthd, vthq = vthdq2(float(t[k]), p_wl)
                vpd, vpq = v_pcc_from_grid_impedance(vthd, vthq, float(sol.y[0][k]), float(sol.y[1][k]), d["r_th"], d["x_th"])
                pg = 1.5 * (vpd * float(sol.y[0][k]) + vpq * float(sol.y[1][k]))
                pdraw[k] = -pg

            ax = axes[idx // 2, idx % 2]
            ax.plot(t, p_load, color="gray", linewidth=0.8, alpha=0.7, label="IT Load")
            ax.plot(t, pdraw, color=color, linewidth=1.1, label=f"Grid draw ({label})")
            ax.plot(t, sol.y[5], color=color, linewidth=0.8, linestyle="--", alpha=0.7, label="$P_{bess}$")
            ax.set_ylabel("Power (p.u.)")
            ax.set_title(f"({chr(97+idx)}) {label}")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=7.5, framealpha=0.9)
            if idx >= 2:
                ax.set_xlabel("Time (s)")

        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.96, wspace=0.22, hspace=0.32)
        plt.savefig(args.broadband_out, dpi=500)
        print(f"Broadband workload plot saved to {args.broadband_out}")

    if args.switching_out:
        import matplotlib.pyplot as plt
        import numpy as np
        from emt_sim import run_emt_case

        print("Running switching vs averaged EMT comparison...")

        p_sw = SimParams(**{
            **p.__dict__,
            "t_end_s": 7.0, "points": 14001,
            "fault_start_s": 6.0, "fault_end_s": 6.15,
        })

        # Averaged (non-switching) EMT
        emt_avg = run_emt_case("GFM", p_sw, dt=50e-6, t_end_s=p_sw.t_end_s, ds_rate=10)
        # Switching EMT
        p_sw_on = SimParams(**{**p_sw.__dict__, "enable_switching": True, "switching_freq_hz": 3060.0})
        emt_sw = run_emt_case("GFM", p_sw_on, dt=20e-6, t_end_s=p_sw_on.t_end_s, ds_rate=25)

        fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"font.size": 11})

        # Convert abc to dq for both runs
        for emt_data, lbl, col, ls in [(emt_avg, "Averaged", "black", "--"), (emt_sw, "Switched PWM", "blue", "-")]:
            t = emt_data.t
            w = 2.0 * np.pi * 60.0
            theta = w * t
            va, vb, vc = emt_data.v_pcc_abc[0], emt_data.v_pcc_abc[1], emt_data.v_pcc_abc[2]
            ia, ib, ic = emt_data.i_abc[0], emt_data.i_abc[1], emt_data.i_abc[2]
            v_alpha = (2.0/3.0) * (va - 0.5*vb - 0.5*vc)
            v_beta = (2.0/3.0) * ((np.sqrt(3)/2.0) * (vb - vc))
            i_alpha = (2.0/3.0) * (ia - 0.5*ib - 0.5*ic)
            i_beta = (2.0/3.0) * ((np.sqrt(3)/2.0) * (ib - ic))
            c, s = np.cos(theta), np.sin(theta)
            i_d = i_alpha * c + i_beta * s
            i_q = -i_alpha * s + i_beta * c
            # Compute Vpcc and Pdraw via Thevenin
            d = p_sw.derived()
            v_th_d_arr = np.where((t >= p_sw.fault_start_s) & (t <= p_sw.fault_end_s), p_sw.fault_v_pu, 1.0)
            vpd = v_th_d_arr + d["r_th"] * i_d - d["x_th"] * i_q
            vpq = d["r_th"] * i_q + d["x_th"] * i_d
            v_abs = np.sqrt(vpd**2 + vpq**2)
            pdraw = -(1.5 * (vpd * i_d + vpq * i_q))
            i_abs = np.sqrt(i_d**2 + i_q**2)

            axes[0].plot(t, pdraw, color=col, linestyle=ls, linewidth=1.0, label=lbl)
            axes[1].plot(t, v_abs, color=col, linestyle=ls, linewidth=1.0, label=lbl)
            axes[2].plot(t, i_abs, color=col, linestyle=ls, linewidth=1.0, label=lbl)

        for ax in axes:
            ax.axvspan(p_sw.fault_start_s, p_sw.fault_end_s, color="yellow", alpha=0.2)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(framealpha=0.9)

        axes[0].set_ylabel("$P_{draw}$ (p.u.)")
        axes[1].set_ylabel("$|V_{pcc}|$ (p.u.)")
        axes[2].set_ylabel("$|I|$ (p.u.)")
        axes[2].set_xlabel("Time (s)")
        axes[0].set_title("Switching EMT vs Averaged EMT Comparison")

        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.95, hspace=0.25)
        plt.savefig(args.switching_out, dpi=450)
        print(f"Switching comparison saved to {args.switching_out}")

    if args.reverse_out:
        import matplotlib.pyplot as plt
        import numpy as np
        from common import v_th_dq_grid_frame as vthdq3
        from dq_plant import v_pcc_from_grid_impedance

        print("Running reverse transition (Mode 2 -> Mode 1) demonstration...")

        # Single continuous simulation with two sequential fault windows.
        # Fault 1: t=0.50-0.65 s, Fault 2: t=1.50-1.65 s.
        # Mode 1 enabled throughout so SoC recovery is visible between faults.
        fault_w = ((0.50, 0.65), (1.50, 1.65))
        p_rev = SimParams(**{
            **p.__dict__,
            "fault_start_s": fault_w[0][0], "fault_end_s": fault_w[0][1],
            "fault_windows": fault_w,
            "t_end_s": 3.0, "points": 12001,
            "stage1_enable": True,
            "soc_init": 0.80,
        })

        y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_rev.soc_init, p_rev.v_dc_init_pu, 0.0]
        sol = run_case(
            lambda t, y: case_gfm_proposed(t, y, p_rev),
            y0, p_rev.t_end_s, p_rev.points,
            method="RK45", rtol=1e-4, atol=1e-6, max_step=1e-3,
        )

        t = sol.t
        d = p_rev.derived()
        vpcc_abs = np.zeros_like(t)
        pdraw = np.zeros_like(t)
        for k in range(t.size):
            vthd, vthq = vthdq3(float(t[k]), p_rev)
            vpd, vpq = v_pcc_from_grid_impedance(vthd, vthq, float(sol.y[0][k]), float(sol.y[1][k]), d["r_th"], d["x_th"])
            vpcc_abs[k] = np.sqrt(vpd**2 + vpq**2)
            pdraw[k] = -(1.5 * (vpd * float(sol.y[0][k]) + vpq * float(sol.y[1][k])))

        # Compute mode flag using the controller's actual fault-detection logic
        mode_flag = np.zeros_like(t)
        for k in range(t.size):
            y_k = sol.y[:, k] if hasattr(sol.y, 'shape') and sol.y.ndim == 2 else np.array([sol.y[j][k] for j in range(len(sol.y))])
            mode_flag[k] = 1.0 if gfm_in_fault_mode(float(t[k]), y_k, p_rev) else 0.0

        fig, axes = plt.subplots(5, 1, figsize=(8.5, 8.5), sharex=True)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"font.size": 11})

        axes[0].plot(t, pdraw, color="black", linewidth=1.1)
        axes[0].set_ylabel("$P_{draw}$ (p.u.)")
        axes[0].set_title("Bidirectional Mode Transition: Two Sequential Faults")

        axes[1].plot(t, vpcc_abs, color="blue", linewidth=1.1)
        axes[1].set_ylabel("$|V_{pcc}|$ (p.u.)")
        axes[1].axhline(y=p_rev.v_thresh_pu, color="red", linestyle=":", linewidth=0.8, label="$V_{thresh}$")
        axes[1].legend(fontsize=8, framealpha=0.9)

        axes[2].plot(t, sol.y[7], color="green", linewidth=1.1)
        axes[2].set_ylabel("$V_{dc}$ (p.u.)")

        axes[3].plot(t, sol.y[6], color="purple", linewidth=1.1)
        axes[3].set_ylabel("SoC (p.u.)")

        axes[4].plot(t, mode_flag, color="red", linewidth=1.1)
        axes[4].set_ylabel("Mode Flag")
        axes[4].set_yticks([0, 1])
        axes[4].set_yticklabels(["Mode 1", "Mode 2"])
        axes[4].set_xlabel("Time (s)")

        for ax in axes:
            for fs, fe in fault_w:
                ax.axvspan(fs, fe, color="yellow", alpha=0.2)
            ax.grid(True, linestyle=":", alpha=0.6)

        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.06, top=0.96, hspace=0.30)
        plt.savefig(args.reverse_out, dpi=500)
        print(f"Reverse transition plot saved to {args.reverse_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
