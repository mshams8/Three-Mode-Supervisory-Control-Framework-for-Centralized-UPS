"""
R1 addition: minimum-draw policy sweep with aggregate-frequency response.

Reviewer 2 (point 1) asked for a quantified relationship between the
minimum-draw parameter P_draw,min^fault and (a) PCC voltage recovery,
(b) BESS stress, (c) inverter current loading, and (d) aggregate frequency
response from the swing-equation proxy. This script sweeps the parameter
across {0.0, 0.10, 0.20, 0.30, 0.40} p.u. on the data-center base, runs
case_gfm_proposed for each value, and computes all four metric families.

Outputs:
  - artifacts/pdrawmin_sweep.csv: one row per sweep value with all metrics
  - figures/simulation_results_pdrawmin.png: 2x3 panel summary figure
"""

import argparse
import csv
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import SimParams
from controllers import case_gfm_proposed
from frequency_proxy import run_frequency_proxy
from integration import run_case
from metrics import (
    compute_fault_max_abs_pbess,
    compute_fault_max_current,
    compute_fault_min_soc,
    compute_fault_min_vdc,
    compute_fault_min_vpcc,
    compute_fault_min_vpcc_settled,
    compute_stage1_unserved_energy_mwh,
    compute_time_series_pdraw,
)


def _run_one(pmin: float, p_template: SimParams) -> dict:
    p_case = SimParams(**{**p_template.__dict__, "p_fault_min_grid_draw_pu": pmin})

    sol = run_case(
        lambda t, y: case_gfm_proposed(t, y, p_case),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_case.soc_init, p_case.v_dc_init_pu, 0.0],
        p_case.t_end_s, p_case.points,
        method="RK45", rtol=1e-4, atol=1e-6, max_step=1e-3,
    )

    if not sol.success:
        return {"pmin": pmin, "ok": False, "msg": sol.message}

    settle_s = 1.0 / 60.0
    min_vpcc = compute_fault_min_vpcc(sol, p_case)
    min_vpcc_1c = compute_fault_min_vpcc_settled(sol, p_case, settle_s=settle_s)
    max_i = compute_fault_max_current(sol, p_case)
    min_vdc = compute_fault_min_vdc(sol, p_case, vdc_index=7)
    unserved = compute_stage1_unserved_energy_mwh(sol, p_case, is_gfm=True)
    max_pbess = compute_fault_max_abs_pbess(sol, p_case, pbess_index=5)
    min_soc = compute_fault_min_soc(sol, p_case, soc_index=6)

    # BESS energy used during fault
    t_full = sol.t
    pbess = np.array(sol.y[5])
    fault_mask = (t_full >= p_case.fault_start_s) & (t_full <= p_case.fault_end_s)
    batt_energy = float(np.trapezoid(np.abs(pbess[fault_mask]), t_full[fault_mask])) if np.any(fault_mask) else 0.0

    # Aggregate-frequency response: drive the swing-equation proxy with the per-block P_draw
    # trajectory, scaled by N_blocks blocks on the bulk system base.
    t_pdraw, pdraw = compute_time_series_pdraw(sol, p_case)
    _, f_hz = run_frequency_proxy(t_pdraw, pdraw, p_case, n_blocks=p_case.sys_n_blocks)
    f_nadir = float(np.min(f_hz))
    df_nadir_hz = float(60.0 - f_nadir)
    # Aggregate active-power deviation during fault on the data-center base, summed over N blocks
    pdraw_pre = float(pdraw[max(0, np.searchsorted(t_pdraw, p_case.fault_start_s) - 1)])
    pdraw_min_in_fault = float(np.min(pdraw[fault_mask])) if np.any(fault_mask) else float("nan")
    aggregate_dp_mw = (pdraw_pre - pdraw_min_in_fault) * p_case.sys_n_blocks * p_case.base_mw

    return {
        "pmin": pmin, "ok": True,
        "min_vpcc": min_vpcc, "min_vpcc_1cyc": min_vpcc_1c,
        "max_i": max_i, "min_vdc": min_vdc,
        "unserved_mwh": unserved, "max_pbess": max_pbess,
        "min_soc": min_soc, "batt_energy_pu_s": batt_energy,
        "f_nadir_hz": f_nadir, "df_nadir_hz": df_nadir_hz,
        "aggregate_dp_mw": aggregate_dp_mw,
    }


def main():
    parser = argparse.ArgumentParser(description="R1: minimum-draw policy sweep.")
    parser.add_argument("--out-csv", type=str, default="artifacts/pdrawmin_sweep.csv")
    parser.add_argument("--out-png", type=str, default="figures/simulation_results_pdrawmin.png")
    parser.add_argument("--values", type=float, nargs="+", default=[0.0, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--scr", type=float, default=1.5)
    parser.add_argument("--xr", type=float, default=5.0)
    parser.add_argument("--fault-v", type=float, default=0.5)
    args = parser.parse_args()

    matplotlib.use("Agg")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or ".", exist_ok=True)

    p_base = SimParams(scr=args.scr, xr_ratio=args.xr, fault_v_pu=args.fault_v,
                       fault_start_s=0.50, fault_end_s=0.65)

    print(f"Sweeping P_draw,min over {args.values}")
    rows = [_run_one(v, p_base) for v in args.values]

    fields = ["pmin", "min_vpcc", "min_vpcc_1cyc", "max_i", "min_vdc",
              "unserved_mwh", "max_pbess", "min_soc", "batt_energy_pu_s",
              "f_nadir_hz", "df_nadir_hz", "aggregate_dp_mw"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            if not r.get("ok", False):
                continue
            w.writerow([f"{r[k]:.6f}" if isinstance(r[k], float) else r[k] for k in fields])
    print(f"Sweep CSV saved to {args.out_csv}")

    # Build summary figure: 2 rows x 3 cols
    pmins = [r["pmin"] for r in rows if r["ok"]]
    fig, axs = plt.subplots(2, 3, figsize=(10.0, 5.5))

    axs[0, 0].plot(pmins, [r["min_vpcc_1cyc"] for r in rows if r["ok"]], "o-", lw=2.0, color="#1f4e8c")
    axs[0, 0].set_ylabel(r"$|V_{pcc}|_{1\text{cyc}}$ (p.u.)")
    axs[0, 0].set_title("Voltage recovery")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(pmins, [r["max_i"] for r in rows if r["ok"]], "s-", lw=2.0, color="#b22222")
    axs[0, 1].set_ylabel(r"$|I|_{\max}$ (p.u.)")
    axs[0, 1].set_title("Inverter loading")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(pmins, [r["max_pbess"] for r in rows if r["ok"]], "d-", lw=2.0, color="#1f8c44")
    axs[0, 2].set_ylabel(r"$|P_{bess}|_{\max}$ (p.u.)")
    axs[0, 2].set_title("BESS stress")
    axs[0, 2].grid(True, alpha=0.3)

    axs[1, 0].plot(pmins, [r["batt_energy_pu_s"] for r in rows if r["ok"]], "^-", lw=2.0, color="#1f8c44")
    axs[1, 0].set_xlabel(r"$P_{draw,min}^{fault}$ (p.u.)")
    axs[1, 0].set_ylabel(r"BESS energy (p.u.$\cdot$s)")
    axs[1, 0].set_title("BESS fault energy")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(pmins, [r["aggregate_dp_mw"] for r in rows if r["ok"]], "v-", lw=2.0, color="#7d3a8a")
    axs[1, 1].set_xlabel(r"$P_{draw,min}^{fault}$ (p.u.)")
    axs[1, 1].set_ylabel("Aggregate $\\Delta P$ (MW)")
    axs[1, 1].set_title("System load drop (N=20)")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(pmins, [r["df_nadir_hz"] for r in rows if r["ok"]], "x-", lw=2.0, color="#cc7a00", ms=8)
    axs[1, 2].set_xlabel(r"$P_{draw,min}^{fault}$ (p.u.)")
    axs[1, 2].set_ylabel(r"$\Delta f_{nadir}$ (Hz)")
    axs[1, 2].set_title("System frequency excursion")
    axs[1, 2].grid(True, alpha=0.3)

    fig.suptitle(r"Minimum-draw policy sweep: trade-off across $P_{draw,min}^{fault}$",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out_png, dpi=300)
    plt.close(fig)
    print(f"Sweep figure saved to {args.out_png}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
