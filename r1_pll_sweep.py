"""
R1 addition: GFL-PLL tuning sensitivity sweep (Reviewer 2, point 3).

The baseline GFL-PLL controller uses Kp=20, Ki=200, which corresponds (assuming
unit-magnitude PCC voltage) to a characteristic equation
  s^2 + Kp s + Ki = 0
=> natural frequency  ω_n = sqrt(Ki)  ≈ 14.14 rad/s (≈ 2.25 Hz)
=> damping            ζ   = Kp / (2 ω_n) ≈ 0.707

This sweep varies the loop bandwidth f_BW ∈ {2.25, 5, 10} Hz at ζ = 0.7 and
reports the GFL-PLL benchmark metrics for each tuning, demonstrating that the
benchmark conclusions in §IV-C-1 (Table II) do not hinge on the specific 20/200
gain choice.
"""

import argparse
import csv
import math
import os

from common import SimParams
from controllers import case_gfl_pll_lvrt
from integration import run_case
from metrics import (
    compute_fault_max_current,
    compute_fault_min_vdc,
    compute_fault_min_vpcc,
    compute_fault_min_vpcc_settled,
    compute_stage1_unserved_energy_mwh,
)


def _kp_ki_from_bw_zeta(f_bw_hz: float, zeta: float) -> tuple[float, float]:
    """SRF-PLL: ω_n = 2π f_BW; Kp = 2ζω_n; Ki = ω_n²."""
    omega_n = 2.0 * math.pi * f_bw_hz
    return 2.0 * zeta * omega_n, omega_n * omega_n


def _run_one(f_bw: float, zeta: float, p_template: SimParams) -> dict:
    kp, ki = _kp_ki_from_bw_zeta(f_bw, zeta)
    p_case = SimParams(**{**p_template.__dict__, "pll_kp": kp, "pll_ki": ki})
    sol = run_case(
        lambda t, y: case_gfl_pll_lvrt(t, y, p_case),
        [-2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_case.v_dc_init_pu],
        p_case.t_end_s, p_case.points,
        method="RK45", rtol=1e-4, atol=1e-6, max_step=1e-3,
    )
    if not sol.success:
        return {"f_bw_hz": f_bw, "zeta": zeta, "ok": False, "msg": sol.message}

    settle_s = 1.0 / 60.0
    return {
        "f_bw_hz": f_bw, "zeta": zeta, "kp": kp, "ki": ki, "ok": True,
        "min_vpcc": compute_fault_min_vpcc(sol, p_case),
        "min_vpcc_1cyc": compute_fault_min_vpcc_settled(sol, p_case, settle_s=settle_s),
        "max_i": compute_fault_max_current(sol, p_case),
        "min_vdc": compute_fault_min_vdc(sol, p_case, vdc_index=6),
        "unserved_mwh": compute_stage1_unserved_energy_mwh(sol, p_case, is_gfm=False),
    }


def main():
    parser = argparse.ArgumentParser(description="R1: GFL-PLL tuning sensitivity sweep.")
    parser.add_argument("--out-csv", type=str, default="artifacts/pll_sweep.csv")
    parser.add_argument("--zeta", type=float, default=0.7,
                        help="Damping ratio (0.7 = paper baseline 20/200).")
    parser.add_argument("--bw-list", type=float, nargs="+",
                        default=[2.25, 5.0, 10.0],
                        help="PLL loop bandwidths in Hz (omega_n / 2pi).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    p_base = SimParams(scr=1.5, xr_ratio=5.0, fault_v_pu=0.5,
                       fault_start_s=0.50, fault_end_s=0.65)

    print("PLL tuning sensitivity sweep")
    rows = [_run_one(bw, args.zeta, p_base) for bw in args.bw_list]

    fields = ["f_bw_hz", "zeta", "kp", "ki", "min_vpcc", "min_vpcc_1cyc",
              "max_i", "min_vdc", "unserved_mwh"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            if not r.get("ok", False):
                continue
            w.writerow([f"{r[k]:.6f}" if isinstance(r[k], float) else r[k] for k in fields])

    print(f"PLL sweep CSV saved to {args.out_csv}")
    print("  f_BW [Hz]   Kp     Ki     min|Vpcc|  |Vpcc|_1c  max|I|   unserved [MWh]")
    for r in rows:
        if not r.get("ok", False):
            print(f"  {r['f_bw_hz']:7.2f}  FAILED: {r.get('msg','')}")
            continue
        print(f"  {r['f_bw_hz']:7.2f}  {r['kp']:6.2f} {r['ki']:6.2f}  "
              f"{r['min_vpcc']:.3f}     {r['min_vpcc_1cyc']:.3f}     "
              f"{r['max_i']:.3f}    {r['unserved_mwh']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
