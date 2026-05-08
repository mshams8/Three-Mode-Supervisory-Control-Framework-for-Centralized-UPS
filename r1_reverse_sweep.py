"""
R1 supplementary sweep for Reviewer 3.2 (bidirectional transition overshoot).

The reviewer noted the visible P_draw and |V_pcc| excursions during the
Mode-2 -> Mode-1 transition. This script reuses the exact continuous two-fault
scenario from run_sim.py --reverse-out and sweeps the Stage-1 SoC-recovery
grid-draw bias cap and a post-fault soft-engage delay. The default controller
remains unchanged unless the caller explicitly selects nonzero delay or a
different bias cap.
"""

import argparse
import csv
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import SimParams, v_th_dq_grid_frame
from controllers import case_gfm_proposed
from dq_plant import v_pcc_from_grid_impedance
from integration import run_case


FAULT_WINDOWS = ((0.50, 0.65), (1.50, 1.65))


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_reverse_case(bias_cap: float, delay_s: float, p_base: SimParams) -> dict:
    p = SimParams(**{
        **p_base.__dict__,
        "fault_start_s": FAULT_WINDOWS[0][0],
        "fault_end_s": FAULT_WINDOWS[0][1],
        "fault_windows": FAULT_WINDOWS,
        "t_end_s": 3.0,
        "points": 12001,
        "stage1_enable": True,
        "soc_init": 0.80,
        "stage1_soc_bias_chg_max_pu": bias_cap,
        "stage1_soc_bias_delay_s": delay_s,
    })

    y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p.soc_init, p.v_dc_init_pu, 0.0]
    sol = run_case(
        lambda t, y: case_gfm_proposed(t, y, p),
        y0,
        p.t_end_s,
        p.points,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
        max_step=1e-3,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    d = p.derived()
    vpcc_abs = np.zeros_like(t)
    pdraw = np.zeros_like(t)
    i_abs = np.sqrt(sol.y[0] ** 2 + sol.y[1] ** 2)

    for k, tk in enumerate(t):
        vthd, vthq = v_th_dq_grid_frame(float(tk), p)
        vpd, vpq = v_pcc_from_grid_impedance(
            vthd, vthq, float(sol.y[0, k]), float(sol.y[1, k]), d["r_th"], d["x_th"]
        )
        vpcc_abs[k] = math.sqrt(vpd * vpd + vpq * vpq)
        pdraw[k] = -(1.5 * (vpd * float(sol.y[0, k]) + vpq * float(sol.y[1, k])))

    post_mask = np.zeros_like(t, dtype=bool)
    for _, end in FAULT_WINDOWS:
        post_mask |= (t >= end) & (t <= end + 0.40)

    final_end = FAULT_WINDOWS[-1][1]
    recovery_mask = t >= final_end
    within = np.abs(pdraw - 1.0) <= 0.05
    recovery_time = float("nan")
    if np.any(recovery_mask):
        start = int(np.searchsorted(t, final_end))
        hold_s = 0.05
        dt_med = float(np.median(np.diff(t))) if t.size > 1 else 1e-3
        hold_n = max(1, int(math.ceil(hold_s / max(1e-9, dt_med))))
        for k in range(start, t.size):
            if not within[k]:
                continue
            if np.all(within[k:min(t.size, k + hold_n)]):
                recovery_time = float(t[k] - final_end)
                break

    return {
        "bias_cap_pu": bias_cap,
        "delay_s": delay_s,
        "min_post_pdraw_pu": float(np.min(pdraw[post_mask])),
        "max_post_pdraw_pu": float(np.max(pdraw[post_mask])),
        "max_post_pdraw_dev_pu": float(np.max(np.abs(pdraw[post_mask] - 1.0))),
        "min_post_vpcc_pu": float(np.min(vpcc_abs[post_mask])),
        "max_post_vpcc_pu": float(np.max(vpcc_abs[post_mask])),
        "max_post_vpcc_dev_pu": float(np.max(np.abs(vpcc_abs[post_mask] - 1.0))),
        "max_i_pu": float(np.max(i_abs)),
        "min_vdc_pu": float(np.min(sol.y[7])),
        "min_soc_pu": float(np.min(sol.y[6])),
        "final_soc_pu": float(sol.y[6, -1]),
        "pdraw_recovery_time_s": recovery_time,
    }


def write_csv(rows: list[dict], out_csv: str) -> None:
    keys = [
        "bias_cap_pu",
        "delay_s",
        "min_post_pdraw_pu",
        "max_post_pdraw_pu",
        "max_post_pdraw_dev_pu",
        "min_post_vpcc_pu",
        "max_post_vpcc_pu",
        "max_post_vpcc_dev_pu",
        "max_i_pu",
        "min_vdc_pu",
        "min_soc_pu",
        "final_soc_pu",
        "pdraw_recovery_time_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})
    print(f"Reverse-transition sweep CSV saved to {out_csv}")


def plot_heatmaps(rows: list[dict], bias_values: list[float], delay_values: list[float], out_png: str) -> None:
    pdraw_grid = np.zeros((len(delay_values), len(bias_values)))
    vpcc_grid = np.zeros_like(pdraw_grid)
    for row in rows:
        i = delay_values.index(row["delay_s"])
        j = bias_values.index(row["bias_cap_pu"])
        pdraw_grid[i, j] = row["max_post_pdraw_dev_pu"]
        vpcc_grid[i, j] = row["max_post_vpcc_dev_pu"]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    for ax, data, title in [
        (axes[0], pdraw_grid, r"max post-fault $|P_{draw}-1|$"),
        (axes[1], vpcc_grid, r"max post-fault $||V_{pcc}|-1|$"),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(bias_values)))
        ax.set_xticklabels([f"{x:.2f}" for x in bias_values])
        ax.set_yticks(range(len(delay_values)))
        ax.set_yticklabels([f"{1000*x:.0f}" for x in delay_values])
        ax.set_xlabel("SoC-bias cap (p.u.)")
        ax.set_ylabel("Soft-engage delay (ms)")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.88)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Reverse-transition sweep figure saved to {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 reverse-transition SoC-bias / soft-engage sweep.")
    parser.add_argument("--out-csv", default="r1_reverse_sweep.csv")
    parser.add_argument("--out", default="simulation_results_reverse_sweep.png")
    parser.add_argument("--bias-values", default="0.05,0.10,0.20")
    parser.add_argument("--delay-values", default="0,0.05,0.10")
    args = parser.parse_args()

    matplotlib.use("Agg")
    bias_values = _parse_float_list(args.bias_values)
    delay_values = _parse_float_list(args.delay_values)

    p_base = SimParams()
    rows = []
    print("Running reverse-transition sweep...")
    for delay_s in delay_values:
        for bias_cap in bias_values:
            row = run_reverse_case(bias_cap, delay_s, p_base)
            rows.append(row)
            print(
                f"  cap={bias_cap:.2f}, delay={delay_s*1000:.0f} ms: "
                f"P=[{row['min_post_pdraw_pu']:.3f},{row['max_post_pdraw_pu']:.3f}], "
                f"V=[{row['min_post_vpcc_pu']:.3f},{row['max_post_vpcc_pu']:.3f}], "
                f"final SoC={row['final_soc_pu']:.6f}"
            )

    write_csv(rows, args.out_csv)
    plot_heatmaps(rows, bias_values, delay_values, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
