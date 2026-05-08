"""
R1 addition: single-converter output-impedance screening (Reviewer 2, point 2).

Plots the magnitude and phase of the inverter terminal impedance Z_out(jω) overlaid
with the grid Thevenin impedance Z_grid(jω) for the SCR=1.5 baseline case, and
reports the magnitude crossover frequency ω_x where |Z_out(jω_x)| = |Z_grid(jω_x)|
together with the phase difference ∠Z_out(jω_x) − ∠Z_grid(jω_x) at that crossover.

Scope: this is single-converter impedance screening, not a full impedance-stability
proof. The aim is to bound high-frequency interaction risk for the present system,
not to certify behavior under multi-converter coupling or protection coordination.
The "phase margin" term is intentionally avoided because the relevant loop
definition for a load-side LCL inverter requires a full impedance-stability
formulation that is outside the scope of this manuscript.

The model used here:

  Closed-loop inverter (PI current control, inverter-side L1):
    Z_inv(s) = R_f1 + s*L_f1 + Kp + Ki/s
  Capacitor branch (with damping resistor):
    Z_cap(s) = R_d + 1/(s*C)
  Grid-side leg of the LCL:
    Z_g_lcl(s) = R_f2 + s*L_f2
  Grid Thevenin (from SCR, X/R):
    Z_grid(s) = R_th + s*L_th

  Output impedance at the LCL grid-side terminal (PCC):
    Z_out(s) = Z_g_lcl(s) + (Z_cap(s) || Z_inv(s))

References:
  - Wang, X. & Blaabjerg, F. (2018). Harmonic stability in power electronic-based power systems.
  - Sun, J. (2011). Impedance-based stability criterion for grid-connected inverters.
"""

import argparse
import csv
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import SimParams, W_BASE, rl_from_scr_xr


def _impedance_inverter(s, kp, ki, r_f1, l_f1):
    """Closed-loop inverter port impedance (PI + inverter-side filter)."""
    return r_f1 + s * l_f1 + kp + ki / s


def _impedance_cap(s, c_pu, r_d):
    """Capacitor branch with series damping resistor."""
    return r_d + 1.0 / (s * c_pu)


def _impedance_grid_side_lcl(s, r_f2, l_f2):
    """Grid-side leg of the LCL filter."""
    return r_f2 + s * l_f2


def _impedance_grid_thevenin(s, r_th, l_th):
    """Thevenin grid impedance derived from SCR / X/R."""
    return r_th + s * l_th


def _parallel(z1, z2):
    return (z1 * z2) / (z1 + z2)


def compute_impedances(p: SimParams, freqs_hz: np.ndarray) -> dict:
    """
    Evaluate Z_out(jω) and Z_grid(jω) at the given frequencies (Hz).

    Uses LCL parameters from p when set; falls back to a representative L-only
    inverter-side filter when LCL is disabled (to keep Z_out plottable).

    Returns dict with keys 'omega', 'z_out', 'z_grid', 'z_inv', 'z_cap', 'z_g_lcl'.
    """
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    s = 1j * omega

    r_th, x_th = rl_from_scr_xr(p.scr, p.xr_ratio)
    l_th = x_th / W_BASE
    z_grid = _impedance_grid_thevenin(s, r_th, l_th)

    # Inverter-side filter
    r_f1 = p.r_f_pu
    l_f1 = p.x_f_pu / W_BASE

    # LCL extension (default 0.0 when not set)
    r_f2 = p.r_f2_pu
    x_f2 = p.x_f2_pu
    cf = p.cf_pu
    r_d = p.r_d_pu

    z_inv = _impedance_inverter(s, p.cc_kp, p.cc_ki, r_f1, l_f1)

    if cf > 0.0 and x_f2 > 0.0:
        l_f2 = x_f2 / W_BASE
        c_pu = cf / W_BASE  # cf is W*C in pu, so C in pu = cf / W_BASE
        z_cap = _impedance_cap(s, c_pu, r_d)
        z_g_lcl = _impedance_grid_side_lcl(s, r_f2, l_f2)
        z_out = z_g_lcl + _parallel(z_cap, z_inv)
    else:
        # Fall back to an L-only filter screening (still useful for the
        # comparison even without the capacitor branch).
        z_cap = np.full_like(s, np.inf)
        z_g_lcl = np.zeros_like(s)
        z_out = z_inv

    return {
        "omega": omega,
        "z_out": z_out,
        "z_grid": z_grid,
        "z_inv": z_inv,
        "z_cap": z_cap,
        "z_g_lcl": z_g_lcl,
    }


def find_magnitude_crossover(omega: np.ndarray, z_out: np.ndarray, z_grid: np.ndarray) -> tuple[float, float]:
    """
    Locate the smallest ω where |Z_out| = |Z_grid| and return (ω_x in rad/s,
    phase_difference in degrees at that crossover). Returns (NaN, NaN) when no
    crossover is found inside the supplied frequency range.

    Phase difference is ∠Z_out(jω_x) − ∠Z_grid(jω_x) in degrees, wrapped to
    (-180, 180].
    """
    diff = np.abs(z_out) - np.abs(z_grid)
    # Look for the first sign change.
    crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
    if crossings.size == 0:
        return float("nan"), float("nan")
    k = int(crossings[0])
    # Linear interpolation in log-frequency space.
    w0, w1 = omega[k], omega[k + 1]
    d0, d1 = diff[k], diff[k + 1]
    if d1 == d0:
        omega_x = w0
    else:
        alpha = -d0 / (d1 - d0)
        omega_x = math.exp(math.log(w0) + alpha * (math.log(w1) - math.log(w0)))

    # Interpolate phase at omega_x.
    phase_out = np.unwrap(np.angle(z_out))
    phase_grid = np.unwrap(np.angle(z_grid))
    phase_diff = phase_out - phase_grid
    phase_x = float(np.interp(omega_x, omega, phase_diff)) * 180.0 / math.pi
    # Wrap to (-180, 180]
    while phase_x > 180.0:
        phase_x -= 360.0
    while phase_x <= -180.0:
        phase_x += 360.0
    return float(omega_x), phase_x


def passivity_metrics(freqs_hz: np.ndarray, z_out: np.ndarray) -> dict:
    """
    Passive output-admittance screening: report min Re{Y_out(jw)} over the
    swept band. Positive Re{Y_out} over the band is a sufficient single-port
    passivity screen for the terminal model used here.
    """
    y_out = 1.0 / z_out
    re_y = np.real(y_out)
    k = int(np.argmin(re_y))
    return {
        "min_re_y": float(re_y[k]),
        "min_re_y_freq_hz": float(freqs_hz[k]),
        "nonpassive_points": int(np.sum(re_y < -1e-9)),
    }


def impedance_summary_row(p: SimParams, freqs_hz: np.ndarray, label: str) -> dict:
    res = compute_impedances(p, freqs_hz)
    omega_x, phase_x = find_magnitude_crossover(res["omega"], res["z_out"], res["z_grid"])
    pm = passivity_metrics(freqs_hz, res["z_out"])
    return {
        "label": label,
        "scr": float(p.scr),
        "cf_pu": float(p.cf_pu),
        "omega_x_rad_s": float(omega_x),
        "f_x_hz": float(omega_x / (2.0 * math.pi)) if math.isfinite(omega_x) else float("nan"),
        "phase_diff_deg": float(phase_x),
        "min_re_y": pm["min_re_y"],
        "min_re_y_freq_hz": pm["min_re_y_freq_hz"],
        "nonpassive_points": pm["nonpassive_points"],
    }


def plot_impedances(p: SimParams, freqs_hz: np.ndarray, out_path: str) -> dict:
    """
    Generate the |Z_out| / |Z_grid| Bode overlay and report crossover metrics.
    """
    res = compute_impedances(p, freqs_hz)
    omega = res["omega"]
    omega_x, phase_x = find_magnitude_crossover(omega, res["z_out"], res["z_grid"])
    pm = passivity_metrics(freqs_hz, res["z_out"])

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)

    ax_mag.loglog(freqs_hz, np.abs(res["z_out"]), label=r"$|Z_{out}(j\omega)|$ (inverter terminal)", lw=2.0, color="#1f4e8c")
    ax_mag.loglog(freqs_hz, np.abs(res["z_grid"]), label=r"$|Z_{grid}(j\omega)|$ (Thevenin, SCR=1.5)", lw=2.0, color="#b22222", linestyle="--")
    ax_mag.set_ylabel("|Z| (p.u.)")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="best", fontsize=9)
    ax_mag.set_title("Single-converter impedance screening")

    ax_phase.semilogx(freqs_hz, np.angle(res["z_out"]) * 180.0 / math.pi, lw=2.0, color="#1f4e8c", label=r"$\angle Z_{out}$")
    ax_phase.semilogx(freqs_hz, np.angle(res["z_grid"]) * 180.0 / math.pi, lw=2.0, color="#b22222", linestyle="--", label=r"$\angle Z_{grid}$")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.legend(loc="best", fontsize=9)
    ax_phase.set_ylim(-180.0, 180.0)

    if math.isfinite(omega_x):
        f_x = omega_x / (2.0 * math.pi)
        for ax in (ax_mag, ax_phase):
            ax.axvline(f_x, color="gray", linestyle=":", alpha=0.7)
        ax_mag.annotate(
            f"crossover\n$f_x = {f_x:.1f}$ Hz\n$\\Delta\\angle = {phase_x:.1f}^\\circ$",
            xy=(f_x, np.interp(omega_x, omega, np.abs(res["z_out"]))),
            xytext=(0.55, 0.20), textcoords="axes fraction",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Output-impedance screening saved to {out_path}")
    if math.isfinite(omega_x):
        f_x = omega_x / (2.0 * math.pi)
        print(f"  Magnitude crossover: omega_x = {omega_x:.2f} rad/s  ( f_x = {f_x:.2f} Hz )")
        print(f"  Phase difference at crossover: angle(Z_out) - angle(Z_grid) = {phase_x:.2f} deg")
    else:
        print("  No |Z_out| = |Z_grid| crossover found in the swept range.")
    print(
        "  Passivity screen: "
        f"min Re(Y_out) = {pm['min_re_y']:.4f} at {pm['min_re_y_freq_hz']:.2f} Hz; "
        f"non-passive grid points = {pm['nonpassive_points']}"
    )

    return {"omega_x": omega_x, "phase_diff_deg": phase_x, "out_path": out_path, **pm}


def plot_cf_sensitivity(p_template: SimParams, cf_values: list[float],
                        freqs_hz: np.ndarray, out_path: str) -> dict:
    """
    Sensitivity of |Z_out(jω)| to the LCL capacitor susceptance C_f.

    Produces a single-panel magnitude-only figure with one trace per C_f value
    and the grid impedance overlaid for reference. Reports the crossover
    frequency and phase difference for each C_f case.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2))

    summary = []
    cmap = plt.colormaps.get_cmap("viridis")
    for i, cf in enumerate(cf_values):
        p_cf = SimParams(
            scr=p_template.scr,
            xr_ratio=p_template.xr_ratio,
            cf_pu=cf,
            x_f2_pu=p_template.x_f2_pu,
            r_d_pu=p_template.r_d_pu,
        )
        res = compute_impedances(p_cf, freqs_hz)
        omega_x, phase_x = find_magnitude_crossover(res["omega"], res["z_out"], res["z_grid"])
        summary.append((cf, omega_x / (2.0 * math.pi) if math.isfinite(omega_x) else float("nan"), phase_x))
        color = cmap(i / max(1, len(cf_values) - 1))
        ax.loglog(freqs_hz, np.abs(res["z_out"]), lw=1.6, color=color,
                  label=fr"$C_f = {cf:.3f}$ p.u.")

    # Grid impedance overlay (constant across Cf)
    res_grid = compute_impedances(p_template, freqs_hz)
    ax.loglog(freqs_hz, np.abs(res_grid["z_grid"]), color="#b22222",
              lw=2.0, linestyle="--", label=r"$|Z_{grid}|$ (SCR=1.5)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Z| (p.u.)")
    ax.set_title(r"LCL filter sensitivity: $|Z_{out}(j\omega)|$ for varying $C_f$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"LCL Cf-sensitivity Bode saved to {out_path}")
    print("  Cf [pu]   f_x [Hz]   dPhase [deg]")
    for cf, fx, dph in summary:
        fx_str = f"{fx:8.2f}" if math.isfinite(fx) else "      --"
        dph_str = f"{dph:8.2f}" if math.isfinite(dph) else "      --"
        print(f"  {cf:7.3f}  {fx_str}  {dph_str}")

    return {"summary": summary, "out_path": out_path}


def plot_scr_sensitivity(p_template: SimParams, scr_values: list[float],
                         freqs_hz: np.ndarray, out_path: str) -> dict:
    """Plot crossover and phase-difference sensitivity versus SCR."""
    rows = []
    for scr in scr_values:
        p_scr = SimParams(
            scr=float(scr),
            xr_ratio=p_template.xr_ratio,
            cf_pu=p_template.cf_pu,
            x_f2_pu=p_template.x_f2_pu,
            r_d_pu=p_template.r_d_pu,
        )
        rows.append(impedance_summary_row(p_scr, freqs_hz, label=f"SCR={scr:g}"))

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
    scr_arr = np.array([r["scr"] for r in rows], dtype=float)
    fx_arr = np.array([r["f_x_hz"] for r in rows], dtype=float)
    ph_arr = np.array([r["phase_diff_deg"] for r in rows], dtype=float)
    re_arr = np.array([r["min_re_y"] for r in rows], dtype=float)

    axes[0].plot(scr_arr, fx_arr, marker="o", color="#1f4e8c", lw=2.0)
    axes[0].set_ylabel(r"$f_x$ (Hz)")
    axes[0].set_title("SCR sensitivity of impedance-screening metrics")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(scr_arr, ph_arr, marker="o", color="#b22222", lw=2.0, label=r"$\Delta\angle$ at $f_x$")
    axes[1].set_ylabel(r"$\Delta\angle$ (deg)")
    axes[1].set_xlabel("SCR")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="best", fontsize=8)
    for scr, re_min in zip(scr_arr, re_arr):
        axes[1].annotate(f"ReY>0\n{re_min:.2f}", xy=(scr, ph_arr[np.where(scr_arr == scr)[0][0]]),
                         xytext=(0, -26), textcoords="offset points",
                         ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"SCR-sensitivity impedance plot saved to {out_path}")
    print("  SCR    f_x [Hz]   dPhase [deg]   min Re(Y_out)")
    for r in rows:
        print(f"  {r['scr']:4.1f}   {r['f_x_hz']:8.2f}   {r['phase_diff_deg']:11.2f}   {r['min_re_y']:12.4f}")
    return {"summary": rows, "out_path": out_path}


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_summary_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = [
        "label", "scr", "cf_pu", "omega_x_rad_s", "f_x_hz",
        "phase_diff_deg", "min_re_y", "min_re_y_freq_hz", "nonpassive_points",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"Impedance summary CSV saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="R1: single-converter output-impedance screening.")
    parser.add_argument("--out", type=str, default="simulation_results_zout.png",
                        help="Output PNG path for the Bode overlay (single Cf).")
    parser.add_argument("--lcl-out", type=str, default=None,
                        help="If set, also produce a Cf-sensitivity sweep figure at this path.")
    parser.add_argument("--scr-out", type=str, default=None,
                        help="If set, produce an SCR-sensitivity figure at this path.")
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="If set, write passivity/crossover summary metrics to this CSV.")
    parser.add_argument("--scr-sweep", type=str, default="1.2,1.5,2.0,3.0,5.0",
                        help="Comma-separated SCR values used with --scr-out/--summary-csv.")
    parser.add_argument("--scr", type=float, default=1.5)
    parser.add_argument("--xr", type=float, default=5.0)
    parser.add_argument("--cf", type=float, default=0.05,
                        help="LCL capacitor susceptance W*C in pu (0 disables LCL).")
    parser.add_argument("--xf2", type=float, default=0.05,
                        help="Grid-side reactance pu at 60 Hz (0 disables LCL).")
    parser.add_argument("--rd", type=float, default=0.10,
                        help="Capacitor damping resistor (pu).")
    parser.add_argument("--f-min", type=float, default=0.1)
    parser.add_argument("--f-max", type=float, default=5000.0)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()

    matplotlib.use("Agg")

    p = SimParams(
        scr=args.scr,
        xr_ratio=args.xr,
        cf_pu=args.cf,
        x_f2_pu=args.xf2,
        r_d_pu=args.rd,
    )

    freqs = np.logspace(math.log10(args.f_min), math.log10(args.f_max), args.n)
    summary_rows = [impedance_summary_row(p, freqs, label="baseline")]
    plot_impedances(p, freqs, args.out)

    if args.lcl_out:
        # Sweep an order-of-magnitude span of Cf values around the nominal
        cf_nominal = args.cf if args.cf > 0 else 0.05
        cf_values = [0.5 * cf_nominal, 0.75 * cf_nominal, cf_nominal, 1.5 * cf_nominal, 2.0 * cf_nominal]
        plot_cf_sensitivity(p, cf_values, freqs, args.lcl_out)
        for cf in cf_values:
            p_cf = SimParams(scr=args.scr, xr_ratio=args.xr, cf_pu=cf, x_f2_pu=args.xf2, r_d_pu=args.rd)
            summary_rows.append(impedance_summary_row(p_cf, freqs, label=f"Cf={cf:.3f}"))

    if args.scr_out or args.summary_csv:
        scr_values = _parse_float_list(args.scr_sweep)
        if args.scr_out:
            plot_scr_sensitivity(p, scr_values, freqs, args.scr_out)
        for scr in scr_values:
            p_scr = SimParams(scr=scr, xr_ratio=args.xr, cf_pu=args.cf, x_f2_pu=args.xf2, r_d_pu=args.rd)
            summary_rows.append(impedance_summary_row(p_scr, freqs, label=f"SCR={scr:g}"))

    if args.summary_csv:
        write_summary_csv(args.summary_csv, summary_rows)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
