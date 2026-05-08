# Simulation code: three-mode supervisory control for AI-data-center UPS

This repository accompanies the manuscript:

> **Mitigating Grid Stability Risks from Large AI Data Centers: A Three-Mode Supervisory Control Framework for Centralized UPS Systems**  
> Mohamed Shamseldein, *IET Generation, Transmission & Distribution*, 2026.  
> Manuscript ID GTD-2026-04-0218.

It contains the averaged-dq and switching-EMT simulation code, parameter sweeps, and figure-generation scripts used for the revised manuscript.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate    # Windows; use .venv/bin/activate on POSIX
pip install -r requirements.txt
python run_sim.py --out simulation_results.png
```

The code uses Python >= 3.10. It was tested on Python 3.11.9 with NumPy 2.2, SciPy 1.13, and Matplotlib 3.9.

## Reproducing Manuscript Figures and Tables

Each manuscript figure is produced by one command. Outputs are written next to the script unless redirected.

| Manuscript item | Command |
|---|---|
| Fig. 2, Mode-1 pulse filtering | `python run_sim.py --pulse-out simulation_results_pulse.png` |
| Fig. 3, four-controller benchmark, plus Table II | `python run_sim.py --out simulation_results.png --comparison-out comparison.csv` |
| Fig. 4, minimum-draw policy sweep | `python r1_pdrawmin_sweep.py --out simulation_results_pdrawmin.png --csv artifacts/pdrawmin_sweep.csv` |
| Fig. 5, averaged dq vs EMT-style abc check | `python run_sim.py --emt-out simulation_results_emt.png` |
| Fig. 6, asymmetric faults | `python run_sim.py --asymmetric-out simulation_results_asymmetric.png` |
| Fig. 7 and Table V, SPWM cross-check | `python run_sim.py --switching-out simulation_results_switching.png` |
| Fig. 8, output-impedance screening | `python output_impedance.py --out simulation_results_zout.png --summary-csv artifacts/impedance_summary.csv` |
| Fig. 9, LCL capacitance sensitivity | `python output_impedance.py --lcl-out simulation_results_lcl.png --scr-out simulation_results_zout_scr.png --summary-csv artifacts/impedance_summary.csv` |
| Fig. 10, bidirectional transition | `python run_sim.py --reverse-out simulation_results_reverse.png` |
| Fig. 11, broadband workload attenuation | `python run_sim.py --broadband-out simulation_results_broadband.png` |
| Fig. 12, frequency proxy | `python run_sim.py --freq-out simulation_results_freq.png` |
| Table III, robustness sweeps | `python validation_study.py --out-dir artifacts/` |
| PLL tuning sensitivity cited in the response | `python r1_pll_sweep.py --out-csv artifacts/pll_sweep.csv` |
| Reverse-transition SoC-bias/soft-engage sweep cited in the response | `python r1_reverse_sweep.py --out-csv artifacts/reverse_sweep.csv --out simulation_results_reverse_sweep.png` |

Run `python run_sim.py --help` for the complete list of command-line options, including SCR, X/R ratio, fault depth, fault duration, workload type, and asymmetric-fault type. Default values match the baseline scenario in Section IV-A of the manuscript.

## Repository Layout

```text
common.py             Parameters, dataclasses, helpers, sign conventions
controllers.py        GFL-MC, GFL-PLL, Threshold, Proposed controllers
dq_plant.py           Series-RL and LCL plants in grid-aligned dq frame
emt_sim.py            Carrier-based SPWM EMT cross-check
emt_utils.py          abc/dq transforms and EMT helper routines
frequency_proxy.py    System-level swing-equation proxy: 2H dx/dt = P_m - P_e - D x
integration.py        RK2/RK4 fixed-step integrators
metrics.py            Post-processing: min Vpcc, peak |I|, BESS energy, etc.
output_impedance.py   Single-converter output-impedance and LCL sensitivity plots
plots.py              Matplotlib figure builders
r1_pdrawmin_sweep.py  R1 minimum-draw sweep with aggregate-frequency proxy
r1_pll_sweep.py       R1 PLL tuning sensitivity sweep
r1_reverse_sweep.py   R1 bidirectional-transition SoC-bias / soft-engage sweep
run_sim.py            Top-level driver for time-domain manuscript figures
validation_study.py   Multi-axis parameter sweeps
artifacts/            Small CSV files used in the R1 response
requirements.txt      Python dependencies
```

## Sign Conventions

- Active power: `P_grid > 0` is inverter-to-grid; `P_draw = -P_grid` is grid-to-load.
- Reactive current: `i_q < 0` corresponds to capacitive voltage-supporting injection at `v_q = 0`.
- Network impedance: `V_pcc = V_th + Z_th * I_grid`; the inverter is the active source and PCC is the grid side of the filter.

## Citing

If you use this code, please cite the IGTD manuscript and the archived code release:
https://doi.org/10.5281/zenodo.20086129.

## License

MIT; see [LICENSE](LICENSE).
