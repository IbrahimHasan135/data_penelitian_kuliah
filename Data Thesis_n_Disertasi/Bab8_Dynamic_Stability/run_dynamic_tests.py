#!/usr/bin/env python3
"""Radial time-evolution checks for the p4-q6 Q-ball candidate.

By design, simulation requires explicit --execute after model assumptions have
been checked against the manuscript. The default command is a non-mutating
preflight. No result is evidence of physical validity by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MASS = 1.0
ALPHA = 1.0
BETA = 0.5
DEFAULT_OMEGA = 0.89
CFL = 0.18
DRIFT_LIMIT = 1.0e-2
RADIUS_MIN, RADIUS_MAX = 0.75, 1.25
CENTER_MIN, CENTER_MAX = 0.65, 1.35


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("../polynomial_pq_hybrid_output/p4_q6"))
    p.add_argument("--output-dir", type=Path, default=Path("bab8_dynamic_output"))
    p.add_argument("--target-omega", type=float, default=DEFAULT_OMEGA)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--quick", action="store_true", help="Reduced grid/time; diagnostic only.")
    p.add_argument("--execute", action="store_true", help="Run PDE evolutions after reviewing preflight assumptions.")
    return p.parse_args()


def load_profile(model_dir: Path, target: float):
    path = model_dir / "branch_profiles.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Profile NPZ tidak ditemukan: {path}")
    with np.load(path) as data:
        required = {"omega", "r", "f", "df", "E", "Q", "E_over_Q", "tail_ratio", "node_count"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Field NPZ tidak lengkap: {sorted(missing)}")
        omega = np.asarray(data["omega"], float)
        valid = (
            np.isfinite(omega) & np.isfinite(data["E_over_Q"]) &
            (np.asarray(data["E_over_Q"]) < MASS) &
            (np.asarray(data["tail_ratio"]) < 1e-6) &
            (np.asarray(data["node_count"]) == 0)
        )
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            raise ValueError("Tidak ditemukan solusi ground-state dengan E/Q<m dan tail<1e-6.")
        idx = int(candidates[np.argmin(np.abs(omega[candidates] - target))])
        return {
            "index": idx, "omega": float(omega[idx]),
            "r": np.asarray(data["r"][idx], float),
            "f": np.asarray(data["f"][idx], float),
            "df": np.asarray(data["df"][idx], float),
            "E": float(data["E"][idx]), "Q": float(data["Q"][idx]),
            "E_over_Q": float(data["E_over_Q"][idx]),
            "tail_ratio": float(data["tail_ratio"][idx]),
        }


def potential(psi):
    s = np.abs(psi) ** 2
    return MASS**2 * s - ALPHA * s**2 + BETA * s**3


def radial_laplacian(psi, dr):
    lap = np.zeros_like(psi)
    i = np.arange(1, len(psi) - 1)
    r = i * dr
    lap[1:-1] = ((psi[2:] - 2 * psi[1:-1] + psi[:-2]) / dr**2
                 + (psi[2:] - psi[:-2]) / (dr * r))
    lap[0] = 6.0 * (psi[1] - psi[0]) / dr**2
    # The last grid point is held at zero (Dirichlet).
    lap[-1] = 0.0
    return lap


def acceleration(psi, dr):
    s = np.abs(psi) ** 2
    dU_ds = MASS**2 - 2.0 * ALPHA * s + 3.0 * BETA * s**2
    out = radial_laplacian(psi, dr) - dU_ds * psi
    out[-1] = 0.0
    return out


def diagnostics(r, psi, pi, dr):
    dpsi = np.gradient(psi, dr, edge_order=2)
    dpsi[-1] = 0.0
    weight = 4.0 * np.pi * r**2
    rho = np.abs(pi)**2 + np.abs(dpsi)**2 + potential(psi)
    j0 = np.real(1j * (np.conj(psi) * pi - psi * np.conj(pi)))
    E = float(np.trapz(rho * weight, r))
    Q = float(np.trapezoid(j0 * weight, r))
    R = math.sqrt(max(0.0, float(np.trapezoid(r**2 * rho * weight, r)) / max(E, 1e-30)))
    return E, Q, R, float(abs(psi[0]))


def make_initial(profile, r, perturbation):
    f = np.interp(r, profile["r"], profile["f"], left=profile["f"][0], right=0.0)
    envelope = np.exp(-(r / max(float(profile["r"][-1]) * 0.15, 1.0)) ** 2)
    if perturbation[0] == "scale":
        f = f * (1.0 + perturbation[1] * envelope)
    elif perturbation[0] == "noise":
        rng = np.random.default_rng(20261125)
        noise = np.convolve(rng.normal(size=len(r)), np.ones(9) / 9.0, mode="same")
        noise /= max(float(np.max(np.abs(noise))), 1e-30)
        f = f * (1.0 + perturbation[1] * noise * envelope)
    psi = f.astype(complex)
    psi[-1] = 0.0
    pi = -1j * profile["omega"] * psi
    pi[-1] = 0.0
    return psi, pi


def evolve(profile, name, perturbation, tmax, nr, dt_factor, output):
    # Keep the outer boundary beyond the causal reach of the soliton during
    # this run. The profile tail is already negligible at its stored endpoint.
    r99_est = max(float(profile["r"][-1]) * 0.35, 5.0)
    rmax = max(float(profile["r"][-1]), r99_est + tmax + 10.0)
    r = np.linspace(0.0, rmax, nr)
    dr = float(r[1] - r[0])
    dt = min(CFL * dr * dt_factor, 0.18 * dr)
    steps = int(math.ceil(tmax / dt))
    dt = tmax / steps
    psi, pi = make_initial(profile, r, perturbation)
    E0, Q0, R0, A0 = diagnostics(r, psi, pi, dr)
    sample_every = max(1, steps // (80 if nr < 1000 else 160))
    rows = []

    for n in range(steps + 1):
        if n % sample_every == 0 or n == steps:
            E, Q, R, A = diagnostics(r, psi, pi, dr)
            rows.append({"time": n * dt, "energy": E, "charge": Q,
                         "radius_rms": R, "center_amplitude": A})
        if n == steps:
            break
        p_half = pi + 0.5 * dt * acceleration(psi, dr)
        psi_new = psi + dt * p_half
        psi_new[-1] = 0.0
        pi_new = p_half + 0.5 * dt * acceleration(psi_new, dr)
        pi_new[-1] = 0.0
        psi, pi = psi_new, pi_new

    E = np.array([x["energy"] for x in rows])
    Q = np.array([x["charge"] for x in rows])
    R = np.array([x["radius_rms"] for x in rows])
    A = np.array([x["center_amplitude"] for x in rows])
    result = {
        "case": name, "dt": dt, "dr": dr, "rmax": rmax, "steps": steps,
        "E0": E0, "Q0": Q0, "R0": R0, "A0": A0,
        "max_energy_drift": float(np.max(np.abs(E - E0)) / max(abs(E0), 1e-30)),
        "max_charge_drift": float(np.max(np.abs(Q - Q0)) / max(abs(Q0), 1e-30)),
        "radius_min_factor": float(np.min(R) / max(R0, 1e-30)),
        "radius_max_factor": float(np.max(R) / max(R0, 1e-30)),
        "center_min_factor": float(np.min(A) / max(A0, 1e-30)),
        "center_max_factor": float(np.max(A) / max(A0, 1e-30)),
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    args = arguments()
    profile = load_profile(args.model_dir, args.target_omega)
    assumptions = {
        "model": "complex scalar Q-ball, radial 3D, polynomial p4-q6",
        "potential": "U(s)=m^2*s-alpha*s^2+beta*s^3",
        "parameters": {"m": MASS, "alpha": ALPHA, "beta": BETA},
        "field_equation": "Psi_tt = Laplacian(Psi) - (m^2 - 2*alpha*|Psi|^2 + 3*beta*|Psi|^4)*Psi",
        "initial_data": "Psi(0,r)=f(r); Psi_t(0,r)=-i*omega*f(r)",
        "outer_boundary": "Dirichlet Psi=0; domain extended beyond tmax+estimated core radius; possible reflection remains a limitation",
        "integrator": "second-order velocity-Verlet / leapfrog; dt <= 0.18*dr",
        "selected_profile": {k: profile[k] for k in ("index", "omega", "E", "Q", "E_over_Q", "tail_ratio")},
        "execution_gate": "Check equation, signs, normalization, and boundary assumptions against manuscript before --execute.",
    }
    print(json.dumps(assumptions, indent=2))
    if not args.execute:
        print("\nPREFLIGHT ONLY: no time evolution run. Review assumptions, then add --execute.")
        return 0

    tmax = 6.0 if args.quick else args.tmax
    nr = 420 if args.quick else 900
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("control", ("none", 0.0), 1.0),
        ("scale_plus_1pct", ("scale", 0.01), 1.0),
        ("scale_minus_1pct", ("scale", -0.01), 1.0),
        ("smooth_noise_1pct", ("noise", 0.01), 1.0),
        ("scale_plus_5pct", ("scale", 0.05), 1.0),
        ("control_dt_half", ("none", 0.0), 0.5),
    ]
    summaries = []
    next_run_rows = {}
    for name, perturbation, dt_factor in cases:
        result = evolve(profile, name, perturbation, tmax, nr, dt_factor, args.output_dir)
        summary = {k: v for k, v in result.items() if k not in {"rows"}}
        summaries.append(summary)
        next_run_rows[name] = result["rows"]
        print(json.dumps(summary))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for result in summaries:
        rows = next_run_rows[ result["case"] ]
        t = [row["time"] for row in rows]
        axes[0, 0].plot(t, [row["energy"] / result["E0"] for row in rows], label=result["case"])
        axes[0, 1].plot(t, [row["charge"] / result["Q0"] for row in rows], label=result["case"])
        axes[1, 0].plot(t, [row["radius_rms"] / result["R0"] for row in rows], label=result["case"])
        axes[1, 1].plot(t, [row["center_amplitude"] / result["A0"] for row in rows], label=result["case"])
    titles = ["Energy / initial", "Charge / initial", "RMS radius / initial", "Core amplitude / initial"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title); ax.set_xlabel("t"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dynamic_diagnostics.png", dpi=180)
    plt.close(fig)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"assumptions": assumptions, "thresholds": {
            "max_relative_energy_drift": DRIFT_LIMIT,
            "max_relative_charge_drift": DRIFT_LIMIT,
            "radius_ratio": [RADIUS_MIN, RADIUS_MAX],
            "center_amplitude_ratio": [CENTER_MIN, CENTER_MAX],
        }, "runs": summaries}, f, indent=2)
    print(f"\nCSV dan summary disimpan di: {args.output_dir}")
    print("Catatan: PNG/field heatmap dan fluks radiasi belum ditambahkan; evaluasi refleksi batas sebelum klaim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
