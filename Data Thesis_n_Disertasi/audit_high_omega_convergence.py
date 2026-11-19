#!/usr/bin/env python3
"""Targeted R_max convergence audit for the existing p4-q6 high-omega Q-ball.

Run from the repository root:
    python3 "Data Thesis_n_Disertasi/audit_high_omega_convergence.py"

The script reuses the saved p4-q6 branch profile and the existing BVP solver.
It does not rescan the full solution branch. Results are written to a new CSV
under Data Thesis_n_Disertasi/rmax_high_omega_audit/.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # Keep the imported solver's relative paths inside this folder.
sys.path.insert(0, str(HERE))

import polynomial_pq_hybrid_solver as solver  # noqa: E402


DEFAULT_MODEL_DIR = HERE / "polynomial_pq_hybrid_output" / "p4_q6"
DEFAULT_OUTPUT = HERE / "rmax_high_omega_audit" / "high_omega_rmax_audit.csv"
DEFAULT_TARGET_OMEGA = 0.9920011103
DEFAULT_RMAX = [70.0, 90.0, 110.0, 140.0]
REFERENCE_RMAX = 50.0
REFERENCE_MESH = solver.N_MESH_BASE


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def load_reference(model_dir: Path, target_omega: float):
    npz_path = model_dir / "branch_profiles.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing saved profiles: {npz_path}")

    with np.load(npz_path) as data:
        if "omega" not in data or len(data["omega"]) == 0:
            raise ValueError(f"No solutions found in {npz_path}")

        index = int(np.argmin(np.abs(data["omega"] - target_omega)))
        omega = float(data["omega"][index])
        if abs(omega - target_omega) > 5.0e-5:
            raise ValueError(
                f"Closest saved omega is {omega:.12f}; "
                f"it is too far from requested {target_omega:.12f}"
            )

        r = np.asarray(data["r"][index], dtype=float)
        f = np.asarray(data["f"][index], dtype=float)
        df = np.asarray(data["df"][index], dtype=float)
        A = float(data["A_center"][index])

    if not (len(r) == len(f) == len(df)) or len(r) < 10:
        raise ValueError("Saved profile arrays are empty or inconsistent.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(f)):
        raise ValueError("Saved profile contains non-finite values.")

    model = next(m for m in solver.construct_models() if m.name == "p4_q6")
    f_star, omega_min, _ = solver.analytic_characteristics(model)
    Q = solver.calculate_charge(r, f, omega)
    E = solver.calculate_energy(r, f, df, omega, model)
    tail = abs(float(f[-1])) / (abs(float(f[0])) + 1.0e-30)
    nodes = solver.count_nodes(f)
    residual = float(np.max(np.abs(
        solver.equation_residual(r, f, df, omega, model)
    )))
    theta = solver.theta_from_omega(omega, omega_min, model.m)

    reference = solver.Solution(
        model.name, model.p, model.q, model.alpha, model.beta,
        A, omega, theta, omega_min, f_star,
        Q, E, E / Q, tail, nodes, float(np.min(f)), residual,
        True, "loaded from saved branch_profiles.npz",
        r, f, df,
    )
    return model, reference


def extend_asymptotic_guess(previous, new_rmax: float):
    """Extend the previous profile with its decaying linear asymptotic tail."""
    old_r = previous.r
    old_f = previous.f
    old_df = previous.df
    if new_rmax <= old_r[-1]:
        return previous

    dr = float(np.median(np.diff(old_r)))
    extra_r = np.arange(old_r[-1] + dr, new_rmax, dr)
    if len(extra_r) == 0 or extra_r[-1] < new_rmax:
        extra_r = np.append(extra_r, new_rmax)

    kappa = math.sqrt(max(solver.MASS**2 - previous.omega**2, 1.0e-14))
    f_edge = float(old_f[-1])
    edge_r = float(old_r[-1])
    extra_f = f_edge * (edge_r / extra_r) * np.exp(-kappa * (extra_r - edge_r))
    extra_df = -(kappa + 1.0 / extra_r) * extra_f

    return solver.Solution(
        previous.model_name, previous.p, previous.q,
        previous.alpha, previous.beta,
        previous.A, previous.omega, previous.theta,
        previous.omega_min, previous.f_star,
        previous.Q, previous.E, previous.E_over_Q,
        previous.tail_ratio, previous.node_count, previous.min_f,
        previous.max_residual, previous.solver_success,
        previous.solver_message,
        np.concatenate((old_r, extra_r)),
        np.concatenate((old_f, extra_f)),
        np.concatenate((old_df, extra_df)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend only the saved p4-q6 high-omega R_max test."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--rmax", type=float, nargs="+", default=DEFAULT_RMAX)
    args = parser.parse_args()

    requested_rmax = sorted(set(float(x) for x in args.rmax))
    if not requested_rmax or requested_rmax[0] <= REFERENCE_RMAX:
        raise ValueError("All requested R_max values must be greater than 50.")

    model, previous = load_reference(args.model_dir, args.target_omega)
    if abs(previous.r[-1] - REFERENCE_RMAX) > 1.0e-6:
        raise ValueError(
            f"Expected the saved branch profile to end at R_max={REFERENCE_RMAX}; "
            f"found {previous.r[-1]:.6g}"
        )

    historical_path = args.model_dir / "RMAX_high_omega.csv"
    historical_70 = None
    if historical_path.exists():
        with historical_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if abs(float(row["RMAX"]) - 70.0) < 1.0e-9:
                historical_70 = {
                    "Q": float(row["Q"]),
                    "E": float(row["E"]),
                    "omega": float(row["omega"]),
                }
                break

    results = []
    for rmax in requested_rmax:
        if rmax <= previous.r[-1]:
            raise ValueError(
                "R_max values must increase, starting above the saved profile's 50."
            )

        # Keep approximately the original radial mesh spacing as the domain grows.
        solver.N_MESH_BASE = max(
            REFERENCE_MESH,
            int(round(REFERENCE_MESH * rmax / REFERENCE_RMAX)),
        )
        seed = extend_asymptotic_guess(previous, rmax)
        result = solver.solve_at_amplitude(
            model=model,
            A=previous.A,
            rmax=rmax,
            previous=seed,
        )
        if not result.solver_success:
            raise RuntimeError(
                f"BVP failed at R_max={rmax:g}: {result.solver_message}"
            )

        row = {
            "RMAX": rmax,
            "mesh_points": solver.N_MESH_BASE,
            "A_center": result.A,
            "omega": result.omega,
            "omega_shift_from_target": result.omega - args.target_omega,
            "Q": result.Q,
            "E": result.E,
            "E_over_Q": result.E_over_Q,
            "tail_ratio": result.tail_ratio,
            "node_count": result.node_count,
            "max_residual": result.max_residual,
            "Q_relative_change_vs_previous": relative_change(result.Q, previous.Q),
            "E_relative_change_vs_previous": relative_change(result.E, previous.E),
            "Q_relative_change_vs_RMAX70_saved": "",
            "E_relative_change_vs_RMAX70_saved": "",
            "historical_RMAX70_omega": "",
        }
        if historical_70 is not None:
            row["Q_relative_change_vs_RMAX70_saved"] = relative_change(
                result.Q, historical_70["Q"]
            )
            row["E_relative_change_vs_RMAX70_saved"] = relative_change(
                result.E, historical_70["E"]
            )
            row["historical_RMAX70_omega"] = historical_70["omega"]

        results.append(row)
        previous = result
        print(
            f"R_max={rmax:6.1f}  omega={result.omega:.12f}  "
            f"Q={result.Q:.9f}  E={result.E:.9f}  "
            f"tail={result.tail_ratio:.3e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved audit table: {args.output}")
    print("This checks numerical convergence only; it does not test time stability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
