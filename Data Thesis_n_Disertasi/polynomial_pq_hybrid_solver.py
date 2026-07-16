
"""
polynomial_pq_hybrid_solver.py
==============================

Robust multi-start continuation solver for polynomial Q-balls.

Models:
    (p,q) = (4,6), (4,8), (6,8)

Potential:
    U(f) = m^2 f^2 - alpha f^p + beta f^q

Radial equation:
    f'' + (2/r) f'
    - (m^2 - omega^2) f
    + (p/2) alpha f^(p-1)
    - (q/2) beta f^(q-1) = 0

Main improvements:
1. Multi-start seeds in both amplitude and omega.
2. Adaptive continuation in A=f(0), upward and downward.
3. Automatic retry when solve_bvp fails or reports singular Jacobian.
4. Branch-jump rejection using profile distance.
5. Deduplication of solutions at nearly identical omega.
6. Strict separation between:
       - branch scan,
       - RMAX convergence test,
       - VK calculation.
7. Full radial profiles are saved to NPZ so later analyses do NOT need
   to solve the BVP again.
8. All figures are exported to PNG. No plt.show().

Run:
    python polynomial_pq_hybrid_solver.py

Dependencies:
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_bvp
from scipy.interpolate import UnivariateSpline


# ============================================================
# GLOBAL SETTINGS
# ============================================================

MASS = 1.0

RMAX_BRANCH = 50.0
RMAX_TESTS = [30.0, 40.0, 50.0, 70.0]

EPS = 1.0e-5
N_MESH_BASE = 700
N_EVAL = 5000
MAX_NODES = 120000

# Primary and fallback tolerances
BVP_TOLERANCES = [5.0e-7, 1.0e-6, 3.0e-6]

# Continuation
ANCHOR_FACTORS = [0.20, 0.35, 0.55, 0.80, 1.00, 1.20, 1.45, 1.65]
OMEGA_SEED_FRACTIONS = [0.55, 0.70, 0.82, 0.92]

A_MIN_FACTOR = 0.06
A_MAX_FACTOR = 1.90
A_STEP_INITIAL = 0.035
A_STEP_MIN = 0.003
A_STEP_MAX = 0.10

# Acceptance / filtering
TAIL_ACCEPT = 2.0e-4
TAIL_STRICT = 1.0e-6
NODE_TOL = 1.0e-5
OMEGA_DEDUP_TOL = 2.0e-5
PROFILE_JUMP_LIMIT = 0.85
RESIDUAL_ACCEPT = 2.0e-3

# Retry perturbations
GUESS_WIDTH_FACTORS = [0.8, 1.0, 1.25, 1.6]
GUESS_AMPLITUDE_FACTORS = [0.92, 1.0, 1.08]

# VK
VK_MIN_POINTS = 5
VK_SPLINE_SMOOTHING = 1.0e-8

OUTPUT_DIR = Path("polynomial_pq_hybrid_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint/resume
CHECKPOINT_ROOT = OUTPUT_DIR / "_checkpoints"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

# Increment this only when the checkpoint structure becomes incompatible.
CHECKPOINT_FORMAT_VERSION = 1


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class PQModel:
    p: int
    q: int
    alpha: float
    beta: float
    m: float = MASS

    @property
    def name(self) -> str:
        return f"p{self.p}_q{self.q}"


@dataclass
class Solution:
    model_name: str
    p: int
    q: int
    alpha: float
    beta: float

    A: float
    omega: float
    theta: float
    omega_min: float
    f_star: float

    Q: float
    E: float
    E_over_Q: float

    tail_ratio: float
    node_count: int
    min_f: float
    max_residual: float

    solver_success: bool
    solver_message: str

    r: np.ndarray
    f: np.ndarray
    df: np.ndarray


# ============================================================
# MODELS
# ============================================================

def construct_models() -> list[PQModel]:
    """
    Normalize the three models to:
        f_star = 1
        omega_min^2 = 0.5
    """
    pairs = [(4, 6), (4, 8), (6, 8)]
    models: list[PQModel] = []

    for p, q in pairs:
        alpha = 0.5 * (q - 2) / (q - p)
        beta = 0.5 * (p - 2) / (q - p)
        models.append(PQModel(p=p, q=q, alpha=alpha, beta=beta))

    return models


def analytic_characteristics(model: PQModel) -> tuple[float, float, float]:
    numerator = (model.p - 2) * model.alpha
    denominator = (model.q - 2) * model.beta

    f_star = (numerator / denominator) ** (1.0 / (model.q - model.p))

    omega_min_sq = (
        model.m**2
        - model.alpha * f_star ** (model.p - 2)
        + model.beta * f_star ** (model.q - 2)
    )

    if not (0.0 <= omega_min_sq < model.m**2):
        raise ValueError(
            f"{model.name}: invalid omega_min^2={omega_min_sq:.8g}"
        )

    return f_star, math.sqrt(omega_min_sq), omega_min_sq


# ============================================================
# PARAMETERIZATION OF OMEGA
# ============================================================

def stable_sigmoid(x: float | np.ndarray):
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr)
    pos = arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return float(out) if np.isscalar(x) else out


def omega_from_theta(theta: float, omega_min: float, mass: float) -> float:
    margin = 1.0e-6
    lower = omega_min + margin
    upper = mass - margin
    return lower + (upper - lower) * stable_sigmoid(theta)


def theta_from_omega(omega: float, omega_min: float, mass: float) -> float:
    margin = 1.0e-6
    lower = omega_min + margin
    upper = mass - margin
    x = (omega - lower) / (upper - lower)
    x = np.clip(x, 1.0e-10, 1.0 - 1.0e-10)
    return float(np.log(x / (1.0 - x)))


# ============================================================
# PHYSICS
# ============================================================

def potential(f: np.ndarray, model: PQModel) -> np.ndarray:
    return (
        model.m**2 * f**2
        - model.alpha * f**model.p
        + model.beta * f**model.q
    )


def radial_force(f: np.ndarray, omega: float, model: PQModel) -> np.ndarray:
    return (
        (model.m**2 - omega**2) * f
        - 0.5 * model.p * model.alpha * f ** (model.p - 1)
        + 0.5 * model.q * model.beta * f ** (model.q - 1)
    )


def calculate_charge(r: np.ndarray, f: np.ndarray, omega: float) -> float:
    return float(
        8.0 * np.pi * omega * simpson(f**2 * r**2, x=r)
    )


def calculate_energy(
    r: np.ndarray,
    f: np.ndarray,
    df: np.ndarray,
    omega: float,
    model: PQModel,
) -> float:
    density = omega**2 * f**2 + df**2 + potential(f, model)
    return float(4.0 * np.pi * simpson(density * r**2, x=r))


def count_nodes(f: np.ndarray, tolerance: float = NODE_TOL) -> int:
    scale = max(float(np.max(np.abs(f))), 1.0e-30)
    significant = f[np.abs(f) > tolerance * scale]

    if significant.size < 2:
        return 0

    signs = np.sign(significant)
    return int(np.sum(signs[:-1] * signs[1:] < 0))


def equation_residual(
    r: np.ndarray,
    f: np.ndarray,
    df: np.ndarray,
    omega: float,
    model: PQModel,
) -> np.ndarray:
    d2f = np.gradient(df, r, edge_order=2)
    safe_r = np.maximum(r, EPS)

    residual = (
        d2f
        + 2.0 * df / safe_r
        - (model.m**2 - omega**2) * f
        + 0.5 * model.p * model.alpha * f ** (model.p - 1)
        - 0.5 * model.q * model.beta * f ** (model.q - 1)
    )

    residual[:8] = 0.0
    return residual


# ============================================================
# INITIAL GUESSES
# ============================================================

def gaussian_guess(
    r: np.ndarray,
    amplitude: float,
    omega_guess: float,
    model: PQModel,
    width_factor: float = 1.0,
    amplitude_factor: float = 1.0,
) -> np.ndarray:
    kappa = math.sqrt(max(model.m**2 - omega_guess**2, 1.0e-8))
    width = width_factor * max(2.5 / kappa, 2.0)

    f = amplitude_factor * amplitude * np.exp(-(r / width) ** 2)
    df = -2.0 * r * f / width**2

    # Enforce desired central amplitude exactly.
    scale = amplitude / max(f[0], 1.0e-30)
    return np.vstack((f * scale, df * scale))


def previous_solution_guess(
    previous: Solution,
    new_r: np.ndarray,
    new_A: float,
) -> np.ndarray:
    scale = new_A / max(previous.A, 1.0e-30)
    f = np.interp(new_r, previous.r, previous.f) * scale
    df = np.interp(new_r, previous.r, previous.df) * scale
    return np.vstack((f, df))


# ============================================================
# CORE BVP SOLVER WITH RETRIES
# ============================================================

def _solve_bvp_once(
    model: PQModel,
    A: float,
    rmax: float,
    theta_guess: float,
    y_guess: np.ndarray,
    tolerance: float,
    n_mesh: int,
) -> Solution:
    f_star, omega_min, _ = analytic_characteristics(model)
    r_mesh = np.linspace(EPS, rmax, n_mesh)

    # Interpolate guess if its mesh differs.
    if y_guess.shape[1] != n_mesh:
        old_x = np.linspace(EPS, rmax, y_guess.shape[1])
        y_guess = np.vstack(
            [
                np.interp(r_mesh, old_x, y_guess[0]),
                np.interp(r_mesh, old_x, y_guess[1]),
            ]
        )

    def ode(r: np.ndarray, y: np.ndarray, pars: np.ndarray) -> np.ndarray:
        theta = float(pars[0])
        omega = omega_from_theta(theta, omega_min, model.m)
        f, df = y
        safe_r = np.maximum(r, EPS)
        d2f = -(2.0 / safe_r) * df + radial_force(f, omega, model)
        return np.vstack((df, d2f))

    def bc(ya: np.ndarray, yb: np.ndarray, pars: np.ndarray) -> np.ndarray:
        theta = float(pars[0])
        omega = omega_from_theta(theta, omega_min, model.m)
        kappa = math.sqrt(max(model.m**2 - omega**2, 1.0e-14))

        return np.array(
            [
                ya[0] - A,
                ya[1],
                yb[1] + (kappa + 1.0 / rmax) * yb[0],
            ]
        )

    result = solve_bvp(
        ode,
        bc,
        r_mesh,
        y_guess,
        p=np.array([theta_guess]),
        tol=tolerance,
        max_nodes=MAX_NODES,
        verbose=0,
    )

    if not result.success:
        return Solution(
            model.name, model.p, model.q, model.alpha, model.beta,
            A, np.nan, np.nan, omega_min, f_star,
            np.nan, np.nan, np.nan,
            np.nan, -1, np.nan, np.nan,
            False, str(result.message),
            np.array([]), np.array([]), np.array([]),
        )

    theta = float(result.p[0])
    omega = omega_from_theta(theta, omega_min, model.m)

    r = np.linspace(EPS, rmax, N_EVAL)
    f, df = result.sol(r)

    Q = calculate_charge(r, f, omega)
    E = calculate_energy(r, f, df, omega, model)
    E_over_Q = E / Q if Q > 0 else np.inf
    tail_ratio = abs(f[-1]) / (abs(f[0]) + 1.0e-30)
    nodes = count_nodes(f)

    residual = equation_residual(r, f, df, omega, model)
    max_residual = float(np.max(np.abs(residual)))

    return Solution(
        model.name, model.p, model.q, model.alpha, model.beta,
        A, omega, theta, omega_min, f_star,
        Q, E, E_over_Q,
        tail_ratio, nodes, float(np.min(f)), max_residual,
        True, str(result.message),
        r, f, df,
    )


def solve_at_amplitude(
    model: PQModel,
    A: float,
    rmax: float,
    previous: Solution | None = None,
    omega_seed: float | None = None,
) -> Solution:
    """
    Robust solve with multiple retry strategies.
    """

    f_star, omega_min, _ = analytic_characteristics(model)

    if previous is not None:
        theta_base = previous.theta
        base_mesh = np.linspace(EPS, rmax, N_MESH_BASE)
        base_guess = previous_solution_guess(previous, base_mesh, A)

        trial_specs = []

        for tol in BVP_TOLERANCES:
            for amp_factor in [1.0, 0.98, 1.02]:
                guess = base_guess.copy()
                guess[0] *= amp_factor
                guess[1] *= amp_factor

                # Re-enforce the center amplitude.
                center_scale = A / max(guess[0, 0], 1.0e-30)
                guess *= center_scale

                trial_specs.append(
                    (theta_base, guess, tol, N_MESH_BASE)
                )

        # Fallback Gaussian seeds near previous omega.
        omega_candidates = [
            previous.omega,
            max(omega_min + 1.0e-5, previous.omega - 0.01),
            min(model.m - 1.0e-5, previous.omega + 0.01),
        ]

    else:
        if omega_seed is None:
            omega_seed = omega_min + 0.78 * (model.m - omega_min)

        omega_candidates = [
            omega_seed,
            omega_min + 0.60 * (model.m - omega_min),
            omega_min + 0.75 * (model.m - omega_min),
            omega_min + 0.88 * (model.m - omega_min),
            omega_min + 0.94 * (model.m - omega_min),
        ]
        trial_specs = []

    for omega_guess in omega_candidates:
        theta_guess = theta_from_omega(
            omega_guess,
            omega_min,
            model.m,
        )

        r_mesh = np.linspace(EPS, rmax, N_MESH_BASE)

        for width_factor in GUESS_WIDTH_FACTORS:
            for amp_factor in GUESS_AMPLITUDE_FACTORS:
                guess = gaussian_guess(
                    r_mesh,
                    A,
                    omega_guess,
                    model,
                    width_factor=width_factor,
                    amplitude_factor=amp_factor,
                )

                for tol in BVP_TOLERANCES:
                    trial_specs.append(
                        (theta_guess, guess, tol, N_MESH_BASE)
                    )

    best_failure: Solution | None = None

    for theta_guess, guess, tol, n_mesh in trial_specs:
        try:
            result = _solve_bvp_once(
                model=model,
                A=A,
                rmax=rmax,
                theta_guess=theta_guess,
                y_guess=guess,
                tolerance=tol,
                n_mesh=n_mesh,
            )
        except Exception as exc:
            best_failure = Solution(
                model.name, model.p, model.q, model.alpha, model.beta,
                A, np.nan, np.nan, omega_min, f_star,
                np.nan, np.nan, np.nan,
                np.nan, -1, np.nan, np.nan,
                False, str(exc),
                np.array([]), np.array([]), np.array([]),
            )
            continue

        if result.solver_success:
            return result

        best_failure = result

    if best_failure is not None:
        return best_failure

    return Solution(
        model.name, model.p, model.q, model.alpha, model.beta,
        A, np.nan, np.nan, omega_min, f_star,
        np.nan, np.nan, np.nan,
        np.nan, -1, np.nan, np.nan,
        False, "All solver retries failed.",
        np.array([]), np.array([]), np.array([]),
    )


# ============================================================
# CONTINUATION
# ============================================================

def normalized_profile_distance(a: Solution, b: Solution) -> float:
    common_r = np.linspace(EPS, min(a.r[-1], b.r[-1]), 2500)
    fa = np.interp(common_r, a.r, a.f)
    fb = np.interp(common_r, b.r, b.f)

    numerator = math.sqrt(
        float(simpson((fa - fb) ** 2, x=common_r))
    )
    denominator = (
        math.sqrt(float(simpson(fa**2, x=common_r)))
        + 1.0e-30
    )

    return numerator / denominator


def accept_solution(
    candidate: Solution,
    previous: Solution | None,
) -> bool:
    if not candidate.solver_success:
        return False
    if not np.isfinite(candidate.omega):
        return False
    if candidate.node_count != 0:
        return False
    if candidate.tail_ratio > TAIL_ACCEPT:
        return False
    if candidate.max_residual > RESIDUAL_ACCEPT:
        return False

    if previous is not None:
        jump = normalized_profile_distance(previous, candidate)
        if jump > PROFILE_JUMP_LIMIT:
            return False

    return True


def continue_direction(
    model: PQModel,
    anchor_A: float,
    direction: int,
    rmax: float,
    omega_seed: float,
) -> list[Solution]:
    f_star, _, _ = analytic_characteristics(model)

    A_min = A_MIN_FACTOR * f_star
    A_max = A_MAX_FACTOR * f_star

    anchor = solve_at_amplitude(
        model=model,
        A=anchor_A,
        rmax=rmax,
        previous=None,
        omega_seed=omega_seed,
    )

    if not accept_solution(anchor, None):
        return []

    solutions = [anchor]
    previous = anchor

    A = anchor_A
    step = A_STEP_INITIAL * f_star
    success_streak = 0

    while True:
        A_trial = A + direction * step

        if A_trial < A_min or A_trial > A_max:
            break

        candidate = solve_at_amplitude(
            model=model,
            A=A_trial,
            rmax=rmax,
            previous=previous,
        )

        if accept_solution(candidate, previous):
            solutions.append(candidate)
            previous = candidate
            A = A_trial
            success_streak += 1

            if success_streak >= 3:
                step = min(
                    step * 1.25,
                    A_STEP_MAX * f_star,
                )
                success_streak = 0
        else:
            step *= 0.5
            success_streak = 0

            if step < A_STEP_MIN * f_star:
                break

    return solutions


def deduplicate_by_omega(
    solutions: Iterable[Solution],
) -> list[Solution]:
    valid = sorted(
        [
            s for s in solutions
            if s.solver_success and np.isfinite(s.omega)
        ],
        key=lambda s: s.omega,
    )

    if not valid:
        return []

    groups: list[list[Solution]] = [[valid[0]]]

    for sol in valid[1:]:
        if abs(sol.omega - groups[-1][-1].omega) < OMEGA_DEDUP_TOL:
            groups[-1].append(sol)
        else:
            groups.append([sol])

    merged: list[Solution] = []

    for group in groups:
        best = min(
            group,
            key=lambda s: (
                s.tail_ratio,
                s.max_residual,
                abs(s.min_f),
            ),
        )
        merged.append(best)

    return merged



def checkpoint_settings_signature(model: PQModel) -> dict:
    """
    Metadata used to reject stale checkpoints created with incompatible
    numerical settings.
    """
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.name,
        "p": model.p,
        "q": model.q,
        "alpha": model.alpha,
        "beta": model.beta,
        "mass": model.m,
        "rmax_branch": RMAX_BRANCH,
        "n_mesh_base": N_MESH_BASE,
        "n_eval": N_EVAL,
        "anchor_factors": list(ANCHOR_FACTORS),
        "omega_seed_fractions": list(OMEGA_SEED_FRACTIONS),
        "a_min_factor": A_MIN_FACTOR,
        "a_max_factor": A_MAX_FACTOR,
        "a_step_initial": A_STEP_INITIAL,
        "a_step_min": A_STEP_MIN,
        "a_step_max": A_STEP_MAX,
        "tail_accept": TAIL_ACCEPT,
        "residual_accept": RESIDUAL_ACCEPT,
    }


def model_checkpoint_dir(model: PQModel) -> Path:
    path = CHECKPOINT_ROOT / model.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def anchor_checkpoint_path(
    model: PQModel,
    anchor_index: int,
    seed_index: int,
) -> Path:
    return model_checkpoint_dir(model) / (
        f"anchor_{anchor_index:02d}_seed_{seed_index:02d}.pkl"
    )


def atomic_pickle_dump(payload: object, path: Path) -> None:
    """
    Write to a temporary file and rename atomically. This prevents a corrupt
    checkpoint if the program or laptop stops during the save operation.
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()

    temp_path.replace(path)


def save_anchor_checkpoint(
    model: PQModel,
    anchor_index: int,
    seed_index: int,
    anchor_factor: float,
    omega_seed_fraction: float,
    anchor_A: float,
    omega_seed: float,
    solutions: list[Solution],
) -> Path:
    path = anchor_checkpoint_path(model, anchor_index, seed_index)

    payload = {
        "settings": checkpoint_settings_signature(model),
        "anchor_index": anchor_index,
        "seed_index": seed_index,
        "anchor_factor": anchor_factor,
        "omega_seed_fraction": omega_seed_fraction,
        "anchor_A": anchor_A,
        "omega_seed": omega_seed,
        "raw_count": len(solutions),
        "solutions": solutions,
    }

    atomic_pickle_dump(payload, path)
    return path


def load_anchor_checkpoint(
    model: PQModel,
    anchor_index: int,
    seed_index: int,
) -> dict | None:
    path = anchor_checkpoint_path(model, anchor_index, seed_index)

    if not path.exists():
        return None

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(
            f"\nCheckpoint rusak dan akan dihitung ulang: {path.name} "
            f"({exc})"
        )
        path.unlink(missing_ok=True)
        return None

    if payload.get("settings") != checkpoint_settings_signature(model):
        print(
            f"\nCheckpoint tidak cocok dengan setting saat ini dan "
            f"akan dihitung ulang: {path.name}"
        )
        return None

    return payload


def save_progress_manifest(
    model: PQModel,
    completed: int,
    total: int,
    current_anchor_index: int | None = None,
    current_seed_index: int | None = None,
) -> None:
    manifest = {
        "settings": checkpoint_settings_signature(model),
        "completed_anchors": completed,
        "total_anchors": total,
        "current_anchor_index": current_anchor_index,
        "current_seed_index": current_seed_index,
    }

    path = model_checkpoint_dir(model) / "progress.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.replace(path)


def clear_checkpoints(selected_models: list[PQModel] | None = None) -> None:
    if selected_models is None:
        targets = [CHECKPOINT_ROOT]
    else:
        targets = [CHECKPOINT_ROOT / model.name for model in selected_models]

    for target in targets:
        if target.exists():
            shutil.rmtree(target)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


def count_completed_anchor_checkpoints(model: PQModel) -> int:
    count = 0

    for anchor_index, _ in enumerate(ANCHOR_FACTORS):
        for seed_index, _ in enumerate(OMEGA_SEED_FRACTIONS):
            if load_anchor_checkpoint(
                model,
                anchor_index,
                seed_index,
            ) is not None:
                count += 1

    return count


def merge_all_anchor_checkpoints(model: PQModel) -> list[Solution]:
    all_solutions: list[Solution] = []

    for anchor_index, _ in enumerate(ANCHOR_FACTORS):
        for seed_index, _ in enumerate(OMEGA_SEED_FRACTIONS):
            payload = load_anchor_checkpoint(
                model,
                anchor_index,
                seed_index,
            )

            if payload is not None:
                all_solutions.extend(payload.get("solutions", []))

    return all_solutions


def run_multistart_branch(model: PQModel) -> list[Solution]:
    """
    Multi-start scan with automatic checkpoint/resume.

    A checkpoint is written after every complete (anchor, omega_seed) job.
    When the script is run again, completed jobs are loaded and skipped.
    The currently interrupted job is the only one that may be recomputed.
    """
    f_star, omega_min, _ = analytic_characteristics(model)

    omega_seeds = [
        omega_min + fraction * (model.m - omega_min)
        for fraction in OMEGA_SEED_FRACTIONS
    ]

    total_jobs = len(ANCHOR_FACTORS) * len(omega_seeds)
    completed_before = count_completed_anchor_checkpoints(model)

    print("\n" + "=" * 78)
    print(f"MODEL {model.name}")
    print("=" * 78)
    print(
        f"p={model.p}, q={model.q}, "
        f"alpha={model.alpha:.8f}, beta={model.beta:.8f}, "
        f"f*={f_star:.8f}, omega_min={omega_min:.8f}"
    )

    if completed_before:
        print(
            f"Resume checkpoint: {completed_before}/{total_jobs} "
            f"anchor-seed jobs sudah selesai."
        )

    completed = completed_before

    try:
        for anchor_index, factor in enumerate(ANCHOR_FACTORS):
            anchor_A = factor * f_star

            for seed_index, omega_seed in enumerate(omega_seeds):
                payload = load_anchor_checkpoint(
                    model,
                    anchor_index,
                    seed_index,
                )

                job_number = (
                    anchor_index * len(omega_seeds)
                    + seed_index
                    + 1
                )

                if payload is not None:
                    print(
                        f"[{job_number:02d}/{total_jobs:02d}] "
                        f"anchor A={anchor_A:.5f}, "
                        f"omega_seed={omega_seed:.6f} ... "
                        f"SKIP checkpoint "
                        f"({payload.get('raw_count', 0)} raw solutions)"
                    )
                    continue

                save_progress_manifest(
                    model,
                    completed,
                    total_jobs,
                    anchor_index,
                    seed_index,
                )

                print(
                    f"[{job_number:02d}/{total_jobs:02d}] "
                    f"anchor A={anchor_A:.5f}, "
                    f"omega_seed={omega_seed:.6f} ... ",
                    end="",
                    flush=True,
                )

                up = continue_direction(
                    model,
                    anchor_A,
                    +1,
                    RMAX_BRANCH,
                    omega_seed,
                )

                down = continue_direction(
                    model,
                    anchor_A,
                    -1,
                    RMAX_BRANCH,
                    omega_seed,
                )

                raw_solutions = up + down
                raw_count = len(raw_solutions)

                checkpoint_path = save_anchor_checkpoint(
                    model=model,
                    anchor_index=anchor_index,
                    seed_index=seed_index,
                    anchor_factor=factor,
                    omega_seed_fraction=OMEGA_SEED_FRACTIONS[
                        seed_index
                    ],
                    anchor_A=anchor_A,
                    omega_seed=omega_seed,
                    solutions=raw_solutions,
                )

                completed += 1

                save_progress_manifest(
                    model,
                    completed,
                    total_jobs,
                    None,
                    None,
                )

                print(
                    f"{raw_count} raw solutions | "
                    f"checkpoint={checkpoint_path.name}"
                )

    except KeyboardInterrupt:
        save_progress_manifest(
            model,
            completed,
            total_jobs,
            None,
            None,
        )

        print(
            "\n\nProgram dihentikan pengguna."
            "\nSemua anchor-seed yang telah selesai sudah tersimpan."
            "\nSaat program dijalankan kembali, solver akan melewati "
            "checkpoint tersebut dan melanjutkan job yang belum selesai."
        )
        raise

    all_solutions = merge_all_anchor_checkpoints(model)
    branch = deduplicate_by_omega(all_solutions)

    print(f"Unique accepted solutions: {len(branch)}")

    if branch:
        print(
            f"omega coverage: "
            f"{branch[0].omega:.8f} -> {branch[-1].omega:.8f}"
        )

    return branch


# ============================================================
# VK
# ============================================================

def vk_gradient(branch: list[Solution]) -> np.ndarray:
    n = len(branch)

    if n < VK_MIN_POINTS:
        return np.full(n, np.nan)

    omega = np.array([s.omega for s in branch])
    Q = np.array([s.Q for s in branch])

    keep = np.concatenate(
        ([True], np.diff(omega) > OMEGA_DEDUP_TOL / 2.0)
    )

    omega_u = omega[keep]
    Q_u = Q[keep]

    if len(omega_u) < VK_MIN_POINTS:
        return np.full(n, np.nan)

    degree = min(3, len(omega_u) - 1)

    spline = UnivariateSpline(
        omega_u,
        Q_u,
        k=degree,
        s=VK_SPLINE_SMOOTHING * len(omega_u),
    )

    derivative = spline.derivative()(omega_u)
    return np.interp(omega, omega_u, derivative)


# ============================================================
# RMAX TESTS
# ============================================================

def rmax_test(
    model: PQModel,
    reference: Solution,
) -> list[tuple[float, Solution]]:
    results: list[tuple[float, Solution]] = []
    previous: Solution | None = None

    for rmax in RMAX_TESTS:
        seed = previous if previous is not None else reference

        result = solve_at_amplitude(
            model=model,
            A=reference.A,
            rmax=rmax,
            previous=seed,
        )

        if result.solver_success:
            results.append((rmax, result))
            previous = result

    return results


# ============================================================
# SAVING DATA
# ============================================================

def save_branch_csv(
    branch: list[Solution],
    path: Path,
) -> None:
    gradient = vk_gradient(branch)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "model", "p", "q", "alpha", "beta",
            "A_center", "omega", "omega_min",
            "Q", "E", "E_over_Q", "dQ_domega",
            "tail_ratio", "node_count", "min_f",
            "max_residual", "localized_strict",
            "ground_state", "energetically_stable",
            "VK_candidate",
        ])

        for solution, derivative in zip(branch, gradient):
            writer.writerow([
                solution.model_name,
                solution.p,
                solution.q,
                solution.alpha,
                solution.beta,
                solution.A,
                solution.omega,
                solution.omega_min,
                solution.Q,
                solution.E,
                solution.E_over_Q,
                derivative,
                solution.tail_ratio,
                solution.node_count,
                solution.min_f,
                solution.max_residual,
                solution.tail_ratio < TAIL_STRICT,
                solution.node_count == 0,
                solution.E_over_Q < MASS,
                bool(np.isfinite(derivative) and derivative < 0),
            ])


def save_branch_profiles_npz(
    branch: list[Solution],
    path: Path,
) -> None:
    """
    Save full radial profiles.
    Later scripts should load this file instead of solving BVP again.
    """
    if not branch:
        np.savez_compressed(
            path,
            omega=np.array([]),
            A_center=np.array([]),
            r=np.empty((0, N_EVAL)),
            f=np.empty((0, N_EVAL)),
            df=np.empty((0, N_EVAL)),
        )
        return

    np.savez_compressed(
        path,
        omega=np.array([s.omega for s in branch]),
        A_center=np.array([s.A for s in branch]),
        Q=np.array([s.Q for s in branch]),
        E=np.array([s.E for s in branch]),
        E_over_Q=np.array([s.E_over_Q for s in branch]),
        tail_ratio=np.array([s.tail_ratio for s in branch]),
        node_count=np.array([s.node_count for s in branch]),
        max_residual=np.array([s.max_residual for s in branch]),
        r=np.stack([s.r for s in branch]),
        f=np.stack([s.f for s in branch]),
        df=np.stack([s.df for s in branch]),
    )


def save_rmax_csv(
    results: list[tuple[float, Solution]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "RMAX",
            "A_center",
            "omega",
            "Q",
            "E",
            "E_over_Q",
            "tail_ratio",
            "node_count",
            "max_residual",
        ])

        for rmax, solution in results:
            writer.writerow([
                rmax,
                solution.A,
                solution.omega,
                solution.Q,
                solution.E,
                solution.E_over_Q,
                solution.tail_ratio,
                solution.node_count,
                solution.max_residual,
            ])


def branch_summary(branch: list[Solution]) -> dict[str, float]:
    if not branch:
        return {
            "count": 0,
            "omega_min_found": np.nan,
            "omega_max_found": np.nan,
            "localized_fraction": 0.0,
            "ground_fraction": 0.0,
            "energetic_fraction": 0.0,
            "VK_fraction": 0.0,
        }

    gradient = vk_gradient(branch)

    return {
        "count": len(branch),
        "omega_min_found": min(s.omega for s in branch),
        "omega_max_found": max(s.omega for s in branch),
        "localized_fraction": float(
            np.mean([s.tail_ratio < TAIL_STRICT for s in branch])
        ),
        "ground_fraction": float(
            np.mean([s.node_count == 0 for s in branch])
        ),
        "energetic_fraction": float(
            np.mean([s.E_over_Q < MASS for s in branch])
        ),
        "VK_fraction": float(
            np.mean(gradient < 0)
        ) if np.any(np.isfinite(gradient)) else 0.0,
    }


def save_model_comparison(
    all_branches: dict[str, list[Solution]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "model",
            "count",
            "omega_min_found",
            "omega_max_found",
            "localized_fraction",
            "ground_fraction",
            "energetic_fraction",
            "VK_fraction",
        ])

        for name, branch in all_branches.items():
            summary = branch_summary(branch)

            writer.writerow([
                name,
                summary["count"],
                summary["omega_min_found"],
                summary["omega_max_found"],
                summary["localized_fraction"],
                summary["ground_fraction"],
                summary["energetic_fraction"],
                summary["VK_fraction"],
            ])


# ============================================================
# PLOTS
# ============================================================

def plot_branch(
    model: PQModel,
    branch: list[Solution],
    model_dir: Path,
) -> None:
    if not branch:
        return

    omega = np.array([s.omega for s in branch])
    A = np.array([s.A for s in branch])
    Q = np.array([s.Q for s in branch])
    E = np.array([s.E for s in branch])
    E_over_Q = np.array([s.E_over_Q for s in branch])
    tail = np.array([s.tail_ratio for s in branch])
    gradient = vk_gradient(branch)

    # Profiles
    plt.figure(figsize=(10, 6))
    indices = np.linspace(
        0,
        len(branch) - 1,
        min(10, len(branch)),
        dtype=int,
    )

    for index in indices:
        solution = branch[index]
        plt.plot(
            solution.r,
            solution.f,
            label=(
                f"omega={solution.omega:.5f}, "
                f"A={solution.A:.3f}"
            ),
        )

    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.title(f"Ground-state profiles: {model.name}")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(model_dir / "01_profiles.png", dpi=300)
    plt.close()

    # omega versus A
    plt.figure(figsize=(8, 5))
    plt.plot(A, omega, "o-")
    plt.axhline(
        branch[0].omega_min,
        linestyle="--",
        label="analytic omega_min",
    )
    plt.axhline(model.m, linestyle=":", label="m")
    plt.xlabel("A=f(0)")
    plt.ylabel("omega")
    plt.title(f"Continuation branch: {model.name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "02_omega_vs_A.png", dpi=300)
    plt.close()

    # Q
    plt.figure(figsize=(8, 5))
    plt.plot(omega, Q, "o-")
    plt.xlabel("omega")
    plt.ylabel("Q")
    plt.title(f"Q(omega): {model.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir / "03_Q_vs_omega.png", dpi=300)
    plt.close()

    # VK
    plt.figure(figsize=(8, 5))
    plt.plot(omega, gradient, "o-")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("omega")
    plt.ylabel("dQ/domega")
    plt.title(f"VK indicator: {model.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir / "04_VK.png", dpi=300)
    plt.close()

    # Energy
    plt.figure(figsize=(8, 5))
    plt.plot(omega, E, "o-")
    plt.xlabel("omega")
    plt.ylabel("E")
    plt.title(f"E(omega): {model.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_dir / "05_E_vs_omega.png", dpi=300)
    plt.close()

    # E/Q
    plt.figure(figsize=(8, 5))
    plt.plot(omega, E_over_Q, "o-")
    plt.axhline(model.m, linestyle="--", label="m")
    plt.xlabel("omega")
    plt.ylabel("E/Q")
    plt.title(f"Energetic stability: {model.name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "06_E_over_Q.png", dpi=300)
    plt.close()

    # Tail
    plt.figure(figsize=(8, 5))
    plt.semilogy(omega, tail, "o-")
    plt.axhline(
        TAIL_STRICT,
        linestyle="--",
        label="strict limit",
    )
    plt.xlabel("omega")
    plt.ylabel("|f(R)|/|f(0)|")
    plt.title(f"Tail localization: {model.name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "07_tail.png", dpi=300)
    plt.close()


def plot_rmax_results(
    model: PQModel,
    results: list[tuple[float, Solution]],
    label: str,
    model_dir: Path,
) -> None:
    if not results:
        return

    rmax = np.array([item[0] for item in results])
    Q = np.array([item[1].Q for item in results])
    E = np.array([item[1].E for item in results])
    tail = np.array([item[1].tail_ratio for item in results])

    plt.figure(figsize=(8, 5))
    plt.plot(rmax, Q, "o-")
    plt.xlabel("RMAX")
    plt.ylabel("Q")
    plt.title(f"Q convergence ({label}): {model.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / f"08_Q_RMAX_{label}.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rmax, E, "o-")
    plt.xlabel("RMAX")
    plt.ylabel("E")
    plt.title(f"E convergence ({label}): {model.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / f"09_E_RMAX_{label}.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(rmax, tail, "o-")
    plt.xlabel("RMAX")
    plt.ylabel("tail ratio")
    plt.title(f"Tail convergence ({label}): {model.name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / f"10_tail_RMAX_{label}.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Polynomial p-q Q-ball solver with automatic checkpoint/resume."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["p4_q6", "p4_q8", "p6_q8"],
        default=["p4_q6", "p4_q8", "p6_q8"],
        help=(
            "Model yang dijalankan. Default: semua model. "
            "Contoh: --models p4_q6"
        ),
    )

    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help=(
            "Hapus checkpoint untuk model yang dipilih lalu mulai dari awal."
        ),
    )

    parser.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Jangan menghitung anchor baru. Hanya gabungkan checkpoint "
            "yang sudah ada menjadi CSV, NPZ, dan grafik."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    print("=" * 78)
    print("ROBUST MULTI-START CONTINUATION: POLYNOMIAL p-q")
    print("AUTOMATIC CHECKPOINT / RESUME")
    print("=" * 78)

    all_models = construct_models()
    models = [
        model
        for model in all_models
        if model.name in args.models
    ]

    if args.reset_checkpoints:
        clear_checkpoints(models)
        print("Checkpoint model terpilih telah dihapus.")

    all_branches: dict[str, list[Solution]] = {}

    for model in models:
        if args.merge_only:
            raw_solutions = merge_all_anchor_checkpoints(model)
            branch = deduplicate_by_omega(raw_solutions)

            print(
                f"\nMERGE ONLY {model.name}: "
                f"{len(raw_solutions)} raw -> "
                f"{len(branch)} unique solutions"
            )
        else:
            branch = run_multistart_branch(model)

        all_branches[model.name] = branch

        model_dir = OUTPUT_DIR / model.name
        model_dir.mkdir(parents=True, exist_ok=True)

        # These files are regenerated from every completed checkpoint.
        save_branch_csv(
            branch,
            model_dir / "branch_results.csv",
        )

        save_branch_profiles_npz(
            branch,
            model_dir / "branch_profiles.npz",
        )

        plot_branch(
            model,
            branch,
            model_dir,
        )

        if not args.merge_only and len(branch) >= 3:
            representative_indices = {
                "low_omega": 0,
                "mid_omega": len(branch) // 2,
                "high_omega": len(branch) - 1,
            }

            for label, index in representative_indices.items():
                reference = branch[index]

                print(
                    f"RMAX test {model.name} {label}: "
                    f"omega={reference.omega:.8f}, "
                    f"A={reference.A:.6f}"
                )

                results = rmax_test(
                    model,
                    reference,
                )

                save_rmax_csv(
                    results,
                    model_dir / f"RMAX_{label}.csv",
                )

                plot_rmax_results(
                    model,
                    results,
                    label,
                    model_dir,
                )

    # Preserve previous model rows when only selected models are run.
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_branches = dict(all_branches)

    if comparison_path.exists():
        try:
            with comparison_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                old_rows = list(csv.DictReader(handle))

            selected_names = set(comparison_branches)

            for row in old_rows:
                name = row.get("model", "")
                if name and name not in selected_names:
                    # An empty placeholder avoids deleting old output rows.
                    comparison_branches.setdefault(name, [])
        except Exception:
            pass

    save_model_comparison(
        comparison_branches,
        comparison_path,
    )

    print("\n" + "=" * 78)
    print("FINISHED")
    print("=" * 78)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Checkpoint directory: {CHECKPOINT_ROOT.resolve()}")
    print("")
    print("Menjalankan ulang perintah yang sama akan otomatis resume.")
    print("")
    print("Perintah berguna:")
    print("  python polynomial_pq_hybrid_solver_checkpoint.py")
    print("  python polynomial_pq_hybrid_solver_checkpoint.py --models p4_q6")
    print(
        "  python polynomial_pq_hybrid_solver_checkpoint.py "
        "--models p4_q6 --merge-only"
    )
    print(
        "  python polynomial_pq_hybrid_solver_checkpoint.py "
        "--models p4_q6 --reset-checkpoints"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nKeluar dengan aman. Checkpoint anchor-seed yang sudah "
            "selesai tetap tersimpan."
        )
        sys.exit(130)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        sys.exit(1)
