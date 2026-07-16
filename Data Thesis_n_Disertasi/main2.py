"""
Benchmark Piecewise-Parabolic Q-ball
====================================

Formulasi dua-domain dengan unknown matching radius R.

Transformasi:
    u(r) = r f(r)

Wilayah dalam:
    u_in'' + k^2 u_in = 0

Wilayah luar:
    u_out'' - kappa^2 u_out = 0

dengan:
    k      = sqrt(omega^2 - m_inner^2)
    kappa  = sqrt(M_outer^2 - omega^2)

Kondisi batas:
    u_in(0) = 0
    f_in(R) = v
    f_out(R) = v
    f_in'(R) = f_out'(R)
    u_out'(RMAX) + kappa*u_out(RMAX) = 0

Radius matching R dicari sebagai parameter solve_bvp.

Tidak menggunakan plt.show().
Semua grafik disimpan ke PNG.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_bvp
from scipy.optimize import brentq


# ============================================================
# KONFIGURASI
# ============================================================

@dataclass(frozen=True)
class ModelParameters:
    M_outer: float = 1.0
    m_inner: float = 0.40
    v_transition: float = 1.0


PARAMS = ModelParameters()

OMEGAS = np.array([
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
])

RMAX_MAIN = 50.0

RMAX_TESTS = [
    20.0,
    30.0,
    40.0,
    50.0,
    70.0,
]

TARGET_OMEGA = 0.70

N_MESH = 800
N_EVAL = 5000

BVP_TOL = 1e-8
MAX_NODES = 50000

OUTPUT_DIR = Path("piecewise_parabolic_output_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA HASIL
# ============================================================

@dataclass
class BenchmarkResult:
    omega: float
    rmax: float

    radius_exact: float
    radius_numeric: float

    center_exact: float
    center_numeric: float

    q_exact: float
    q_numeric: float

    energy_exact: float
    energy_numeric: float

    relative_l2_error: float
    relative_q_error: float
    relative_energy_error: float

    tail_ratio: float
    matching_value_error: float
    matching_derivative_error: float

    bvp_success: bool
    bvp_message: str

    r: np.ndarray
    f_exact: np.ndarray
    df_exact: np.ndarray
    f_numeric: np.ndarray
    df_numeric: np.ndarray


# ============================================================
# VALIDASI
# ============================================================

def validate_parameters(
    omega: float,
    rmax: float,
    params: ModelParameters,
) -> None:
    if params.M_outer <= 0:
        raise ValueError("M_outer harus positif.")

    if params.m_inner < 0:
        raise ValueError("m_inner tidak boleh negatif.")

    if params.v_transition <= 0:
        raise ValueError("v_transition harus positif.")

    if params.m_inner >= params.M_outer:
        raise ValueError(
            "Harus berlaku m_inner < M_outer."
        )

    if not params.m_inner < omega < params.M_outer:
        raise ValueError(
            f"omega={omega:.6f} harus berada pada "
            f"{params.m_inner} < omega < {params.M_outer}."
        )

    if rmax <= 0:
        raise ValueError("RMAX harus positif.")


# ============================================================
# BESARAN DASAR
# ============================================================

def wave_numbers(
    omega: float,
    params: ModelParameters,
) -> tuple[float, float]:

    k = np.sqrt(
        omega**2 - params.m_inner**2
    )

    kappa = np.sqrt(
        params.M_outer**2 - omega**2
    )

    return k, kappa


# ============================================================
# SOLUSI ANALITIK
# ============================================================

def matching_equation(
    radius: float,
    k: float,
    kappa: float,
) -> float:
    """
    Kondisi matching:

        k cot(kR) + kappa = 0
    """
    return (
        k / np.tan(k * radius)
        + kappa
    )


def exact_matching_radius(
    omega: float,
    params: ModelParameters,
) -> float:

    k, kappa = wave_numbers(
        omega,
        params,
    )

    eps = 1e-10

    lower = np.pi / (2.0 * k) + eps
    upper = np.pi / k - eps

    return brentq(
        matching_equation,
        lower,
        upper,
        args=(k, kappa),
        xtol=1e-14,
        rtol=1e-14,
        maxiter=500,
    )


def exact_constants(
    omega: float,
    params: ModelParameters,
) -> tuple[float, float, float]:

    k, _ = wave_numbers(
        omega,
        params,
    )

    radius = exact_matching_radius(
        omega,
        params,
    )

    amplitude = (
        params.v_transition
        * radius
        / np.sin(k * radius)
    )

    center = amplitude * k

    return radius, amplitude, center


def exact_profile(
    r: np.ndarray,
    omega: float,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, float]:

    k, kappa = wave_numbers(
        omega,
        params,
    )

    radius, amplitude, center = exact_constants(
        omega,
        params,
    )

    r = np.asarray(r, dtype=float)

    f = np.zeros_like(r)
    df = np.zeros_like(r)

    inner = r <= radius
    outer = ~inner

    r_inner = r[inner]

    near_zero = r_inner < 1e-6
    normal = ~near_zero

    # Taylor dekat pusat
    f[inner][near_zero] = center * (
        1.0
        - k**2 * r_inner[near_zero]**2 / 6.0
    )

    df[inner][near_zero] = (
        -center
        * k**2
        * r_inner[near_zero]
        / 3.0
    )

    rr = r_inner[normal]

    f_inner = np.zeros_like(r_inner)
    df_inner = np.zeros_like(r_inner)

    f_inner[near_zero] = center * (
        1.0
        - k**2 * r_inner[near_zero]**2 / 6.0
    )

    df_inner[near_zero] = (
        -center
        * k**2
        * r_inner[near_zero]
        / 3.0
    )

    f_inner[normal] = (
        amplitude
        * np.sin(k * rr)
        / rr
    )

    df_inner[normal] = amplitude * (
        k * np.cos(k * rr) / rr
        - np.sin(k * rr) / rr**2
    )

    f[inner] = f_inner
    df[inner] = df_inner

    r_outer = r[outer]

    if r_outer.size > 0:
        f_outer = (
            params.v_transition
            * radius
            * np.exp(
                -kappa * (r_outer - radius)
            )
            / r_outer
        )

        df_outer = f_outer * (
            -kappa - 1.0 / r_outer
        )

        f[outer] = f_outer
        df[outer] = df_outer

    return f, df, radius


# ============================================================
# POTENSIAL
# ============================================================

def potential(
    f: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:

    f_abs = np.abs(f)

    outer_value = (
        params.M_outer**2
        * f_abs**2
    )

    inner_value = (
        params.m_inner**2
        * f_abs**2
        + (
            params.M_outer**2
            - params.m_inner**2
        )
        * params.v_transition**2
    )

    return np.where(
        f_abs < params.v_transition,
        outer_value,
        inner_value,
    )


# ============================================================
# SOLVER DUA DOMAIN
# ============================================================

def solve_two_domain_bvp(
    omega: float,
    rmax: float,
    params: ModelParameters,
):
    """
    Domain numerik x berada pada [0, 1].

    Wilayah dalam:
        r_in = R*x

    Wilayah luar:
        r_out = R + (RMAX-R)*x

    State:
        y[0] = u_in
        y[1] = du_in/dr
        y[2] = u_out
        y[3] = du_out/dr

    Parameter:
        p[0] = matching radius R
    """

    validate_parameters(
        omega,
        rmax,
        params,
    )

    k, kappa = wave_numbers(
        omega,
        params,
    )

    radius_exact, _, _ = exact_constants(
        omega,
        params,
    )

    if radius_exact >= rmax:
        raise ValueError(
            f"RMAX={rmax} terlalu kecil. "
            f"Radius exact={radius_exact:.6f}."
        )

    x = np.linspace(
        0.0,
        1.0,
        N_MESH,
    )

    # Tebakan radius dibuat sedikit berbeda
    # agar solver benar-benar mencarinya.
    radius_guess = 0.93 * radius_exact

    if radius_guess <= 0:
        radius_guess = radius_exact

    if radius_guess >= rmax:
        radius_guess = 0.8 * rmax

    # Tebakan awal memakai bentuk mendekati analitik.
    r_in_guess = radius_guess * x

    r_out_guess = (
        radius_guess
        + (rmax - radius_guess) * x
    )

    f_in_guess, df_in_guess, _ = exact_profile(
        np.maximum(r_in_guess, 1e-12),
        omega,
        params,
    )

    f_out_guess, df_out_guess, _ = exact_profile(
        r_out_guess,
        omega,
        params,
    )

    u_in_guess = (
        r_in_guess * f_in_guess
    )

    du_in_guess = (
        f_in_guess
        + r_in_guess * df_in_guess
    )

    u_out_guess = (
        r_out_guess * f_out_guess
    )

    du_out_guess = (
        f_out_guess
        + r_out_guess * df_out_guess
    )

    y_guess = np.vstack([
        u_in_guess,
        du_in_guess,
        u_out_guess,
        du_out_guess,
    ])

    def ode(
        x_values: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:

        radius = float(p[0])

        if not 0 < radius < rmax:
            radius = np.clip(
                radius,
                1e-6,
                rmax - 1e-6,
            )

        outer_length = rmax - radius

        u_in = y[0]
        du_in_dr = y[1]

        u_out = y[2]
        du_out_dr = y[3]

        # Turunan terhadap x
        du_in_dx = (
            radius * du_in_dr
        )

        ddu_in_dr_dx = (
            radius * (-k**2 * u_in)
        )

        du_out_dx = (
            outer_length * du_out_dr
        )

        ddu_out_dr_dx = (
            outer_length
            * (kappa**2 * u_out)
        )

        return np.vstack([
            du_in_dx,
            ddu_in_dr_dx,
            du_out_dx,
            ddu_out_dr_dx,
        ])

    def boundary_conditions(
        ya: np.ndarray,
        yb: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:

        radius = float(p[0])
        v = params.v_transition

        return np.array([
            # u_in(0)=0
            ya[0],

            # f_in(R)=v  => u_in(R)=vR
            yb[0] - v * radius,

            # f_out(R)=v => u_out(R)=vR
            ya[2] - v * radius,

            # continuity derivative f
            # karena u sama pada R,
            # continuity f' ekuivalen u'_in=u'_out
            yb[1] - ya[3],

            # asymptotic outer Robin
            # u_out' + kappa*u_out = 0
            yb[3] + kappa * yb[2],
        ])

    result = solve_bvp(
        ode,
        boundary_conditions,
        x,
        y_guess,
        p=np.array([radius_guess]),
        tol=BVP_TOL,
        max_nodes=MAX_NODES,
        verbose=0,
    )

    return result


# ============================================================
# REKONSTRUKSI PROFIL NUMERIK
# ============================================================

def reconstruct_numeric_profile(
    result,
    rmax: float,
    n_eval: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:

    if not result.success:
        raise RuntimeError(
            f"BVP gagal: {result.message}"
        )

    radius = float(result.p[0])

    x_eval = np.linspace(
        0.0,
        1.0,
        n_eval,
    )

    values = result.sol(x_eval)

    u_in = values[0]
    du_in = values[1]

    u_out = values[2]
    du_out = values[3]

    r_in = radius * x_eval

    r_out = (
        radius
        + (rmax - radius) * x_eval
    )

    # f_in = u/r
    f_in = np.empty_like(r_in)
    df_in = np.empty_like(r_in)

    # Nilai pusat:
    # u ≈ u'(0) r
    f_in[0] = du_in[0]

    # Dari persamaan regular:
    # f'(0)=0
    df_in[0] = 0.0

    f_in[1:] = (
        u_in[1:] / r_in[1:]
    )

    df_in[1:] = (
        du_in[1:] / r_in[1:]
        - u_in[1:] / r_in[1:]**2
    )

    f_out = (
        u_out / r_out
    )

    df_out = (
        du_out / r_out
        - u_out / r_out**2
    )

    # Hindari menggandakan titik matching.
    r = np.concatenate([
        r_in,
        r_out[1:],
    ])

    f = np.concatenate([
        f_in,
        f_out[1:],
    ])

    df = np.concatenate([
        df_in,
        df_out[1:],
    ])

    return r, f, df, radius


# ============================================================
# Q, ENERGI, GALAT
# ============================================================

def calculate_charge(
    r: np.ndarray,
    f: np.ndarray,
    omega: float,
) -> float:

    return float(
        8.0
        * np.pi
        * omega
        * simpson(
            f**2 * r**2,
            x=r,
        )
    )


def calculate_energy(
    r: np.ndarray,
    f: np.ndarray,
    df: np.ndarray,
    omega: float,
    params: ModelParameters,
) -> float:

    density = (
        omega**2 * f**2
        + df**2
        + potential(f, params)
    )

    return float(
        4.0
        * np.pi
        * simpson(
            density * r**2,
            x=r,
        )
    )


def interpolate_to_grid(
    source_r: np.ndarray,
    source_y: np.ndarray,
    target_r: np.ndarray,
) -> np.ndarray:

    return np.interp(
        target_r,
        source_r,
        source_y,
    )


def run_benchmark(
    omega: float,
    rmax: float,
    params: ModelParameters,
) -> BenchmarkResult:

    solver_result = solve_two_domain_bvp(
        omega,
        rmax,
        params,
    )

    if not solver_result.success:
        return BenchmarkResult(
            omega=omega,
            rmax=rmax,
            radius_exact=np.nan,
            radius_numeric=np.nan,
            center_exact=np.nan,
            center_numeric=np.nan,
            q_exact=np.nan,
            q_numeric=np.nan,
            energy_exact=np.nan,
            energy_numeric=np.nan,
            relative_l2_error=np.nan,
            relative_q_error=np.nan,
            relative_energy_error=np.nan,
            tail_ratio=np.nan,
            matching_value_error=np.nan,
            matching_derivative_error=np.nan,
            bvp_success=False,
            bvp_message=str(
                solver_result.message
            ),
            r=np.array([]),
            f_exact=np.array([]),
            df_exact=np.array([]),
            f_numeric=np.array([]),
            df_numeric=np.array([]),
        )

    r_numeric, f_numeric, df_numeric, radius_numeric = (
        reconstruct_numeric_profile(
            solver_result,
            rmax,
            N_EVAL,
        )
    )

    f_exact, df_exact, radius_exact = exact_profile(
        r_numeric,
        omega,
        params,
    )

    _, _, center_exact = exact_constants(
        omega,
        params,
    )

    center_numeric = float(
        f_numeric[0]
    )

    q_exact = calculate_charge(
        r_numeric,
        f_exact,
        omega,
    )

    q_numeric = calculate_charge(
        r_numeric,
        f_numeric,
        omega,
    )

    energy_exact = calculate_energy(
        r_numeric,
        f_exact,
        df_exact,
        omega,
        params,
    )

    energy_numeric = calculate_energy(
        r_numeric,
        f_numeric,
        df_numeric,
        omega,
        params,
    )

    numerator = np.sqrt(
        simpson(
            (f_numeric - f_exact)**2,
            x=r_numeric,
        )
    )

    denominator = np.sqrt(
        simpson(
            f_exact**2,
            x=r_numeric,
        )
    )

    relative_l2_error = (
        numerator
        / (denominator + 1e-30)
    )

    relative_q_error = (
        abs(q_numeric - q_exact)
        / (abs(q_exact) + 1e-30)
    )

    relative_energy_error = (
        abs(
            energy_numeric - energy_exact
        )
        / (abs(energy_exact) + 1e-30)
    )

    tail_ratio = (
        abs(f_numeric[-1])
        / (
            np.max(np.abs(f_numeric))
            + 1e-30
        )
    )

    # Cek matching dari kiri dan kanan.
    left_index = np.searchsorted(
        r_numeric,
        radius_numeric,
        side="right",
    ) - 1

    right_index = min(
        left_index + 1,
        len(r_numeric) - 1,
    )

    matching_value_error = abs(
        f_numeric[left_index]
        - f_numeric[right_index]
    )

    matching_derivative_error = abs(
        df_numeric[left_index]
        - df_numeric[right_index]
    )

    return BenchmarkResult(
        omega=omega,
        rmax=rmax,
        radius_exact=radius_exact,
        radius_numeric=radius_numeric,
        center_exact=center_exact,
        center_numeric=center_numeric,
        q_exact=q_exact,
        q_numeric=q_numeric,
        energy_exact=energy_exact,
        energy_numeric=energy_numeric,
        relative_l2_error=relative_l2_error,
        relative_q_error=relative_q_error,
        relative_energy_error=relative_energy_error,
        tail_ratio=tail_ratio,
        matching_value_error=matching_value_error,
        matching_derivative_error=matching_derivative_error,
        bvp_success=True,
        bvp_message=str(
            solver_result.message
        ),
        r=r_numeric,
        f_exact=f_exact,
        df_exact=df_exact,
        f_numeric=f_numeric,
        df_numeric=df_numeric,
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    results: list[BenchmarkResult],
    filepath: Path,
) -> None:

    headers = [
        "omega",
        "RMAX",
        "success",
        "message",
        "R_exact",
        "R_numeric",
        "center_exact",
        "center_numeric",
        "Q_exact",
        "Q_numeric",
        "E_exact",
        "E_numeric",
        "L2_relative_error",
        "Q_relative_error",
        "E_relative_error",
        "tail_ratio",
        "matching_value_error",
        "matching_derivative_error",
        "E_over_Q_numeric",
    ]

    with filepath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)
        writer.writerow(headers)

        for result in results:
            if (
                np.isfinite(result.q_numeric)
                and result.q_numeric != 0
            ):
                e_over_q = (
                    result.energy_numeric
                    / result.q_numeric
                )
            else:
                e_over_q = np.nan

            writer.writerow([
                result.omega,
                result.rmax,
                result.bvp_success,
                result.bvp_message,
                result.radius_exact,
                result.radius_numeric,
                result.center_exact,
                result.center_numeric,
                result.q_exact,
                result.q_numeric,
                result.energy_exact,
                result.energy_numeric,
                result.relative_l2_error,
                result.relative_q_error,
                result.relative_energy_error,
                result.tail_ratio,
                result.matching_value_error,
                result.matching_derivative_error,
                e_over_q,
            ])


# ============================================================
# PLOT
# ============================================================

def plot_profiles(
    results: list[BenchmarkResult],
) -> None:

    plt.figure(figsize=(10, 6))

    for result in results:
        if not result.bvp_success:
            continue

        plt.plot(
            result.r,
            result.f_numeric,
            linewidth=1.8,
            label=(
                f"numerik omega={result.omega:.2f}"
            ),
        )

        plt.plot(
            result.r,
            result.f_exact,
            linestyle="--",
            linewidth=1.2,
            label=(
                f"analitik omega={result.omega:.2f}"
            ),
        )

    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.title(
        "Piecewise-parabolic dua-domain: "
        "analitik vs numerik"
    )
    plt.grid(True)
    plt.legend(
        fontsize=8,
        ncol=2,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "01_profiles.png",
        dpi=300,
    )
    plt.close()


def plot_errors(
    results: list[BenchmarkResult],
) -> None:

    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    omega = np.array([
        result.omega
        for result in valid
    ])

    l2 = np.array([
        result.relative_l2_error
        for result in valid
    ])

    q_error = np.array([
        result.relative_q_error
        for result in valid
    ])

    e_error = np.array([
        result.relative_energy_error
        for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.semilogy(
        omega,
        l2,
        marker="o",
        label="galat L2 profil",
    )

    plt.semilogy(
        omega,
        q_error,
        marker="s",
        label="galat Q",
    )

    plt.semilogy(
        omega,
        e_error,
        marker="^",
        label="galat E",
    )

    plt.xlabel("omega")
    plt.ylabel("Galat relatif")
    plt.title(
        "Akurasi solver dua-domain"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "02_errors.png",
        dpi=300,
    )
    plt.close()


def plot_q_energy(
    results: list[BenchmarkResult],
) -> None:

    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    omega = np.array([
        result.omega
        for result in valid
    ])

    q_exact = np.array([
        result.q_exact
        for result in valid
    ])

    q_numeric = np.array([
        result.q_numeric
        for result in valid
    ])

    e_exact = np.array([
        result.energy_exact
        for result in valid
    ])

    e_numeric = np.array([
        result.energy_numeric
        for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.plot(
        omega,
        q_exact,
        linestyle="--",
        marker="o",
        label="Q analitik",
    )

    plt.plot(
        omega,
        q_numeric,
        marker="s",
        label="Q numerik",
    )

    plt.xlabel("omega")
    plt.ylabel("Q")
    plt.title("Muatan Noether")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "03_Q.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        omega,
        e_exact,
        linestyle="--",
        marker="o",
        label="E analitik",
    )

    plt.plot(
        omega,
        e_numeric,
        marker="s",
        label="E numerik",
    )

    plt.xlabel("omega")
    plt.ylabel("E")
    plt.title("Energi")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "04_E.png",
        dpi=300,
    )
    plt.close()


def plot_radius(
    results: list[BenchmarkResult],
) -> None:

    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    omega = np.array([
        result.omega
        for result in valid
    ])

    exact = np.array([
        result.radius_exact
        for result in valid
    ])

    numeric = np.array([
        result.radius_numeric
        for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.plot(
        omega,
        exact,
        linestyle="--",
        marker="o",
        label="R analitik",
    )

    plt.plot(
        omega,
        numeric,
        marker="s",
        label="R numerik",
    )

    plt.xlabel("omega")
    plt.ylabel("Matching radius R")
    plt.title(
        "Radius matching analitik vs numerik"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "05_matching_radius.png",
        dpi=300,
    )
    plt.close()


def plot_log_tail(
    result: BenchmarkResult,
) -> None:

    if not result.bvp_success:
        return

    exact_abs = (
        np.abs(result.f_exact) + 1e-30
    )

    numeric_abs = (
        np.abs(result.f_numeric) + 1e-30
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        result.r,
        np.log(exact_abs),
        linestyle="--",
        label="analitik",
    )

    plt.plot(
        result.r,
        np.log(numeric_abs),
        label="numerik",
    )

    plt.axvline(
        result.radius_numeric,
        linestyle=":",
        label="R numerik",
    )

    plt.xlabel("r")
    plt.ylabel("log |f(r)|")
    plt.title(
        f"Log-tail omega={result.omega:.2f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "06_log_tail.png",
        dpi=300,
    )
    plt.close()


def plot_rmax_tests(
    results: list[BenchmarkResult],
) -> None:

    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    rmax = np.array([
        result.rmax
        for result in valid
    ])

    q = np.array([
        result.q_numeric
        for result in valid
    ])

    e = np.array([
        result.energy_numeric
        for result in valid
    ])

    l2 = np.array([
        result.relative_l2_error
        for result in valid
    ])

    tail = np.array([
        result.tail_ratio
        for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.plot(
        rmax,
        q,
        marker="o",
    )

    plt.xlabel("RMAX")
    plt.ylabel("Q")
    plt.title(
        f"Konvergensi Q, omega={TARGET_OMEGA:.2f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "07_Q_RMAX.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        rmax,
        e,
        marker="o",
    )

    plt.xlabel("RMAX")
    plt.ylabel("E")
    plt.title(
        f"Konvergensi E, omega={TARGET_OMEGA:.2f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "08_E_RMAX.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.semilogy(
        rmax,
        l2,
        marker="o",
        label="galat L2",
    )

    plt.semilogy(
        rmax,
        tail,
        marker="s",
        label="tail ratio",
    )

    plt.xlabel("RMAX")
    plt.ylabel("Diagnostik")
    plt.title(
        f"Galat dan tail, omega={TARGET_OMEGA:.2f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "09_error_tail_RMAX.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 72)
    print("PIECEWISE-PARABOLIC BENCHMARK V2")
    print("FORMULASI DUA DOMAIN")
    print("=" * 72)

    print(
        f"M_outer      = {PARAMS.M_outer}"
    )
    print(
        f"m_inner      = {PARAMS.m_inner}"
    )
    print(
        f"v_transition = {PARAMS.v_transition}"
    )
    print(
        f"RMAX utama   = {RMAX_MAIN}"
    )
    print(
        f"Output       = {OUTPUT_DIR.resolve()}"
    )

    omega_results: list[
        BenchmarkResult
    ] = []

    print("\nScan omega:")

    for omega in OMEGAS:

        print(
            f"omega={omega:.3f} ... ",
            end="",
            flush=True,
        )

        try:
            result = run_benchmark(
                omega,
                RMAX_MAIN,
                PARAMS,
            )
        except Exception as error:
            print(f"GAGAL: {error}")
            continue

        omega_results.append(result)

        if result.bvp_success:
            print(
                "berhasil | "
                f"R_num={result.radius_numeric:.8f} | "
                f"R_exact={result.radius_exact:.8f} | "
                f"L2={result.relative_l2_error:.3e} | "
                f"dQ={result.relative_q_error:.3e} | "
                f"dE={result.relative_energy_error:.3e} | "
                f"tail={result.tail_ratio:.3e}"
            )
        else:
            print(
                f"solver gagal: "
                f"{result.bvp_message}"
            )

    if not omega_results:
        print(
            "Tidak ada hasil omega."
        )
        return 1

    save_csv(
        omega_results,
        OUTPUT_DIR
        / "benchmark_omega_summary.csv",
    )

    plot_profiles(
        omega_results,
    )

    plot_errors(
        omega_results,
    )

    plot_q_energy(
        omega_results,
    )

    plot_radius(
        omega_results,
    )

    target_result = next(
        (
            result
            for result in omega_results
            if abs(
                result.omega - TARGET_OMEGA
            ) < 1e-12
        ),
        None,
    )

    if target_result is not None:
        plot_log_tail(
            target_result,
        )

    print("\nUji RMAX:")

    rmax_results: list[
        BenchmarkResult
    ] = []

    for rmax in RMAX_TESTS:

        print(
            f"RMAX={rmax:.1f} ... ",
            end="",
            flush=True,
        )

        try:
            result = run_benchmark(
                TARGET_OMEGA,
                rmax,
                PARAMS,
            )
        except Exception as error:
            print(f"GAGAL: {error}")
            continue

        rmax_results.append(result)

        if result.bvp_success:
            print(
                "berhasil | "
                f"Q={result.q_numeric:.8f} | "
                f"E={result.energy_numeric:.8f} | "
                f"L2={result.relative_l2_error:.3e} | "
                f"tail={result.tail_ratio:.3e}"
            )
        else:
            print(
                f"solver gagal: "
                f"{result.bvp_message}"
            )

    if rmax_results:
        save_csv(
            rmax_results,
            OUTPUT_DIR
            / "benchmark_RMAX_summary.csv",
        )

        plot_rmax_tests(
            rmax_results,
        )

    print("\n" + "=" * 72)
    print("SELESAI")
    print("=" * 72)

    for filepath in sorted(
        OUTPUT_DIR.glob("*")
    ):
        print(f"- {filepath}")

    print("\nKriteria benchmark lulus:")
    print("L2 relatif       < 1e-3")
    print("Galat Q relatif  < 1e-3")
    print("Galat E relatif  < 1e-3")
    print("Tail ratio       < 1e-6")
    print("Q dan E konvergen terhadap RMAX")
    print("Radius numerik mendekati radius analitik")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            "\nProgram dihentikan pengguna."
        )
        sys.exit(130)

    except Exception as error:
        print(
            f"\nKesalahan fatal: {error}"
        )
        sys.exit(1)
