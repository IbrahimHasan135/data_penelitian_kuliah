"""
Benchmark numerik Q-ball dengan potensial piecewise-parabolic.

Tujuan:
1. Menghasilkan solusi analitik piecewise.
2. Menyelesaikan persamaan yang sama menggunakan solve_bvp.
3. Membandingkan solusi numerik dengan solusi analitik.
4. Menghitung muatan Noether Q dan energi E.
5. Menguji konvergensi terhadap RMAX.
6. Mengekspor seluruh grafik ke PNG dan hasil ke CSV.

Tidak menggunakan plt.show().
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Backend noninteraktif untuk Linux/venv/server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_bvp
from scipy.optimize import brentq


# ============================================================
# KONFIGURASI MODEL
# ============================================================

@dataclass(frozen=True)
class ModelParameters:
    # Massa/kelengkungan potensial di wilayah luar.
    M_outer: float = 1.0

    # Massa/kelengkungan potensial di wilayah inti.
    m_inner: float = 0.40

    # Nilai medan pada batas pergantian potensial.
    v_transition: float = 1.0


PARAMS = ModelParameters()

# Frekuensi harus memenuhi:
# m_inner < omega < M_outer
OMEGAS = np.array([0.50, 0.60, 0.70, 0.80, 0.90])

# Domain utama untuk scan omega.
RMAX_MAIN = 50.0

# Pengujian sensitivitas domain.
RMAX_TESTS = [20.0, 30.0, 40.0, 50.0, 70.0]

# Frekuensi contoh untuk uji RMAX dan log-tail.
TARGET_OMEGA = 0.70

# Grid awal BVP.
N_GRID = 1500

# Grid evaluasi integral/plot.
N_EVAL = 6000

# Hindari singularitas 2/r tepat di r=0.
EPS = 1.0e-6

# Toleransi solver.
BVP_TOL = 1.0e-7
MAX_NODES = 50000

# Direktori output.
OUTPUT_DIR = Path("piecewise_parabolic_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STRUKTUR HASIL
# ============================================================

@dataclass
class BenchmarkResult:
    omega: float
    rmax: float
    matching_radius_exact: float
    matching_radius_numeric: float
    center_exact: float
    center_numeric: float
    q_exact: float
    q_numeric: float
    energy_exact: float
    energy_numeric: float
    relative_l2_error: float
    max_absolute_error: float
    relative_q_error: float
    relative_energy_error: float
    tail_ratio: float
    bvp_success: bool
    bvp_message: str
    r: np.ndarray
    f_exact: np.ndarray
    df_exact: np.ndarray
    f_numeric: np.ndarray
    df_numeric: np.ndarray


# ============================================================
# VALIDASI PARAMETER
# ============================================================

def validate_parameters(omega: float, params: ModelParameters) -> None:
    if params.M_outer <= 0:
        raise ValueError("M_outer harus positif.")

    if params.m_inner < 0:
        raise ValueError("m_inner tidak boleh negatif.")

    if params.v_transition <= 0:
        raise ValueError("v_transition harus positif.")

    if params.m_inner >= params.M_outer:
        raise ValueError("Harus berlaku m_inner < M_outer.")

    if not params.m_inner < omega < params.M_outer:
        raise ValueError(
            f"omega={omega:.6f} tidak berada dalam jendela "
            f"({params.m_inner}, {params.M_outer})."
        )


# ============================================================
# BESARAN DASAR
# ============================================================

def radial_wave_numbers(
    omega: float,
    params: ModelParameters,
) -> tuple[float, float]:
    """
    kappa: laju peluruhan eksponensial di luar.
    k: bilangan gelombang osilatori di dalam.
    """
    validate_parameters(omega, params)

    k = np.sqrt(omega**2 - params.m_inner**2)
    kappa = np.sqrt(params.M_outer**2 - omega**2)

    return k, kappa


# ============================================================
# RADIUS PENCOCOKAN ANALITIK
# ============================================================

def matching_equation(
    radius: float,
    k: float,
    kappa: float,
) -> float:
    """
    Syarat kontinuitas turunan:

        k cot(kR) + kappa = 0
    """
    return k / np.tan(k * radius) + kappa


def find_ground_state_matching_radius(
    omega: float,
    params: ModelParameters,
) -> float:
    """
    Ground state dipilih dari akar pertama pada interval:

        pi/(2k) < R < pi/k

    Pada interval ini profil inti positif dan belum memiliki node.
    """
    k, kappa = radial_wave_numbers(omega, params)

    small_offset = 1.0e-8

    lower = np.pi / (2.0 * k) + small_offset
    upper = np.pi / k - small_offset

    f_lower = matching_equation(lower, k, kappa)
    f_upper = matching_equation(upper, k, kappa)

    if not np.isfinite(f_lower) or not np.isfinite(f_upper):
        raise RuntimeError("Nilai fungsi matching tidak finite.")

    if f_lower * f_upper >= 0:
        raise RuntimeError(
            "Akar matching ground-state tidak ter-bracket. "
            "Periksa parameter model."
        )

    return brentq(
        matching_equation,
        lower,
        upper,
        args=(k, kappa),
        xtol=1.0e-13,
        rtol=1.0e-13,
        maxiter=500,
    )


# ============================================================
# SOLUSI ANALITIK
# ============================================================

def analytic_constants(
    omega: float,
    params: ModelParameters,
) -> tuple[float, float, float]:
    """
    Mengembalikan:
    - R: radius pencocokan,
    - A: konstanta solusi inti A sin(kr)/r,
    - f0: amplitudo pusat.
    """
    k, _ = radial_wave_numbers(omega, params)
    radius = find_ground_state_matching_radius(omega, params)

    sine_value = np.sin(k * radius)

    if abs(sine_value) < 1.0e-14:
        raise RuntimeError("sin(kR) terlalu dekat nol.")

    amplitude_constant = (
        params.v_transition * radius / sine_value
    )

    center_amplitude = amplitude_constant * k

    return radius, amplitude_constant, center_amplitude


def analytic_profile(
    r: np.ndarray,
    omega: float,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Profil dan turunannya:

    inti:
        f = A sin(kr)/r

    luar:
        f = v R exp[-kappa(r-R)]/r
    """
    k, kappa = radial_wave_numbers(omega, params)
    radius, amplitude_constant, center_amplitude = analytic_constants(
        omega,
        params,
    )

    r = np.asarray(r, dtype=float)

    f = np.empty_like(r)
    df = np.empty_like(r)

    inner_mask = r <= radius
    outer_mask = ~inner_mask

    r_inner = r[inner_mask]

    # Formula regular dekat pusat.
    near_zero = r_inner < 1.0e-5
    regular_inner = ~near_zero

    f_inner = np.empty_like(r_inner)
    df_inner = np.empty_like(r_inner)

    # Ekspansi Taylor:
    # sin(kr)/r ≈ k - k^3 r^2/6
    f_inner[near_zero] = center_amplitude * (
        1.0 - (k**2 * r_inner[near_zero] ** 2) / 6.0
    )

    df_inner[near_zero] = (
        -center_amplitude
        * k**2
        * r_inner[near_zero]
        / 3.0
    )

    rr = r_inner[regular_inner]

    f_inner[regular_inner] = (
        amplitude_constant * np.sin(k * rr) / rr
    )

    df_inner[regular_inner] = amplitude_constant * (
        k * np.cos(k * rr) / rr
        - np.sin(k * rr) / rr**2
    )

    f[inner_mask] = f_inner
    df[inner_mask] = df_inner

    r_outer = r[outer_mask]

    if r_outer.size > 0:
        f_outer = (
            params.v_transition
            * radius
            * np.exp(-kappa * (r_outer - radius))
            / r_outer
        )

        df_outer = f_outer * (
            -kappa - 1.0 / r_outer
        )

        f[outer_mask] = f_outer
        df[outer_mask] = df_outer

    return f, df, radius


# ============================================================
# POTENSIAL PIECEWISE-PARABOLIC
# ============================================================

def potential(
    f: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    U(f) kontinu pada |f| = v.

    luar:
        U = M^2 f^2

    inti:
        U = m_in^2 f^2
            + (M^2 - m_in^2) v^2
    """
    f_abs = np.abs(f)

    outer = params.M_outer**2 * f_abs**2

    inner = (
        params.m_inner**2 * f_abs**2
        + (
            params.M_outer**2
            - params.m_inner**2
        )
        * params.v_transition**2
    )

    return np.where(
        f_abs < params.v_transition,
        outer,
        inner,
    )


def potential_curvature(
    f: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    U_s atau massa efektif dalam persamaan radial.

    Karena U = mass^2 |Psi|^2 + konstanta,
    turunan terhadap |Psi|^2 hanyalah mass^2.
    """
    return np.where(
        np.abs(f) < params.v_transition,
        params.M_outer**2,
        params.m_inner**2,
    )


# ============================================================
# SOLVER NUMERIK BVP
# ============================================================

def solve_numeric_bvp(
    omega: float,
    rmax: float,
    params: ModelParameters,
) -> solve_bvp:
    validate_parameters(omega, params)

    _, kappa = radial_wave_numbers(omega, params)

    r_grid = np.linspace(EPS, rmax, N_GRID)

    f_guess, df_guess, _ = analytic_profile(
        r_grid,
        omega,
        params,
    )

    # Sedikit perturbasi agar solver benar-benar melakukan koreksi.
    perturbation = 1.0 + 0.01 * np.exp(-r_grid / 4.0)
    f_guess = f_guess * perturbation

    y_guess = np.vstack((f_guess, df_guess))

    def ode(r: np.ndarray, y: np.ndarray) -> np.ndarray:
        f = y[0]
        df = y[1]

        mass_sq = potential_curvature(f, params)

        safe_r = np.maximum(r, EPS)

        d2f = (
            -(2.0 / safe_r) * df
            + (mass_sq - omega**2) * f
        )

        return np.vstack((df, d2f))

    def boundary_conditions(
        ya: np.ndarray,
        yb: np.ndarray,
    ) -> np.ndarray:
        # Simetri pusat:
        center_condition = ya[1]

        # Robin asymptotic:
        # f' + (kappa + 1/R) f = 0
        tail_condition = (
            yb[1]
            + (kappa + 1.0 / rmax) * yb[0]
        )

        return np.array([
            center_condition,
            tail_condition,
        ])

    result = solve_bvp(
        ode,
        boundary_conditions,
        r_grid,
        y_guess,
        tol=BVP_TOL,
        max_nodes=MAX_NODES,
        verbose=0,
    )

    return result


# ============================================================
# Q, ENERGI, DAN DIAGNOSTIK
# ============================================================

def calculate_charge(
    r: np.ndarray,
    f: np.ndarray,
    omega: float,
) -> float:
    """
    Muatan Noether:

        Q = 8 pi omega ∫ f^2 r^2 dr
    """
    integrand = f**2 * r**2

    return float(
        8.0
        * np.pi
        * omega
        * simpson(integrand, x=r)
    )


def calculate_energy(
    r: np.ndarray,
    f: np.ndarray,
    df: np.ndarray,
    omega: float,
    params: ModelParameters,
) -> float:
    """
    Energi:

        E = 4 pi ∫ [omega^2 f^2 + f'^2 + U(f)] r^2 dr
    """
    energy_density = (
        omega**2 * f**2
        + df**2
        + potential(f, params)
    )

    return float(
        4.0
        * np.pi
        * simpson(
            energy_density * r**2,
            x=r,
        )
    )


def find_numeric_matching_radius(
    r: np.ndarray,
    f: np.ndarray,
    transition_value: float,
) -> float:
    """
    Mencari crossing pertama dari atas ke bawah:
        f(r) = v
    """
    difference = f - transition_value

    indices = np.where(
        (difference[:-1] >= 0.0)
        & (difference[1:] < 0.0)
    )[0]

    if indices.size == 0:
        return np.nan

    i = int(indices[0])

    x0, x1 = r[i], r[i + 1]
    y0, y1 = difference[i], difference[i + 1]

    if abs(y1 - y0) < 1.0e-15:
        return float(x0)

    return float(
        x0 - y0 * (x1 - x0) / (y1 - y0)
    )


def run_single_benchmark(
    omega: float,
    rmax: float,
    params: ModelParameters,
) -> BenchmarkResult:
    bvp = solve_numeric_bvp(
        omega,
        rmax,
        params,
    )

    r_eval = np.linspace(EPS, rmax, N_EVAL)

    f_exact, df_exact, radius_exact = analytic_profile(
        r_eval,
        omega,
        params,
    )

    if bvp.success:
        numerical = bvp.sol(r_eval)
        f_numeric = numerical[0]
        df_numeric = numerical[1]
    else:
        f_numeric = np.full_like(r_eval, np.nan)
        df_numeric = np.full_like(r_eval, np.nan)

    q_exact = calculate_charge(
        r_eval,
        f_exact,
        omega,
    )

    energy_exact = calculate_energy(
        r_eval,
        f_exact,
        df_exact,
        omega,
        params,
    )

    if bvp.success:
        q_numeric = calculate_charge(
            r_eval,
            f_numeric,
            omega,
        )

        energy_numeric = calculate_energy(
            r_eval,
            f_numeric,
            df_numeric,
            omega,
            params,
        )

        error = f_numeric - f_exact

        norm_exact = np.sqrt(
            simpson(f_exact**2, x=r_eval)
        )

        norm_error = np.sqrt(
            simpson(error**2, x=r_eval)
        )

        relative_l2_error = (
            norm_error / (norm_exact + 1.0e-30)
        )

        max_absolute_error = float(
            np.max(np.abs(error))
        )

        relative_q_error = abs(
            q_numeric - q_exact
        ) / (abs(q_exact) + 1.0e-30)

        relative_energy_error = abs(
            energy_numeric - energy_exact
        ) / (abs(energy_exact) + 1.0e-30)

        tail_ratio = abs(f_numeric[-1]) / (
            np.max(np.abs(f_numeric)) + 1.0e-30
        )

        radius_numeric = find_numeric_matching_radius(
            r_eval,
            f_numeric,
            params.v_transition,
        )

        center_numeric = float(f_numeric[0])

    else:
        q_numeric = np.nan
        energy_numeric = np.nan
        relative_l2_error = np.nan
        max_absolute_error = np.nan
        relative_q_error = np.nan
        relative_energy_error = np.nan
        tail_ratio = np.nan
        radius_numeric = np.nan
        center_numeric = np.nan

    _, _, center_exact = analytic_constants(
        omega,
        params,
    )

    return BenchmarkResult(
        omega=omega,
        rmax=rmax,
        matching_radius_exact=radius_exact,
        matching_radius_numeric=radius_numeric,
        center_exact=center_exact,
        center_numeric=center_numeric,
        q_exact=q_exact,
        q_numeric=q_numeric,
        energy_exact=energy_exact,
        energy_numeric=energy_numeric,
        relative_l2_error=relative_l2_error,
        max_absolute_error=max_absolute_error,
        relative_q_error=relative_q_error,
        relative_energy_error=relative_energy_error,
        tail_ratio=tail_ratio,
        bvp_success=bool(bvp.success),
        bvp_message=str(bvp.message),
        r=r_eval,
        f_exact=f_exact,
        df_exact=df_exact,
        f_numeric=f_numeric,
        df_numeric=df_numeric,
    )


# ============================================================
# SIMPAN CSV
# ============================================================

def save_summary_csv(
    results: list[BenchmarkResult],
    filepath: Path,
) -> None:
    headers = [
        "omega",
        "RMAX",
        "bvp_success",
        "bvp_message",
        "R_match_exact",
        "R_match_numeric",
        "center_exact",
        "center_numeric",
        "Q_exact",
        "Q_numeric",
        "E_exact",
        "E_numeric",
        "relative_L2_error",
        "max_absolute_error",
        "relative_Q_error",
        "relative_E_error",
        "tail_ratio",
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
            e_over_q = (
                result.energy_numeric
                / result.q_numeric
                if (
                    np.isfinite(result.energy_numeric)
                    and np.isfinite(result.q_numeric)
                    and result.q_numeric != 0
                )
                else np.nan
            )

            writer.writerow([
                result.omega,
                result.rmax,
                result.bvp_success,
                result.bvp_message,
                result.matching_radius_exact,
                result.matching_radius_numeric,
                result.center_exact,
                result.center_numeric,
                result.q_exact,
                result.q_numeric,
                result.energy_exact,
                result.energy_numeric,
                result.relative_l2_error,
                result.max_absolute_error,
                result.relative_q_error,
                result.relative_energy_error,
                result.tail_ratio,
                e_over_q,
            ])


# ============================================================
# PLOT
# ============================================================

def plot_profiles(
    results: list[BenchmarkResult],
    filepath: Path,
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
                f"numerik, omega={result.omega:.2f}"
            ),
        )

        plt.plot(
            result.r,
            result.f_exact,
            linestyle="--",
            linewidth=1.2,
            label=(
                f"analitik, omega={result.omega:.2f}"
            ),
        )

    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.title(
        "Piecewise-parabolic: profil analitik vs numerik"
    )
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_errors(
    results: list[BenchmarkResult],
    filepath: Path,
) -> None:
    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    omegas = np.array([
        result.omega for result in valid
    ])

    l2_errors = np.array([
        result.relative_l2_error
        for result in valid
    ])

    q_errors = np.array([
        result.relative_q_error
        for result in valid
    ])

    energy_errors = np.array([
        result.relative_energy_error
        for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.semilogy(
        omegas,
        l2_errors,
        marker="o",
        label="galat relatif L2 profil",
    )

    plt.semilogy(
        omegas,
        q_errors,
        marker="s",
        label="galat relatif Q",
    )

    plt.semilogy(
        omegas,
        energy_errors,
        marker="^",
        label="galat relatif E",
    )

    plt.xlabel("omega")
    plt.ylabel("Galat relatif")
    plt.title("Akurasi solver terhadap solusi analitik")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_charge_energy(
    results: list[BenchmarkResult],
) -> None:
    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    omegas = np.array([
        result.omega for result in valid
    ])

    q_exact = np.array([
        result.q_exact for result in valid
    ])

    q_numeric = np.array([
        result.q_numeric for result in valid
    ])

    energy_exact = np.array([
        result.energy_exact for result in valid
    ])

    energy_numeric = np.array([
        result.energy_numeric for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.plot(
        omegas,
        q_exact,
        linestyle="--",
        marker="o",
        label="Q analitik",
    )

    plt.plot(
        omegas,
        q_numeric,
        marker="s",
        label="Q numerik",
    )

    plt.xlabel("omega")
    plt.ylabel("Q")
    plt.title("Muatan Noether Q(omega)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "03_Q_vs_omega.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        omegas,
        energy_exact,
        linestyle="--",
        marker="o",
        label="E analitik",
    )

    plt.plot(
        omegas,
        energy_numeric,
        marker="s",
        label="E numerik",
    )

    plt.xlabel("omega")
    plt.ylabel("E")
    plt.title("Energi E(omega)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "04_E_vs_omega.png",
        dpi=300,
    )
    plt.close()


def plot_log_tail(
    result: BenchmarkResult,
    filepath: Path,
) -> None:
    if not result.bvp_success:
        return

    absolute_exact = (
        np.abs(result.f_exact) + 1.0e-30
    )

    absolute_numeric = (
        np.abs(result.f_numeric) + 1.0e-30
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        result.r,
        np.log(absolute_exact),
        linestyle="--",
        label="log |f| analitik",
    )

    plt.plot(
        result.r,
        np.log(absolute_numeric),
        label="log |f| numerik",
    )

    plt.axvline(
        result.matching_radius_exact,
        linestyle=":",
        label="radius matching",
    )

    plt.xlabel("r")
    plt.ylabel("log |f(r)|")
    plt.title(
        f"Log-tail benchmark, omega={result.omega:.2f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_rmax_convergence(
    results: list[BenchmarkResult],
) -> None:
    valid = [
        result
        for result in results
        if result.bvp_success
    ]

    rmax_values = np.array([
        result.rmax for result in valid
    ])

    q_values = np.array([
        result.q_numeric for result in valid
    ])

    energy_values = np.array([
        result.energy_numeric
        for result in valid
    ])

    l2_values = np.array([
        result.relative_l2_error
        for result in valid
    ])

    tail_values = np.array([
        result.tail_ratio
        for result in valid
    ])

    plt.figure(figsize=(9, 5))

    plt.plot(
        rmax_values,
        q_values,
        marker="o",
        label="Q numerik",
    )

    plt.xlabel("RMAX")
    plt.ylabel("Q")
    plt.title(
        f"Konvergensi Q terhadap RMAX, "
        f"omega={TARGET_OMEGA:.2f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "06_Q_RMAX_convergence.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        rmax_values,
        energy_values,
        marker="o",
        label="E numerik",
    )

    plt.xlabel("RMAX")
    plt.ylabel("E")
    plt.title(
        f"Konvergensi energi terhadap RMAX, "
        f"omega={TARGET_OMEGA:.2f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "07_E_RMAX_convergence.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.semilogy(
        rmax_values,
        l2_values,
        marker="o",
        label="galat L2",
    )

    plt.semilogy(
        rmax_values,
        tail_values,
        marker="s",
        label="tail ratio",
    )

    plt.xlabel("RMAX")
    plt.ylabel("Nilai diagnostik")
    plt.title(
        f"Galat dan tail ratio terhadap RMAX, "
        f"omega={TARGET_OMEGA:.2f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "08_error_tail_RMAX.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> int:
    print("=" * 72)
    print("BENCHMARK PIECEWISE-PARABOLIC")
    print("=" * 72)

    print(
        f"M_outer       = {PARAMS.M_outer}"
    )
    print(
        f"m_inner       = {PARAMS.m_inner}"
    )
    print(
        f"v_transition  = {PARAMS.v_transition}"
    )
    print(
        f"RMAX utama    = {RMAX_MAIN}"
    )
    print(
        f"Output         = {OUTPUT_DIR.resolve()}"
    )

    main_results: list[BenchmarkResult] = []

    print("\nScan frekuensi:")

    for omega in OMEGAS:
        print(
            f"  omega={omega:.3f} ... ",
            end="",
            flush=True,
        )

        try:
            result = run_single_benchmark(
                omega,
                RMAX_MAIN,
                PARAMS,
            )
        except Exception as error:
            print(f"GAGAL: {error}")
            continue

        main_results.append(result)

        if result.bvp_success:
            print(
                "berhasil | "
                f"R={result.matching_radius_numeric:.6f} | "
                f"L2={result.relative_l2_error:.3e} | "
                f"dQ={result.relative_q_error:.3e} | "
                f"dE={result.relative_energy_error:.3e} | "
                f"tail={result.tail_ratio:.3e}"
            )
        else:
            print(
                f"solver gagal: {result.bvp_message}"
            )

    if not main_results:
        print("Tidak ada hasil yang dapat dianalisis.")
        return 1

    save_summary_csv(
        main_results,
        OUTPUT_DIR / "benchmark_omega_summary.csv",
    )

    plot_profiles(
        main_results,
        OUTPUT_DIR / "01_profiles_analytic_numeric.png",
    )

    plot_errors(
        main_results,
        OUTPUT_DIR / "02_relative_errors.png",
    )

    plot_charge_energy(main_results)

    target_candidates = [
        result
        for result in main_results
        if abs(
            result.omega - TARGET_OMEGA
        ) < 1.0e-12
    ]

    if target_candidates:
        plot_log_tail(
            target_candidates[0],
            OUTPUT_DIR / "05_log_tail.png",
        )

    print("\nUji konvergensi RMAX:")

    rmax_results: list[BenchmarkResult] = []

    for rmax in RMAX_TESTS:
        print(
            f"  RMAX={rmax:.1f} ... ",
            end="",
            flush=True,
        )

        try:
            result = run_single_benchmark(
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
                f"solver gagal: {result.bvp_message}"
            )

    if rmax_results:
        save_summary_csv(
            rmax_results,
            OUTPUT_DIR / "benchmark_RMAX_summary.csv",
        )

        plot_rmax_convergence(rmax_results)

    print("\n" + "=" * 72)
    print("SELESAI")
    print("=" * 72)

    print("File output:")

    for filepath in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {filepath}")

    print("\nPatokan awal benchmark:")
    print("  relative_L2_error  < 1e-3")
    print("  relative_Q_error   < 1e-3")
    print("  relative_E_error   < 1e-3")
    print("  tail_ratio         < 1e-6")
    print("  Q dan E konvergen ketika RMAX diperbesar")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nKesalahan fatal: {exc}")
        sys.exit(1)
