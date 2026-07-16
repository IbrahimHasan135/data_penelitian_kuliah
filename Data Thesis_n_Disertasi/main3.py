"""
ANALISIS NUMERIK Q-BALL POLYNOMIAL p-q
======================================

Potensial:
    U(f) = m^2 f^2 - alpha f^p + beta f^q

Persamaan radial:
    f'' + (2/r) f'
    - (m^2 - omega^2) f
    + (p/2) alpha f^(p-1)
    - (q/2) beta f^(q-1) = 0

Strategi numerik:
1. Hitung f_star dan omega_min secara analitik.
2. Pilih amplitudo pusat A = f(0) sebagai continuation parameter.
3. Untuk setiap A, solve_bvp mencari omega sebagai eigenparameter.
4. Omega dibatasi otomatis pada:
       omega_min < omega < m
5. Solusi sebelumnya dipakai sebagai initial guess berikutnya.
6. Hitung:
       Q, E, E/Q, dQ/domega, tail ratio, node, residual.
7. Uji tiga model:
       (p,q) = (4,6), (4,8), (6,8)
8. Ekspor seluruh grafik ke PNG dan hasil ke CSV.

Tidak menggunakan plt.show().
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_bvp


# ============================================================
# KONFIGURASI GLOBAL
# ============================================================

MASS = 1.0

RMAX = 40.0
EPS = 1.0e-5

N_MESH = 700
N_EVAL = 5000

BVP_TOL = 1.0e-6
MAX_NODES = 60000

# Banyak titik continuation untuk tiap model.
N_AMPLITUDES = 28

# Rentang amplitudo relatif terhadap f_star.
A_MIN_FACTOR = 0.20
A_MAX_FACTOR = 1.55

# Untuk uji konvergensi domain.
RMAX_TESTS = [25.0, 35.0, 45.0, 60.0]

# Ambang diagnostik.
TAIL_LIMIT = 1.0e-6
CONVERGENCE_LIMIT = 1.0e-3
NODE_TOL = 1.0e-5

OUTPUT_DIR = Path("polynomial_pq_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
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
class SolutionResult:
    model_name: str
    p: int
    q: int
    alpha: float
    beta: float

    amplitude_center: float
    omega: float
    omega_min: float
    f_star: float

    q_charge: float
    energy: float
    energy_per_charge: float

    tail_ratio: float
    node_count: int
    min_profile: float
    max_residual: float

    solver_success: bool
    solver_message: str

    r: np.ndarray
    f: np.ndarray
    df: np.ndarray

    theta: float


# ============================================================
# PARAMETER MODEL
# ============================================================

def construct_models() -> list[PQModel]:
    """
    Parameter dipilih agar, untuk semua kandidat:

        f_star = 1
        omega_min^2 = 0.5

    Dari:
        beta = ((p-2)/(q-2)) alpha

    dan:
        m^2 - omega_min^2
        = alpha - beta
        = 0.5

    diperoleh:
        alpha = 0.5 (q-2)/(q-p)
        beta  = 0.5 (p-2)/(q-p)
    """

    exponent_pairs = [
        (4, 6),
        (4, 8),
        (6, 8),
    ]

    models: list[PQModel] = []

    for p, q in exponent_pairs:
        alpha = (
            0.5
            * (q - 2)
            / (q - p)
        )

        beta = (
            0.5
            * (p - 2)
            / (q - p)
        )

        models.append(
            PQModel(
                p=p,
                q=q,
                alpha=alpha,
                beta=beta,
                m=MASS,
            )
        )

    return models


# ============================================================
# ANALISIS ANALITIK
# ============================================================

def analytic_characteristics(
    model: PQModel,
) -> tuple[float, float, float]:
    """
    Menghitung:

        f_star =
        [((p-2) alpha)/((q-2) beta)]^(1/(q-p))

        omega_min^2 =
        m^2
        - alpha f_star^(p-2)
        + beta f_star^(q-2)
    """

    numerator = (
        (model.p - 2)
        * model.alpha
    )

    denominator = (
        (model.q - 2)
        * model.beta
    )

    if numerator <= 0 or denominator <= 0:
        raise ValueError(
            "alpha, beta, p, dan q tidak valid."
        )

    f_star = (
        numerator / denominator
    ) ** (
        1.0 / (model.q - model.p)
    )

    omega_min_sq = (
        model.m**2
        - model.alpha
        * f_star ** (model.p - 2)
        + model.beta
        * f_star ** (model.q - 2)
    )

    if omega_min_sq < 0:
        raise ValueError(
            f"{model.name}: omega_min^2 negatif: "
            f"{omega_min_sq:.8f}"
        )

    if omega_min_sq >= model.m**2:
        raise ValueError(
            f"{model.name}: jendela omega kosong."
        )

    omega_min = math.sqrt(
        omega_min_sq
    )

    return (
        f_star,
        omega_min,
        omega_min_sq,
    )


# ============================================================
# TRANSFORMASI PARAMETER OMEGA
# ============================================================

def sigmoid(x: float | np.ndarray):
    """
    Sigmoid stabil secara numerik.
    """
    x_array = np.asarray(x)

    result = np.empty_like(
        x_array,
        dtype=float,
    )

    positive = x_array >= 0
    negative = ~positive

    result[positive] = (
        1.0
        / (
            1.0
            + np.exp(-x_array[positive])
        )
    )

    exp_x = np.exp(
        x_array[negative]
    )

    result[negative] = (
        exp_x
        / (1.0 + exp_x)
    )

    if np.isscalar(x):
        return float(result)

    return result


def omega_from_theta(
    theta: float,
    omega_min: float,
    mass: float,
) -> float:
    """
    Memaksa omega selalu berada pada:

        omega_min < omega < mass
    """

    small_margin = 1.0e-5

    lower = (
        omega_min
        + small_margin
    )

    upper = (
        mass
        - small_margin
    )

    return (
        lower
        + (upper - lower)
        * sigmoid(theta)
    )


def theta_from_omega(
    omega: float,
    omega_min: float,
    mass: float,
) -> float:
    small_margin = 1.0e-5

    lower = (
        omega_min
        + small_margin
    )

    upper = (
        mass
        - small_margin
    )

    fraction = (
        (omega - lower)
        / (upper - lower)
    )

    fraction = np.clip(
        fraction,
        1.0e-9,
        1.0 - 1.0e-9,
    )

    return math.log(
        fraction
        / (1.0 - fraction)
    )


# ============================================================
# POTENSIAL DAN TURUNAN
# ============================================================

def potential(
    f: np.ndarray,
    model: PQModel,
) -> np.ndarray:
    """
    U(f) = m^2 f^2 - alpha f^p + beta f^q
    """

    return (
        model.m**2 * f**2
        - model.alpha * f**model.p
        + model.beta * f**model.q
    )


def nonlinear_force(
    f: np.ndarray,
    omega: float,
    model: PQModel,
) -> np.ndarray:
    """
    Dari:

        f'' =
        -(2/r)f'
        + (m^2-omega^2)f
        - (p/2) alpha f^(p-1)
        + (q/2) beta f^(q-1)

    Fungsi ini mengembalikan bagian tanpa friction radial.
    """

    return (
        (model.m**2 - omega**2) * f
        - 0.5
        * model.p
        * model.alpha
        * f ** (model.p - 1)
        + 0.5
        * model.q
        * model.beta
        * f ** (model.q - 1)
    )


# ============================================================
# SOLVER CONTINUATION
# ============================================================

def make_initial_guess(
    r: np.ndarray,
    amplitude: float,
    omega_guess: float,
    model: PQModel,
) -> np.ndarray:
    """
    Initial guess lokal berbentuk Gaussian.

    Bukan asumsi solusi final; hanya seed solver.
    """

    kappa = math.sqrt(
        max(
            model.m**2
            - omega_guess**2,
            1.0e-8,
        )
    )

    width = max(
        2.0 / kappa,
        2.0,
    )

    f_guess = (
        amplitude
        * np.exp(
            -(r / width) ** 2
        )
    )

    df_guess = (
        -2.0
        * r
        / width**2
        * f_guess
    )

    return np.vstack([
        f_guess,
        df_guess,
    ])


def rescale_previous_guess(
    previous_result: SolutionResult,
    new_r: np.ndarray,
    new_amplitude: float,
) -> np.ndarray:
    """
    Menggunakan solusi sebelumnya sebagai continuation guess.
    """

    old_amplitude = max(
        previous_result.amplitude_center,
        1.0e-12,
    )

    scale = (
        new_amplitude
        / old_amplitude
    )

    f_guess = np.interp(
        new_r,
        previous_result.r,
        previous_result.f,
    ) * scale

    df_guess = np.interp(
        new_r,
        previous_result.r,
        previous_result.df,
    ) * scale

    return np.vstack([
        f_guess,
        df_guess,
    ])


def solve_for_center_amplitude(
    model: PQModel,
    amplitude: float,
    rmax: float,
    previous_result: SolutionResult | None = None,
) -> SolutionResult:
    """
    Untuk amplitudo pusat A yang diberikan, solver mencari omega.

    State:
        y[0] = f
        y[1] = f'

    Unknown parameter:
        p[0] = theta

    dengan omega(theta) selalu berada di jendela analitik.

    Boundary condition:
        f(EPS) = A
        f'(EPS) = 0
        f'(R) + (kappa + 1/R) f(R) = 0
    """

    (
        f_star,
        omega_min,
        _,
    ) = analytic_characteristics(model)

    r_mesh = np.linspace(
        EPS,
        rmax,
        N_MESH,
    )

    if previous_result is None:
        omega_guess = (
            omega_min
            + 0.72
            * (model.m - omega_min)
        )

        theta_guess = theta_from_omega(
            omega_guess,
            omega_min,
            model.m,
        )

        y_guess = make_initial_guess(
            r_mesh,
            amplitude,
            omega_guess,
            model,
        )

    else:
        theta_guess = (
            previous_result.theta
        )

        omega_guess = (
            previous_result.omega
        )

        y_guess = rescale_previous_guess(
            previous_result,
            r_mesh,
            amplitude,
        )

    def ode(
        r: np.ndarray,
        y: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:

        theta = float(parameters[0])

        omega = omega_from_theta(
            theta,
            omega_min,
            model.m,
        )

        f = y[0]
        df = y[1]

        safe_r = np.maximum(
            r,
            EPS,
        )

        d2f = (
            -(2.0 / safe_r) * df
            + nonlinear_force(
                f,
                omega,
                model,
            )
        )

        return np.vstack([
            df,
            d2f,
        ])

    def boundary_conditions(
        ya: np.ndarray,
        yb: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:

        theta = float(parameters[0])

        omega = omega_from_theta(
            theta,
            omega_min,
            model.m,
        )

        kappa = math.sqrt(
            max(
                model.m**2
                - omega**2,
                1.0e-14,
            )
        )

        return np.array([
            ya[0] - amplitude,
            ya[1],
            yb[1]
            + (
                kappa
                + 1.0 / rmax
            )
            * yb[0],
        ])

    solver = solve_bvp(
        ode,
        boundary_conditions,
        r_mesh,
        y_guess,
        p=np.array([theta_guess]),
        tol=BVP_TOL,
        max_nodes=MAX_NODES,
        verbose=0,
    )

    if not solver.success:
        return SolutionResult(
            model_name=model.name,
            p=model.p,
            q=model.q,
            alpha=model.alpha,
            beta=model.beta,
            amplitude_center=amplitude,
            omega=np.nan,
            omega_min=omega_min,
            f_star=f_star,
            q_charge=np.nan,
            energy=np.nan,
            energy_per_charge=np.nan,
            tail_ratio=np.nan,
            node_count=-1,
            min_profile=np.nan,
            max_residual=np.nan,
            solver_success=False,
            solver_message=str(
                solver.message
            ),
            r=np.array([]),
            f=np.array([]),
            df=np.array([]),
            theta=np.nan,
        )

    theta_solution = float(
        solver.p[0]
    )

    omega = omega_from_theta(
        theta_solution,
        omega_min,
        model.m,
    )

    r_eval = np.linspace(
        EPS,
        rmax,
        N_EVAL,
    )

    values = solver.sol(
        r_eval
    )

    f = values[0]
    df = values[1]

    q_charge = calculate_charge(
        r_eval,
        f,
        omega,
    )

    energy = calculate_energy(
        r_eval,
        f,
        df,
        omega,
        model,
    )

    energy_per_charge = (
        energy / q_charge
        if q_charge > 0
        else np.inf
    )

    tail_ratio = (
        abs(f[-1])
        / (
            abs(f[0])
            + 1.0e-30
        )
    )

    node_count = count_nodes(
        f,
        NODE_TOL,
    )

    residual = calculate_equation_residual(
        r_eval,
        f,
        df,
        omega,
        model,
    )

    max_residual = float(
        np.max(np.abs(residual))
    )

    return SolutionResult(
        model_name=model.name,
        p=model.p,
        q=model.q,
        alpha=model.alpha,
        beta=model.beta,
        amplitude_center=amplitude,
        omega=omega,
        omega_min=omega_min,
        f_star=f_star,
        q_charge=q_charge,
        energy=energy,
        energy_per_charge=energy_per_charge,
        tail_ratio=tail_ratio,
        node_count=node_count,
        min_profile=float(
            np.min(f)
        ),
        max_residual=max_residual,
        solver_success=True,
        solver_message=str(
            solver.message
        ),
        r=r_eval,
        f=f,
        df=df,
        theta=theta_solution,
    )


# ============================================================
# BESARAN FISIK
# ============================================================

def calculate_charge(
    r: np.ndarray,
    f: np.ndarray,
    omega: float,
) -> float:
    """
    Q = 8 pi omega integral f^2 r^2 dr
    """

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
    model: PQModel,
) -> float:
    """
    E = 4 pi integral
        [omega^2 f^2 + f'^2 + U(f)] r^2 dr
    """

    density = (
        omega**2 * f**2
        + df**2
        + potential(
            f,
            model,
        )
    )

    return float(
        4.0
        * np.pi
        * simpson(
            density * r**2,
            x=r,
        )
    )


def count_nodes(
    f: np.ndarray,
    tolerance: float,
) -> int:
    """
    Menghitung perubahan tanda yang signifikan.
    Tail noise diabaikan.
    """

    scale = max(
        np.max(np.abs(f)),
        1.0e-30,
    )

    mask = (
        np.abs(f)
        > tolerance * scale
    )

    filtered = f[mask]

    if filtered.size < 2:
        return 0

    signs = np.sign(filtered)

    return int(
        np.sum(
            signs[:-1]
            * signs[1:]
            < 0
        )
    )


def calculate_equation_residual(
    r: np.ndarray,
    f: np.ndarray,
    df: np.ndarray,
    omega: float,
    model: PQModel,
) -> np.ndarray:
    """
    Residual dihitung dengan second derivative numerik.
    """

    d2f_numeric = np.gradient(
        df,
        r,
        edge_order=2,
    )

    safe_r = np.maximum(
        r,
        EPS,
    )

    residual = (
        d2f_numeric
        + (2.0 / safe_r) * df
        - (
            model.m**2
            - omega**2
        )
        * f
        + 0.5
        * model.p
        * model.alpha
        * f ** (model.p - 1)
        - 0.5
        * model.q
        * model.beta
        * f ** (model.q - 1)
    )

    # Hilangkan beberapa titik dekat pusat,
    # tempat gradient numerik paling sensitif.
    residual[:5] = 0.0

    return residual


# ============================================================
# CONTINUATION SATU MODEL
# ============================================================

def run_model_continuation(
    model: PQModel,
    rmax: float = RMAX,
) -> list[SolutionResult]:

    (
        f_star,
        omega_min,
        omega_min_sq,
    ) = analytic_characteristics(model)

    print("\n" + "=" * 72)
    print(
        f"MODEL {model.name}"
    )
    print("=" * 72)

    print(
        f"p={model.p}, q={model.q}, "
        f"alpha={model.alpha:.8f}, "
        f"beta={model.beta:.8f}"
    )

    print(
        f"f_star={f_star:.8f}"
    )

    print(
        f"omega_min^2={omega_min_sq:.8f}"
    )

    print(
        f"jendela omega: "
        f"{omega_min:.8f} < omega < {model.m:.8f}"
    )

    # Mulai dari amplitudo kecil menuju besar.
    amplitudes = np.linspace(
        A_MIN_FACTOR * f_star,
        A_MAX_FACTOR * f_star,
        N_AMPLITUDES,
    )

    results: list[SolutionResult] = []

    previous_result: SolutionResult | None = None

    for amplitude in amplitudes:
        print(
            f"A={amplitude:.6f} ... ",
            end="",
            flush=True,
        )

        try:
            result = solve_for_center_amplitude(
                model=model,
                amplitude=float(amplitude),
                rmax=rmax,
                previous_result=previous_result,
            )
        except Exception as error:
            print(
                f"ERROR: {error}"
            )
            continue

        if not result.solver_success:
            print(
                f"GAGAL: {result.solver_message}"
            )
            continue

        print(
            f"omega={result.omega:.8f} | "
            f"Q={result.q_charge:.6f} | "
            f"E={result.energy:.6f} | "
            f"E/Q={result.energy_per_charge:.6f} | "
            f"tail={result.tail_ratio:.3e} | "
            f"node={result.node_count}"
        )

        results.append(result)

        # Continuation hanya memakai solusi
        # ground-state yang cukup layak.
        if (
            result.node_count == 0
            and np.isfinite(result.omega)
        ):
            previous_result = result

    # Urut berdasarkan omega untuk analisis VK.
    results.sort(
        key=lambda item: item.omega
    )

    return results


# ============================================================
# UJI KONVERGENSI RMAX
# ============================================================

def run_rmax_test(
    model: PQModel,
    reference: SolutionResult,
) -> list[SolutionResult]:

    results: list[SolutionResult] = []

    previous: SolutionResult | None = None

    for rmax in RMAX_TESTS:
        print(
            f"  RMAX={rmax:.1f} ... ",
            end="",
            flush=True,
        )

        seed = (
            previous
            if previous is not None
            else reference
        )

        try:
            result = solve_for_center_amplitude(
                model=model,
                amplitude=reference.amplitude_center,
                rmax=rmax,
                previous_result=seed,
            )
        except Exception as error:
            print(
                f"ERROR: {error}"
            )
            continue

        if not result.solver_success:
            print(
                f"GAGAL: {result.solver_message}"
            )
            continue

        print(
            f"omega={result.omega:.8f} | "
            f"Q={result.q_charge:.8f} | "
            f"E={result.energy:.8f} | "
            f"tail={result.tail_ratio:.3e}"
        )

        results.append(result)
        previous = result

    return results


# ============================================================
# CSV
# ============================================================

def calculate_vk_gradient(
    results: list[SolutionResult],
) -> np.ndarray:

    if len(results) < 3:
        return np.full(
            len(results),
            np.nan,
        )

    omega = np.array([
        result.omega
        for result in results
    ])

    q_values = np.array([
        result.q_charge
        for result in results
    ])

    return np.gradient(
        q_values,
        omega,
    )


def save_model_csv(
    results: list[SolutionResult],
    filepath: Path,
) -> None:

    vk_gradient = calculate_vk_gradient(
        results
    )

    with filepath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "model",
            "p",
            "q",
            "alpha",
            "beta",
            "A_center",
            "omega",
            "omega_min",
            "Q",
            "E",
            "E_over_Q",
            "dQ_domega",
            "tail_ratio",
            "node_count",
            "min_profile",
            "max_residual",
            "localized",
            "ground_state",
            "energetically_stable",
            "VK_candidate",
        ])

        for result, derivative in zip(
            results,
            vk_gradient,
        ):
            writer.writerow([
                result.model_name,
                result.p,
                result.q,
                result.alpha,
                result.beta,
                result.amplitude_center,
                result.omega,
                result.omega_min,
                result.q_charge,
                result.energy,
                result.energy_per_charge,
                derivative,
                result.tail_ratio,
                result.node_count,
                result.min_profile,
                result.max_residual,
                result.tail_ratio < TAIL_LIMIT,
                result.node_count == 0,
                result.energy_per_charge < MASS,
                derivative < 0
                if np.isfinite(derivative)
                else False,
            ])


# ============================================================
# PLOT PER MODEL
# ============================================================

def plot_model_results(
    model: PQModel,
    results: list[SolutionResult],
) -> None:

    if not results:
        return

    model_dir = (
        OUTPUT_DIR / model.name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    omega = np.array([
        result.omega
        for result in results
    ])

    q_values = np.array([
        result.q_charge
        for result in results
    ])

    energy_values = np.array([
        result.energy
        for result in results
    ])

    e_over_q = np.array([
        result.energy_per_charge
        for result in results
    ])

    tail = np.array([
        result.tail_ratio
        for result in results
    ])

    amplitudes = np.array([
        result.amplitude_center
        for result in results
    ])

    vk_gradient = calculate_vk_gradient(
        results
    )

    # --------------------------------------------------------
    # Profil
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Pilih maksimal 8 profil agar grafik tidak penuh.
    indices = np.linspace(
        0,
        len(results) - 1,
        min(8, len(results)),
        dtype=int,
    )

    for index in indices:
        result = results[index]

        plt.plot(
            result.r,
            result.f,
            label=(
                f"omega={result.omega:.4f}, "
                f"A={result.amplitude_center:.3f}"
            ),
        )

    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.title(
        f"Profil ground-state kandidat {model.name}"
    )
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        model_dir / "01_profiles.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Omega vs amplitudo pusat
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        amplitudes,
        omega,
        marker="o",
    )

    plt.axhline(
        results[0].omega_min,
        linestyle="--",
        label="omega_min",
    )

    plt.axhline(
        model.m,
        linestyle=":",
        label="m",
    )

    plt.xlabel("A = f(0)")
    plt.ylabel("omega")
    plt.title(
        f"Continuation branch: {model.name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        model_dir / "02_omega_vs_amplitude.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Q(omega)
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        omega,
        q_values,
        marker="o",
    )

    plt.xlabel("omega")
    plt.ylabel("Q")
    plt.title(
        f"Muatan Noether Q(omega): {model.name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "03_Q_vs_omega.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # VK
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        omega,
        vk_gradient,
        marker="o",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("omega")
    plt.ylabel("dQ/domega")
    plt.title(
        f"Indikator VK: {model.name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "04_VK_gradient.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Energi
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        omega,
        energy_values,
        marker="o",
    )

    plt.xlabel("omega")
    plt.ylabel("E")
    plt.title(
        f"Energi E(omega): {model.name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "05_E_vs_omega.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # E/Q
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        omega,
        e_over_q,
        marker="o",
    )

    plt.axhline(
        model.m,
        linestyle="--",
        label="massa kuanta bebas m",
    )

    plt.xlabel("omega")
    plt.ylabel("E/Q")
    plt.title(
        f"Stabilitas energetik: {model.name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        model_dir / "06_E_over_Q.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Tail
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        omega,
        tail,
        marker="o",
    )

    plt.axhline(
        TAIL_LIMIT,
        linestyle="--",
        label="batas tail",
    )

    plt.xlabel("omega")
    plt.ylabel("|f(RMAX)| / |f(0)|")
    plt.title(
        f"Lokalisasi tail: {model.name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        model_dir / "07_tail_ratio.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Log-tail untuk solusi tengah
    # --------------------------------------------------------

    middle = results[
        len(results) // 2
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        middle.r,
        np.log(
            np.abs(middle.f)
            + 1.0e-30
        ),
    )

    plt.xlabel("r")
    plt.ylabel("log |f(r)|")
    plt.title(
        f"Log-tail {model.name}, "
        f"omega={middle.omega:.6f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "08_log_tail.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# PLOT RMAX
# ============================================================

def plot_rmax_test(
    model: PQModel,
    results: list[SolutionResult],
) -> None:

    if not results:
        return

    model_dir = (
        OUTPUT_DIR / model.name
    )

    rmax_values = np.array(
        RMAX_TESTS[:len(results)]
    )

    q_values = np.array([
        result.q_charge
        for result in results
    ])

    energy_values = np.array([
        result.energy
        for result in results
    ])

    tail_values = np.array([
        result.tail_ratio
        for result in results
    ])

    plt.figure(figsize=(8, 5))

    plt.plot(
        rmax_values,
        q_values,
        marker="o",
    )

    plt.xlabel("RMAX")
    plt.ylabel("Q")
    plt.title(
        f"Konvergensi Q terhadap RMAX: {model.name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "09_Q_RMAX.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        rmax_values,
        energy_values,
        marker="o",
    )

    plt.xlabel("RMAX")
    plt.ylabel("E")
    plt.title(
        f"Konvergensi E terhadap RMAX: {model.name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "10_E_RMAX.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        rmax_values,
        tail_values,
        marker="o",
    )

    plt.xlabel("RMAX")
    plt.ylabel("Tail ratio")
    plt.title(
        f"Tail terhadap RMAX: {model.name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        model_dir / "11_tail_RMAX.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# RINGKASAN ANTARMODEL
# ============================================================

def model_score(
    results: list[SolutionResult],
) -> dict[str, float]:

    if len(results) < 3:
        return {
            "success_count": len(results),
            "localized_fraction": 0.0,
            "ground_fraction": 0.0,
            "energetic_fraction": 0.0,
            "vk_fraction": 0.0,
            "score": -np.inf,
        }

    vk = calculate_vk_gradient(
        results
    )

    localized_fraction = np.mean([
        result.tail_ratio < TAIL_LIMIT
        for result in results
    ])

    ground_fraction = np.mean([
        result.node_count == 0
        for result in results
    ])

    energetic_fraction = np.mean([
        result.energy_per_charge < MASS
        for result in results
    ])

    vk_fraction = np.mean(
        vk < 0
    )

    score = (
        2.0 * localized_fraction
        + 2.0 * ground_fraction
        + 2.0 * energetic_fraction
        + 2.0 * vk_fraction
        + 0.05 * len(results)
    )

    return {
        "success_count": len(results),
        "localized_fraction": localized_fraction,
        "ground_fraction": ground_fraction,
        "energetic_fraction": energetic_fraction,
        "vk_fraction": vk_fraction,
        "score": score,
    }


def save_comparison_csv(
    all_results: dict[
        str,
        list[SolutionResult],
    ],
) -> None:

    filepath = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    with filepath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "model",
            "success_count",
            "localized_fraction",
            "ground_fraction",
            "energetic_fraction",
            "VK_fraction",
            "score",
        ])

        for name, results in all_results.items():
            score_data = model_score(
                results
            )

            writer.writerow([
                name,
                score_data["success_count"],
                score_data["localized_fraction"],
                score_data["ground_fraction"],
                score_data["energetic_fraction"],
                score_data["vk_fraction"],
                score_data["score"],
            ])


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 72)
    print("ANALISIS NUMERIK POLYNOMIAL p-q")
    print("=" * 72)

    models = construct_models()

    all_results: dict[
        str,
        list[SolutionResult],
    ] = {}

    for model in models:

        try:
            results = run_model_continuation(
                model=model,
                rmax=RMAX,
            )
        except Exception as error:
            print(
                f"\nMODEL {model.name} ERROR: {error}"
            )
            continue

        all_results[
            model.name
        ] = results

        model_dir = (
            OUTPUT_DIR / model.name
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_model_csv(
            results,
            model_dir
            / "branch_results.csv",
        )

        plot_model_results(
            model,
            results,
        )

        # Pilih solusi ground-state tengah
        # untuk uji RMAX.
        valid = [
            result
            for result in results
            if (
                result.node_count == 0
                and result.tail_ratio
                < 1.0e-4
            )
        ]

        if valid:
            reference = valid[
                len(valid) // 2
            ]

            print(
                f"\nUji RMAX {model.name}, "
                f"A={reference.amplitude_center:.6f}:"
            )

            rmax_results = run_rmax_test(
                model,
                reference,
            )

            save_model_csv(
                rmax_results,
                model_dir
                / "RMAX_test.csv",
            )

            plot_rmax_test(
                model,
                rmax_results,
            )

    save_comparison_csv(
        all_results
    )

    print("\n" + "=" * 72)
    print("SELESAI")
    print("=" * 72)

    print(
        f"Output disimpan di:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print("\nFile utama:")

    print(
        "- model_comparison.csv"
    )

    for model in models:
        print(
            f"- {model.name}/branch_results.csv"
        )
        print(
            f"- {model.name}/01_profiles.png"
        )
        print(
            f"- {model.name}/03_Q_vs_omega.png"
        )
        print(
            f"- {model.name}/04_VK_gradient.png"
        )
        print(
            f"- {model.name}/06_E_over_Q.png"
        )
        print(
            f"- {model.name}/07_tail_ratio.png"
        )

    print("\nKriteria awal kandidat lolos:")

    print(
        f"- tail ratio < {TAIL_LIMIT:.1e}"
    )

    print(
        "- node_count = 0"
    )

    print(
        "- E/Q < m"
    )

    print(
        "- dQ/domega < 0"
    )

    print(
        "- Q dan E konvergen terhadap RMAX"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            "\nProgram dihentikan oleh pengguna."
        )
        sys.exit(130)

    except Exception as error:
        print(
            f"\nKesalahan fatal: {error}"
        )
        sys.exit(1)
