
"""
particle_identity_static_tests.py
=================================

Analisis identitas partikel statik untuk solusi Q-ball polynomial p4-q6.

Input default:
    polynomial_pq_hybrid_output/p4_q6/branch_profiles.npz
    polynomial_pq_hybrid_output/p4_q6/branch_results.csv
    polynomial_pq_hybrid_output/p4_q6/RMAX_low_omega.csv
    polynomial_pq_hybrid_output/p4_q6/RMAX_mid_omega.csv
    polynomial_pq_hybrid_output/p4_q6/RMAX_high_omega.csv

Script ini TIDAK menjalankan solve_bvp lagi.
Ia langsung membaca profil radial lengkap dari branch_profiles.npz.

Pengujian:
1. Pemilihan solusi referensi yang memenuhi:
       ground state,
       tail kecil,
       E/Q < m,
       dQ/domega < 0.
2. Finite energy dan finite Noether charge.
3. Radius energi, radius muatan, R50, R90, R99.
4. Densitas energi dan densitas muatan.
5. Tekanan radial, tangensial, dan tekanan rata-rata.
6. Identitas virial:
       T + 3 V_omega = 0.
7. Kondisi von Laue:
       integral p_average d^3x = 0.
8. Cek ulang residual ODE dari profil tersimpan.
9. Konsistensi NPZ terhadap branch_results.csv.
10. Konvergensi RMAX pada solusi low/mid/high omega.
11. Prediksi Lorentz boost:
       E(v)=gamma M,
       P(v)=gamma M v,
       E^2-P^2=M^2.
12. Ringkasan PASS/FAIL otomatis.

Tidak memakai plt.show().
Semua grafik diekspor ke PNG.

Jalankan:
    python particle_identity_static_tests.py

Opsional:
    python particle_identity_static_tests.py --target-omega 0.89
    python particle_identity_static_tests.py --solution-index 0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson


# ============================================================
# MODEL AND PATH SETTINGS
# ============================================================

MASS = 1.0
P_EXPONENT = 4
Q_EXPONENT = 6
ALPHA = 1.0
BETA = 0.5

DEFAULT_MODEL_DIR = Path(
    "polynomial_pq_hybrid_output/p4_q6"
)

DEFAULT_OUTPUT_DIR = Path(
    "particle_identity_static_output"
)

# Default target chosen inside the stable low-omega branch.
DEFAULT_TARGET_OMEGA = 0.89

BOOST_VELOCITIES = np.array(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
)

# Preliminary gates
TAIL_LIMIT = 1.0e-6
ODE_RESIDUAL_LIMIT = 1.0e-4
VIRIAL_LIMIT = 1.0e-3
LAUE_LIMIT = 1.0e-3
RMAX_CONVERGENCE_LIMIT = 1.0e-3
NPZ_CSV_RELATIVE_LIMIT = 1.0e-6

EPS = 1.0e-30

# Extended tests
TRANSLATION_RELATIVE_LIMIT = 3.0e-3
BOOST_NUMERIC_LIMIT = 5.0e-3
DYNAMIC_ENERGY_DRIFT_LIMIT = 1.0e-2
DYNAMIC_CHARGE_DRIFT_LIMIT = 1.0e-2
DYNAMIC_RADIUS_MIN = 0.75
DYNAMIC_RADIUS_MAX = 1.35
DYNAMIC_CENTER_MIN = 0.35
DYNAMIC_CENTER_MAX = 1.75


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ProfileData:
    index: int
    omega: float
    A_center: float
    Q_saved: float
    E_saved: float
    E_over_Q_saved: float
    tail_saved: float
    node_count_saved: int
    max_residual_saved: float
    r: np.ndarray
    f: np.ndarray
    df: np.ndarray


@dataclass
class StaticDiagnostics:
    E: float
    Q: float
    E_over_Q: float

    R_energy_rms: float
    R_charge_rms: float
    R50_energy: float
    R90_energy: float
    R99_energy: float

    T_gradient: float
    V_omega: float
    virial_raw: float
    virial_relative: float

    laue_integral: float
    laue_relative: float

    ode_residual_max: float
    tail_ratio: float
    node_count: int

    npz_E_relative_error: float
    npz_Q_relative_error: float

    energy_density: np.ndarray
    charge_density: np.ndarray
    radial_pressure: np.ndarray
    tangential_pressure: np.ndarray
    average_pressure: np.ndarray
    potential: np.ndarray
    ode_residual: np.ndarray


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static particle-identity analysis from saved p4-q6 profiles."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Folder output p4_q6 dari polynomial solver.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder penyimpanan hasil analisis.",
    )

    parser.add_argument(
        "--target-omega",
        type=float,
        default=DEFAULT_TARGET_OMEGA,
        help=(
            "Pilih solusi valid yang paling dekat dengan omega ini."
        ),
    )

    parser.add_argument(
        "--solution-index",
        type=int,
        default=None,
        help=(
            "Pilih indeks NPZ langsung. Jika diisi, target omega diabaikan."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grid lebih kecil dan evolusi lebih singkat.",
    )
    parser.add_argument(
        "--skip-dynamics",
        action="store_true",
        help="Lewati evolusi PDE radial.",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Lewati uji translasi dan boost numerik 3D.",
    )
    return parser.parse_args()


# ============================================================
# BASIC PHYSICS
# ============================================================

def potential(f: np.ndarray) -> np.ndarray:
    return (
        MASS**2 * f**2
        - ALPHA * f**P_EXPONENT
        + BETA * f**Q_EXPONENT
    )


def radial_equation_residual(
    r: np.ndarray,
    f: np.ndarray,
    df: np.ndarray,
    omega: float,
) -> np.ndarray:
    d2f = np.gradient(df, r, edge_order=2)
    safe_r = np.maximum(r, 1.0e-10)

    residual = (
        d2f
        + 2.0 * df / safe_r
        - (MASS**2 - omega**2) * f
        + 0.5 * P_EXPONENT * ALPHA * f ** (P_EXPONENT - 1)
        - 0.5 * Q_EXPONENT * BETA * f ** (Q_EXPONENT - 1)
    )

    # Numerical differentiation is least reliable at the first points.
    residual[:8] = 0.0
    return residual


def count_nodes(f: np.ndarray, tolerance: float = 1.0e-5) -> int:
    scale = max(float(np.max(np.abs(f))), EPS)
    significant = f[np.abs(f) > tolerance * scale]

    if significant.size < 2:
        return 0

    signs = np.sign(significant)
    return int(np.sum(signs[:-1] * signs[1:] < 0))


# ============================================================
# INPUT LOADING
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")


def load_npz(model_dir: Path) -> dict[str, np.ndarray]:
    path = model_dir / "branch_profiles.npz"
    require_file(path)

    data = np.load(path)

    required = {
        "omega",
        "A_center",
        "Q",
        "E",
        "E_over_Q",
        "tail_ratio",
        "node_count",
        "max_residual",
        "r",
        "f",
        "df",
    }

    missing = required.difference(data.files)

    if missing:
        raise KeyError(
            f"Field NPZ tidak lengkap. Hilang: {sorted(missing)}"
        )

    arrays = {key: data[key] for key in data.files}

    count = len(arrays["omega"])

    for key in [
        "A_center",
        "Q",
        "E",
        "E_over_Q",
        "tail_ratio",
        "node_count",
        "max_residual",
    ]:
        if len(arrays[key]) != count:
            raise ValueError(
                f"Panjang field {key} tidak sama dengan omega."
            )

    for key in ["r", "f", "df"]:
        if arrays[key].shape[0] != count:
            raise ValueError(
                f"Dimensi pertama {key} tidak sama dengan jumlah solusi."
            )

    return arrays


def load_branch_csv(model_dir: Path) -> list[dict[str, str]]:
    path = model_dir / "branch_results.csv"
    require_file(path)

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return rows


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def select_profile_index(
    arrays: dict[str, np.ndarray],
    branch_rows: list[dict[str, str]],
    target_omega: float,
    explicit_index: int | None,
) -> int:
    count = len(arrays["omega"])

    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= count:
            raise IndexError(
                f"solution-index harus antara 0 dan {count - 1}."
            )
        return explicit_index

    valid_indices: list[int] = []

    # NPZ and CSV are expected to use the same branch order.
    csv_count = min(len(branch_rows), count)

    for index in range(csv_count):
        row = branch_rows[index]

        try:
            ground = bool_from_csv(row["ground_state"])
            energetic = bool_from_csv(row["energetically_stable"])
            vk = bool_from_csv(row["VK_candidate"])
            localized = bool_from_csv(row["localized_strict"])
        except KeyError:
            ground = arrays["node_count"][index] == 0
            energetic = arrays["E_over_Q"][index] < MASS
            vk = False
            localized = arrays["tail_ratio"][index] < TAIL_LIMIT

        if ground and energetic and vk and localized:
            valid_indices.append(index)

    if not valid_indices:
        # Relax VK only if no strict point exists.
        for index in range(count):
            if (
                arrays["node_count"][index] == 0
                and arrays["E_over_Q"][index] < MASS
                and arrays["tail_ratio"][index] < TAIL_LIMIT
            ):
                valid_indices.append(index)

    if not valid_indices:
        raise RuntimeError(
            "Tidak ada solusi yang memenuhi ground-state, E/Q<m, "
            "dan tail strict."
        )

    return min(
        valid_indices,
        key=lambda idx: abs(
            float(arrays["omega"][idx]) - target_omega
        ),
    )


def build_profile(
    arrays: dict[str, np.ndarray],
    index: int,
) -> ProfileData:
    return ProfileData(
        index=index,
        omega=float(arrays["omega"][index]),
        A_center=float(arrays["A_center"][index]),
        Q_saved=float(arrays["Q"][index]),
        E_saved=float(arrays["E"][index]),
        E_over_Q_saved=float(arrays["E_over_Q"][index]),
        tail_saved=float(arrays["tail_ratio"][index]),
        node_count_saved=int(arrays["node_count"][index]),
        max_residual_saved=float(arrays["max_residual"][index]),
        r=np.asarray(arrays["r"][index], dtype=float),
        f=np.asarray(arrays["f"][index], dtype=float),
        df=np.asarray(arrays["df"][index], dtype=float),
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def radius_at_fraction(
    r: np.ndarray,
    cumulative: np.ndarray,
    fraction: float,
) -> float:
    total = float(cumulative[-1])

    if total <= 0:
        return np.nan

    normalized = cumulative / total
    index = int(np.searchsorted(normalized, fraction))
    index = min(index, len(r) - 1)

    if index == 0:
        return float(r[0])

    x0 = normalized[index - 1]
    x1 = normalized[index]
    r0 = r[index - 1]
    r1 = r[index]

    if abs(x1 - x0) < EPS:
        return float(r1)

    weight = (fraction - x0) / (x1 - x0)
    return float(r0 + weight * (r1 - r0))


def compute_static_diagnostics(
    profile: ProfileData,
) -> StaticDiagnostics:
    r = profile.r
    f = profile.f
    df = profile.df
    omega = profile.omega

    U = potential(f)

    energy_density = (
        omega**2 * f**2
        + df**2
        + U
    )

    # j^0 = 2 omega f^2
    charge_density = 2.0 * omega * f**2

    radial_pressure = (
        omega**2 * f**2
        + df**2
        - U
    )

    tangential_pressure = (
        omega**2 * f**2
        - df**2
        - U
    )

    average_pressure = (
        radial_pressure
        + 2.0 * tangential_pressure
    ) / 3.0

    volume_weight = 4.0 * np.pi * r**2

    E = float(
        simpson(
            energy_density * volume_weight,
            x=r,
        )
    )

    Q = float(
        simpson(
            charge_density * volume_weight,
            x=r,
        )
    )

    R_energy_rms = math.sqrt(
        float(
            simpson(
                r**2 * energy_density * volume_weight,
                x=r,
            )
        ) / (E + EPS)
    )

    R_charge_rms = math.sqrt(
        float(
            simpson(
                r**2 * charge_density * volume_weight,
                x=r,
            )
        ) / (Q + EPS)
    )

    cumulative_energy = cumulative_trapezoid(
        energy_density * volume_weight,
        r,
        initial=0.0,
    )

    R50 = radius_at_fraction(
        r,
        cumulative_energy,
        0.50,
    )

    R90 = radius_at_fraction(
        r,
        cumulative_energy,
        0.90,
    )

    R99 = radius_at_fraction(
        r,
        cumulative_energy,
        0.99,
    )

    T_gradient = float(
        simpson(
            df**2 * volume_weight,
            x=r,
        )
    )

    V_omega = float(
        simpson(
            (U - omega**2 * f**2) * volume_weight,
            x=r,
        )
    )

    virial_raw = T_gradient + 3.0 * V_omega

    virial_relative = abs(virial_raw) / (
        abs(T_gradient)
        + 3.0 * abs(V_omega)
        + EPS
    )

    laue_integral = float(
        simpson(
            average_pressure * volume_weight,
            x=r,
        )
    )

    laue_relative = abs(laue_integral) / (
        E + EPS
    )

    ode_residual = radial_equation_residual(
        r,
        f,
        df,
        omega,
    )

    ode_residual_max = float(
        np.max(np.abs(ode_residual))
    )

    tail_ratio = abs(f[-1]) / (
        abs(f[0]) + EPS
    )

    node_count = count_nodes(f)

    npz_E_relative_error = abs(
        E - profile.E_saved
    ) / (
        abs(profile.E_saved) + EPS
    )

    npz_Q_relative_error = abs(
        Q - profile.Q_saved
    ) / (
        abs(profile.Q_saved) + EPS
    )

    return StaticDiagnostics(
        E=E,
        Q=Q,
        E_over_Q=E / (Q + EPS),
        R_energy_rms=R_energy_rms,
        R_charge_rms=R_charge_rms,
        R50_energy=R50,
        R90_energy=R90,
        R99_energy=R99,
        T_gradient=T_gradient,
        V_omega=V_omega,
        virial_raw=virial_raw,
        virial_relative=virial_relative,
        laue_integral=laue_integral,
        laue_relative=laue_relative,
        ode_residual_max=ode_residual_max,
        tail_ratio=tail_ratio,
        node_count=node_count,
        npz_E_relative_error=npz_E_relative_error,
        npz_Q_relative_error=npz_Q_relative_error,
        energy_density=energy_density,
        charge_density=charge_density,
        radial_pressure=radial_pressure,
        tangential_pressure=tangential_pressure,
        average_pressure=average_pressure,
        potential=U,
        ode_residual=ode_residual,
    )


# ============================================================
# RMAX CONVERGENCE
# ============================================================

def load_rmax_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            try:
                rows.append({
                    key: float(value)
                    for key, value in row.items()
                    if value not in {"", None}
                })
            except ValueError:
                continue

    return rows


def convergence_metrics(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    if len(rows) < 2:
        return {
            "Q_relative_change": np.nan,
            "E_relative_change": np.nan,
            "tail_last": np.nan,
            "omega_relative_change": np.nan,
        }

    rows = sorted(rows, key=lambda row: row["RMAX"])

    first = rows[0]
    last = rows[-1]

    return {
        "Q_relative_change": abs(
            last["Q"] - first["Q"]
        ) / (
            abs(last["Q"]) + EPS
        ),
        "E_relative_change": abs(
            last["E"] - first["E"]
        ) / (
            abs(last["E"]) + EPS
        ),
        "tail_last": last["tail_ratio"],
        "omega_relative_change": abs(
            last["omega"] - first["omega"]
        ) / (
            abs(last["omega"]) + EPS
        ),
    }


# ============================================================
# BOOST
# ============================================================

def boost_table(rest_mass: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for velocity in BOOST_VELOCITIES:
        gamma = 1.0 / math.sqrt(
            1.0 - velocity**2
        )

        energy = gamma * rest_mass
        momentum = gamma * rest_mass * velocity

        invariant = (
            energy**2
            - momentum**2
        )

        rows.append({
            "velocity": float(velocity),
            "gamma": gamma,
            "boosted_energy": energy,
            "boosted_momentum": momentum,
            "E2_minus_P2": invariant,
            "rest_mass_squared": rest_mass**2,
            "relative_invariant_error": abs(
                invariant - rest_mass**2
            ) / (
                rest_mass**2 + EPS
            ),
        })

    return rows


# ============================================================
# PASS / FAIL
# ============================================================

def evaluate_gates(
    diagnostics: StaticDiagnostics,
    convergence: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    gates: list[dict[str, object]] = []

    def add(
        name: str,
        value: float | int,
        criterion: str,
        passed: bool,
    ) -> None:
        gates.append({
            "criterion": name,
            "value": value,
            "required": criterion,
            "status": "PASS" if passed else "FAIL",
        })

    add(
        "Ground state",
        diagnostics.node_count,
        "node_count = 0",
        diagnostics.node_count == 0,
    )

    add(
        "Localized tail",
        diagnostics.tail_ratio,
        f"tail_ratio < {TAIL_LIMIT:.1e}",
        diagnostics.tail_ratio < TAIL_LIMIT,
    )

    add(
        "Finite energy",
        diagnostics.E,
        "finite and positive",
        np.isfinite(diagnostics.E) and diagnostics.E > 0,
    )

    add(
        "Finite charge",
        diagnostics.Q,
        "finite and positive",
        np.isfinite(diagnostics.Q) and diagnostics.Q > 0,
    )

    add(
        "Energetic stability",
        diagnostics.E_over_Q,
        "E/Q < m = 1",
        diagnostics.E_over_Q < MASS,
    )

    add(
        "ODE residual",
        diagnostics.ode_residual_max,
        f"max residual < {ODE_RESIDUAL_LIMIT:.1e}",
        diagnostics.ode_residual_max < ODE_RESIDUAL_LIMIT,
    )

    add(
        "Virial identity",
        diagnostics.virial_relative,
        f"relative error < {VIRIAL_LIMIT:.1e}",
        diagnostics.virial_relative < VIRIAL_LIMIT,
    )

    add(
        "Von Laue condition",
        diagnostics.laue_relative,
        f"relative error < {LAUE_LIMIT:.1e}",
        diagnostics.laue_relative < LAUE_LIMIT,
    )

    add(
        "NPZ energy consistency",
        diagnostics.npz_E_relative_error,
        f"relative error < {NPZ_CSV_RELATIVE_LIMIT:.1e}",
        diagnostics.npz_E_relative_error < NPZ_CSV_RELATIVE_LIMIT,
    )

    add(
        "NPZ charge consistency",
        diagnostics.npz_Q_relative_error,
        f"relative error < {NPZ_CSV_RELATIVE_LIMIT:.1e}",
        diagnostics.npz_Q_relative_error < NPZ_CSV_RELATIVE_LIMIT,
    )

    for label, metrics in convergence.items():
        if np.isfinite(metrics["Q_relative_change"]):
            add(
                f"RMAX Q convergence ({label})",
                metrics["Q_relative_change"],
                (
                    "relative change < "
                    f"{RMAX_CONVERGENCE_LIMIT:.1e}"
                ),
                (
                    metrics["Q_relative_change"]
                    < RMAX_CONVERGENCE_LIMIT
                ),
            )

            add(
                f"RMAX E convergence ({label})",
                metrics["E_relative_change"],
                (
                    "relative change < "
                    f"{RMAX_CONVERGENCE_LIMIT:.1e}"
                ),
                (
                    metrics["E_relative_change"]
                    < RMAX_CONVERGENCE_LIMIT
                ),
            )

    return gates


# ============================================================
# OUTPUT
# ============================================================

def write_summary_csv(
    output_dir: Path,
    profile: ProfileData,
    diagnostics: StaticDiagnostics,
) -> None:
    path = output_dir / "particle_static_summary.csv"

    rows = [
        ("solution_index", profile.index),
        ("omega", profile.omega),
        ("A_center", profile.A_center),
        ("E", diagnostics.E),
        ("Q", diagnostics.Q),
        ("E_over_Q", diagnostics.E_over_Q),
        ("R_energy_rms", diagnostics.R_energy_rms),
        ("R_charge_rms", diagnostics.R_charge_rms),
        ("R50_energy", diagnostics.R50_energy),
        ("R90_energy", diagnostics.R90_energy),
        ("R99_energy", diagnostics.R99_energy),
        ("T_gradient", diagnostics.T_gradient),
        ("V_omega", diagnostics.V_omega),
        ("virial_raw", diagnostics.virial_raw),
        ("virial_relative", diagnostics.virial_relative),
        ("laue_integral", diagnostics.laue_integral),
        ("laue_relative", diagnostics.laue_relative),
        ("ode_residual_max", diagnostics.ode_residual_max),
        ("tail_ratio", diagnostics.tail_ratio),
        ("node_count", diagnostics.node_count),
        (
            "npz_E_relative_error",
            diagnostics.npz_E_relative_error,
        ),
        (
            "npz_Q_relative_error",
            diagnostics.npz_Q_relative_error,
        ),
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        writer.writerows(rows)


def write_profile_csv(
    output_dir: Path,
    profile: ProfileData,
    diagnostics: StaticDiagnostics,
) -> None:
    path = output_dir / "particle_profile_diagnostics.csv"

    matrix = np.column_stack([
        profile.r,
        profile.f,
        profile.df,
        diagnostics.potential,
        diagnostics.energy_density,
        diagnostics.charge_density,
        diagnostics.radial_pressure,
        diagnostics.tangential_pressure,
        diagnostics.average_pressure,
        diagnostics.ode_residual,
    ])

    np.savetxt(
        path,
        matrix,
        delimiter=",",
        comments="",
        header=(
            "r,f,df,potential,energy_density,charge_density,"
            "radial_pressure,tangential_pressure,"
            "average_pressure,ode_residual"
        ),
    )


def write_gates_csv(
    output_dir: Path,
    gates: list[dict[str, object]],
) -> None:
    path = output_dir / "particle_identity_gates.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "criterion",
                "value",
                "required",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(gates)


def write_boost_csv(
    output_dir: Path,
    rows: list[dict[str, float]],
) -> None:
    path = output_dir / "lorentz_boost_test.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_rmax_summary_csv(
    output_dir: Path,
    convergence: dict[str, dict[str, float]],
) -> None:
    path = output_dir / "RMAX_convergence_summary.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "region",
            "Q_relative_change",
            "E_relative_change",
            "tail_last",
            "omega_relative_change",
        ])

        for label, metrics in convergence.items():
            writer.writerow([
                label,
                metrics["Q_relative_change"],
                metrics["E_relative_change"],
                metrics["tail_last"],
                metrics["omega_relative_change"],
            ])


def write_text_report(
    output_dir: Path,
    profile: ProfileData,
    diagnostics: StaticDiagnostics,
    gates: list[dict[str, object]],
) -> None:
    path = output_dir / "particle_identity_report.txt"

    pass_count = sum(
        gate["status"] == "PASS"
        for gate in gates
    )

    total_count = len(gates)

    lines = [
        "PARTICLE IDENTITY STATIC ANALYSIS",
        "=" * 60,
        "",
        f"Selected NPZ index : {profile.index}",
        f"omega              : {profile.omega:.12g}",
        f"A_center           : {profile.A_center:.12g}",
        f"Rest energy M=E    : {diagnostics.E:.12g}",
        f"Noether charge Q   : {diagnostics.Q:.12g}",
        f"E/Q                : {diagnostics.E_over_Q:.12g}",
        f"R_energy_rms       : {diagnostics.R_energy_rms:.12g}",
        f"R_charge_rms       : {diagnostics.R_charge_rms:.12g}",
        f"R50                : {diagnostics.R50_energy:.12g}",
        f"R90                : {diagnostics.R90_energy:.12g}",
        f"R99                : {diagnostics.R99_energy:.12g}",
        "",
        "STATIC BALANCE",
        "-" * 60,
        f"Virial relative    : {diagnostics.virial_relative:.6e}",
        f"Laue relative      : {diagnostics.laue_relative:.6e}",
        f"ODE residual max   : {diagnostics.ode_residual_max:.6e}",
        f"Tail ratio         : {diagnostics.tail_ratio:.6e}",
        "",
        "GATES",
        "-" * 60,
    ]

    for gate in gates:
        lines.append(
            f"[{gate['status']}] {gate['criterion']}: "
            f"{gate['value']} | required: {gate['required']}"
        )

    lines.extend([
        "",
        f"PASS COUNT: {pass_count}/{total_count}",
        "",
        (
            "PRELIMINARY CONCLUSION: "
            + (
                "STATIC PARTICLE CRITERIA PASSED"
                if pass_count == total_count
                else "SOME STATIC PARTICLE CRITERIA FAILED"
            )
        ),
        "",
        (
            "Interpretation limit: passing this test supports an "
            "effective classical-relativistic particle interpretation. "
            "It does not identify the solution as an electron, proton, "
            "or another Standard Model particle, and it does not replace "
            "a time-evolution stability test."
        ),
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# PLOTS
# ============================================================

def make_plots(
    output_dir: Path,
    profile: ProfileData,
    diagnostics: StaticDiagnostics,
    boost_rows: list[dict[str, float]],
) -> None:
    r = profile.r

    plt.figure(figsize=(9, 5))
    plt.plot(r, profile.f)
    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.title(
        f"p4-q6 reference profile, omega={profile.omega:.6f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "01_profile.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(r, diagnostics.energy_density)
    plt.xlabel("r")
    plt.ylabel("energy density")
    plt.title("Localized energy density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "02_energy_density.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(r, diagnostics.charge_density)
    plt.xlabel("r")
    plt.ylabel("Noether charge density")
    plt.title("Localized Noether charge density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "03_charge_density.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        r,
        diagnostics.radial_pressure,
        label="radial pressure",
    )
    plt.plot(
        r,
        diagnostics.tangential_pressure,
        label="tangential pressure",
    )
    plt.plot(
        r,
        diagnostics.average_pressure,
        label="average pressure",
    )
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("r")
    plt.ylabel("pressure")
    plt.title("Internal stress distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "04_internal_pressure.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.semilogy(
        r,
        np.maximum(
            np.abs(diagnostics.ode_residual),
            1.0e-30,
        ),
    )
    plt.xlabel("r")
    plt.ylabel("|ODE residual|")
    plt.title("Radial equation residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "05_ode_residual.png",
        dpi=300,
    )
    plt.close()

    velocities = np.array([
        row["velocity"]
        for row in boost_rows
    ])

    energies = np.array([
        row["boosted_energy"]
        for row in boost_rows
    ])

    momenta = np.array([
        row["boosted_momentum"]
        for row in boost_rows
    ])

    plt.figure(figsize=(9, 5))
    plt.plot(
        velocities,
        energies,
        marker="o",
        label="E(v)",
    )
    plt.plot(
        velocities,
        momenta,
        marker="s",
        label="P(v)",
    )
    plt.xlabel("v")
    plt.ylabel("energy / momentum")
    plt.title("Lorentz-boosted energy and momentum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "06_boost_energy_momentum.png",
        dpi=300,
    )
    plt.close()

    invariant_error = np.array([
        row["relative_invariant_error"]
        for row in boost_rows
    ])

    plt.figure(figsize=(9, 5))
    plt.semilogy(
        velocities,
        np.maximum(
            invariant_error,
            1.0e-30,
        ),
        marker="o",
    )
    plt.xlabel("v")
    plt.ylabel("relative invariant error")
    plt.title("Check E^2 - P^2 = M^2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "07_lorentz_invariant.png",
        dpi=300,
    )
    plt.close()



# ============================================================
# EXTENDED PARTICLE TESTS
# ============================================================

def write_generic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_cartesian_grid(half_size: float, points: int):
    axis = np.linspace(-half_size, half_size, points)
    dx = float(axis[1] - axis[0])
    return (*np.meshgrid(axis, axis, axis, indexing="ij"), dx)


def interpolate_radial(profile: ProfileData, radius: np.ndarray):
    flat = radius.ravel()
    f = np.interp(flat, profile.r, profile.f, left=profile.f[0], right=0.0)
    df = np.interp(flat, profile.r, profile.df, left=0.0, right=0.0)
    return f.reshape(radius.shape), df.reshape(radius.shape)


def translation_invariance_test(profile, diagnostics, points):
    shifts = [0.0, 0.75 * diagnostics.R90_energy, 1.5 * diagnostics.R90_energy]
    half = max(4.0 * diagnostics.R99_energy, shifts[-1] + 3.0 * diagnostics.R99_energy)
    X, Y, Z, dx = make_cartesian_grid(half, points)
    dV = dx**3
    rows=[]
    for shift in shifts:
        radius=np.sqrt((X-shift)**2+Y**2+Z**2)
        f,df=interpolate_radial(profile,radius)
        rho=profile.omega**2*f**2+df**2+potential(f)
        j0=2.0*profile.omega*f**2
        rows.append({
            'shift_x':shift,
            'grid_points':points,
            'box_half_size':half,
            'E':float(np.sum(rho)*dV),
            'Q':float(np.sum(j0)*dV),
        })
    E0,Q0=rows[0]['E'],rows[0]['Q']
    for row in rows:
        row['E_relative_to_centered']=abs(row['E']-E0)/(abs(E0)+EPS)
        row['Q_relative_to_centered']=abs(row['Q']-Q0)/(abs(Q0)+EPS)
    return rows


def boosted_snapshot_test(profile, diagnostics, points):
    half=max(4.0*diagnostics.R99_energy,18.0)
    X,Y,Z,dx=make_cartesian_grid(half,points)
    dV=dx**3
    rows=[]
    for v in [0.3,0.6]:
        gamma=1.0/math.sqrt(1.0-v*v)
        R=np.sqrt((gamma*X)**2+Y**2+Z**2)
        f,df=interpolate_radial(profile,R)
        safe=np.maximum(R,1e-14)
        phase=np.exp(1j*profile.omega*gamma*v*X)
        Rt=-(gamma**2)*v*X/safe
        Rx=(gamma**2)*X/safe
        Ry=Y/safe; Rz=Z/safe
        zero=R<1e-12
        Rt[zero]=Rx[zero]=Ry[zero]=Rz[zero]=0.0
        psi=f*phase
        psit=(df*Rt-1j*profile.omega*gamma*f)*phase
        psix=(df*Rx+1j*profile.omega*gamma*v*f)*phase
        psiy=df*Ry*phase; psiz=df*Rz*phase
        U=MASS**2*np.abs(psi)**2-ALPHA*np.abs(psi)**4+BETA*np.abs(psi)**6
        rho=np.abs(psit)**2+np.abs(psix)**2+np.abs(psiy)**2+np.abs(psiz)**2+U
        px=-2.0*np.real(np.conj(psit)*psix)
        j0=np.real(1j*(np.conj(psi)*psit-psi*np.conj(psit)))
        E=float(np.sum(rho)*dV); P=float(np.sum(px)*dV); Q=float(np.sum(j0)*dV)
        inv=E*E-P*P
        rows.append({
            'velocity':v,'gamma':gamma,'E_numeric':E,'P_numeric':P,'Q_numeric':Q,
            'E_expected':gamma*diagnostics.E,'P_expected':gamma*diagnostics.E*v,
            'E_relative_error':abs(E-gamma*diagnostics.E)/(gamma*diagnostics.E+EPS),
            'P_relative_error':abs(P-gamma*diagnostics.E*v)/(gamma*diagnostics.E*v+EPS),
            'Q_relative_error':abs(Q-diagnostics.Q)/(diagnostics.Q+EPS),
            'invariant_relative_error':abs(inv-diagnostics.E**2)/(diagnostics.E**2+EPS),
        })
    return rows


def radial_laplacian(psi,dr):
    lap=np.zeros_like(psi)
    idx=np.arange(1,len(psi)-1)
    r=idx*dr
    lap[1:-1]=(psi[2:]-2*psi[1:-1]+psi[:-2])/dr**2+(psi[2:]-psi[:-2])/(dr*r)
    lap[0]=6.0*(psi[1]-psi[0])/dr**2
    return lap


def complex_potential(psi):
    s=np.abs(psi)**2
    return MASS**2*s-ALPHA*s**2+BETA*s**3


def nonlinear_coeff(psi):
    s=np.abs(psi)**2
    return MASS**2-2.0*ALPHA*s+3.0*BETA*s**2


def dynamic_diagnostics(r,psi,pi,dr):
    dpsi=np.gradient(psi,dr,edge_order=2)
    w=4*np.pi*r**2
    rho=np.abs(pi)**2+np.abs(dpsi)**2+complex_potential(psi)
    j0=np.real(1j*(np.conj(psi)*pi-psi*np.conj(pi)))
    E=float(simpson(rho*w,x=r)); Q=float(simpson(j0*w,x=r))
    R=math.sqrt(float(simpson(r**2*rho*w,x=r))/(E+EPS))
    return E,Q,R


def run_dynamic_case(profile, diagnostics, name, kind, magnitude, quick):
    rmax=min(float(profile.r[-1]),max(25.0,4.0*diagnostics.R99_energy))
    nr=650 if quick else 1100
    tmax=50.0 if quick else 120.0
    r=np.linspace(0,rmax,nr); dr=float(r[1]-r[0]); dt=0.18*dr
    steps=int(tmax/dt); sample=max(1,steps//450)
    f0=np.interp(r,profile.r,profile.f,left=profile.f[0],right=0.0)
    env=np.exp(-(r/max(diagnostics.R90_energy,1.0))**2)
    if kind=='scale':
        psi=f0*(1.0+magnitude*env)
    else:
        rng=np.random.default_rng(12345)
        noise=rng.normal(size=nr); noise=np.convolve(noise,np.ones(11)/11,mode='same')
        noise/=max(np.max(np.abs(noise)),EPS)
        psi=f0*(1.0+magnitude*noise*env)
    psi=psi.astype(complex); pi=-1j*profile.omega*psi
    sponge=np.zeros_like(r); start=.82*rmax; mask=r>start
    sponge[mask]=0.7*((r[mask]-start)/(rmax-start))**2
    def acc(field): return radial_laplacian(field,dr)-nonlinear_coeff(field)*field
    E0,Q0,R0=dynamic_diagnostics(r,psi,pi,dr); A0=abs(psi[0])
    times=[]; Es=[]; Qs=[]; Rs=[]; As=[]
    for n in range(steps+1):
        if n%sample==0:
            E,Q,R=dynamic_diagnostics(r,psi,pi,dr)
            times.append(n*dt); Es.append(E); Qs.append(Q); Rs.append(R); As.append(abs(psi[0]))
        if n==steps: break
        ph=pi+0.5*dt*acc(psi)
        newpsi=psi+dt*ph; newpsi[-1]=0
        newpi=ph+0.5*dt*acc(newpsi)
        newpi*=np.exp(-sponge*dt); newpi[-1]=0
        psi,pi=newpsi,newpi
    Es=np.array(Es); Qs=np.array(Qs); Rs=np.array(Rs); As=np.array(As)
    return {
        'case':name,'time':np.array(times),'energy':Es,'charge':Qs,'radius':Rs,'center':As,
        'E0':E0,'Q0':Q0,'R0':R0,'A0':A0,
        'max_energy_drift':float(np.max(np.abs(Es-E0))/(abs(E0)+EPS)),
        'max_charge_drift':float(np.max(np.abs(Qs-Q0))/(abs(Q0)+EPS)),
        'radius_min_factor':float(np.min(Rs)/(R0+EPS)),'radius_max_factor':float(np.max(Rs)/(R0+EPS)),
        'center_min_factor':float(np.min(As)/(A0+EPS)),'center_max_factor':float(np.max(As)/(A0+EPS)),
    }


def save_dynamic(output,result):
    rows=[{'time':t,'energy':E,'charge':Q,'radius':R,'center_amplitude':A}
          for t,E,Q,R,A in zip(result['time'],result['energy'],result['charge'],result['radius'],result['center'])]
    write_generic_csv(output/f"dynamic_{result['case']}.csv",rows)
    plt.figure(figsize=(9,5))
    plt.plot(result['time'],result['energy']/result['E0'],label='E/E0')
    plt.plot(result['time'],result['charge']/result['Q0'],label='Q/Q0')
    plt.plot(result['time'],result['radius']/result['R0'],label='R/R0')
    plt.plot(result['time'],result['center']/result['A0'],label='|Psi(0)|/A0')
    plt.xlabel('t'); plt.ylabel('normalized'); plt.title(f"Dynamic perturbation: {result['case']}")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(output/f"dynamic_{result['case']}.png",dpi=300); plt.close()


def rmax_extended_metrics(rows):
    if len(rows)<2:
        return {'omega_reference':np.nan,'Q_last':np.nan,'E_last':np.nan,'Q_full':np.nan,'E_full':np.nan}
    rows=sorted(rows,key=lambda x:x['RMAX']); first,prev,last=rows[0],rows[-2],rows[-1]
    return {
        'omega_reference':float(np.mean([x['omega'] for x in rows])),
        'Q_last':abs(last['Q']-prev['Q'])/(abs(last['Q'])+EPS),
        'E_last':abs(last['E']-prev['E'])/(abs(last['E'])+EPS),
        'Q_full':abs(last['Q']-first['Q'])/(abs(last['Q'])+EPS),
        'E_full':abs(last['E']-first['E'])/(abs(last['E'])+EPS),
        'tail_last':last['tail_ratio'],
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args=parse_arguments(); output_dir:Path=args.output_dir
    output_dir.mkdir(parents=True,exist_ok=True)
    arrays=load_npz(args.model_dir); branch_rows=load_branch_csv(args.model_dir)
    index=select_profile_index(arrays,branch_rows,args.target_omega,args.solution_index)
    profile=build_profile(arrays,index); diagnostics=compute_static_diagnostics(profile)
    boost_rows=boost_table(diagnostics.E)
    make_plots(output_dir,profile,diagnostics,boost_rows)

    gates=[]
    def gate(category,name,value,required,passed,gate_type='primary'):
        gates.append({'category':category,'criterion':name,'value':value,'required':required,'gate_type':gate_type,'status':'PASS' if passed else 'FAIL'})

    # Existing static checks
    gate('static','Ground state',diagnostics.node_count,'node_count = 0',diagnostics.node_count==0)
    gate('static','Localized tail',diagnostics.tail_ratio,f'< {TAIL_LIMIT}',diagnostics.tail_ratio<TAIL_LIMIT)
    gate('static','Finite energy',diagnostics.E,'positive finite',np.isfinite(diagnostics.E) and diagnostics.E>0)
    gate('static','Finite charge',diagnostics.Q,'positive finite',np.isfinite(diagnostics.Q) and diagnostics.Q>0)
    gate('static','Energetic stability',diagnostics.E_over_Q,'E/Q < 1',diagnostics.E_over_Q<MASS)
    gate('static','ODE residual',diagnostics.ode_residual_max,f'< {ODE_RESIDUAL_LIMIT}',diagnostics.ode_residual_max<ODE_RESIDUAL_LIMIT)
    gate('static','Virial identity',diagnostics.virial_relative,f'< {VIRIAL_LIMIT}',diagnostics.virial_relative<VIRIAL_LIMIT)
    gate('static','Von Laue',diagnostics.laue_relative,f'< {LAUE_LIMIT}',diagnostics.laue_relative<LAUE_LIMIT)
    gate('static','NPZ E consistency',diagnostics.npz_E_relative_error,f'< {NPZ_CSV_RELATIVE_LIMIT}',diagnostics.npz_E_relative_error<NPZ_CSV_RELATIVE_LIMIT)
    gate('static','NPZ Q consistency',diagnostics.npz_Q_relative_error,f'< {NPZ_CSV_RELATIVE_LIMIT}',diagnostics.npz_Q_relative_error<NPZ_CSV_RELATIVE_LIMIT)

    # RMAX: compare the last two largest domains; only closest omega dataset is primary.
    conv={}
    for label in ['low_omega','mid_omega','high_omega']:
        conv[label]=rmax_extended_metrics(load_rmax_csv(args.model_dir/f'RMAX_{label}.csv'))
    valid=[(k,v) for k,v in conv.items() if np.isfinite(v['omega_reference'])]
    ref_region=min(valid,key=lambda kv:abs(kv[1]['omega_reference']-profile.omega))[0] if valid else None
    rrows=[]
    for label,m in conv.items():
        gt='primary' if label==ref_region else 'diagnostic'
        if np.isfinite(m['Q_last']):
            gate('RMAX',f'Q convergence {label} (last interval)',m['Q_last'],f'< {RMAX_CONVERGENCE_LIMIT}',m['Q_last']<RMAX_CONVERGENCE_LIMIT,gt)
            gate('RMAX',f'E convergence {label} (last interval)',m['E_last'],f'< {RMAX_CONVERGENCE_LIMIT}',m['E_last']<RMAX_CONVERGENCE_LIMIT,gt)
        rrows.append({'region':label,'gate_type':gt,**m})
    write_generic_csv(output_dir/'RMAX_convergence_extended.csv',rrows)

    # Translation and numerical boost
    if not args.skip_3d:
        points=49 if args.quick else 65
        tr=translation_invariance_test(profile,diagnostics,points)
        write_generic_csv(output_dir/'translation_test.csv',tr)
        maxe=max(x['E_relative_to_centered'] for x in tr); maxq=max(x['Q_relative_to_centered'] for x in tr)
        gate('translation','Energy unchanged after spatial shift',maxe,f'< {TRANSLATION_RELATIVE_LIMIT}',maxe<TRANSLATION_RELATIVE_LIMIT)
        gate('translation','Charge unchanged after spatial shift',maxq,f'< {TRANSLATION_RELATIVE_LIMIT}',maxq<TRANSLATION_RELATIVE_LIMIT)
        plt.figure(figsize=(9,5)); shifts=[x['shift_x'] for x in tr]
        plt.semilogy(shifts,np.maximum([x['E_relative_to_centered'] for x in tr],1e-30),'o-',label='E')
        plt.semilogy(shifts,np.maximum([x['Q_relative_to_centered'] for x in tr],1e-30),'s-',label='Q')
        plt.xlabel('shift x0'); plt.ylabel('relative change'); plt.title('Translation invariance'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(output_dir/'08_translation_invariance.png',dpi=300); plt.close()

        br=boosted_snapshot_test(profile,diagnostics,points)
        write_generic_csv(output_dir/'boosted_snapshot_test.csv',br)
        maxinv=max(x['invariant_relative_error'] for x in br); maxbq=max(x['Q_relative_error'] for x in br)
        gate('boost','Numerical E²-P² invariant',maxinv,f'< {BOOST_NUMERIC_LIMIT}',maxinv<BOOST_NUMERIC_LIMIT)
        gate('boost','Numerical boost charge invariant',maxbq,f'< {BOOST_NUMERIC_LIMIT}',maxbq<BOOST_NUMERIC_LIMIT)

    # Radial time evolution
    if not args.skip_dynamics:
        cases=[('plus_1pct','scale',.01),('minus_1pct','scale',-.01),('noise_1pct','noise',.01)]
        if not args.quick: cases.append(('plus_5pct','scale',.05))
        for name,kind,mag in cases:
            print('Running dynamic case:',name)
            res=run_dynamic_case(profile,diagnostics,name,kind,mag,args.quick); save_dynamic(output_dir,res)
            gate('dynamics',f'Energy drift {name}',res['max_energy_drift'],f'< {DYNAMIC_ENERGY_DRIFT_LIMIT}',res['max_energy_drift']<DYNAMIC_ENERGY_DRIFT_LIMIT)
            gate('dynamics',f'Charge drift {name}',res['max_charge_drift'],f'< {DYNAMIC_CHARGE_DRIFT_LIMIT}',res['max_charge_drift']<DYNAMIC_CHARGE_DRIFT_LIMIT)
            rp=res['radius_min_factor']>DYNAMIC_RADIUS_MIN and res['radius_max_factor']<DYNAMIC_RADIUS_MAX
            cp=res['center_min_factor']>DYNAMIC_CENTER_MIN and res['center_max_factor']<DYNAMIC_CENTER_MAX
            gate('dynamics',f'Radius bounded {name}',f"{res['radius_min_factor']:.5g}..{res['radius_max_factor']:.5g}",f'{DYNAMIC_RADIUS_MIN}<R/R0<{DYNAMIC_RADIUS_MAX}',rp)
            gate('dynamics',f'Core bounded {name}',f"{res['center_min_factor']:.5g}..{res['center_max_factor']:.5g}",f'{DYNAMIC_CENTER_MIN}<A/A0<{DYNAMIC_CENTER_MAX}',cp)

    # Preserve standard outputs and add extended report.
    write_summary_csv(output_dir,profile,diagnostics)
    write_profile_csv(output_dir,profile,diagnostics)
    write_boost_csv(output_dir,boost_rows)
    write_generic_csv(output_dir/'particle_identity_gates_extended.csv',gates)
    primary=[g for g in gates if g['gate_type']=='primary']; passed=sum(g['status']=='PASS' for g in primary)
    lines=['EXTENDED PARTICLE IDENTITY ANALYSIS','='*72,f'index={profile.index}',f'omega={profile.omega}',f'E={diagnostics.E}',f'Q={diagnostics.Q}',f'E/Q={diagnostics.E_over_Q}',f'reference_RMAX_region={ref_region}','','PRIMARY GATES','-'*72]
    lines += [f"[{g['status']}] {g['category']} | {g['criterion']} | value={g['value']} | required={g['required']}" for g in primary]
    diagnostics_failed=[g for g in gates if g['gate_type']=='diagnostic' and g['status']=='FAIL']
    if diagnostics_failed:
        lines += ['','NON-PRIMARY DIAGNOSTIC FAILURES','-'*72] + [f"[FAIL] {g['criterion']} | {g['value']}" for g in diagnostics_failed]
    lines += ['',f'PRIMARY PASS COUNT: {passed}/{len(primary)}','',('CONCLUSION: ALL PRIMARY TESTS PASSED' if passed==len(primary) else 'CONCLUSION: SOME PRIMARY TESTS FAILED'),'','Limits: radial dynamics does not test non-spherical modes or collisions.']
    (output_dir/'particle_identity_report_extended.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('='*72); print('EXTENDED PARTICLE IDENTITY ANALYSIS'); print('='*72)
    print('omega                 =',profile.omega); print('reference RMAX region =',ref_region); print(f'primary PASS count    = {passed}/{len(primary)}'); print('output                 =',output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgram dihentikan pengguna.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        sys.exit(1)
