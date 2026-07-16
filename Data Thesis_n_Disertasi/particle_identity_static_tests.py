
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
# MAIN
# ============================================================

def main() -> int:
    args = parse_arguments()

    model_dir: Path = args.model_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = load_npz(model_dir)
    branch_rows = load_branch_csv(model_dir)

    index = select_profile_index(
        arrays=arrays,
        branch_rows=branch_rows,
        target_omega=args.target_omega,
        explicit_index=args.solution_index,
    )

    profile = build_profile(
        arrays,
        index,
    )

    diagnostics = compute_static_diagnostics(
        profile
    )

    rmax_files = {
        "low_omega": model_dir / "RMAX_low_omega.csv",
        "mid_omega": model_dir / "RMAX_mid_omega.csv",
        "high_omega": model_dir / "RMAX_high_omega.csv",
    }

    convergence: dict[str, dict[str, float]] = {}

    for label, path in rmax_files.items():
        rows = load_rmax_csv(path)
        convergence[label] = convergence_metrics(rows)

    boost_rows = boost_table(
        diagnostics.E
    )

    gates = evaluate_gates(
        diagnostics,
        convergence,
    )

    write_summary_csv(
        output_dir,
        profile,
        diagnostics,
    )

    write_profile_csv(
        output_dir,
        profile,
        diagnostics,
    )

    write_gates_csv(
        output_dir,
        gates,
    )

    write_boost_csv(
        output_dir,
        boost_rows,
    )

    write_rmax_summary_csv(
        output_dir,
        convergence,
    )

    write_text_report(
        output_dir,
        profile,
        diagnostics,
        gates,
    )

    make_plots(
        output_dir,
        profile,
        diagnostics,
        boost_rows,
    )

    pass_count = sum(
        gate["status"] == "PASS"
        for gate in gates
    )

    print("=" * 72)
    print("PARTICLE IDENTITY STATIC ANALYSIS")
    print("=" * 72)
    print(f"Selected NPZ index    = {profile.index}")
    print(f"omega                 = {profile.omega:.10f}")
    print(f"A_center              = {profile.A_center:.10f}")
    print(f"Rest energy M=E       = {diagnostics.E:.10f}")
    print(f"Noether charge Q      = {diagnostics.Q:.10f}")
    print(f"E/Q                   = {diagnostics.E_over_Q:.10f}")
    print(f"R_energy_rms          = {diagnostics.R_energy_rms:.10f}")
    print(f"R_charge_rms          = {diagnostics.R_charge_rms:.10f}")
    print(f"R50 / R90 / R99       = "
          f"{diagnostics.R50_energy:.6f} / "
          f"{diagnostics.R90_energy:.6f} / "
          f"{diagnostics.R99_energy:.6f}")
    print(f"Tail ratio            = {diagnostics.tail_ratio:.3e}")
    print(f"Node count            = {diagnostics.node_count}")
    print(f"ODE residual max      = {diagnostics.ode_residual_max:.3e}")
    print(f"Virial relative error = {diagnostics.virial_relative:.3e}")
    print(f"Laue relative error   = {diagnostics.laue_relative:.3e}")
    print(f"NPZ E relative error  = "
          f"{diagnostics.npz_E_relative_error:.3e}")
    print(f"NPZ Q relative error  = "
          f"{diagnostics.npz_Q_relative_error:.3e}")
    print("")
    print(f"PASS COUNT            = {pass_count}/{len(gates)}")
    print("")
    print(f"Output directory      = {output_dir.resolve()}")

    if pass_count == len(gates):
        print(
            "\nPRELIMINARY RESULT: "
            "STATIC PARTICLE CRITERIA PASSED."
        )
    else:
        print(
            "\nPRELIMINARY RESULT: "
            "SOME STATIC PARTICLE CRITERIA FAILED. "
            "Periksa particle_identity_gates.csv."
        )

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
