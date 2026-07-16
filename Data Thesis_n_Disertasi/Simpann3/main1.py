import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# ==========================
# Parameter terbaik dari scan
# ==========================

m = 1.0
alpha = 2.5
beta = 0.5

omegas = np.linspace(0.20, 0.55, 8)

RMAX_LIST = [40.0, 60.0, 80.0, 100.0]

EPS = 1e-6
RTOL = 1e-7
ATOL = 1e-9
MAX_STEP = 0.05

A_MIN = 0.05
A_MAX = 5.0
N_SCAN = 120


# ==========================
# ODE
# ==========================

def make_ode(omega):
    Omega_sq = m**2 - omega**2

    def ode(r, y):
        psi, dpsi = y

        d2psi = (
            -(2.0 / r) * dpsi
            + Omega_sq * psi
            - 2.0 * alpha * psi**3
            + 3.0 * beta * psi**5
        )

        return [dpsi, d2psi]

    return ode


# ==========================
# Shooting Solver
# ==========================

def solve_profile(omega, RMAX):
    ode = make_ode(omega)

    def shoot(A):
        sol = solve_ivp(
            ode,
            [EPS, RMAX],
            [A, 0.0],
            rtol=RTOL,
            atol=ATOL,
            max_step=MAX_STEP
        )

        if not sol.success:
            return np.nan

        return sol.y[0, -1]

    As = np.linspace(A_MIN, A_MAX, N_SCAN)
    vals = []

    for A in As:
        try:
            vals.append(shoot(A))
        except Exception:
            vals.append(np.nan)

    bracket = None

    for i in range(len(As) - 1):
        if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
            continue

        if vals[i] * vals[i + 1] < 0:
            bracket = [As[i], As[i + 1]]
            break

    if bracket is None:
        return None

    try:
        root = root_scalar(
            shoot,
            bracket=bracket,
            xtol=1e-7
        )
    except Exception:
        return None

    if not root.converged:
        return None

    A_star = root.root

    sol = solve_ivp(
        ode,
        [EPS, RMAX],
        [A_star, 0.0],
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP
    )

    if not sol.success:
        return None

    r = sol.t
    psi = sol.y[0]
    dpsi = sol.y[1]

    V = m**2 * psi**2 - alpha * psi**4 + beta * psi**6

    Q = 4.0 * np.pi * simpson(
        psi**2 * r**2,
        x=r
    )

    H = 4.0 * np.pi * simpson(
        (omega**2 * psi**2 + dpsi**2 + V) * r**2,
        x=r
    )

    tail_abs = abs(psi[-1])
    max_abs = np.max(np.abs(psi))

    # cek apakah tail besar dibanding amplitudo pusat
    tail_ratio = tail_abs / (max_abs + 1e-12)

    return {
        "omega": omega,
        "RMAX": RMAX,
        "A": A_star,
        "r": r,
        "psi": psi,
        "dpsi": dpsi,
        "Q": Q,
        "H": H,
        "tail_abs": tail_abs,
        "tail_ratio": tail_ratio,
        "max_abs": max_abs
    }


# ==========================
# Main Run
# ==========================

all_results = []

for RMAX in RMAX_LIST:
    print(f"\n===== RMAX = {RMAX} =====")

    for omega in omegas:
        print(f"omega={omega:.3f}", end=" | ")

        res = solve_profile(omega, RMAX)

        if res is None:
            print("gagal")
            continue

        print(
            f"A={res['A']:.6f} | "
            f"Q={res['Q']:.3f} | "
            f"H={res['H']:.3f} | "
            f"tail_ratio={res['tail_ratio']:.3e}"
        )

        all_results.append(res)


if len(all_results) == 0:
    print("Tidak ada solusi ditemukan.")
    exit()


# ==========================
# Save CSV
# ==========================

with open("boundary_test_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "RMAX",
        "omega",
        "A_center",
        "Q",
        "H",
        "tail_abs",
        "tail_ratio",
        "max_abs"
    ])

    for res in all_results:
        writer.writerow([
            res["RMAX"],
            res["omega"],
            res["A"],
            res["Q"],
            res["H"],
            res["tail_abs"],
            res["tail_ratio"],
            res["max_abs"]
        ])

print("\nCSV disimpan: boundary_test_summary.csv")


# ==========================
# Plot Profil Per RMAX
# ==========================

for RMAX in RMAX_LIST:
    subset = [res for res in all_results if res["RMAX"] == RMAX]

    if len(subset) == 0:
        continue

    plt.figure(figsize=(9, 5))

    for res in subset:
        plt.plot(
            res["r"],
            res["psi"],
            label=f"omega={res['omega']:.2f}"
        )

    plt.xlabel("r")
    plt.ylabel("psi(r)")
    plt.title(f"Profil Soliton - RMAX={RMAX}, alpha={alpha}, beta={beta}")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.tight_layout()
    filename = f"profiles_RMAX_{int(RMAX)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Grafik disimpan: {filename}")


# ==========================
# Plot Q dan H untuk tiap RMAX
# ==========================

for quantity, ylabel, filename_base in [
    ("Q", "Q", "Q_vs_omega"),
    ("H", "H", "H_vs_omega"),
    ("tail_ratio", "tail_ratio", "tail_ratio_vs_omega")
]:
    plt.figure(figsize=(9, 5))

    for RMAX in RMAX_LIST:
        subset = [res for res in all_results if res["RMAX"] == RMAX]

        if len(subset) == 0:
            continue

        subset = sorted(subset, key=lambda x: x["omega"])

        omega_vals = np.array([res["omega"] for res in subset])
        vals = np.array([res[quantity] for res in subset])

        plt.plot(
            omega_vals,
            vals,
            marker="o",
            linewidth=2,
            label=f"RMAX={RMAX}"
        )

    plt.xlabel("omega")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs omega untuk variasi RMAX")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    filename = f"{filename_base}_RMAX_compare.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Grafik disimpan: {filename}")


# ==========================
# Plot log-tail untuk satu omega contoh
# ==========================

target_omega = 0.40

plt.figure(figsize=(9, 5))

for RMAX in RMAX_LIST:
    candidates = [
        res for res in all_results
        if res["RMAX"] == RMAX and abs(res["omega"] - target_omega) < 1e-9
    ]

    if len(candidates) == 0:
        continue

    res = candidates[0]

    r = res["r"]
    psi_abs = np.abs(res["psi"]) + 1e-20

    plt.plot(
        r,
        np.log(psi_abs),
        label=f"RMAX={RMAX}"
    )

plt.xlabel("r")
plt.ylabel("log(|psi(r)|)")
plt.title(f"Log-tail check untuk omega={target_omega}")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("log_tail_check.png", dpi=300)
plt.close()

print("Grafik disimpan: log_tail_check.png")


# ==========================
# Ringkasan final
# ==========================

print("\nSelesai.")
print("Output utama:")
print("- boundary_test_summary.csv")
print("- profiles_RMAX_40.png / 60 / 80 / 100")
print("- Q_vs_omega_RMAX_compare.png")
print("- H_vs_omega_RMAX_compare.png")
print("- tail_ratio_vs_omega_RMAX_compare.png")
print("- log_tail_check.png")
