import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

# ==========================
# Global Setting
# ==========================

m = 1.0

omegas = np.linspace(0.20, 0.55, 8)

alpha_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
beta_list  = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5]

RMAX = 40.0
EPS = 1e-6

RTOL = 1e-6
ATOL = 1e-8
MAX_STEP = 0.05

A_MIN = 0.05
A_MAX = 5.0
N_SCAN = 80


# ==========================
# ODE Maker
# ==========================

def make_ode(omega, alpha, beta):
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
# Solve Profile
# ==========================

def solve_profile(omega, alpha, beta):
    ode = make_ode(omega, alpha, beta)

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
            xtol=1e-6
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

    V = (
        m**2 * psi**2
        - alpha * psi**4
        + beta * psi**6
    )

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

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "A": A_star,
        "r": r,
        "psi": psi,
        "dpsi": dpsi,
        "Q": Q,
        "H": H,
        "tail_abs": tail_abs,
        "max_abs": max_abs
    }


# ==========================
# Evaluate Parameter Pair
# ==========================

def evaluate_pair(alpha, beta):
    results = []

    for omega in omegas:
        res = solve_profile(omega, alpha, beta)

        if res is not None:
            results.append(res)

    if len(results) < 4:
        return None

    omega_vals = np.array([r["omega"] for r in results])
    Q_vals = np.array([r["Q"] for r in results])
    H_vals = np.array([r["H"] for r in results])
    tail_vals = np.array([r["tail_abs"] for r in results])
    amp_vals = np.array([r["max_abs"] for r in results])

    dQ = np.gradient(Q_vals, omega_vals)

    # ==========================
    # Scoring
    # ==========================
    # Semakin banyak dQ negatif, semakin baik untuk VK.
    # Semakin halus Q, semakin baik.
    # Semakin kecil tail, semakin baik.
    # Energi harus finite dan positif.
    # ==========================

    negative_fraction = np.mean(dQ < 0)

    if np.any(np.isnan(Q_vals)) or np.any(np.isnan(H_vals)):
        return None

    if np.any(H_vals <= 0):
        energy_positive = 0
    else:
        energy_positive = 1

    # ukuran "lompat" kurva Q
    q_jump = np.max(np.abs(np.diff(Q_vals))) / (np.mean(np.abs(Q_vals)) + 1e-12)

    tail_score = np.mean(tail_vals)

    amp_score = np.mean(amp_vals)

    score = (
        3.0 * negative_fraction
        + 1.0 * energy_positive
        - 1.5 * q_jump
        - 0.5 * tail_score
    )

    return {
        "alpha": alpha,
        "beta": beta,
        "results": results,
        "omega_vals": omega_vals,
        "Q_vals": Q_vals,
        "H_vals": H_vals,
        "dQ_vals": dQ,
        "negative_fraction": negative_fraction,
        "energy_positive": energy_positive,
        "q_jump": q_jump,
        "tail_score": tail_score,
        "amp_score": amp_score,
        "score": score
    }


# ==========================
# Main Scan
# ==========================

all_summaries = []
best = None

for alpha in alpha_list:
    for beta in beta_list:
        print(f"Uji alpha={alpha}, beta={beta}")

        summary = evaluate_pair(alpha, beta)

        if summary is None:
            print("  gagal / solusi terlalu sedikit")
            continue

        print(
            f"  score={summary['score']:.4f} | "
            f"neg_frac={summary['negative_fraction']:.2f} | "
            f"q_jump={summary['q_jump']:.4f} | "
            f"tail={summary['tail_score']:.4e}"
        )

        all_summaries.append(summary)

        if best is None or summary["score"] > best["score"]:
            best = summary


if best is None:
    print("Tidak ada parameter yang layak.")
    exit()


# ==========================
# Save Summary CSV
# ==========================

with open("scan_parameter_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "alpha",
        "beta",
        "score",
        "negative_fraction",
        "energy_positive",
        "q_jump",
        "tail_score",
        "amp_score",
        "n_success"
    ])

    for s in sorted(all_summaries, key=lambda x: x["score"], reverse=True):
        writer.writerow([
            s["alpha"],
            s["beta"],
            s["score"],
            s["negative_fraction"],
            s["energy_positive"],
            s["q_jump"],
            s["tail_score"],
            s["amp_score"],
            len(s["results"])
        ])

print("\nParameter terbaik:")
print(f"alpha = {best['alpha']}")
print(f"beta  = {best['beta']}")
print(f"score = {best['score']}")
print("CSV disimpan: scan_parameter_summary.csv")


# ==========================
# Plot Best Results
# ==========================

omega_vals = best["omega_vals"]
Q_vals = best["Q_vals"]
H_vals = best["H_vals"]
dQ_vals = best["dQ_vals"]

# 1. Profil
plt.figure(figsize=(8, 5))

for res in best["results"]:
    plt.plot(
        res["r"],
        res["psi"],
        label=f"omega={res['omega']:.2f}"
    )

plt.xlabel("r")
plt.ylabel("psi(r)")
plt.title(f"Best Profiles: alpha={best['alpha']}, beta={best['beta']}")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("best_profiles.png", dpi=300)
plt.close()

# 2. Q omega
plt.figure(figsize=(8, 5))
plt.plot(omega_vals, Q_vals, "o-", linewidth=2)
plt.xlabel("omega")
plt.ylabel("Q")
plt.title(f"Best VK Curve Q(omega): alpha={best['alpha']}, beta={best['beta']}")
plt.grid(True)
plt.tight_layout()
plt.savefig("best_vk_Q.png", dpi=300)
plt.close()

# 3. dQ/domega
plt.figure(figsize=(8, 5))
plt.plot(omega_vals, dQ_vals, "o-", linewidth=2)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("omega")
plt.ylabel("dQ/domega")
plt.title(f"Best VK Gradient: alpha={best['alpha']}, beta={best['beta']}")
plt.grid(True)
plt.tight_layout()
plt.savefig("best_vk_gradient.png", dpi=300)
plt.close()

# 4. H omega
plt.figure(figsize=(8, 5))
plt.plot(omega_vals, H_vals, "o-", linewidth=2)
plt.xlabel("omega")
plt.ylabel("H")
plt.title(f"Best Energy Curve H(omega): alpha={best['alpha']}, beta={best['beta']}")
plt.grid(True)
plt.tight_layout()
plt.savefig("best_energy_H.png", dpi=300)
plt.close()

print("Grafik disimpan:")
print("- best_profiles.png")
print("- best_vk_Q.png")
print("- best_vk_gradient.png")
print("- best_energy_H.png")
