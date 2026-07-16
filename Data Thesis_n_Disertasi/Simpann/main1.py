import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar
import csv

# ==========================
# Parameter Model
# ==========================

m = 1.0
alpha = 2.0
beta = 0.5

RMAX = 30.0
EPS = 1e-6

omegas = np.linspace(0.15, 0.9, 16)

# ==========================
# Fungsi ODE
# ==========================

def make_ode(omega):
    Omega_sq = m**2 - omega**2

    def ode(r, y):
        psi, dpsi = y

        d2psi = (
            -(2.0 / r) * dpsi
            + Omega_sq * psi
            - 2 * alpha * psi**3
            + 3 * beta * psi**5
        )

        return [dpsi, d2psi]

    return ode


def solve_profile(omega, A_min=0.1, A_max=4.0, n_scan=200):
    ode = make_ode(omega)

    def shoot(A):
        sol = solve_ivp(
            ode,
            [EPS, RMAX],
            [A, 0.0],
            rtol=1e-8,
            atol=1e-10,
            max_step=0.02
        )
        return sol.y[0, -1]

    As = np.linspace(A_min, A_max, n_scan)
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

    root = root_scalar(
        shoot,
        bracket=bracket,
        xtol=1e-8
    )

    A_star = root.root

    sol = solve_ivp(
        ode,
        [EPS, RMAX],
        [A_star, 0.0],
        rtol=1e-8,
        atol=1e-10,
        max_step=0.02
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

    Q = 4 * np.pi * simpson(
        psi**2 * r**2,
        x=r
    )

    H = 4 * np.pi * simpson(
        (omega**2 * psi**2 + dpsi**2 + V) * r**2,
        x=r
    )

    return {
        "omega": omega,
        "A": A_star,
        "r": r,
        "psi": psi,
        "dpsi": dpsi,
        "Q": Q,
        "H": H
    }


# ==========================
# Jalankan Scan Omega
# ==========================

results = []

for omega in omegas:
    print(f"Mencari solusi untuk omega = {omega:.3f}")

    res = solve_profile(omega)

    if res is None:
        print(f"  gagal")
    else:
        print(
            f"  berhasil | A = {res['A']:.6f} | "
            f"Q = {res['Q']:.6f} | H = {res['H']:.6f}"
        )
        results.append(res)


if len(results) == 0:
    print("Tidak ada solusi ditemukan.")
    exit()


# ==========================
# Simpan CSV
# ==========================

with open("soliton_data.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["omega", "A_center", "Q", "H"])

    for res in results:
        writer.writerow([
            res["omega"],
            res["A"],
            res["Q"],
            res["H"]
        ])

print("Data disimpan ke soliton_data.csv")


# ==========================
# Plot 1: Profil Soliton
# ==========================

plt.figure(figsize=(8, 5))

for res in results:
    plt.plot(
        res["r"],
        res["psi"],
        label=f"omega={res['omega']:.2f}"
    )

plt.xlabel("r")
plt.ylabel("psi(r)")
plt.title("Profil Soliton untuk Berbagai Omega")
plt.grid(True)
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig("profil_soliton.png", dpi=300)
plt.close()

print("Grafik disimpan: profil_soliton.png")


# ==========================
# Plot 2: Kurva VK Q(omega)
# ==========================

omega_vals = np.array([res["omega"] for res in results])
Q_vals = np.array([res["Q"] for res in results])
H_vals = np.array([res["H"] for res in results])

dQ_domega = np.gradient(Q_vals, omega_vals)

plt.figure(figsize=(8, 5))

plt.plot(
    omega_vals,
    Q_vals,
    marker="o",
    linewidth=2,
    label="Q(omega)"
)

plt.xlabel("omega")
plt.ylabel("Q")
plt.title("Kurva VK: Q(omega)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("kurva_vk_Q.png", dpi=300)
plt.close()

print("Grafik disimpan: kurva_vk_Q.png")


# ==========================
# Plot 3: Gradien VK dQ/domega
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    omega_vals,
    dQ_domega,
    marker="o",
    linewidth=2,
    label="dQ/domega"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("omega")
plt.ylabel("dQ/domega")
plt.title("Uji Stabilitas VK: dQ/domega")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("kurva_vk_gradient.png", dpi=300)
plt.close()

print("Grafik disimpan: kurva_vk_gradient.png")


# ==========================
# Plot 4: Energi H(omega)
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    omega_vals,
    H_vals,
    marker="o",
    linewidth=2,
    label="H(omega)"
)

plt.xlabel("omega")
plt.ylabel("H")
plt.title("Finite Energy Check: H(omega)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("kurva_energi_H.png", dpi=300)
plt.close()

print("Grafik disimpan: kurva_energi_H.png")


# ==========================
# Ringkasan Numerik
# ==========================

print("\nRingkasan:")
print("omega\t\tA\t\tQ\t\tH\t\tdQ/domega")

for omega, res, grad in zip(omega_vals, results, dQ_domega):
    print(
        f"{omega:.3f}\t\t"
        f"{res['A']:.6f}\t"
        f"{res['Q']:.6f}\t"
        f"{res['H']:.6f}\t"
        f"{grad:.6f}"
    )
