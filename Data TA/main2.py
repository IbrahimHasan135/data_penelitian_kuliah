import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Mencari semua file E_*.xlsx dan P_*.xlsx yang ada di folder
energy_files = sorted(glob.glob("E_*.xlsx"))
power_files = sorted(glob.glob("P_*.xlsx"))

print("Energy files to process:", energy_files)
print("Power files to process:", power_files)


def normalize_columns(df):
    cols = [str(c).strip() for c in df.columns]

    # Format 1: Excel normal, 3 kolom
    if {"entity_id", "state", "last_changed"}.issubset(set(cols)):
        df = df.rename(columns={
            "entity_id": "entity",
            "last_changed": "time"
        })
        return df[["entity", "state", "time"]].copy()

    # Format 2: Semua data nyatu di 1 kolom
    if len(cols) == 1 and cols[0] == "entity_id,state,last_changed":
        first_col = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        first_col.columns = ["entity", "state", "time"]
        return first_col.copy()

    # Format 3: Jika header kolom pertamanya sedikit berbeda
    if len(cols) == 1:
        first_col = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        first_col.columns = ["entity", "state", "time"]
        return first_col.copy()

    return df


def process_single_file(file_path):
    df = pd.read_excel(file_path)
    df = normalize_columns(df)

    # Membersihkan data dari status unknown/unavailable
    df = df[~df["state"].astype(str).str.lower().isin(["unknown", "unavailable"])].copy()

    df["state"] = pd.to_numeric(df["state"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.dropna(subset=["state", "time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ==========================================
# 1. PROSES DAN PLOT DATA ENERGY (E_0, E_1, E_2)
# ==========================================
for file in energy_files:
    energy = process_single_file(file)
    if energy.empty:
        print(f"Skipping {file} karena datanya kosong.")
        continue

    # Kalkulasi data Energy
    energy_start = energy["state"].iloc[0]
    energy_end = energy["state"].iloc[-1]
    energy_min = energy["state"].min()
    total_energy = energy_end - energy_start

    # Plot grafik
    plt.figure(figsize=(14, 6))
    plt.plot(energy["time"], energy["state"], linewidth=1.5, label="Energy (kWh)")
    plt.title(f"Energy Consumption - {Path(file).stem}")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")

    # Skala dinamis Energy sesuai permintaanmu sebelumnya
    plt.ylim(energy_min - 0.1, energy_min + 1.0)

    plt.grid(True, linestyle="--", alpha=0.5)

    energy_info = (
        f"Start : {energy_start:.2f} kWh\n"
        f"End   : {energy_end:.2f} kWh\n"
        f"Total : {total_energy:.2f} kWh"
    )
    plt.gcf().text(
        0.98, 0.02, energy_info,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white")
    )

    plt.legend(loc="lower left")
    plt.tight_layout()

    # Menyimpan file sesuai nama asli file datanya (misal: E_0.png)
    output_name = f"{Path(file).stem}.png"
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_name}")


# ==========================================
# 2. PROSES DAN PLOT DATA POWER (P_0, P_1, P_2)
# ==========================================
for file in power_files:
    power = process_single_file(file)
    if power.empty:
        print(f"Skipping {file} karena datanya kosong.")
        continue

    # Kalkulasi data Power
    power_avg = power["state"].mean()
    power_min = power["state"].min()
    power_max = power["state"].max()

    # Plot grafik
    plt.figure(figsize=(14, 6))
    plt.plot(power["time"], power["state"], linewidth=1.5, color="orange", label="Power (W)")
    plt.title(f"Power Usage - {Path(file).stem}")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")

    # Mengunci skala y Power dari 0 sampai 900
    plt.ylim(0, 900)

    plt.grid(True, linestyle="--", alpha=0.5)

    power_info = (
        f"Avg : {power_avg:.2f} W\n"
        f"Min : {power_min:.2f} W\n"
        f"Max : {power_max:.2f} W"
    )
    plt.gcf().text(
        0.98, 0.02, power_info,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white")
    )

    plt.legend(loc="lower left")
    plt.tight_layout()

    # Menyimpan file sesuai nama asli file datanya (misal: P_0.png)
    output_name = f"{Path(file).stem}.png"
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_name}")

print("\nDONE ✅ Semua file data E dan P telah diproses terpisah!")
