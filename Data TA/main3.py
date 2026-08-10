import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib.ticker import FormatStrFormatter

FIG_SIZE = (16, 8)
LINE_WIDTH = 2.5
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13
INFO_SIZE = 13
X_TICKS = [0, 20, 40, 60]

energy_files = sorted(glob.glob("Normal_Energy.xlsx"))
power_files = sorted(glob.glob("Nomal_Power.xlsx"))

print("Energy files found:", energy_files)
print("Power files found:", power_files)


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

    # Format 3: kadang header kolom pertamanya bukan persis sama, tapi tetap 1 kolom isi CSV
    if len(cols) == 1:
        first_col = df.iloc[:, 0].astype(str).str.split(",", expand=True)
        first_col.columns = ["entity", "state", "time"]
        return first_col.copy()

    return df


def load_data(files):
    all_df = []
    for file in files:
        df = pd.read_excel(file)
        df = normalize_columns(df)

        df = df[~df["state"].astype(str).str.lower().isin(["unknown", "unavailable"])].copy()

        df["state"] = pd.to_numeric(df["state"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        df = df.dropna(subset=["state", "time"])
        all_df.append(df)

    if not all_df:
        return pd.DataFrame(columns=["entity", "state", "time"])

    df = pd.concat(all_df)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def elapsed_minutes(df):
    return (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 60.0


def apply_readable_axes(ymin, ymax, y_format):
    plt.xlim(0, 60)
    plt.xticks(X_TICKS, fontsize=TICK_SIZE)
    plt.yticks(np.linspace(ymin, ymax, 5), fontsize=TICK_SIZE)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(y_format))
    plt.tick_params(axis="both", labelsize=TICK_SIZE)


# LOAD DATA
energy = load_data(energy_files)
power = load_data(power_files)

if energy.empty or power.empty:
    print("Error: Energy atau Power data kosong!")
    exit()

# ENERGY CALCULATION
energy_start = energy["state"].iloc[0]
energy_end = energy["state"].iloc[-1]
energy_min = energy["state"].min()
energy_max = energy["state"].max()
total_energy = energy_end - energy_start

# POWER CALCULATION
power_avg = power["state"].mean()
power_min = power["state"].min()
power_max = power["state"].max()

print("\n=== ENERGY ===")
print(f"Start Energy : {energy_start:.2f} kWh")
print(f"End Energy   : {energy_end:.2f} kWh")
print(f"Min Energy   : {energy_min:.2f} kWh")
print(f"Max Energy   : {energy_max:.2f} kWh")
print(f"Total Energy : {total_energy:.2f} kWh")

print("\n=== POWER ===")
print(f"Average Power : {power_avg:.2f} W")
print(f"Min Power     : {power_min:.2f} W")
print(f"Max Power     : {power_max:.2f} W")

# PLOT ENERGY (DENGAN BATAS SUMBU Y DARI MIN SAMPAI MIN + 1)
energy_minutes = elapsed_minutes(energy)

plt.figure(figsize=FIG_SIZE)
plt.plot(energy_minutes, energy["state"], linewidth=LINE_WIDTH, label="Energy (kWh)")
plt.title("Energy Consumption", fontsize=TITLE_SIZE, fontweight="bold")
plt.xlabel("Time (minutes)", fontsize=LABEL_SIZE)
plt.ylabel("Energy (kWh)", fontsize=LABEL_SIZE)

# --- INI CODE YANG DIUBAH UNTUK SKALA ENERGY ---
y_min = energy_min - 0.1
y_max = energy_min + 1.0
plt.ylim(y_min, y_max)
apply_readable_axes(y_min, y_max, "%.2f")
# -----------------------------------------------

plt.grid(True, linestyle="--", alpha=0.5)

energy_info = (
    f"Start : {energy_start:.2f} kWh\n"
    f"End   : {energy_end:.2f} kWh\n"
    f"Total : {total_energy:.2f} kWh"
)

plt.gcf().text(
    0.98, 0.04,
    energy_info,
    ha="right",
    va="bottom",
    fontsize=INFO_SIZE,
    bbox=dict(boxstyle="round", facecolor="white")
)

plt.legend(loc="lower left", fontsize=LEGEND_SIZE)
plt.tight_layout()
plt.savefig("energy_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# PLOT POWER (DENGAN BATAS SUMBU Y 0 - 900)
power_minutes = elapsed_minutes(power)

plt.figure(figsize=FIG_SIZE)
plt.plot(power_minutes, power["state"], linewidth=LINE_WIDTH, color="orange", label="Power (W)")
plt.title("Power Usage", fontsize=TITLE_SIZE, fontweight="bold")
plt.xlabel("Time (minutes)", fontsize=LABEL_SIZE)
plt.ylabel("Power (W)", fontsize=LABEL_SIZE)
y_min = 0
y_max = 900
plt.ylim(y_min, y_max)
apply_readable_axes(y_min, y_max, "%.0f")
plt.grid(True, linestyle="--", alpha=0.5)

power_info = (
    f"Avg : {power_avg:.2f} W\n"
    f"Min : {power_min:.2f} W\n"
    f"Max : {power_max:.2f} W"
)

plt.gcf().text(
    0.98, 0.04,
    power_info,
    ha="right",
    va="bottom",
    fontsize=INFO_SIZE,
    bbox=dict(boxstyle="round", facecolor="white")
)

plt.legend(loc="lower left", fontsize=LEGEND_SIZE)
plt.tight_layout()
plt.savefig("power_plot.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nDONE ✅\nSaved:\n- energy_plot.png\n- power_plot.png")
