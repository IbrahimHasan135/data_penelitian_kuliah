import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.ticker import FormatStrFormatter

FIG_SIZE = (16, 8)
LINE_WIDTH = 2.5
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 13
X_TICKS = [0, 20, 40, 60]

files = ["T_N.xlsx", "T_0.xlsx", "T_1.xlsx", "T_2.xlsx"]

for file in files:
    # Memastikan file ada sebelum diproses agar script tidak crash
    if not Path(file).exists():
        print(f"File {file} tidak ditemukan, skip.")
        continue

    df = pd.read_excel(file)

    # Ambil kolom sesuai format data
    waktu = pd.to_datetime(df.iloc[:, 1])  # kolom waktu
    suhu = df.iloc[:, 2]                  # kolom suhu
    menit = (waktu - waktu.iloc[0]).dt.total_seconds() / 60.0

    plt.figure(figsize=FIG_SIZE)

    plt.plot(menit, suhu, linewidth=LINE_WIDTH, color="crimson")

    plt.title(f"Temperature - {Path(file).stem}", fontsize=TITLE_SIZE, fontweight="bold")
    plt.xlabel("Time (minutes)", fontsize=LABEL_SIZE)
    plt.ylabel("Temperature (°C)", fontsize=LABEL_SIZE)

    # --- KODE UNTUK SKALA SUHU 23 - 26 ---
    y_min = 21
    y_max = 26
    plt.ylim(y_min, y_max)
    # -------------------------------------

    # Format sumbu dibuat sederhana agar mudah dibaca di skripsi.
    plt.xlim(0, 60)
    plt.xticks(X_TICKS, fontsize=TICK_SIZE)
    plt.yticks(np.linspace(y_min, y_max, 5), fontsize=TICK_SIZE)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.tick_params(axis="both", labelsize=TICK_SIZE)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Menyimpan hasil grafik suhu
    output_name = f"{Path(file).stem}_temp.png"
    plt.savefig(output_name, dpi=300)
    print(f"Saved: {output_name}")

    # Menutup plot untuk menghemat RAM dan mencegah timbulnya Warning di terminal
    plt.close()
