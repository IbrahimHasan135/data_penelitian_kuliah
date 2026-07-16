import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

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

    plt.figure(figsize=(12, 5))

    plt.plot(waktu, suhu, linewidth=2, color="crimson")

    plt.title(f"Temperature - {Path(file).stem}")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    # --- KODE UNTUK SKALA SUHU 23 - 26 ---
    plt.ylim(21, 26)
    # -------------------------------------

    # FORMAT JAM AJA
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # NGGAK TERLALU DETAIL (interval tiap 5 menit)
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Menyimpan hasil grafik suhu
    output_name = f"{Path(file).stem}_temp.png"
    plt.savefig(output_name, dpi=300)
    print(f"Saved: {output_name}")

    # Menutup plot untuk menghemat RAM dan mencegah timbulnya Warning di terminal
    plt.close()
