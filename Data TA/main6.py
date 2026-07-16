import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# 1. Dataset based on PIR testing Table 4.2.1.2 in your thesis
# Format: [X (Lateral), Y (Forward), 0_deg_time, 90_deg_time]
# 'None' represents 'FAIL' or Undetected (Blind Spot)
data = [
    [0, 1, 1.55, None],
    [0, 2, 2.47, None],
    [0, 3, 3.15, None],
    [1, 1, 2.56, 4.04],
    [1, 2, 3.31, 10.23],
    [1, 3, 18.20, None],
    [2, 1, None, None],
    [2, 2, None, 10.76],
    [2, 3, None, None]
]

columns = ['X', 'Y', '0°', '90°']
df = pd.DataFrame(data, columns=columns)

# ==========================================
# FIGURE: PIR SENSOR RESPONSE TIME HEATMAP
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PIR Sensor Response Time Map (seconds) by Orientation', fontsize=16, fontweight='bold')

# --- PERBAIKAN WARNA KUSTOM ---
# Mendefinisikan warna: Hijau (Cepat) -> Kuning (Sedang) -> Oranye (Lambat)
cmap_colors = ["#66bd63", "#ffffbf", "#fdae61"] 
custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreenYellowOrange", cmap_colors)

# Memaksa warna merah pekat KHUSUS untuk sel yang tidak terdeteksi (FAIL / NaN)
custom_cmap.set_bad(color="#d7191c") 

angles = ['0°', '90°']

for ax, angle in zip(axes, angles):
    # Create pivot table for the specific angle
    pivot_angle = df.pivot(index='Y', columns='X', values=angle)
    pivot_angle = pivot_angle.sort_index(ascending=False) # Invert Y-axis so Y=1 is at the bottom
    
    # Custom annotations: format to 2 decimals, or "FAIL" if NaN
    annot_data = pivot_angle.copy()
    annot_labels = annot_data.map(lambda v: f"{v:.2f}" if pd.notnull(v) else "FAIL")
    
    # Plot heatmap
    # Tambahan vmin=0 dan vmax=20 agar waktu 18.20s maksimal berwarna oranye, bukan merah
    sns.heatmap(pivot_angle, ax=ax, annot=annot_labels, fmt="", cmap=custom_cmap, 
                linewidths=0.5, linecolor='gray', cbar=True, 
                cbar_kws={'label': 'Response Time (s)'},
                vmin=0, vmax=20)
    
    ax.set_title(f'Sensor Orientation: {angle}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lateral Distance - X Axis (m)', fontsize=12)
    ax.set_ylabel('Forward Distance - Y Axis (m)', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Gambar akan langsung di-save ke folder Anda
plt.savefig('PIR_Response_Time_Map.png', dpi=300) 

# plt.show() # Diblok agar tidak muncul warning di terminal Linux
