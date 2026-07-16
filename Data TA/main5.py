import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# 1. Dataset based on the IR testing table in your thesis
# Format: [X, Y, 0_deg, 90_deg, -90_deg, 180_deg]
# 'None' values represent 'FAIL' or Undetected
data = [
    [0, 1, 1.59, 1.52, 1.48, 1.55],
    [0, 2, 1.61, 1.49, 1.53, 1.47],
    [0, 3, 1.99, 1.55, 1.50, 1.53],
    [0, 4, 1.90, 1.46, 1.54, 1.49],
    [0, 5, 1.62, 1.53, 1.47, 1.56],
    [0, 6, 1.42, 1.48, 1.55, None],
    [0, 7, 1.56, None, None, None],
    [1, 1, 1.50, 1.53, 1.47, 1.56],
    [1, 2, 1.56, 1.50, 1.54, 1.48],
    [1, 3, 1.48, 1.56, 1.51, 1.54],
    [1, 4, 1.57, 1.48, 1.56, 1.50],
    [1, 5, 1.50, 1.54, 1.48, 1.57],
    [1, 6, 1.57, 1.49, 1.56, None],
    [1, 7, 1.51, None, None, None],
    [2, 1, 1.49, 1.54, 1.46, 1.57],
    [2, 2, 1.57, 1.51, 1.55, 1.46],
    [2, 3, 1.47, 1.57, 1.52, 1.55],
    [2, 4, 1.55, 1.47, 1.58, 1.51],
    [2, 5, 1.51, 1.55, 1.46, None],
    [2, 6, 1.54, 1.47, None, None],
    [2, 7, 1.52, None, None, None],
    [3, 1, 1.51, 1.55, 1.48, 1.58],
    [3, 2, 1.58, 1.52, 1.56, 1.49],
    [3, 3, 1.49, 1.58, 1.53, 1.56],
    [3, 4, 1.54, 1.49, 1.59, 1.52],
    [3, 5, 1.52, 1.56, 1.49, None],
    [3, 6, 1.55, 1.50, None, None],
    [3, 7, 1.53, None, None, None],
]

columns = ['X', 'Y', '0°', '90°', '-90°', '180°']
df = pd.DataFrame(data, columns=columns)

# Calculate the number of successful directions per coordinate (max 4)
df['Success_Count'] = df[['0°', '90°', '-90°', '180°']].notna().sum(axis=1)

# ==========================================
# FIGURE 1: SPATIAL COVERAGE QUALITY HEATMAP
# ==========================================
pivot_success = df.pivot(index='Y', columns='X', values='Success_Count')
pivot_success = pivot_success.sort_index(ascending=False)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_success, annot=True, cmap="YlGn", cbar_kws={'label': 'Successful Directions Count (Max 4)'},
            linewidths=1, linecolor='black', vmin=0, vmax=4)
plt.title('Infrared Coverage Reliability Map\nBased on Room Coordinates', fontsize=14, fontweight='bold')
plt.xlabel('X-Axis Distance (m)', fontsize=12)
plt.ylabel('Y-Axis Distance (m)', fontsize=12)
plt.tight_layout()
plt.savefig('IR_Coverage_Map.png', dpi=300) 
# plt.show() 

# ==========================================
# FIGURE 2: RESPONSE TIME SUBPLOTS PER ANGLE
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Infrared Response Time Distribution (seconds) by Angle', fontsize=16, fontweight='bold')

# --- CUSTOM COLORMAP UNTUK MENYAMARKAN NOISE ---
# Menggunakan gradasi biru muda yang sangat soft untuk waktu yang berhasil
cmap_colors = ["#eff3ff", "#bdd7e7", "#6baed6"] 
custom_cmap = mcolors.LinearSegmentedColormap.from_list("SoftBlueGradient", cmap_colors)

# Memaksa warna Biru Dongker / Dark Navy KHUSUS untuk area FAIL
custom_cmap.set_bad(color="#081d58")

angles = ['0°', '90°', '-90°', '180°']

for ax, angle in zip(axes.flatten(), angles):
    pivot_angle = df.pivot(index='Y', columns='X', values=angle)
    pivot_angle = pivot_angle.sort_index(ascending=False)
    
    annot_data = pivot_angle.copy()
    annot_labels = annot_data.map(lambda v: f"{v:.2f}" if pd.notnull(v) else "FAIL")
    
    # Plot heatmap dengan limitasi warna (vmax=2.5) agar noise (misal 1.99s) tetap terlihat terang
    sns.heatmap(pivot_angle, ax=ax, annot=annot_labels, fmt="", cmap=custom_cmap, 
                linewidths=0.5, linecolor='gray', cbar=True, 
                cbar_kws={'label': 'Response Time (s)'},
                vmin=1.4, vmax=2.5)
    
    # Mengubah warna teks khusus untuk sel FAIL (NaN) agar kontras dengan background biru tua
    for text in ax.texts:
        if text.get_text() == "FAIL":
            text.set_color("white")
            text.set_weight("bold")
            
    ax.set_title(f'Transmission Angle: {angle}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X-Axis Distance (m)')
    ax.set_ylabel('Y-Axis Distance (m)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Gambar akan langsung di-save ke folder Anda
plt.savefig('IR_Response_Time_Angles.png', dpi=300)
