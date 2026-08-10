import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
df['Detected_Count'] = df[['0°', '90°']].notna().sum(axis=1)


def plot_matrix_heatmap(ax, matrix, annot_labels, cmap, vmin, vmax,
                        cbar_label, linecolor="gray", linewidth=0.5,
                        annot_size=11):
    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    image = ax.imshow(masked_values, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color=linecolor, linestyle="-",
            linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(len(matrix.index)):
        for col_idx in range(len(matrix.columns)):
            label = str(annot_labels.iloc[row_idx, col_idx])
            text_color = "white" if label == "FAIL" else "black"
            weight = "bold" if label == "FAIL" else "normal"
            ax.text(col_idx, row_idx, label, ha="center", va="center",
                    fontsize=annot_size, color=text_color,
                    fontweight=weight)

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    return image


def mirrored_axis_points(x, y):
    if x == 0:
        return [(0, y)]
    return [(-x, y), (x, y)]


def transform_pir_points(x, y, angle_label):
    """
    Mengubah titik uji lokal menjadi titik tampak atas PIR.
    Untuk PIR, kolom 90 derajat tetap dipakai sebagai tambahan area depan,
    bukan diputar ke belakang. Karena itu koordinatnya sama dengan area depan
    dan hanya dipakai jika titik 0 derajat pada lokasi itu tidak terdeteksi.
    """
    if angle_label in {'0°', '90°'}:
        return mirrored_axis_points(x, y)
    return []


def build_top_view_points(source_df, value_columns):
    """
    Membuat titik coverage pandang atas PIR.
    Data 0 derajat dipakai sebagai data utama. Data 90 derajat hanya dipakai
    untuk melengkapi titik yang gagal pada 0 derajat, sehingga titik utama
    tidak tertimpa oleh pengujian sudut 90 derajat.
    """
    points = []
    for _, row in source_df.iterrows():
        primary_time = row['0°']
        primary_detected = pd.notnull(primary_time)

        for point_x, point_y in transform_pir_points(row['X'], row['Y'], '0°'):
            points.append({
                'X': point_x,
                'Y': point_y,
                'Detected': primary_detected,
                'Response_Time': primary_time if primary_detected else np.nan,
                'Direction': '0° primary',
                'Radius': np.hypot(point_x, point_y),
                'Angle': np.arctan2(point_y, point_x),
            })

        supplement_time = row['90°']
        if not primary_detected and pd.notnull(supplement_time):
            for point_x, point_y in transform_pir_points(row['X'], row['Y'], '90°'):
                points.append({
                    'X': point_x,
                    'Y': point_y,
                    'Detected': True,
                    'Response_Time': supplement_time,
                    'Direction': '90° supplement',
                    'Radius': np.hypot(point_x, point_y),
                    'Angle': np.arctan2(point_y, point_x),
                })

    return pd.DataFrame(points)


def coverage_boundary(points_df):
    detected = points_df[points_df['Detected']].copy()
    if detected.empty:
        return pd.DataFrame(columns=['X', 'Y'])

    points = [(0.0, 0.0)]
    points.extend(detected[['X', 'Y']].drop_duplicates().itertuples(index=False, name=None))
    points = sorted(set(points))

    def cross(origin, point_a, point_b):
        return ((point_a[0] - origin[0]) * (point_b[1] - origin[1])
                - (point_a[1] - origin[1]) * (point_b[0] - origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return pd.DataFrame(hull, columns=['X', 'Y'])


def plot_top_view_coverage(points_df, title, output_name, max_range):
    boundary = coverage_boundary(points_df)

    fig, ax = plt.subplots(figsize=(10, 10))

    for radius in range(1, max_range + 1):
        circle = plt.Circle((0, 0), radius, fill=False, color="#b8b8b8",
                            linestyle="--", linewidth=1, alpha=0.75)
        ax.add_patch(circle)
        ax.text(0.08, radius, f"{radius} m", fontsize=10, color="#666666",
                va="bottom")

    for angle_deg in range(0, 360, 30):
        angle = np.deg2rad(angle_deg)
        ax.plot([0, max_range * np.cos(angle)], [0, max_range * np.sin(angle)],
                color="#dddddd", linewidth=1, zorder=0)

    if not boundary.empty:
        envelope_x = boundary['X'].tolist()
        envelope_y = boundary['Y'].tolist()
        ax.fill(envelope_x, envelope_y, color="#80c78a", alpha=0.22,
                label="Estimated coverage area", zorder=1)
        ax.plot(envelope_x + [envelope_x[0]], envelope_y + [envelope_y[0]], color="#26823a",
                linewidth=2.5, zorder=2)

    detected_points = (
        points_df[points_df['Detected']]
        .groupby(['X', 'Y'], as_index=False)['Response_Time']
        .min()
    )
    failed_points = points_df[~points_df['Detected']][['X', 'Y']].drop_duplicates()
    detected_keys = set(detected_points[['X', 'Y']].itertuples(index=False, name=None))
    failed_points = failed_points[
        ~failed_points.apply(lambda row: (row['X'], row['Y']) in detected_keys, axis=1)
    ]

    scatter = ax.scatter(
        detected_points['X'], detected_points['Y'],
        c=detected_points['Response_Time'], cmap="YlGn_r", vmin=1, vmax=20,
        s=160, edgecolor="#1f1f1f", linewidth=1.2, zorder=3
    )
    if not failed_points.empty:
        ax.scatter(
            failed_points['X'], failed_points['Y'],
            marker="x", s=170, color="#d7191c", linewidth=2.5, zorder=4
        )

    ax.scatter(0, 0, marker="^", s=260, color="#111111", zorder=5)
    ax.text(0.18, 0.15, "PIR Device\n(0,0)", ha="left", va="bottom",
            fontsize=12, fontweight="bold")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Response time (s)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Lateral distance - X axis (m)", fontsize=13)
    ax.set_ylabel("Forward distance - Y axis (m)", fontsize=13)
    ax.set_xlim(-max_range - 0.4, max_range + 0.4)
    ax.set_ylim(-0.4, max_range + 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.tick_params(axis="both", labelsize=12)

    legend_items = [
        Patch(facecolor="#80c78a", alpha=0.22, edgecolor="#26823a",
              label="Estimated coverage area"),
        Line2D([0], [0], marker="o", color="w", label="Detected test point",
               markerfacecolor="#80c78a", markeredgecolor="#1f1f1f",
               markersize=11),
        Line2D([0], [0], marker="x", color="#d7191c",
               label="Undetected test point", markersize=11,
               linestyle="None", markeredgewidth=2.5),
        Line2D([0], [0], marker="^", color="w", label="Device position",
               markerfacecolor="#111111", markeredgecolor="#111111",
               markersize=12),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=11,
              frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    
    # Plot heatmap.
    # Tambahan vmin=0 dan vmax=20 agar waktu 18.20s maksimal berwarna oranye, bukan merah
    plot_matrix_heatmap(
        ax,
        pivot_angle,
        annot_labels,
        cmap=custom_cmap,
        vmin=0,
        vmax=20,
        cbar_label='Response Time (s)',
        linecolor='gray',
        linewidth=0.5,
        annot_size=11,
    )
    
    ax.set_title(f'Sensor Orientation: {angle}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lateral Distance - X Axis (m)', fontsize=12)
    ax.set_ylabel('Forward Distance - Y Axis (m)', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Gambar akan langsung di-save ke folder Anda
plt.savefig('PIR_Response_Time_Map.png', dpi=300) 

# plt.show() # Diblok agar tidak muncul warning di terminal Linux

# ==========================================
# FIGURE 2: PIR TOP VIEW COVERAGE MAP
# ==========================================
pir_top_view_points = build_top_view_points(df, ['0°', '90°'])
plot_top_view_coverage(
    pir_top_view_points,
    'PIR Front Coverage Pattern with 90-Degree Supplement',
    'PIR_Top_View_Coverage.png',
    max_range=3
)
