import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


def transform_ir_points(x, y, angle_label):
    """
    Mengubah titik uji lokal menjadi titik tampak atas 360 derajat.
    0 derajat = depan, 90 derajat = kanan, -90 derajat = kiri,
    180 derajat = belakang.
    """
    if angle_label == '0°':
        return mirrored_axis_points(x, y)
    if angle_label == '180°':
        return mirrored_axis_points(x, -y)
    if angle_label == '90°':
        return [(py, px) for px, py in mirrored_axis_points(x, y)]
    if angle_label == '-90°':
        return [(-py, px) for px, py in mirrored_axis_points(x, y)]
    return []


def build_top_view_points(source_df, value_columns):
    """
    Membuat titik coverage pandang atas dari tiap kolom sudut pengambilan.
    Setiap nilai response time dipetakan ke posisi arah pengambilan aslinya.
    """
    points = []
    for _, row in source_df.iterrows():
        for angle_label in value_columns:
            response_time = row[angle_label]
            detected = pd.notnull(response_time)
            for point_x, point_y in transform_ir_points(row['X'], row['Y'], angle_label):
                points.append({
                    'X': point_x,
                    'Y': point_y,
                    'Detected': detected,
                    'Response_Time': response_time if detected else np.nan,
                    'Direction': angle_label,
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
        ax.text(0.12, radius, f"{radius} m", fontsize=10, color="#666666",
                va="bottom")

    for angle_deg in range(0, 360, 30):
        angle = np.deg2rad(angle_deg)
        ax.plot([0, max_range * np.cos(angle)], [0, max_range * np.sin(angle)],
                color="#dddddd", linewidth=1, zorder=0)

    if not boundary.empty:
        envelope_x = boundary['X'].tolist()
        envelope_y = boundary['Y'].tolist()
        ax.fill(envelope_x, envelope_y, color="#5ca8ff", alpha=0.18,
                label="Estimated coverage area", zorder=1)
        ax.plot(envelope_x + [envelope_x[0]], envelope_y + [envelope_y[0]], color="#1f6fb2",
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
        c=detected_points['Response_Time'], cmap="YlGnBu_r", vmin=1.4, vmax=2.5,
        s=150, edgecolor="#1f1f1f", linewidth=1.2, zorder=3
    )
    if not failed_points.empty:
        ax.scatter(
            failed_points['X'], failed_points['Y'],
            marker="x", s=150, color="#d7191c", linewidth=2.5, zorder=4
        )

    ax.scatter(0, 0, marker="^", s=260, color="#111111", zorder=5)
    ax.text(0, -0.35, "IR Device\n(0,0)", ha="center", va="top",
            fontsize=12, fontweight="bold")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Response time (s)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Lateral distance - X axis (m)", fontsize=13)
    ax.set_ylabel("Forward distance - Y axis (m)", fontsize=13)
    ax.set_xlim(-max_range - 0.6, max_range + 0.6)
    ax.set_ylim(-max_range - 0.6, max_range + 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.tick_params(axis="both", labelsize=12)

    legend_items = [
        Patch(facecolor="#5ca8ff", alpha=0.18, edgecolor="#1f6fb2",
              label="Estimated coverage area"),
        Line2D([0], [0], marker="o", color="w", label="Detected test point",
               markerfacecolor="#5ca8ff", markeredgecolor="#1f1f1f",
               markersize=11),
        Line2D([0], [0], marker="^", color="w", label="Device position",
               markerfacecolor="#111111", markeredgecolor="#111111",
               markersize=12),
    ]
    if not failed_points.empty:
        legend_items.append(
            Line2D([0], [0], marker="x", color="#d7191c",
                   label="Undetected test point", markersize=11,
                   linestyle="None", markeredgewidth=2.5)
        )
    ax.legend(handles=legend_items, loc="upper right", fontsize=11,
              frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==========================================
# FIGURE 1: SPATIAL COVERAGE QUALITY HEATMAP
# ==========================================
pivot_success = df.pivot(index='Y', columns='X', values='Success_Count')
pivot_success = pivot_success.sort_index(ascending=False)

plt.figure(figsize=(8, 6))
ax = plt.gca()
plot_matrix_heatmap(
    ax,
    pivot_success,
    pivot_success.astype(int).astype(str),
    cmap="YlGn",
    vmin=0,
    vmax=4,
    cbar_label='Successful Directions Count (Max 4)',
    linecolor="black",
    linewidth=1,
    annot_size=12,
)
plt.title('Infrared Coverage Reliability Map\nBased on Room Coordinates', fontsize=14, fontweight='bold')
plt.xlabel('X-Axis Distance (m)', fontsize=12)
plt.ylabel('Y-Axis Distance (m)', fontsize=12)
plt.tight_layout()
plt.savefig('IR_Coverage_Map.png', dpi=300) 
# plt.show() 

# ==========================================
# FIGURE 1B: TOP VIEW COVERAGE MAP
# ==========================================
ir_top_view_points = build_top_view_points(df, ['0°', '90°', '-90°', '180°'])
plot_top_view_coverage(
    ir_top_view_points,
    'Infrared 360-Degree Top-View Coverage Pattern',
    'IR_Top_View_Coverage.png',
    max_range=7
)

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
    plot_matrix_heatmap(
        ax,
        pivot_angle,
        annot_labels,
        cmap=custom_cmap,
        vmin=1.4,
        vmax=2.5,
        cbar_label='Response Time (s)',
        linecolor='gray',
        linewidth=0.5,
        annot_size=10,
    )
            
    ax.set_title(f'Transmission Angle: {angle}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X-Axis Distance (m)')
    ax.set_ylabel('Y-Axis Distance (m)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Gambar akan langsung di-save ke folder Anda
plt.savefig('IR_Response_Time_Angles.png', dpi=300)
