import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# ==========================================
# NUMERICAL GRID SETUP
# ==========================================
x = np.linspace(-4, 4, 250)
y = np.linspace(-4, 4, 250)
X, Y = np.meshgrid(x, y)

# Positions of two wave packets (charges)
x1, y1 = -1.0, 0.0
x2, y2 = 1.0, 0.0

# Function to calculate overlapping wave energy (interference pressure)
def calculate_interference(phase_diff_deg):
    phi_rad = np.radians(phase_diff_deg)
    
    # Radial distances from both packets
    r1 = np.sqrt((X - x1)**2 + (Y - y1)**2) + 1e-3
    r2 = np.sqrt((X - x2)**2 + (Y - y2)**2) + 1e-3
    
    # Spherically decaying wave amplitudes (Stationary wave analogy)
    A1 = np.cos(2.0 * np.pi * r1) / r1
    A2 = (np.cos(2.0 * np.pi * r2 + phi_rad)) / r2 # Phase shifted packet 2
    
    # Interference pressure term u_int = 2*A1*A2*cos(delta_phi)
    # Using superposition amplitude density |A1 + A2|^2
    density = (A1 + A2)**2
    return density

# Generate states
density_like = calculate_interference(0.0)    # In-phase (Constructive - Repulsion)
density_opp = calculate_interference(180.0)  # Out-of-phase (Destructive - Attraction)

# ==========================================
# VISUALIZATION (PNG STATIC - DUAL PANEL)
# ==========================================
fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

ax_like = plt.subplot(gs[0])
ax_opp = plt.subplot(gs[1])

# Plot Like Charges (Repulsion)
heatmap_like = ax_like.contourf(X, Y, density_like, levels=60, cmap='magma')
ax_like.plot(x1, y1, 'ro', markersize=10, label=r'Wave Packet A ($+$)')
ax_like.plot(x2, y2, 'ro', markersize=10, label=r'Wave Packet B ($+$)')
ax_like.set_aspect('equal')
ax_like.set_title("EM Interaction: Like Charges (In-Phase / $\Delta\phi=0^\circ$)")
ax_like.set_xlabel("X (meters)")
ax_like.set_ylabel("Y (meters)")
ax_like.legend(loc='upper right')

info_like = (
    "INTERFERES CONSTRUCTIVE (REPULSION):\n"
    "----------------------------------------\n"
    "Phase alignment creates a peak energy\n"
    "barrier. Wave packets are pushed away\n"
    "seeking minimum phase resistance.\n"
    "Force Vector: F = + k (q1*q2 / r²)"
)
ax_like.text(0.03, 0.03, info_like, transform=ax_like.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# Plot Opposite Charges (Attraction)
heatmap_opp = ax_opp.contourf(X, Y, density_opp, levels=60, cmap='magma')
ax_opp.plot(x1, y1, 'ro', markersize=10, label=r'Wave Packet A ($+$)')
ax_opp.plot(x2, y2, 'co', markersize=10, label=r'Wave Packet B ($-$)')
ax_opp.set_aspect('equal')
ax_opp.set_title("EM Interaction: Opposite Charges (Out-of-Phase / $\Delta\phi=180^\circ$)")
ax_opp.set_xlabel("X (meters)")
ax_opp.legend(loc='upper right')

info_opp = (
    "INTERFER DESTRUCTIVE (ATTRACTION):\n"
    "----------------------------------------\n"
    "Phase cancellation forms a valley of\n"
    "vacuum pressure, causing packets to be\n"
    "sucked into the low energy gradient.\n"
    "Force Vector: F = - k (q1*q2 / r²)"
)
ax_opp.text(0.03, 0.03, info_opp, transform=ax_opp.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.colorbar(heatmap_like, ax=[ax_like, ax_opp], fraction=0.025, pad=0.04, label=r'Interference Energy Density ($|A_{total}|^2$)')
plt.savefig('em_interference_states.png', dpi=300, bbox_inches='tight')
print("✅ PNG Saved: em_interference_states.png")

# ==========================================
# ANIMATION EXPORT (GIF - FLUCTUATING PHASE)
# ==========================================
print("⏳ Rendering EM Wave Fluctuation GIF (Harap tunggu)...")
fig_anim, (ax_ani1, ax_ani2) = plt.subplots(1, 2, figsize=(16, 8))

def update_frame(frame):
    ax_ani1.clear(); ax_ani2.clear()
    
    # Time varying oscillation cos(omega * t)
    t = frame * 0.2
    
    r1 = np.sqrt((X - x1)**2 + (Y - y1)**2) + 1e-3
    r2 = np.sqrt((X - x2)**2 + (Y - y2)**2) + 1e-3
    
    # Fluktuasi fasa searah jarum jam vs berlawanan (Spin + / Spin -)
    A1 = (np.cos(2*np.pi*r1 - t)) / r1
    A2_like = (np.cos(2*np.pi*r2 - t)) / r2            # In phase
    A2_opp = (np.cos(2*np.pi*r2 - t + np.pi)) / r2    # Out of phase
    
    dens_t_like = (A1 + A2_like)**2
    dens_t_opp = (A1 + A2_opp)**2
    
    # Panel 1
    ax_ani1.contourf(X, Y, dens_t_like, levels=40, cmap='inferno')
    ax_ani1.plot(x1, y1, 'ro', markersize=8); ax_ani1.plot(x2, y2, 'ro', markersize=8)
    ax_ani1.set_aspect('equal')
    ax_ani1.set_title(f"Dynamic Repulsion (t={t:.1f})")
    ax_ani1.set_xlabel("X"); ax_ani1.set_ylabel("Y")
    
    # Panel 2
    ax_ani2.contourf(X, Y, dens_t_opp, levels=40, cmap='inferno')
    ax_ani2.plot(x1, y1, 'ro', markersize=8); ax_ani2.plot(x2, y2, 'co', markersize=8)
    ax_ani2.set_aspect('equal')
    ax_ani2.set_title(f"Dynamic Attraction (t={t:.1f})")
    ax_ani2.set_xlabel("X")

    return ax_ani1, ax_ani2

ani = animation.FuncAnimation(fig_anim, update_frame, frames=30, interval=100, blit=False)
ani.save('em_wave_fluctuation.gif', writer='pillow', fps=10)
print("✅ GIF Saved: em_wave_fluctuation.gif")
