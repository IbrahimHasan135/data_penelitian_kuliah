import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# ==========================================
# UNIVERSE & EARTH PARAMETERS
# ==========================================
G = 6.674e-11           
c = 3e8                 
M_earth = 5.972e24       

E_wave = M_earth * c**2  
R_earth = 6371000        
H_initial = 400000         
r_initial = R_earth + H_initial

# ==========================================
# NUMERICAL SIMULATION SETUP
# ==========================================
dt = 1.0                
total_time = 1000      
steps = int(total_time / dt)

x = r_initial
y = 0.0
vx = 0.0
vy = 0.0

traj_x, traj_y = [], []

for i in range(steps):
    r = np.sqrt(x**2 + y**2)
    if r <= R_earth:
        print(f"Impact at t = {i*dt} seconds")
        break
        
    a_mag = (G * E_wave) / ((c**2) * (r**2))
    ax = -a_mag * (x / r)
    ay = -a_mag * (y / r)
    
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt
    
    traj_x.append(x)
    traj_y.append(y)

# Calculate gravity values for infographic
g_surface = (G * E_wave) / ((c**2) * (R_earth**2))
g_initial = (G * E_wave) / ((c**2) * (r_initial**2))

# ==========================================
# VISUALIZATION (PNG STATIC)
# ==========================================
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
ax_main = plt.subplot(gs[0])
ax_zoom = plt.subplot(gs[1])

# Grid for Heatmap
grid_size = np.linspace(-r_initial*1.2, r_initial*1.2, 200)
X, Y = np.meshgrid(grid_size, grid_size)
R_grid = np.sqrt(X**2 + Y**2)
R_grid[R_grid < R_earth] = R_earth 
delta_n = (G / c**4) * (E_wave / R_grid)

# --- PANEL 1: FULL VIEW ---
heatmap1 = ax_main.contourf(X, Y, delta_n, levels=50, cmap='inferno')
ax_main.add_patch(plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5, label='Earth Surface'))
ax_main.plot(traj_x, traj_y, color='cyan', linewidth=2, label='Free Fall Trajectory')
ax_main.plot(traj_x[0], traj_y[0], 'go', markersize=8, label='Initial Position')
ax_main.set_aspect('equal')
ax_main.set_title("Simulation 1: Wave Packet Free Fall (Global View)")
ax_main.set_xlabel("X (meters)")
ax_main.set_ylabel("Y (meters)")
ax_main.legend(loc='upper left')

# --- INFOGRAPHIC TEXT BOX (Dipindah ke ax_main) ---
info_text = (
    "SIMULATION PARAMETERS:\n"
    "----------------------------------\n"
    f"Initial Altitude (H): {H_initial/1000:.1f} km\n"
    f"Initial Velocity (v): 0.0 m/s\n\n"
    "CALCULATED WAVE DYNAMICS:\n"
    "----------------------------------\n"
    f"Accel at H (a): {g_initial:.2f} m/s²\n"
    f"Accel at Surface (g): {g_surface:.2f} m/s²\n"
    f"Impact Time: {len(traj_x)*dt:.1f} s"
)
# Diletakkan di sudut kiri bawah panel utama
ax_main.text(0.03, 0.03, info_text, transform=ax_main.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='left', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# --- PANEL 2: ZOOM VIEW (Sekarang bersih dari teks) ---
heatmap2 = ax_zoom.contourf(X, Y, delta_n, levels=50, cmap='inferno')
ax_zoom.add_patch(plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5))
line_zoom, = ax_zoom.plot(traj_x, traj_y, color='cyan', linewidth=3, marker='.', markevery=10)
dot_zoom, = ax_zoom.plot(traj_x[0], traj_y[0], 'go', markersize=10)
ax_zoom.set_aspect('equal')
# Zoom difokuskan ke area jatuhnya
ax_zoom.set_xlim(R_earth - 50000, r_initial + 50000)
ax_zoom.set_ylim(-100000, 100000)
ax_zoom.set_title("Simulation 1: Zoomed View on Kinematics")
ax_zoom.set_xlabel("X (meters)")

plt.colorbar(heatmap1, ax=[ax_main, ax_zoom], fraction=0.02, pad=0.04, label=r'Vacuum Refractive Index Distortion ($\Delta n$)')

plt.savefig('sim1_freefall_english.png', dpi=300, bbox_inches='tight')
print("✅ PNG Saved: sim1_freefall_english.png")

# ==========================================
# ANIMATION EXPORT (GIF)
# ==========================================
print("⏳ Rendering GIF Animation (This may take a minute)...")
def update(frame):
    # Update line and dot for animation
    idx = frame * 10 
    if idx >= len(traj_x):
        idx = len(traj_x) - 1
    
    line_zoom.set_data(traj_x[:idx], traj_y[:idx])
    dot_zoom.set_data([traj_x[idx]], [traj_y[idx]]) 
    return line_zoom, dot_zoom

ani = animation.FuncAnimation(fig, update, frames=len(traj_x)//10, interval=50, blit=True)
ani.save('sim1_freefall_anim.gif', writer='pillow', fps=20)
print("✅ GIF Saved: sim1_freefall_anim.gif")
