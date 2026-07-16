import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
total_time = 5600      # ~93 mins (1 orbit)
steps = int(total_time / dt)

x = r_initial
y = 0.0
vx = 0.0
vy = 7672.0             # Orbital Velocity

traj_x, traj_y = [], []

for i in range(steps):
    r = np.sqrt(x**2 + y**2)
    if r <= R_earth:
        break
        
    a_mag = (G * E_wave) / ((c**2) * (r**2))
    ax = -a_mag * (x / r)
    ay = -a_mag * (y / r)
    
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt
    
    if i % 10 == 0:
        traj_x.append(x)
        traj_y.append(y)

g_initial = (G * E_wave) / ((c**2) * (r_initial**2))

# ==========================================
# VISUALIZATION (PNG STATIC)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))

grid_size = np.linspace(-r_initial*1.5, r_initial*1.5, 200)
X, Y = np.meshgrid(grid_size, grid_size)
R_grid = np.sqrt(X**2 + Y**2)
R_grid[R_grid < R_earth] = R_earth 
delta_n = (G / c**4) * (E_wave / R_grid)

heatmap = ax.contourf(X, Y, delta_n, levels=50, cmap='inferno')
plt.colorbar(heatmap, fraction=0.046, pad=0.04, label=r'Vacuum Refractive Index Distortion ($\Delta n$)')

ax.add_patch(plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5, label='Earth Surface'))
line_orbit, = ax.plot(traj_x, traj_y, color='lime', linewidth=2, label='Orbital Trajectory')
dot_orbit, = ax.plot(traj_x[0], traj_y[0], 'wo', markersize=8, label='Satellite/Wave Packet')

ax.set_aspect('equal')
ax.set_title("Simulation 2: Wave Packet Locked in Orbital Resonance")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.legend(loc='upper right')

# --- INFOGRAPHIC TEXT BOX 1 (Top Left) ---
info_text_top = (
    "ORBITAL PARAMETERS:\n"
    "----------------------------------\n"
    f"Altitude (H): {H_initial/1000:.1f} km\n"
    f"Tangential Velocity: {vy/1000:.3f} km/s"
)
ax.text(0.03, 0.97, info_text_top, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# --- INFOGRAPHIC TEXT BOX 2 (Bottom Left) ---
info_text_bottom = (
    "RESONANCE DYNAMICS:\n"
    "----------------------------------\n"
    f"Centripetal Accel req: {(vy**2)/r_initial:.2f} m/s²\n"
    f"Phase Tilt Accel (a): {g_initial:.2f} m/s²\n"
    "Status: Phase Locked (Stable Orbit)"
)
ax.text(0.03, 0.03, info_text_bottom, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.savefig('sim2_orbit_english.png', dpi=300, bbox_inches='tight')
print("✅ PNG Saved: sim2_orbit_english.png")

# ==========================================
# ANIMATION EXPORT (GIF)
# ==========================================
print("⏳ Rendering GIF Animation (This may take a minute)...")
def update(frame):
    idx = frame * 2 # Speed multiplier
    if idx >= len(traj_x):
        idx = len(traj_x) - 1
        
    line_orbit.set_data(traj_x[:idx], traj_y[:idx])
    dot_orbit.set_data([traj_x[idx]], [traj_y[idx]]) 
    return line_orbit, dot_orbit

ani = animation.FuncAnimation(fig, update, frames=len(traj_x)//2, interval=30, blit=True)
ani.save('sim2_orbit_anim.gif', writer='pillow', fps=30)
print("✅ GIF Saved: sim2_orbit_anim.gif")
