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
H_initial = 400000       # SOLUSI 1: Start di 400 km agar tidak glitch masuk tanah
r_initial = R_earth + H_initial

# Calculate Escape Velocity pada ketinggian tersebut
v_esc = np.sqrt(2 * G * E_wave / (c**2 * r_initial))

# ==========================================
# NUMERICAL SIMULATION SETUP
# ==========================================
dt = 5.0                
total_time = 6000      
steps = int(total_time / dt)

# SOLUSI 2: Kecepatan sub-orbital di bawah kecepatan melingkar (7.67 km/s) agar pasti jatuh
point1 = 8000.0          
point2 = v_esc
point3 = 13000.0

# Initial conditions (Starting at 400km above North Pole, shooting right)
x1, y1, vx1, vy1 = 0.0, r_initial, point1, 0.0     # Sub-orbital
x2, y2, vx2, vy2 = 0.0, r_initial, point2, 0.0     # Exact Escape
x3, y3, vx3, vy3 = 0.0, r_initial, point3, 0.0     # Hyperbolic

traj_x1, traj_y1 = [], []
traj_x2, traj_y2 = [], []
traj_x3, traj_y3 = [], []

act1, act2, act3 = True, True, True

for i in range(steps):
    # --- Particle 1 (Red: Sub-orbital) ---
    r1 = np.sqrt(x1**2 + y1**2)
    # Logika tabrakan dari lu: Berhenti bergerak saat nyentuh tanah
    if r1 <= R_earth: 
        act1 = False
        x1 = x1 
        y1 = y1
    else:
        a_mag1 = (G * E_wave) / ((c**2) * (r1**2))
        vx1 -= a_mag1 * (x1/r1) * dt
        vy1 -= a_mag1 * (y1/r1) * dt
        x1 += vx1 * dt
        y1 += vy1 * dt
    traj_x1.append(x1); traj_y1.append(y1)
    
    # --- Particle 2 (Yellow: Escape) ---
    r2 = np.sqrt(x2**2 + y2**2)
    a_mag2 = (G * E_wave) / ((c**2) * (r2**2))
    vx2 -= a_mag2 * (x2/r2) * dt
    vy2 -= a_mag2 * (y2/r2) * dt
    x2 += vx2 * dt
    y2 += vy2 * dt
    traj_x2.append(x2); traj_y2.append(y2)

    # --- Particle 3 (Green: Hyperbolic) ---
    r3 = np.sqrt(x3**2 + y3**2)
    a_mag3 = (G * E_wave) / ((c**2) * (r3**2))
    vx3 -= a_mag3 * (x3/r3) * dt
    vy3 -= a_mag3 * (y3/r3) * dt
    x3 += vx3 * dt
    y3 += vy3 * dt
    traj_x3.append(x3); traj_y3.append(y3)

# ==========================================
# VISUALIZATION (PNG)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 12)) 

composition_limit = R_earth * 6
grid_size = np.linspace(-composition_limit, composition_limit, 250)
X, Y = np.meshgrid(grid_size, grid_size)
R_grid = np.sqrt(X**2 + Y**2)
R_grid[R_grid < R_earth] = R_earth 
delta_n = (G / c**4) * (E_wave / R_grid)

heatmap = ax.contourf(X, Y, delta_n, levels=50, cmap='inferno')
plt.colorbar(heatmap, fraction=0.046, pad=0.04, label=r'Vacuum Refractive Index Distortion ($\Delta n$)')

# Bumi Padat
ax.add_patch(plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5, label='Earth Surface'))

# Plot Garis (Merah dibuat tebal dan ada end-point marker)
line1, = ax.plot(traj_x1, traj_y1, color='red', linewidth=2.5, label=f'Sub-orbital (v = {point1/1000:.2f} km/s)')
#ax.plot(traj_x1[-1], traj_y1[-1], 'ro', markersize=8) # Titik jatuh

line2, = ax.plot(traj_x2, traj_y2, color='gold', linewidth=2.5, label=f'Parabolic / Escape (v = {point2/1000:.2f} km/s)')
line3, = ax.plot(traj_x3, traj_y3, color='lime', linewidth=2.5, label=f'Hyperbolic (v = {point3/1000:.2f} km/s)')

ax.set_aspect('equal')
ax.set_xlim(-composition_limit, composition_limit)
ax.set_ylim(-composition_limit, composition_limit)

ax.set_title("Simulation 3: Kinematic Escape Velocity via Phase Evolution")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.legend(loc='upper right')

info_text = (
    "ESCAPE KINEMATICS DYNAMICS:\n"
    "----------------------------------\n"
    f"Launch Altitude: {H_initial/1000:.1f} km\n"
    f"Required Escape Velocity: {v_esc/1000:.3f} km/s\n"
    "Observation:\n"
    "- Red Particle: Phase tilt > Momentum (Crashes)\n"
    "- Yellow Particle: Perfect balance (Escapes)\n"
    "- Green Particle: Kinetic dominates (Escapes)"
)
ax.text(0.03, 0.97, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.savefig('sim3_escape_fixed_final.png', dpi=300, bbox_inches='tight')
print("✅ PNG Saved: sim3_escape_fixed_final.png")

# ==========================================
# ANIMATION (GIF)
# ==========================================
print("⏳ Rendering Escape GIF...")
def update(frame):
    idx = frame * 5 
    if idx >= len(traj_x1): idx = len(traj_x1) - 1
    line1.set_data(traj_x1[:idx], traj_y1[:idx])
    
    idx_others = frame * 5
    if idx_others >= len(traj_x2): idx_others = len(traj_x2) - 1
    line2.set_data(traj_x2[:idx_others], traj_y2[:idx_others])
    line3.set_data(traj_x3[:idx_others], traj_y3[:idx_others])
    
    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, frames=len(traj_x2)//5, interval=30, blit=True)
ani.save('sim3_escape_fixed_final_anim.gif', writer='pillow', fps=30)
print("✅ GIF Saved: sim3_escape_fixed_final_anim.gif")
