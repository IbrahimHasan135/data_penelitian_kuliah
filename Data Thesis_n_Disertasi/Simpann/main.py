import matplotlib
matplotlib.use('Agg')  # Wajib ditaruh paling atas untuk lingkungan Linux headless
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup Dashboard Figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Simulasi Model ToE: Mekanika Gelombang Stasioner", fontsize=16, fontweight='bold')

# Membuat 5 Subplot
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233, polar=True) # Polar untuk rotasi fasa
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)

# ==========================================
# 1. MEDAN VAKUM KUANTUM (Fluktuasi Dasar)
# ==========================================
grid_size = 50
im1 = ax1.imshow(np.random.rand(grid_size, grid_size), cmap='magma', vmin=0, vmax=1)
ax1.set_title("1. Fluktuasi Vakum Kuantum\n(Medan Dasar Ruang)")
ax1.axis('off')

# ==========================================
# 2. MATERI: PAKET GELOMBANG STASIONER
# ==========================================
x2 = np.linspace(-5, 5, 500)
line2_env, = ax2.plot(x2, np.exp(-x2**2), 'y--', alpha=0.5, label='Amplop (Massa)')
line2_wave, = ax2.plot(x2, np.zeros_like(x2), color='cyan', label='Osilasi Stasioner')
ax2.set_ylim(-1.5, 1.5)
ax2.set_title("2. Materi sebagai Paket Gelombang")
ax2.set_facecolor('black')
ax2.legend(loc='upper right', fontsize=8)

# ==========================================
# 3. MUATAN: POLARISASI FASA KOMPLEKS
# ==========================================
line3_pos, = ax3.plot([0, 0], [0, 1], color='red', linewidth=3, label='Muatan (+) rotasi maju')
line3_neg, = ax3.plot([0, np.pi], [0, 1], color='blue', linewidth=3, label='Muatan (-) rotasi mundur')
ax3.set_title("3. Muatan sbg Arah Rotasi Fasa")
ax3.set_yticklabels([])
ax3.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8)

# ==========================================
# 4. GRAVITASI: DISTORSI IMPEDANSI/INDEKS BIAS
# ==========================================
x4 = np.linspace(-2, 2, 100)
y4 = np.linspace(-2, 2, 100)
X4, Y4 = np.meshgrid(x4, y4)
R4 = np.sqrt(X4**2 + Y4**2)
# Simulasi indeks bias n(r)
Z4 = 1 + 0.5 / (R4 + 0.1) 
im4 = ax4.imshow(Z4, extent=[-2, 2, -2, 2], cmap='plasma', origin='lower')
dot4, = ax4.plot([2], [0], 'wo', markersize=6, label='Jatuh Bebas (Fasa)')
ax4.set_title("4. Gravitasi: Gradien Indeks Bias")
ax4.legend(loc='upper right', fontsize=8)

# ==========================================
# 5. GAYA NUKLIR: SUMUR POTENSIAL & INSTABILITAS
# ==========================================
x5 = np.linspace(-3, 3, 200)
# Sumur potensial sangat dalam dan sempit
V5 = -5 * np.exp(-x5**2 / 0.05) 
ax5.plot(x5, V5, 'r-', alpha=0.7, label='Sumur Potensial (Gaya Kuat)')
line5, = ax5.plot([0], [0.3], 'bo', markersize=8, label='Gelombang Terkurung')
ax5.set_ylim(-6, 2)
ax5.set_title("5. Gaya Nuklir (Kuat & Lemah)")
ax5.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

# ==========================================
# FUNGSI UPDATE ANIMASI (Frame per Frame)
# ==========================================
def update(frame):
    # Waktu simulasi diperkecil agar pergerakan jauh lebih halus per framenya
    t = frame * 0.05  
    
    # 1. Update Vakum: Fluktuasi noise (dibuat lebih dinamis namun transisinya halus)
    noise = np.random.rand(grid_size, grid_size) * (0.5 + 0.5 * np.sin(t * 3)**2)
    im1.set_array(noise)
    
    # 2. Update Paket Gelombang
    envelope = np.exp(-x2**2)
    carrier = np.sin(15 * x2) * np.cos(t * 8)
    line2_wave.set_ydata(envelope * carrier)
    
    # 3. Update Fasa (Muatan): Rotasi berlawanan arah dengan kecepatan konstan yang halus
    line3_pos.set_data([0, t * 2], [0, 1])            
    line3_neg.set_data([0, -t * 2 + np.pi], [0, 1])   
    
    # 4. Update Gravitasi: Spiral jatuh yang rapat dan mulus
    r_grav = max(0.05, 2 - 0.15 * t)
    theta_grav = t * 3
    dot4.set_data([r_grav * np.cos(theta_grav)], [r_grav * np.sin(theta_grav)])
    
    # 5. Update Gaya Nuklir
    # Karena frame total diperbesar, waktu peluruhan disesuaikan ke frame 350
    if frame < 350:
        # Gaya Kuat (Terkurung) - osilasi cepat di dasar sumur
        x_pos = 0.2 * np.sin(t * 15)
        y_pos = -5 * np.exp(-x_pos**2 / 0.05) + 0.3
    else:
        # Gaya Lemah (Peluruhan) - transisi lepas perlahan
        escape_velocity = (frame - 350) * 0.015
        x_pos = escape_velocity
        y_pos = -5 * np.exp(-x_pos**2 / 0.05) + 0.3
        
    line5.set_data([x_pos], [y_pos])
    
    return im1, line2_wave, line3_pos, line3_neg, dot4, line5

# ---------------------------------------------------------
# RENDER KE GIF (Animasi)
# ---------------------------------------------------------
# Frame ditambah jadi 600 agar pergerakan mikro bisa tertangkap
print("Memulai proses render animasi resolusi tinggi ke 'toe_simulation.gif' (tunggu sebentar)...")
ani = animation.FuncAnimation(fig, update, frames=200, interval=16, blit=False, repeat=False)

# Dirender pada 60 fps untuk animasi super smooth
ani.save('toe_simulation.gif', writer='pillow', fps=20, dpi=100)
print("Render selesai! Animasi super halus berhasil disimpan.")
