import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup grid 2D
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2) + 1e-5

# Fungsi spasial gelombang stasioner (Bentuk Orbital)
# s-orbital (l=0)
Z_s = np.exp(-R)
# p_x-orbital (l=1)
Z_p = X * np.exp(-R/2)
# d_{x^2-y^2}-orbital (l=2)
Z_d = (X**2 - Y**2) * np.exp(-R/2)
# f-orbital (l=3 simplified)
Z_f = (X**3 - 3*X*Y**2) * np.exp(-R/3)

orbitals = {
    "s_orbital": {"spatial": Z_s, "title": "Dynamic Phase Fluctuation: s-orbital (l=0)"},
    "p_orbital": {"spatial": Z_p, "title": "Dynamic Phase Fluctuation: p-orbital (l=1)"},
    "d_orbital": {"spatial": Z_d, "title": "Dynamic Phase Fluctuation: d-orbital (l=2)"},
    "f_orbital": {"spatial": Z_f, "title": "Dynamic Phase Fluctuation: f-orbital (l=3)"}
}

# Membuat GIF untuk masing-masing orbital
t_array = np.linspace(0, 2*np.pi, 25)

for name, data in orbitals.items():
    print(f"⏳ Merender animasi {name}.gif ...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ims = []
    
    for t in t_array:
        # Fluktuasi fasa stasioner cos(omega * t)
        Z_t = data["spatial"] * np.cos(t)
        im = ax.pcolormesh(X, Y, Z_t, cmap='twilight', shading='auto', animated=True, vmin=-1.5, vmax=1.5)
        
        if t == 0:
            fig.colorbar(im, label='Amplitudo Gelombang $\psi$')
            ax.set_title(data["title"])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save(f'{name}_oscillation.gif', writer='pillow', fps=10)
    plt.close()
    print(f"✅ Berhasil menyimpan: {name}_oscillation.gif")

print("Semua animasi orbital berhasil dirender!")
