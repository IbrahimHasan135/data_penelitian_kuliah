# Audit Bab 6–7 — urutan pemeriksaan

Catatan kerja untuk mencocokkan klaim roadmap dengan skrip dan data di repository. Hasil sebelumnya tetap disimpan sebagai checkpoint; audit berikutnya menambahkan bukti baru tanpa menghapus hasil lama.

## Pemeriksaan statik yang sudah dilakukan

Skrip `particle_identity_static_tests.py` membaca profil tersimpan dari model p4-q6, lalu menghitung ulang energi, muatan, residu ODE, node, rasio ekor, identitas virial, kondisi von Laue, dan konvergensi RMAX.

Hasil yang dilaporkan: 14/16 gerbang lolos. Dua kegagalan adalah konvergensi Q dan E pada titik frekuensi tinggi. Titik tersebut berada dekat batas lokalisasi dan memiliki E/Q > m, sehingga jangan menyebut seluruh cabang sebagai solusi terikat yang lolos.

## Langkah berikutnya: uji konvergensi frekuensi tinggi

Jalankan skrip terarah berikut dari repository root:

```bash
python3 "Data Thesis_n_Disertasi/audit_high_omega_convergence.py"
```

Skrip ini memulai dari profil p4-q6 yang tersimpan dan menguji ulang cabang frekuensi tinggi pada RMAX 70, 90, 110, dan 140. Ia menggunakan solver BVP yang sama, melanjutkan solusi dari satu domain ke domain yang lebih besar, mempertahankan resolusi radial kira-kira sama, dan menulis hasil ke:

`Data Thesis_n_Disertasi/rmax_high_omega_audit/high_omega_rmax_audit.csv`

Keluaran tidak menimpa hasil sebelumnya. Skrip memerlukan `numpy`, `scipy`, dan `matplotlib`, sama seperti solver utama. Ia tidak menjalankan ulang pemindaian penuh semua model.

## Target keputusan checkpoint

Bandingkan perubahan Q dan E antar-domain berturut-turut, pergeseran omega, tail ratio, node count, dan residu ODE. Catat apakah hasil terus mendekati konvergensi, gagal konvergen, atau solver gagal. Pemeriksaan ini hanya menilai konvergensi numerik untuk titik frekuensi tinggi; bukan uji kestabilan dinamik atau bukti transformasi Lorentz aktual.

Setelah hasil CSV ini diperiksa, checkpoint berikutnya dapat dirancang: uji evolusi waktu pada solusi referensi p4-q6.
