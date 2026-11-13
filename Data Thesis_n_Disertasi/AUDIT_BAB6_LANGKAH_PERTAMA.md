# Audit Bab 6 — langkah pertama

Catatan kerja untuk mencocokkan hasil Bab 6 di Google Docs dengan skrip dan data yang tersimpan di repository. Ini checkpoint audit; tidak mengubah atau menghapus catatan hasil sebelumnya.

## Status yang ditemukan

- Solver cabang polinomial: `polynomial_pq_hybrid_solver.py`.
- Data profil dan hasil p4-q6 sudah tersedia di `polynomial_pq_hybrid_output/p4_q6/`, termasuk `branch_profiles.npz` dan `branch_results.csv`.
- Ada skrip analisis statik yang membaca profil tersimpan tanpa menjalankan ulang BVP: `particle_identity_static_tests.py`.
- Karena keluaran solver sudah tersedia, jangan mulai dengan menjalankan ulang solver penuh. Langkah pertama adalah mereproduksi pemeriksaan statik dari data tersimpan.

## Jalankan satu skrip ini dulu

Dari folder `Data Thesis_n_Disertasi`:

```bash
python3 particle_identity_static_tests.py --output-dir particle_identity_static_output_audit
```

Skrip memerlukan `numpy`, `scipy`, dan `matplotlib`. Opsi output terpisah menjaga hasil audit baru agar tidak menimpa keluaran lama.

## Apa yang diperiksa

Skrip memilih solusi referensi p4-q6 lalu menghitung ulang energi dan muatan, residu ODE, jumlah node, rasio ekor, identitas virial, kondisi von Laue, konsistensi NPZ terhadap CSV, dan konvergensi terhadap RMAX. Ia juga membuat pemeriksaan prediksi boost Lorentz dari besaran statik.

## Target checkpoint

Simpan keluaran baru, lalu bandingkan dengan hasil yang sudah ada. Catat setiap gerbang sebagai **cocok**, **berbeda**, atau **belum dapat dinilai**. Jangan menyimpulkan kestabilan dinamik penuh hanya dari analisis statik atau pemeriksaan boost prediktif.

Jika skrip gagal karena file masukan tidak ditemukan, pastikan perintah dijalankan dari folder yang disebutkan. Jika gerbang gagal, jangan jalankan ulang solver dulu; periksa rumus/definisi dan data yang dipakai skrip.

## Setelah langkah pertama

Baru setelah hasil reproduksi statik cocok, audit solver sumbernya dan metode benchmark secara terpisah. Solver penuh `polynomial_pq_hybrid_solver.py` menjalankan pemindaian multi-model dan continuation, sehingga tidak diperlukan untuk pemeriksaan pertama.
