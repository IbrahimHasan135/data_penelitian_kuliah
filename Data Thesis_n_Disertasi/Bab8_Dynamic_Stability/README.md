# Bab 8 — Rancangan Uji Kestabilan Dinamik

Folder ini menyiapkan protokol dan kode untuk menguji evolusi waktu kandidat solusi Q-ball polinomial p4-q6 dari Bab 6–7. Bab 8 menetapkan protokol; hasil eksekusi dan klaim stabilitas masuk Bab 9.

## Status sebelum eksekusi

Roadmap Bab 7.7 meminta persamaan medan waktu penuh, konvensi tanda, kondisi awal, syarat batas, skema numerik, diagnostik, dan ambang penilaian dipastikan terlebih dahulu. Kode ini sudah memuat satu implementasi kerja berdasarkan solver statik yang ada, tetapi implementasi tersebut harus dibandingkan dengan penurunan Lagrangian di naskah sebelum simulasi dianggap sebagai bukti. Perintah tanpa `--execute` hanya memeriksa berkas input dan mencetak asumsi; evolusi PDE tidak dijalankan.

## Model kerja yang dikodekan

Potensial yang dipakai solver statik repo:

`U(|Psi|^2) = m^2 |Psi|^2 - alpha |Psi|^4 + beta |Psi|^6`

dengan parameter kerja `m=1, alpha=1, beta=0.5`. Persamaan kerja:

`d2_t Psi = Laplacian(Psi) - (m^2 - 2 alpha |Psi|^2 + 3 beta |Psi|^4) Psi`

Ansatz solusi statik: `Psi(t,r)=f(r) exp(-i omega t)`; maka kondisi awal `Psi(0,r)=f(r)` dan `d_t Psi(0,r)=-i omega f(r)`. Program menguji evolusi sferis 3D (laplacian radial). Ini tidak menguji mode nonsferis, tumbukan, atau kestabilan Lorentz.

## Uji yang disiapkan

1. **Kontrol tanpa gangguan:** evolusi dari profil statik untuk memeriksa drift numerik.
2. **Gangguan skala ±1%:** mengubah amplitudo profil secara halus untuk menguji respons terhadap kompresi/ekspansi kecil.
3. **Gangguan skala +5%:** menguji apakah respons tetap terbatas pada gangguan lebih besar.
4. **Gangguan halus acak 1%:** menguji sensitivitas terhadap perubahan bentuk profil.
5. **Konvergensi waktu:** mengulang kontrol tanpa gangguan dengan langkah waktu lebih kecil.
6. **Diagnostik:** energi, muatan Noether, radius RMS energi, amplitudo inti, dan rasio energi/muatan terhadap nilai awal. Domain diperpanjang agar radiasi belum mencapai batas luar selama waktu simulasi; batas Dirichlet luar tetap dapat memantulkan radiasi jika durasi terlalu panjang.

Program menghasilkan CSV dan PNG ke folder keluaran. Ambang pada laporan adalah ambang diagnostik numerik awal, bukan kriteria fisik universal: energy/charge drift 1%, variasi radius 25%, dan amplitudo inti 35%. Interpretasi hasil harus mempertimbangkan konvergensi dan refleksi batas.

## Instalasi dan penggunaan

Dari root repository:

```bash
python3 -m pip install -r "Data Thesis_n_Disertasi/Bab8_Dynamic_Stability/requirements.txt"
python3 "Data Thesis_n_Disertasi/Bab8_Dynamic_Stability/run_dynamic_tests.py" \
  --model-dir "Data Thesis_n_Disertasi/polynomial_pq_hybrid_output/p4_q6"
```

Perintah pertama kali hanya melakukan preflight. Setelah persamaan, kondisi awal, dan batas diverifikasi terhadap naskah, jalankan:

```bash
python3 "Data Thesis_n_Disertasi/Bab8_Dynamic_Stability/run_dynamic_tests.py" \
  --model-dir "Data Thesis_n_Disertasi/polynomial_pq_hybrid_output/p4_q6" \
  --execute
```

Untuk uji cepat:

```bash
python3 "Data Thesis_n_Disertasi/Bab8_Dynamic_Stability/run_dynamic_tests.py" \
  --model-dir "Data Thesis_n_Disertasi/polynomial_pq_hybrid_output/p4_q6" \
  --execute --quick
```

Jangan gunakan hasil `--quick` sebagai hasil final Bab 9. Jalankan uji penuh dan simpan folder output yang dihasilkan bersama versi kode, nilai parameter, dan commit repo.
