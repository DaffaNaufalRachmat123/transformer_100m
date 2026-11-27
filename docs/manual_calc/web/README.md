# Manual interaktif

Buka `docs/manual_calc/manual_calc.html` langsung di browser. CSS, JavaScript,
font, dan data sudah di-embed; tidak membutuhkan server atau koneksi internet.

## Kontrol

- **Putar / Jeda**: animasi otomatis satu operasi per jeda.
- **Next / Prev / Ulangi**: kendali manual satu operasi.
- **Lompat operasi**: slider untuk meninjau operasi yang sudah dilihat; tidak
  melewati operasi yang belum dijalankan.
- **Jeda (ms)**: tempo masing-masing proses. Default dihasilkan sebagai
  `<html data-step-delay="850">`; edit angka itu di HTML atau di `sections.py`
  jika ingin mengubahnya secara permanen saat build berikutnya.
- **Lanjutkan tur berurutan**: tur otomatis seluruh 32 proses. Tur berhenti pada
  tahap 4 sampai pengguna menekan tombol Generate.
- **Generate embedding & bobot dari nol**: menggunakan seed yang diisi, mengganti semua
  bobot awal dan menghitung ulang forward, loss, backward, clipping/AdamW,
  training 600 langkah, dan inferensi. **Seed acak & generate** memilih seed baru.
- **Gunakan contoh NumPy 42**: memulihkan bobot build awal.

Proses wajib diselesaikan berurutan. Pada pembukaan halaman, angka animasi
embedding belum ditampilkan. Selesaikan tahap 0–3 lalu Generate. Animasi tahap 4
menunjukkan uniform u1/u2, transformasi Box–Muller, perkalian skala, dan pengisian
sel E satu per satu; forward tetap terkunci sampai pembangkitan selesai.
Generate ulang menghapus kemajuan tahap 4 dan seluruh tahap sesudahnya, sehingga
perhitungan dengan seed baru harus dijalankan ulang. Setiap perkalian titik
memperlihatkan operand aktif, hasil kali, dan akumulator secara visual.

Angka aktif ditampilkan di animasi. Perhitungan statis build
dipindahkan ke panel **Arsip perhitungan awal NumPy seed 42**, supaya angka lama
tidak terbaca sebagai hasil seed baru. Tabel eksperimen kontrol (bagian 12) adalah
hasil eksperimen seed tetap pada waktu build, bukan run regenerate.

## Asal angka

`sim.py` menghasilkan bobot awal dengan `numpy.default_rng(42)`, distribusi normal
dan skala yang mengikuti `init_params()` repo. Browser memakai Mulberry32 untuk
bilangan uniform dan Box–Muller untuk normal:

```text
z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
E = 0.02 * z
Wq/Wk/Wv/Wo/W1/W3 = z / sqrt(D)
W2 = z / sqrt(FFN)
gamma = 1
```

Seed yang sama di browser menghasilkan angka yang sama. Seed 42 browser berbeda
dari NumPy seed 42 karena algoritma generator berbeda; gunakan tombol restore
untuk kembali ke angka NumPy. ID token berasal dari vocab/merge tokenizer repo,
bukan generator bobot. Jejak tokenizer di-embed dari implementasi Python.

## Build dan verifikasi

```sh
python3 docs/manual_calc/build_manual.py
python3 docs/manual_calc/verify_engine.py
python3 docs/manual_calc/verify_browser.py
```

Build dan verifikasi engine memerlukan NumPy dan Node.js. Verifikasi browser
memerlukan Python Playwright beserta Chromium. `engine.js` mengikuti `sim.py`
(float64). `app.js` merakit operasi pedagogis dari angka presisi penuh; tampilan
dibulatkan. Verifikasi membandingkan forward, gradien batch, backward satu posisi,
AdamW, serta inferensi dengan implementasi NumPy.
