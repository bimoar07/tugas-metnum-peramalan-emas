# Tugas 2: Peramalan Harga Emas Antam (Regresi Multivariabel)
**Mata Kuliah:** Metode Numerik (PACS262309) — FMIPA  
**Dosen Pengampu:** Faizal Makhrus, S.Kom., M.Sc., Ph.D.  
**Metode:** Regresi Multivariabel Autoregressive (AR(3)) & Multi-Layer Perceptron (MLP) dengan optimasi *Gradient Descent* dan turunan numerik *Central Finite Difference* ($h = 10^{-10}$).

---

## 📁 Struktur Berkas Proyek

```text
tugas-metnum-peramalan-emas/
├── fetch_data.py                  # Script otomatisasi pengambil data API Antam
├── train_jan_agu_2026.csv         # Data Latih (Januari - Agustus 2026, 243 hari)
├── test_sep_2026.csv              # Data Uji (September 2026, s.d. saat ini 28 hari)
├── full_jan_sep_2026.csv          # Dataset gabungan Januari - September 2026
├── normalization_params.json      # Parameter Min-Max (dihitung HANYA dari data latih)
├── main.cpp                       # Kode C++ tunggal (Linear AR(3) & MLP)
├── notebook_peramalan_emas.ipynb  # Jupyter Notebook untuk Google Colab & visualisasi
└── README.md                      # Dokumentasi & panduan eksekusi
```

---

## 🚀 Cara Menjalankan

### 1. Eksekusi Program C++ (Lokal / Terminal)
Kompilasi dengan optimasi `-O3`:
```bash
g++ -O3 main.cpp -o model_emas
```

Jalankan Model Linear AR(3):
```bash
./model_emas linear 0.0005 3000
```
- Argumen: `[model] [learning_rate] [max_iterations]`
- Default: `model=linear`, `lr=0.0005`, `max_iter=3000`

Jalankan Model Neural Network (MLP 3-4-1):
```bash
./model_emas mlp 0.0001 2000
```

Program akan menghasilkan file luaran CSV:
- `history_loss_linear.csv` / `history_loss_mlp.csv` (jejak iterasi loss & gradien)
- `predictions_sep_linear.csv` / `predictions_sep_mlp.csv` (perbandingan harga aktual vs prediksi)

---

### 2. Eksekusi di Google Colab
1. Upload folder ini ke repositori GitHub pribadimu (misal `github.com/bimoar07/tugas-metnum-peramalan-emas`).
2. Buka file `notebook_peramalan_emas.ipynb` di Google Colab.
3. Jalankan sel berurutan. Notebook akan secara otomatis:
   - Me-`git clone` repositori.
   - Mengompilasi program C++ via shell.
   - Menjalankan model Linear dan MLP.
   - Menampilkan grafik historis deret waktu, kurva konvergensi fungsi loss, plot perbandingan peramalan September, serta tabel metrik RMSE dan MAPE.

---

### 3. Pembaruan Data di Akhir Bulan (30 September 2026)
Pada tanggal 30 September nanti, jalankan perintah satu baris berikut untuk memperbarui data September menjadi 30 hari penuh secara otomatis:
```bash
python3 fetch_data.py
```
Script akan langsung mengunduh data terbaru dari API resmi Logam Mulia dan menyinkronkan file CSV terkait.

---

## 🧠 Pembagian Bagian Kode di `main.cpp`

Di dalam file `main.cpp`, kode dibagi secara terstruktur:
1. **Scaffolding I/O (Oleh Agent):** Penanganan parsing CSV, pembuatan sliding window 3-hari, normalisasi Min-Max, serta fungsi prediksi *Walk-Forward* (1-step-ahead).
2. **Logika Inti Algoritma (Oleh Bimo):**
   - `compute_loss()`: Perhitungan fungsi objektif galat kuadrat ($SSE = \sum (y - \hat{y})^2$).
   - `compute_gradients()`: Turunan parsial numerik dengan *Central Finite Difference*:
     $$\frac{\partial E}{\partial p} \approx \frac{E(p + h) - E(p - h)}{2h}, \quad h = 10^{-10}$$
   - `train()`: Pembaruan parameter via *Gradient Descent*:
     $$p \leftarrow p - \alpha \cdot \frac{\partial E}{\partial p}$$
   - `predict_rollout()`: Peramalan otonom rekursif 30 hari penuh di bulan September tanpa bocoran data riil masa depan.
