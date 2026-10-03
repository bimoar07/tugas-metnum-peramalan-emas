# Tugas 2: Peramalan Harga Emas Antam (Regresi Multivariabel)
**Mata Kuliah:** Metode Numerik (PACS262309) — FMIPA  
**Dosen Pengampu:** Faizal Makhrus, S.Kom., M.Sc., Ph.D.  
**Metode:** Regresi Multivariabel Autoregressive (AR(3)) & Linear Neural Network (16 parameter) dengan optimasi *Gradient Descent* dan turunan numerik *Central Finite Difference* ($h = 10^{-10}$).

---

## 📁 Struktur Berkas Proyek

```text
tugas-metnum-peramalan-emas/
├── fetch_data.py                  # Script otomatisasi pengambil data API Antam
├── train_jan_agu_2026.csv         # Data Latih (Januari - Agustus 2026, 243 hari)
├── test_sep_2026.csv              # Data Uji (September 2026, penuh 30 hari)
├── full_jan_sep_2026.csv          # Dataset gabungan Januari - September 2026
├── normalization_params.json      # Parameter Min-Max (dihitung HANYA dari data latih)
├── linear.cpp                     # Implementasi Model Linear AR(3) (4 parameter)
├── nn.cpp                         # Implementasi Linear Neural Network (16 parameter)
├── notebook_peramalan_emas.ipynb  # Jupyter Notebook untuk Google Colab & visualisasi
└── README.md                      # Dokumentasi & panduan eksekusi
```

---

## 🚀 Cara Menjalankan

### 1. Model Linear AR(3) (`linear.cpp`)
```bash
g++ -O3 linear.cpp -o model_linear
./model_linear
```

### 2. Model Neural Network (`nn.cpp`)
```bash
g++ -O3 nn.cpp -o model_nn
./model_nn
```

---

## 🧠 Struktur Model Neural Network (`nn.cpp`)

Sesuai catatan materi:
- **Input:** $u_1, u_2, u_3$ (harga 3 hari sebelumnya)
- **Hidden Layer:**
  $$z_1 = w_1 u_1 + w_4 u_2 + w_7 u_3 + b_1$$
  $$z_2 = w_2 u_1 + w_5 u_2 + w_8 u_3 + b_2$$
  $$z_3 = w_3 u_1 + w_6 u_2 + w_9 u_3 + b_3$$
- **Output Layer:**
  $$\hat{y} = w_{10} z_1 + w_{11} z_2 + w_{12} z_3 + b_4$$
- **Optimasi:** Central Finite Difference ($h = 10^{-10}$) untuk 12 bobot ($w_1 \dots w_{12}$) dan 4 bias ($b_1 \dots b_4$).
