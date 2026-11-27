# ablation_bhoa (v0.2.0) — Kode Eksperimen Studi Ablasi Lévy Flight pada BHOA

Implementasi terkunci (Lampiran A proposal) untuk **"Studi Ablasi Lévy Flight pada Binary Horse Herd Optimization Algorithm (BHOA) untuk Seleksi Fitur Network Intrusion Detection System Berbasis Machine Learning"**.

## Struktur

| Berkas | Isi |
|---|---|
| `bhoa.py` | BHOA bersama + saklar LF/LS (V0–V3), formulasi Lampiran MATLAB terkunci (`pbest - x`, roam inersia `V_prev`, kelompok δ = G + R), `S2`, `lf_scale=0.05`, anggaran terpusat + cache, 6 aliran RNG terpisah |
| `data.py` | Pemuat UNSW-NB15 (dedup data pengembangan + sensitivitas `test_dedup`) dan CIC-IDS2017 (mode berkas tunggal & mode uji eksternal *disjoint* dengan asersi 0 irisan hash), `GroupedPreprocessor` |
| `build_cicids_disjoint.py` | Utilitas pembuat `CICIDS2017_main_dev.csv` & `CICIDS2017_main_test.csv` yang 100% saling lepas (*disjoint*) dari berkas pilot menggunakan 8 berkas CSV harian `MachineLearningCVE` |
| `evaluation.py` | Fitness `0.99·(1−BA) + 0.01·|b|/d` (5-NN Euclidean uniform), metrik uji utama & pendukung |
| `stats_analysis.py` | Δ₀, Δ₁, interaksi I, Wilcoxon eksak (`method="exact"`), uji tanda, rank-biserial, bootstrap CI 95%, uji permutasi 100.000×, Holm |
| `sanity_check.py` | Validasi implementasi pada dataset kecil (9 pemeriksaan otomatis) |
| `run_pilot.py` | Orkestrasi eksperimen *Latin square* 4×4 + keluaran CSV/PNG/JSON (termasuk sensitivitas `*_test_dedup.csv` UNSW-NB15) |

## Urutan Menjalankan Eksperimen Utama (dari folder `TA SKRIPSI`)

```powershell
# 1) Validasi implementasi konfigurasi terkunci (±30 detik)
py -3.10 -m ablation_bhoa.sanity_check

# 2) Bangun partisi dev & test CIC-IDS2017 yang saling lepas (disjoint) dari pilot
py -3.10 -m ablation_bhoa.build_cicids_disjoint `
    --raw-dir "C:\path\ke\MachineLearningCVE" `
    --pilot-csv "CICIDS2017_clean_5pct.csv" `
    --out-dev "CICIDS2017_main_dev.csv" `
    --out-test "CICIDS2017_main_test.csv"

# 3) Smoke test cepat (2 seed, anggaran 300)
py -3.10 -m ablation_bhoa.run_pilot --dataset both --quick

# 4) Eksperimen utama terkunci: 30 seed (240 run), B=1.500, search=2.000, final_train=20.000
py -3.10 -m ablation_bhoa.run_pilot --dataset both --seeds 30 `
    --cicids CICIDS2017_main_dev.csv `
    --cicids-test CICIDS2017_main_test.csv `
    --out results/main_ablation
```

*(Untuk mereproduksi ulang pengaturan pilot lama v0.1.0, gunakan flag `--hoa-mode text --lf-scale 1.0 --budget 600 --final-train-size 5000 --no-dedup-unsw`.)*
