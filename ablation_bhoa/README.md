# ablation_bhoa — Kode Uji Coba Awal (Pilot) Studi Ablasi Lévy Flight pada BHOA

Implementasi untuk proposal **"Studi Ablasi Lévy Flight pada Binary Horse Herd Optimization
Algorithm untuk Seleksi Fitur Network Intrusion Detection System"**.

## Struktur

| Berkas | Isi |
|---|---|
| `bhoa.py` | BHOA bersama + saklar LF/LS (V0–V3), anggaran evaluasi terpusat + cache, RNG terpisah, diagnostik |
| `data.py` | Pemuat UNSW-NB15 / CIC-IDS2017, split bebas kebocoran, prapemrosesan berbasis fitur asli (1 bit = 1 fitur asli) |
| `evaluation.py` | Fitness `0.99·(1−BA) + 0.01·|b|/d` (KNN k=5), metrik uji (macro-F1, recall, FPR, BA, akurasi) |
| `stats_analysis.py` | Δ₀, Δ₁, interaksi I, Wilcoxon, uji tanda, rank-biserial, bootstrap CI, uji permutasi, Holm |
| `sanity_check.py` | Validasi implementasi pada dataset kecil (9 pemeriksaan) |
| `run_pilot.py` | Orkestrasi eksperimen + keluaran CSV/PNG/JSON |

## Cara menjalankan (dari folder `TA SKRIPSI`)

```powershell
# 1) Validasi implementasi (±30 detik)
py -3.10 -m ablation_bhoa.sanity_check

# 2) Smoke test cepat: 2 seed, anggaran 300, subsampel 2.000
py -3.10 -m ablation_bhoa.run_pilot --dataset both --quick

# 3) Pilot sesuai proposal: 5 seed, anggaran 3.000, subsampel 5.000
py -3.10 -m ablation_bhoa.run_pilot --dataset both --seeds 5 --budget 3000

# 4) Eksperimen utama (setelah parameter DIKUNCI): 30 seed
py -3.10 -m ablation_bhoa.run_pilot --dataset both --seeds 30 --budget <terkunci> --out results/main_ablation
```

Opsi penting: `--pop`, `--search-size`, `--final-train-size`, `--decay {exponential,linear}`,
`--lf-scale`, `--ls-trials`, `--x-bound`, `--g0`, `--threads`, `--no-cache`.

## Keluaran (`results/pilot_ablation/<dataset>/`)

- `runs.csv` – satu baris per run (fitness, |b|, metrik uji, waktu, permintaan/evaluasi/cache hit, log operator, mask)
- `convergence.csv` – best-so-far fitness per permintaan fitness
- `diagnostics.csv` – keragaman mask (Hamming), proporsi posisi di batas, |x| rata-rata per iterasi
- `feature_frequency.csv` – frekuensi pemilihan tiap fitur asli per varian
- `baseline.csv` – KNN seluruh fitur
- `summary_variants.csv`, `contrasts.csv` – ringkasan dan kontras ablasi
- `plot_convergence.png`, `plot_boxplots.png`, `plot_interaction.png`
- `run_config.json` – argumen, parameter, checksum data, versi pustaka, info perangkat

## Keputusan teknis yang WAJIB dikunci sebelum eksperimen utama

Nilai default di `BHOAParams` adalah nilai awal pilot. Sebelum eksperimen utama, cocokkan
nilainya dengan sumber primer (MiarNaeimi dkk., 2021; Elmanakhly dkk., 2022), lalu catat di Lampiran A:

1. Koefisien enam perilaku HOA (g, h, s, i, d, r), batas grazing (ǔ, ľ), dan proporsi kelompok usia.
2. Jadwal peluruhan. Implementasi ini memakai proporsi anggaran: `ω^(p·T_ref)` atau `1−p`.
3. Domain posisi dan batas (`x_bound`). Pilot menunjukkan grazing multiplikatif mendorong |x| ke batas.
4. Skala, frekuensi, dan clipping langkah Lévy, serta letak LF (sebelum binarisasi, tanpa evaluasi tambahan).
5. Operator LS: jumlah percobaan per iterasi, P(crossover), aturan penerimaan, dan pengodean mask ke posisi.
6. Anggaran fitness, ukuran populasi, ukuran subsampel, dan ukuran data latih akhir.

> Catatan data: file `UNSW_NB15_training-set.csv` di folder ini memiliki 34 prediktor, tidak
> sama dengan partisi resmi yang lazim (±42 prediktor; tidak ada `sttl`, `dttl`, `ct_srv_src`, dst.).
> File `CICIDS2017_clean_5pct.csv` adalah subsampel 5% versi MachineLearningCVE asli, bukan versi
> terkoreksi Engelen dkk. Keduanya perlu dipastikan/diganti sebelum eksperimen utama.
