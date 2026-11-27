"""
build_cicids_disjoint.py
========================
Utilitas pembuat partisi pengembangan (`dev`) dan uji (`test`) CIC-IDS2017 untuk
eksperimen utama yang **100% saling lepas (disjoint)** dari berkas tahap pilot
(`CICIDS2017_clean_5pct.csv`) menggunakan 8 berkas CSV harian mentah `MachineLearningCVE`.

Jaminan metodologis (mitigasi P3 & P5, Arp dkk., 2022):
  1. Membuang kolom identitas (bila ada) + kolom ganda `Fwd Header Length.1`,
     mempertahankan `Destination Port` (diaudit pada Bab IV), menghapus NaN/Inf dan duplikat.
  2. Menghitung hash 64-bit seluruh vektor fitur pada berkas pilot (`CICIDS2017_clean_5pct.csv`)
     lalu membuang seluruh baris ber-hash sama dari korpus gabungan 8 berkas mentah (~95% sisa).
  3. Membentuk partisi `main_dev` dan `main_test` terstratifikasi berbasis kelompok vektor fitur
     identik (`StratifiedGroupKFold`) sehingga:
       - irisan(pilot, main_dev)  == 0 kelompok fitur
       - irisan(pilot, main_test) == 0 kelompok fitur
       - irisan(main_dev, main_test) == 0 kelompok fitur
  4. Menyimpan manifest JSON berisi checksum SHA-256, jumlah baris, dan proporsi kelas.

Contoh pemakaian:
  py -3.10 -m ablation_bhoa.build_cicids_disjoint ^
      --raw-dir "C:\\path\\ke\\MachineLearningCVE" ^
      --pilot-csv "CICIDS2017_clean_5pct.csv" ^
      --out-dev "CICIDS2017_main_dev.csv" ^
      --out-test "CICIDS2017_main_test.csv"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import feature_hashes, sha256_file

ID_DROP_COLS = (
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Timestamp", "External IP", "Fwd Header Length.1",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Bangun partisi dev & test CIC-IDS2017 yang saling lepas dari berkas pilot"
    )
    ap.add_argument("--raw-dir", required=True,
                    help="Folder berisi 8 berkas CSV harian CIC-IDS2017 (MachineLearningCVE)")
    ap.add_argument("--pilot-csv", default="CICIDS2017_clean_5pct.csv",
                    help="Berkas CSV pilot yang seluruh vektor fiturnya wajib dikeluarkan")
    ap.add_argument("--out-dev", default="CICIDS2017_main_dev.csv",
                    help="Berkas keluaran partisi pengembangan eksperimen utama")
    ap.add_argument("--out-test", default="CICIDS2017_main_test.csv",
                    help="Berkas keluaran partisi uji independen eksperimen utama")
    ap.add_argument("--dev-size", type=int, default=88227,
                    help="Target jumlah baris partisi pengembangan utama (selaras skala pilot)")
    ap.add_argument("--test-size", type=int, default=37812,
                    help="Target jumlah baris partisi uji independen utama (selaras skala pilot)")
    ap.add_argument("--seed", type=int, default=202601,
                    help="Seed acak untuk penarikan partisi eksperimen utama")
    return ap.parse_args()


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    drop = [c for c in ID_DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    if "Label" not in df.columns:
        raise ValueError("Kolom 'Label' tidak ditemukan pada berkas CSV.")
    feats = [c for c in df.columns if c != "Label"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _sample_groups_stratified(df: pd.DataFrame, groups: np.ndarray,
                              target_n: int, seed: int) -> pd.Index:
    """
    Ambil subsampel mendekati target_n secara terstratifikasi di tingkat KELOMPOK hash fitur
    (O(N), instan untuk jutaan baris) tanpa pernah memecah kelompok fitur identik.
    """
    if target_n >= len(df):
        return df.index
    y_bin = (df["Label"] != "BENIGN").astype(np.int8).to_numpy()
    gdf = pd.DataFrame({"g": groups, "y": y_bin}).drop_duplicates(subset=["g"], keep="first")
    n_groups_target = min(len(gdf) - 1, max(2, int(round(target_n * (len(gdf) / len(df))))))
    chosen_g, _ = train_test_split(
        gdf["g"].to_numpy(),
        train_size=n_groups_target,
        stratify=gdf["y"].to_numpy(),
        random_state=seed,
    )
    chosen_set = set(chosen_g.tolist())
    mask = pd.Series(groups).isin(chosen_set).to_numpy()
    return df.index[mask]


def main():
    args = parse_args()
    csv_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ditemukan berkas .csv di: {args.raw_dir}")

    print(f"Membaca {len(csv_files)} berkas CSV harian dari {args.raw_dir}...")
    parts = []
    for i, fp in enumerate(csv_files, 1):
        print(f"  [{i}/{len(csv_files)}] {os.path.basename(fp)}")
        parts.append(pd.read_csv(fp, low_memory=False, encoding="utf-8", encoding_errors="replace"))
    full = pd.concat(parts, ignore_index=True)
    n_raw = len(full)
    full = _clean_frame(full)
    feats = [c for c in full.columns if c != "Label"]
    print(f"Korpus bersih gabungan: {len(full):,} baris (dari {n_raw:,} baris mentah), {len(feats)} fitur.")

    print(f"Membaca berkas pilot untuk eksklusi hash fitur: {args.pilot_csv}")
    pilot = _clean_frame(pd.read_csv(args.pilot_csv, low_memory=False))
    if [c for c in pilot.columns if c != "Label"] != feats:
        raise ValueError("Daftar kolom fitur pada pilot CSV tidak cocok dengan 8 berkas mentah!")

    pilot_hashes = set(feature_hashes(pilot, feats).tolist())
    full_hashes = feature_hashes(full, feats)
    keep_mask = ~pd.Series(full_hashes).isin(pilot_hashes).to_numpy()
    pool = full.loc[keep_mask].reset_index(drop=True)
    pool_hashes = feature_hashes(pool, feats)
    print(f"Sisa korpus non-pilot (0 irisan hash dengan pilot): {len(pool):,} baris "
          f"(dibuang {len(full) - len(pool):,} baris beririsan pilot).")

    total_need = args.dev_size + args.test_size
    if total_need > len(pool):
        raise ValueError(f"Permintaan dev+test ({total_need:,}) melebihi sisa pool ({len(pool):,}).")

    # 1. Tarik partisi uji independen terlebih dahulu tanpa memecah kelompok fitur
    test_idx = _sample_groups_stratified(pool, pool_hashes, args.test_size, seed=args.seed)
    test_df = pool.loc[test_idx].reset_index(drop=True)

    rem_pool = pool.drop(index=test_idx).reset_index(drop=True)
    rem_hashes = feature_hashes(rem_pool, feats)

    # 2. Tarik partisi pengembangan dari sisa pool yang saling lepas dari test maupun pilot
    dev_idx = _sample_groups_stratified(rem_pool, rem_hashes, args.dev_size, seed=args.seed + 1)
    dev_df = rem_pool.loc[dev_idx].reset_index(drop=True)

    # 3. Verifikasi tegas 0 irisan hash fitur
    h_dev = set(feature_hashes(dev_df, feats).tolist())
    h_test = set(feature_hashes(test_df, feats).tolist())
    assert len(h_dev & pilot_hashes) == 0, "Irisan pilot vs main_dev != 0!"
    assert len(h_test & pilot_hashes) == 0, "Irisan pilot vs main_test != 0!"
    assert len(h_dev & h_test) == 0, "Irisan main_dev vs main_test != 0!"

    dev_df.to_csv(args.out_dev, index=False)
    test_df.to_csv(args.out_test, index=False)

    attack_dev = float((dev_df["Label"] != "BENIGN").mean())
    attack_test = float((test_df["Label"] != "BENIGN").mean())
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "raw_files": [os.path.basename(f) for f in csv_files],
        "pilot_csv": args.pilot_csv,
        "pilot_sha256": sha256_file(args.pilot_csv),
        "n_raw_total": n_raw,
        "n_clean_total": len(full),
        "n_non_pilot_pool": len(pool),
        "main_dev": {
            "path": args.out_dev,
            "sha256": sha256_file(args.out_dev),
            "rows": len(dev_df),
            "attack_ratio": attack_dev,
        },
        "main_test": {
            "path": args.out_test,
            "sha256": sha256_file(args.out_test),
            "rows": len(test_df),
            "attack_ratio": attack_test,
        },
        "verified_zero_hash_overlap": {
            "pilot_vs_main_dev": 0,
            "pilot_vs_main_test": 0,
            "main_dev_vs_main_test": 0,
        },
        "destination_port_retained": "Destination Port" in feats,
    }
    manifest_path = os.path.splitext(args.out_test)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nSELESAI & TERVERIFIKASI (0 irisan hash fitur):")
    print(f"  main_dev : {args.out_dev} ({len(dev_df):,} baris, {attack_dev*100:.2f}% serangan)")
    print(f"  main_test: {args.out_test} ({len(test_df):,} baris, {attack_test*100:.2f}% serangan)")
    print(f"  manifest : {manifest_path}")


if __name__ == "__main__":
    main()
