"""
data.py
=======
Pemuatan data, pembagian (split) bebas kebocoran, dan prapemrosesan berbasis
"fitur asli" (satu bit seleksi = satu fitur asli, termasuk seluruh kolom one-hot-nya).

Protokol Terkunci Eksperimen Utama (Bab III & Lampiran A proposal):
  * UNSW-NB15  : berkas kurasi Kaggle 36 kolom (34 fitur kandidat setelah buang attack_cat/id).
                 Pada eksperimen utama, data pengembangan (`training-set`) dideduplikasi
                 sebelum pembagian internal pencarian untuk mencegah kebocoran baris duplikat (P3).
                 Evaluasi akhir dilaporkan pada `testing-set` standar utuh sekaligus versi
                 terdeduplikasi (`test_dedup`) sebagai analisis sensitivitas.
  * CIC-IDS2017: mendukung dua mode:
                 (1) Mode disjoint utama (`test_csv` diberikan): `dev` dan `test` berasal dari
                     berkas terpisah yang dibentuk dari 8 berkas mentah MachineLearningCVE,
                     dengan verifikasi tegas 0 irisan kelompok vektor fitur antar-partisi.
                 (2) Mode berkas tunggal (pilot): duplikat dihapus, lalu split 70:30 terstratifikasi
                     berbasis kelompok vektor fitur identik (`StratifiedGroupKFold`).
                 Kolom `Destination Port` dipertahankan sebagai kandidat aktif dan diaudit di Bab IV.
  * Seluruh transformasi (imputasi, one-hot, Min-Max) di-fit hanya pada bagian pelatihan
    tahap yang bersangkutan, lalu diterapkan ke bagian validasi/uji.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# Utilitas
# =============================================================================
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def feature_hashes(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Hash 64-bit deterministik per baris vektor fitur untuk deteksi duplikasi lintas partisi."""
    return pd.util.hash_pandas_object(df[features], index=False).to_numpy()


def _stratified_group_split(df: pd.DataFrame, features: List[str], test_size: float,
                            seed: int):
    """Membagi DataFrame menjadi (train, test) tanpa memecah kelompok vektor fitur identik."""
    groups = feature_hashes(df, features)
    n_groups = int(pd.Series(groups).nunique())
    idx = np.arange(len(df))
    if n_groups == len(df):
        tr_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=df["y"],
                                          random_state=seed)
        return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True), \
            "stratified (semua kelompok unik)"

    n_splits = max(2, int(round(1.0 / test_size * 3))) if test_size <= 0.35 else int(round(1.0 / test_size))
    k_test = max(1, int(round(test_size * n_splits)))
    if k_test >= n_splits:
        k_test = n_splits - 1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_parts = [te for i, (_, te) in enumerate(sgkf.split(idx, df["y"], groups)) if i < k_test]
    te_idx = np.concatenate(test_parts)
    tr_idx = np.setdiff1d(idx, te_idx)
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True), \
        f"StratifiedGroupKFold({n_splits}) -> {k_test} lipatan"


@dataclass
class DatasetBundle:
    name: str
    dev: pd.DataFrame                       # data pengembangan (fitur asli + kolom 'y')
    test: pd.DataFrame                      # data uji utama
    numeric: List[str]
    categorical: List[str]
    info: Dict = field(default_factory=dict)
    test_dedup: Optional[pd.DataFrame] = None   # sensitivitas uji terdeduplikasi (UNSW-NB15)

    @property
    def features(self) -> List[str]:
        return self.numeric + self.categorical


# =============================================================================
# Pemuat dataset
# =============================================================================
def load_unsw(train_csv: str, test_csv: str, dedup_dev: bool = True) -> DatasetBundle:
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    info = {"files": {train_csv: sha256_file(train_csv), test_csv: sha256_file(test_csv)},
            "raw_rows_dev": len(tr), "raw_rows_test": len(te)}
    drop = [c for c in ("id", "attack_cat") if c in tr.columns]
    info["dropped_columns"] = drop
    for df in (tr, te):
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
        df.rename(columns={"label": "y"}, inplace=True)
    cat = [c for c in tr.columns if c != "y" and tr[c].dtype == object]
    num = [c for c in tr.columns if c not in cat and c != "y"]
    for df in (tr, te):
        df[num] = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    info["n_inf_or_nan_dev"] = int(tr[num].isna().any(axis=1).sum())
    info["n_inf_or_nan_test"] = int(te[num].isna().any(axis=1).sum())
    info["n_duplicates_dev"] = int(tr.duplicated().sum())
    info["n_duplicates_test"] = int(te.duplicated().sum())

    if dedup_dev:
        tr = tr.drop_duplicates().reset_index(drop=True)
    else:
        tr = tr.reset_index(drop=True)
    info["dedup_dev_applied"] = bool(dedup_dev)
    info["rows_dev_effective"] = len(tr)

    te = te.reset_index(drop=True)
    te_dedup = te.drop_duplicates().reset_index(drop=True)
    info["rows_test_dedup"] = len(te_dedup)

    return DatasetBundle("UNSW-NB15", tr, te, num, cat, info, test_dedup=te_dedup)


def _clean_cicids_df(csv_path: str, label_col: str = "Label", benign: str = "BENIGN"):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    raw_rows = len(df)
    id_cols = [c for c in ("Flow ID", "Source IP", "Source Port", "Destination IP",
                           "Timestamp", "External IP", "Fwd Header Length.1") if c in df.columns]
    df = df.drop(columns=id_cols)
    y = (df[label_col].astype(str).str.strip() != benign).astype(int)
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    n_inf_or_nan = int(X.isna().any(axis=1).sum())
    data = X.copy()
    data["y"] = y.values
    n0 = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    feats = [c for c in data.columns if c != "y"]
    meta = {
        "raw_rows": raw_rows,
        "dropped_columns": id_cols,
        "n_inf_or_nan": n_inf_or_nan,
        "n_duplicates_removed": n0 - len(data),
        "destination_port_retained": "Destination Port" in feats,
    }
    return data, feats, meta


def load_cicids(csv_path: str, test_size: float = 0.30, split_seed: int = 2026,
                label_col: str = "Label", benign: str = "BENIGN",
                test_csv: Optional[str] = None) -> DatasetBundle:
    """
    Memuat CIC-IDS2017:
      - Jika `test_csv` diberikan (mode eksperimen utama): `csv_path` sebagai data pengembangan
        dan `test_csv` sebagai partisi uji independen yang saling lepas (disjoint) dari pilot,
        dengan pemeriksaan tegas nol irisan hash vektor fitur.
      - Jika `test_csv` None (mode berkas tunggal / pilot): membagi `csv_path` 70:30 dengan
        StratifiedGroupKFold berbasis hash vektor fitur identik.
    """
    if test_csv is not None:
        dev, feats, meta_dev = _clean_cicids_df(csv_path, label_col=label_col, benign=benign)
        test, feats_te, meta_te = _clean_cicids_df(test_csv, label_col=label_col, benign=benign)
        if feats != feats_te:
            raise ValueError("Kolom fitur dev_csv dan test_csv CIC-IDS2017 tidak identik!")
        h_dev = set(feature_hashes(dev, feats).tolist())
        h_te = set(feature_hashes(test, feats).tolist())
        overlap = len(h_dev & h_te)
        if overlap > 0:
            raise ValueError(
                f"Kebocoran terdeteksi: ditemukan {overlap} kelompok vektor fitur identik "
                f"antara dev ({csv_path}) dan test ({test_csv})!"
            )
        info = {
            "mode": "disjoint_external_test",
            "files": {csv_path: sha256_file(csv_path), test_csv: sha256_file(test_csv)},
            "dev_meta": meta_dev,
            "test_meta": meta_te,
            "rows_dev": len(dev),
            "rows_test": len(test),
            "feature_hash_overlap_dev_test": overlap,
            "destination_port_retained": meta_dev["destination_port_retained"],
        }
        return DatasetBundle("CIC-IDS2017", dev, test, feats, [], info)

    data, feats, meta = _clean_cicids_df(csv_path, label_col=label_col, benign=benign)
    groups = feature_hashes(data, feats)
    info = {
        "mode": "single_csv_split",
        "files": {csv_path: sha256_file(csv_path)},
        **meta,
        "n_groups": int(pd.Series(groups).nunique()),
        "n_conflicting_label_groups": int(
            (pd.DataFrame({"g": groups, "y": data["y"]}).groupby("g")["y"].nunique() > 1).sum()
        ),
        "split_seed": split_seed,
    }
    dev, test, method = _stratified_group_split(data, feats, test_size=test_size, seed=split_seed)
    info["split_method"] = method
    info["actual_test_ratio"] = len(test) / len(data)
    return DatasetBundle("CIC-IDS2017", dev, test, feats, [], info)


def drop_constant_features(b: DatasetBundle) -> DatasetBundle:
    """Fitur konstan diidentifikasi HANYA dari data pengembangan, lalu dibuang dari kandidat."""
    const = [c for c in b.features if b.dev[c].nunique(dropna=True) <= 1]
    b.numeric = [c for c in b.numeric if c not in const]
    b.categorical = [c for c in b.categorical if c not in const]
    b.info["constant_features_removed"] = const
    b.info["n_candidate_features"] = len(b.features)
    return b


# =============================================================================
# Prapemrosesan berbasis kelompok fitur asli
# =============================================================================
class GroupedPreprocessor:
    """
    Imputasi median + Min-Max untuk fitur numerik, one-hot untuk kategorikal.
    `groups[j]` = indeks kolom matriks hasil transformasi yang dikendalikan bit fitur asli ke-j.
    """

    def __init__(self, numeric: List[str], categorical: List[str], min_frequency: float = 0.005):
        self.numeric = numeric
        self.categorical = categorical
        self.min_frequency = min_frequency

    def fit(self, df: pd.DataFrame) -> "GroupedPreprocessor":
        num = df[self.numeric].astype(float)
        self.median_ = num.median().fillna(0.0)
        filled = num.fillna(self.median_)
        self.min_ = filled.min()
        rng = filled.max() - self.min_
        self.range_ = rng.replace(0, 1.0)
        self.ohe_ = {}
        for c in self.categorical:
            enc = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=self.min_frequency, sparse_output=False)
            enc.fit(df[[c]].astype(str))
            self.ohe_[c] = enc
        # Peta kelompok kolom
        self.groups: List[np.ndarray] = []
        pos = 0
        for _ in self.numeric:
            self.groups.append(np.array([pos])); pos += 1
        for c in self.categorical:
            w = len(self.ohe_[c].get_feature_names_out())
            self.groups.append(np.arange(pos, pos + w)); pos += w
        self.n_columns = pos
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = df[self.numeric].astype(float).fillna(self.median_)
        Xn = ((num - self.min_) / self.range_).clip(0.0, 1.0).to_numpy(np.float32)
        parts = [Xn]
        for c in self.categorical:
            parts.append(self.ohe_[c].transform(df[[c]].astype(str)).astype(np.float32))
        return np.hstack(parts) if len(parts) > 1 else Xn

    def columns_for(self, mask: np.ndarray) -> np.ndarray:
        sel = [self.groups[j] for j in np.flatnonzero(mask)]
        return np.concatenate(sel) if sel else np.array([], dtype=int)


def stratified_subsample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    sub, _ = train_test_split(df, train_size=n, stratify=df["y"], random_state=seed)
    return sub.reset_index(drop=True)


@dataclass
class SearchData:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    prep: GroupedPreprocessor


def make_search_data(b: DatasetBundle, n_sub: int, val_ratio: float, seed: int) -> SearchData:
    """
    Subsampel terstratifikasi dari data pengembangan -> search-train / search-validation
    dengan pemisahan kelompok vektor fitur identik agar bebas kebocoran internal (P3).
    """
    sub = stratified_subsample(b.dev, n_sub, seed)
    tr, va, _ = _stratified_group_split(sub, b.features, test_size=val_ratio, seed=seed + 1)
    prep = GroupedPreprocessor(b.numeric, b.categorical).fit(tr)
    return SearchData(prep.transform(tr), tr["y"].to_numpy(), prep.transform(va),
                      va["y"].to_numpy(), prep)


@dataclass
class FinalData:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    prep: GroupedPreprocessor


def make_final_data(b: DatasetBundle, n_final_train: Optional[int], seed: int,
                    test_df: Optional[pd.DataFrame] = None) -> FinalData:
    """Data latih akhir (opsional disubsampel, sama untuk semua konfigurasi) + data uji."""
    tr = stratified_subsample(b.dev, n_final_train, seed + 7)
    te = b.test if test_df is None else test_df
    prep = GroupedPreprocessor(b.numeric, b.categorical).fit(tr)
    return FinalData(prep.transform(tr), tr["y"].to_numpy(), prep.transform(te),
                     te["y"].to_numpy(), prep)
