"""
data.py
=======
Pemuatan data, pembagian (split) bebas kebocoran, dan prapemrosesan berbasis
fitur asli (satu bit seleksi = satu fitur asli, termasuk seluruh kolom one-hot-nya).

Protokol (Bab III proposal):
  * UNSW-NB15  : partisi resmi training / testing (data uji).
                 Kolom id dan attack_cat dibuang dari prediktor (attack_cat membocorkan label).
  * CIC-IDS2017: duplikat dihapus, lalu split 70:30 terstratifikasi berbasis kelompok
                 (baris dengan vektor fitur identik berada pada partisi yang sama).
  * Seluruh transformasi (imputasi, one-hot, Min-Max) di-fit hanya pada bagian pelatihan
    tahap yang bersangkutan, lalu diterapkan ke bagian validasi/uji.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


@dataclass
class DatasetBundle:
    name: str
    dev: pd.DataFrame                 # data pengembangan (fitur asli + kolom 'y')
    test: pd.DataFrame                # data uji terkunci
    numeric: List[str]
    categorical: List[str]
    info: Dict = field(default_factory=dict)

    @property
    def features(self) -> List[str]:
        return self.numeric + self.categorical


# =============================================================================
# Pemuat dataset
# =============================================================================
def load_unsw(train_csv: str, test_csv: str) -> DatasetBundle:
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
    return DatasetBundle("UNSW-NB15", tr.reset_index(drop=True), te.reset_index(drop=True),
                         num, cat, info)


def load_cicids(csv_path: str, test_size: float = 0.30, split_seed: int = 2026,
                label_col: str = "Label", benign: str = "BENIGN") -> DatasetBundle:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    info = {"files": {csv_path: sha256_file(csv_path)}, "raw_rows": len(df)}
    id_cols = [c for c in ("Flow ID", "Source IP", "Source Port", "Destination IP",
                           "Timestamp", "External IP", "Fwd Header Length.1") if c in df.columns]
    df = df.drop(columns=id_cols)
    info["dropped_columns"] = id_cols
    y = (df[label_col].astype(str).str.strip() != benign).astype(int)
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    info["n_inf_or_nan"] = int(X.isna().any(axis=1).sum())
    data = X.copy()
    data["y"] = y.values

    n0 = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    info["n_duplicates_removed"] = n0 - len(data)

    # Kelompok = vektor fitur identik (menangkap baris identik dengan label berbeda)
    feats = [c for c in data.columns if c != "y"]
    groups = pd.util.hash_pandas_object(data[feats], index=False).values
    info["n_groups"] = int(pd.Series(groups).nunique())
    info["n_conflicting_label_groups"] = int(
        (pd.DataFrame({"g": groups, "y": data["y"]}).groupby("g")["y"].nunique() > 1).sum())

    idx = np.arange(len(data))
    if info["n_groups"] == len(data):
        dev_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=data["y"],
                                             random_state=split_seed)
        info["split_method"] = "stratified (semua kelompok unik)"
    else:
        n_splits = int(round(1 / test_size * 3))           # 0.3 -> 10 lipatan, ambil 3
        k_test = int(round(test_size * n_splits))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        test_parts = [te for i, (_, te) in enumerate(sgkf.split(idx, data["y"], groups)) if i < k_test]
        test_idx = np.concatenate(test_parts)
        dev_idx = np.setdiff1d(idx, test_idx)
        info["split_method"] = f"StratifiedGroupKFold({n_splits}) -> {k_test} lipatan uji"
    info["split_seed"] = split_seed
    dev = data.iloc[dev_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
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
    """Subsampel terstratifikasi dari data pengembangan -> search-train / search-validation."""
    sub = stratified_subsample(b.dev, n_sub, seed)
    tr, va = train_test_split(sub, test_size=val_ratio, stratify=sub["y"], random_state=seed + 1)
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


def make_final_data(b: DatasetBundle, n_final_train: Optional[int], seed: int) -> FinalData:
    """Data latih akhir (opsional disubsampel, sama untuk semua konfigurasi) + data uji terkunci."""
    tr = stratified_subsample(b.dev, n_final_train, seed + 7)
    prep = GroupedPreprocessor(b.numeric, b.categorical).fit(tr)
    return FinalData(prep.transform(tr), tr["y"].to_numpy(), prep.transform(b.test),
                     b.test["y"].to_numpy(), prep)
