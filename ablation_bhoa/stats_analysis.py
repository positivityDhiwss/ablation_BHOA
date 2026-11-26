"""
stats_analysis.py
=================
Analisis statistik studi ablasi, dijalankan terpisah per dataset.

  Skor kualitas pencarian  : Y = -fitness  (lebih besar = lebih baik)
  Efek LF tanpa LS         : Delta0_r = Y(V1,r) - Y(V0,r)
  Efek LF dengan LS        : Delta1_r = Y(V3,r) - Y(V2,r)
  Kontras interaksi        : I_r = Delta1_r - Delta0_r

  * Wilcoxon signed-rank (dua sisi) + uji tanda eksak sebagai alternatif
  * Ukuran efek rank-biserial (matched pairs)
  * Bootstrap 95% CI (resampling blok seed, 10.000 replikasi)
  * Uji permutasi interaksi: menukar Delta0_r dan Delta1_r dalam blok = membalik tanda I_r
  * Koreksi Holm dalam satu keluarga per dataset {Delta0, Delta1, I}
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def bootstrap_ci(d: np.ndarray, n_boot: int = 10_000, seed: int = 12345,
                 stat=np.mean, level: float = 0.95):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = stat(d[idx], axis=1)
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def wilcoxon_p(d: np.ndarray) -> float:
    nz = d[d != 0]
    if nz.size < 2:
        return 1.0
    try:
        return float(stats.wilcoxon(nz, method="exact", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def sign_test_p(d: np.ndarray) -> float:
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    return float(stats.binomtest(int((nz > 0).sum()), nz.size, 0.5).pvalue)


def signflip_perm_p(d: np.ndarray, n_perm: int = 100_000, seed: int = 54321) -> float:
    """Uji permutasi dua sisi untuk rata-rata I_r (pertukaran Delta0/Delta1 per blok)."""
    obs = abs(d.mean())
    n = d.size
    if n <= 16:  # eksak
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
        null = np.abs((signs * d).mean(axis=1))
        return float(np.mean(null >= obs - 1e-15))
    rng = np.random.default_rng(seed)
    signs = rng.choice([1, -1], size=(n_perm, n))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (n_perm + 1))


def describe_diff(d: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"n": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)),
            "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "ci95_low": lo, "ci95_high": hi,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}


def ablation_contrasts(runs: pd.DataFrame, metric: str, higher_is_better: bool,
                       n_boot: int = 10_000, seed: int = 2026) -> pd.DataFrame:
    """Kontras Delta0, Delta1, dan I untuk satu metrik pada satu dataset."""
    wide = runs.pivot_table(index="seed", columns="variant", values=metric)
    wide = wide.dropna(subset=[v for v in ("V0", "V1", "V2", "V3") if v in wide.columns])
    sgn = 1.0 if higher_is_better else -1.0
    Y = sgn * wide
    d0 = (Y["V1"] - Y["V0"]).to_numpy()
    d1 = (Y["V3"] - Y["V2"]).to_numpy()
    inter = d1 - d0
    rows = []
    for name, d, test in (("Delta0 = V1 - V0 (LF | LS off)", d0, "wilcoxon"),
                          ("Delta1 = V3 - V2 (LF | LS on)", d1, "wilcoxon"),
                          ("I = Delta1 - Delta0 (interaksi)", inter, "permutation")):
        row = {"metric": metric, "contrast": name}
        row.update(describe_diff(d, n_boot, seed))
        if test == "wilcoxon":
            row["p_wilcoxon"] = wilcoxon_p(d)
            row["p_sign"] = sign_test_p(d)
            row["p_primary"] = row["p_wilcoxon"]
            row["rank_biserial"] = rank_biserial(d)
        else:
            row["p_wilcoxon"] = np.nan
            row["p_sign"] = np.nan
            row["p_primary"] = signflip_perm_p(d)
            row["rank_biserial"] = rank_biserial(d)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_primary"].tolist())
    return out


def summarize_variants(runs: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    g = runs.groupby("variant")[metrics]
    parts = []
    for stat in ("mean", "std", "median", "min", "max"):
        s = getattr(g, stat)()
        s.columns = [f"{c}_{stat}" for c in s.columns]
        parts.append(s)
    return pd.concat(parts, axis=1)
