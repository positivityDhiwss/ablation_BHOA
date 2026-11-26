"""
sanity_check.py
===============
Validasi implementasi pada dataset kecil (Breast Cancer Wisconsin, 30 fitur) SEBELUM
eksperimen NIDS (Bab III 3.6 proposal). Memeriksa:

  1. Populasi awal identik antar-varian untuk seed yang sama.
  2. Jumlah permintaan fitness == anggaran (persis) untuk semua varian.
  3. requests == evals + cache_hits.
  4. Semua mask terbaik valid (biner, tidak kosong, dimensi benar).
  5. Log operator: LF hanya aktif pada V1/V3, LS hanya aktif pada V2/V3.
  6. Determinisme: run ulang dengan seed sama menghasilkan hasil identik.
  7. Trace best-so-far monoton tidak naik dan panjangnya == anggaran.
  8. Generator acak terpisah: V0 dan V2 memakai urutan bilangan acak HOA yang sama
     sampai LS pertama kali dipanggil (dicek lewat kesamaan 20 permintaan awal).

Jalankan:  py -3.10 -m ablation_bhoa.sanity_check
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score

from .bhoa import VARIANTS, BHOAParams, run_variant
from .evaluation import make_knn


def main():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = MinMaxScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    d = X.shape[1]

    def fitness(mask):
        cols = np.flatnonzero(mask)
        clf = make_knn(5).fit(Xtr[:, cols], ytr)
        return 0.99 * (1 - balanced_accuracy_score(yva, clf.predict(Xva[:, cols]))) + 0.01 * cols.size / d

    budget, seed = 400, 7
    params = BHOAParams(pop_size=10)
    results = {v: run_variant(v, d, fitness, budget, params, seed) for v in VARIANTS}
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    init0 = results["V0"]["initial_population"]
    check("1. Populasi awal identik V0..V3",
          all(np.array_equal(init0, r["initial_population"]) for r in results.values()))
    check("2. Permintaan fitness == anggaran untuk semua varian",
          all(r["requests"] == budget for r in results.values()))
    check("3. requests == evals + cache_hits",
          all(r["requests"] == r["evals"] + r["cache_hits"] for r in results.values()))
    check("4. Mask terbaik valid (biner, tidak kosong, dimensi benar)",
          all(r["best_mask"].shape == (d,) and set(np.unique(r["best_mask"])) <= {0, 1}
              and r["best_mask"].sum() > 0 for r in results.values()))
    lf_ok = all((r["log"]["lf_applied"] > 0) == VARIANTS[v]["use_lf"] for v, r in results.items())
    ls_ok = all((r["log"]["ls_trials"] > 0) == VARIANTS[v]["use_ls"] for v, r in results.items())
    check("5a. LF hanya aktif pada V1 dan V3", lf_ok)
    check("5b. LS hanya aktif pada V2 dan V3", ls_ok)
    again = {v: run_variant(v, d, fitness, budget, params, seed) for v in VARIANTS}
    check("6. Determinisme (seed sama -> hasil identik)",
          all(np.array_equal(results[v]["trace"], again[v]["trace"])
              and np.array_equal(results[v]["best_mask"], again[v]["best_mask"]) for v in VARIANTS))
    check("7. Trace monoton tidak naik & panjang == anggaran",
          all(len(r["trace"]) == budget and np.all(np.diff(r["trace"]) <= 1e-15)
              for r in results.values()))
    n = params.pop_size
    first_ls = results["V2"]["source_trace"].index("ls")
    check(f"8. V0 dan V2 identik sebelum LS pertama (permintaan 1..{first_ls})",
          np.array_equal(results["V0"]["trace"][:first_ls], results["V2"]["trace"][:first_ls])
          and first_ls >= 2 * n)

    print("\nRingkasan per varian (anggaran =", budget, "):")
    for v, r in results.items():
        last = r["diag"][-1]
        print(f"  {v}: fitness={r['best_fitness']:.5f} |b|={r['n_selected']}/{d} "
              f"evals={r['evals']} hits={r['cache_hits']} iter={r['log']['iterations']} "
              f"LF={r['log']['lf_applied']} LS={r['log']['ls_trials']}({r['log']['ls_improved']}) "
              f"diversity_akhir={last['diversity']:.3f} frac_at_bound={last['frac_at_bound']:.3f}")
    n_fail = sum(not ok for _, ok in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} pemeriksaan lulus.")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
