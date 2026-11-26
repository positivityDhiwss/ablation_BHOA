"""
run_pilot.py
============
Uji coba awal (pilot) studi ablasi LF x LS pada BHOA untuk seleksi fitur NIDS.

Contoh:
  # smoke test cepat (beberapa menit)
  py -3.10 -m ablation_bhoa.run_pilot --dataset both --quick

  # pilot sesuai proposal: 5 seed, anggaran 3.000 permintaan fitness, subsampel 5.000 baris
  py -3.10 -m ablation_bhoa.run_pilot --dataset both --seeds 5 --budget 3000

  # eksperimen utama (setelah parameter dikunci): 30 seed
  py -3.10 -m ablation_bhoa.run_pilot --dataset both --seeds 30 --budget <terkunci> --out results/main

Keluaran per dataset (folder --out/<dataset>/):
  runs.csv, convergence.csv, diagnostics.csv, feature_frequency.csv, baseline.csv,
  summary_variants.csv, contrasts_*.csv, plot_*.png, run_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import __version__
from .bhoa import VARIANT_NAMES, VARIANTS, BHOAParams, run_variant
from .data import (drop_constant_features, load_cicids, load_unsw, make_final_data,
                   make_search_data)
from .evaluation import FinalEvaluator, make_fitness
from .stats_analysis import ablation_contrasts, summarize_variants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser(description="Pilot studi ablasi Lévy Flight pada BHOA (NIDS)")
    ap.add_argument("--dataset", choices=["unsw", "cicids", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=5, help="jumlah blok seed (pilot=5, utama=30)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=3000, help="permintaan fitness per run")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--search-size", type=int, default=5000, help="ukuran subsampel pencarian")
    ap.add_argument("--val-ratio", type=float, default=0.30)
    ap.add_argument("--final-train-size", type=int, default=20000,
                    help="ukuran data latih akhir (0 = seluruh data pengembangan)")
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4, help="batas thread BLAS/OpenMP (sama utk semua varian)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--decay", choices=["exponential", "linear"], default="exponential")
    ap.add_argument("--lf-scale", type=float, default=1.0)
    ap.add_argument("--ls-trials", type=int, default=1)
    ap.add_argument("--x-bound", type=float, default=4.0, help="batas posisi kontinu [-x, x]")
    ap.add_argument("--g0", type=float, default=1.5, help="koefisien grazing awal")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 seed, anggaran 300, subsampel 2.000, latih akhir 5.000")
    ap.add_argument("--unsw-train", default=os.path.join(ROOT, "UNSW_NB15_training-set.csv"))
    ap.add_argument("--unsw-test", default=os.path.join(ROOT, "UNSW_NB15_testing-set.csv"))
    ap.add_argument("--cicids", default=os.path.join(ROOT, "CICIDS2017_clean_5pct.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pilot_ablation"))
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.budget, a.search_size, a.final_train_size = 2, 300, 2000, 5000
        a.n_boot = 2000
    return a


def environment_info(threads: int) -> dict:
    import scipy, sklearn
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(), "threads": threads,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "ablation_bhoa": __version__,
    }


def balanced_order(seed_index: int) -> list:
    """Urutan eksekusi varian diseimbangkan (Latin square 4x4) antar blok seed."""
    base = ["V0", "V1", "V2", "V3"]
    s = seed_index % 4
    return base[s:] + base[:s]


def run_dataset(name: str, bundle, args, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bundle = drop_constant_features(bundle)
    d = len(bundle.features)
    print(f"\n{'=' * 72}\n DATASET {bundle.name}: dev={len(bundle.dev):,} | test={len(bundle.test):,} "
          f"| fitur kandidat d={d}\n{'=' * 72}")

    sd = make_search_data(bundle, args.search_size, args.val_ratio, seed=args.seed_base)
    fd = make_final_data(bundle, args.final_train_size or None, seed=args.seed_base)
    print(f" search-train={sd.X_tr.shape} search-val={sd.X_val.shape} | "
          f"final-train={fd.X_tr.shape} test={fd.X_te.shape}")
    print(f" proporsi serangan: search-train={sd.y_tr.mean():.3f} test={fd.y_te.mean():.3f}")

    fitness_fn = make_fitness(sd, alpha=args.alpha, k=args.k)
    final_eval = FinalEvaluator(fd, k=args.k)
    params = BHOAParams(pop_size=args.pop, decay=args.decay, lf_scale=args.lf_scale,
                        ls_trials=args.ls_trials, x_bound=args.x_bound, g0=args.g0)

    # ---------------- Baseline seluruh fitur ----------------
    full = np.ones(d, dtype=np.uint8)
    base = final_eval.evaluate(full)
    base.update({"variant": "ALL", "n_selected": d, "search_fitness": fitness_fn(full)})
    pd.DataFrame([base]).to_csv(os.path.join(out_dir, "baseline.csv"), index=False)
    print(f" Baseline ALL: macroF1={base['macro_f1']:.4f} BA={base['balanced_accuracy']:.4f} "
          f"FPR={base['fpr']:.4f} | fitness(search-val)={base['search_fitness']:.5f}")

    rows, conv, diags, freq = [], [], [], []
    init_check = {}
    seeds = [args.seed_base + i for i in range(args.seeds)]
    for si, seed in enumerate(seeds):
        for v in balanced_order(si):
            res = run_variant(v, d, fitness_fn, args.budget, params, seed,
                              use_cache=not args.no_cache)
            init_check.setdefault(seed, res["initial_population"])
            assert np.array_equal(init_check[seed], res["initial_population"]), "populasi awal berbeda!"
            ev = final_eval.evaluate(res["best_mask"])
            row = {
                "dataset": bundle.name, "variant": v, "seed": seed,
                "fitness": res["best_fitness"], "n_selected": res["n_selected"],
                "reduction": 1 - res["n_selected"] / d,
                "requests": res["requests"], "evals": res["evals"], "cache_hits": res["cache_hits"],
                "search_wall_s": res["search_wall_s"], "search_cpu_s": res["search_cpu_s"],
                **{f"log_{k}": val for k, val in res["log"].items()},
                **ev,
                "mask": "".join(map(str, res["best_mask"].tolist())),
            }
            rows.append(row)
            tr = res["trace"]
            conv.append(pd.DataFrame({"variant": v, "seed": seed,
                                      "request": np.arange(1, len(tr) + 1), "best_fitness": tr}))
            dg = pd.DataFrame(res["diag"]); dg["variant"], dg["seed"] = v, seed
            diags.append(dg)
            freq.append(res["best_mask"])
            print(f"  seed={seed} {v:<2} ({VARIANT_NAMES[v]:<15}) fit={res['best_fitness']:.5f} "
                  f"|b|={res['n_selected']:>2}/{d} macroF1={ev['macro_f1']:.4f} "
                  f"FPR={ev['fpr']:.4f} req={res['requests']} eval={res['evals']} "
                  f"hit={res['cache_hits']} LF={res['log']['lf_applied']} "
                  f"LS={res['log']['ls_trials']}({res['log']['ls_improved']}) "
                  f"t={res['search_wall_s']:.1f}s")

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    conv_df = pd.concat(conv, ignore_index=True)
    conv_df.to_csv(os.path.join(out_dir, "convergence.csv"), index=False)
    pd.concat(diags, ignore_index=True).to_csv(os.path.join(out_dir, "diagnostics.csv"), index=False)

    # Frekuensi pemilihan fitur per varian
    fr = pd.DataFrame(np.vstack(freq), columns=bundle.features)
    fr["variant"] = runs["variant"].values
    fr.groupby("variant").mean().T.to_csv(os.path.join(out_dir, "feature_frequency.csv"))

    # Ringkasan dan kontras
    metrics = ["fitness", "macro_f1", "n_selected", "reduction", "recall_attack", "fpr",
               "balanced_accuracy", "accuracy", "search_wall_s"]
    summarize_variants(runs, metrics).to_csv(os.path.join(out_dir, "summary_variants.csv"))
    contrasts = []
    for metric, hib in (("fitness", False), ("macro_f1", True), ("n_selected", False),
                        ("search_wall_s", False)):
        c = ablation_contrasts(runs, metric, higher_is_better=hib, n_boot=args.n_boot)
        c["role"] = "primer (konfirmatori)" if metric == "fitness" else "pendukung (eksploratif)"
        contrasts.append(c)
    contrasts = pd.concat(contrasts, ignore_index=True)
    contrasts.to_csv(os.path.join(out_dir, "contrasts.csv"), index=False)

    make_plots(runs, conv_df, base, bundle.name, out_dir)

    cfg = {"args": vars(args), "params": params.to_dict(), "dataset_info": bundle.info,
           "candidate_features": bundle.features, "seeds": seeds,
           "n_columns_search": sd.prep.n_columns, "n_columns_final": fd.prep.n_columns,
           "env": environment_info(args.threads)}
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    print_summary(runs, contrasts, base, bundle.name)
    return runs, contrasts


def make_plots(runs, conv, base, title, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"V0": "#7f7f7f", "V1": "#1f77b4", "V2": "#ff7f0e", "V3": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for v, g in conv.groupby("variant"):
        piv = g.pivot_table(index="request", columns="seed", values="best_fitness")
        med = piv.median(axis=1)
        ax.plot(med.index, med.values, color=colors[v], label=f"{v} {VARIANT_NAMES[v]}")
        ax.fill_between(med.index, piv.quantile(.25, axis=1), piv.quantile(.75, axis=1),
                        color=colors[v], alpha=.15)
    ax.set_xlabel("Permintaan fitness terpakai"); ax.set_ylabel("Best-so-far fitness (median, IQR)")
    ax.set_title(f"Kurva konvergensi terhadap anggaran - {title}")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "plot_convergence.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = ["V0", "V1", "V2", "V3"]
    for ax, (m, lab) in zip(axes, [("fitness", "Fitness (search-val)"),
                                   ("macro_f1", "Macro-F1 (data uji)"),
                                   ("n_selected", "Jumlah fitur terpilih")]):
        data = [runs.loc[runs.variant == v, m].values for v in order]
        try:
            ax.boxplot(data, tick_labels=order)
        except TypeError:
            ax.boxplot(data, labels=order)
        if m == "macro_f1":
            ax.axhline(base["macro_f1"], ls="--", color="red", lw=1, label="ALL")
            ax.legend()
        ax.set_title(lab); ax.grid(alpha=.3)
    fig.suptitle(f"Distribusi hasil per konfigurasi - {title}")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "plot_boxplots.png"), dpi=200)
    plt.close(fig)

    # Interaction plot (rata-rata Y=-fitness)
    m = runs.groupby("variant")["fitness"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [m["V0"], m["V1"]], "o-", label="LS off (V0 -> V1)")
    ax.plot([0, 1], [m["V2"], m["V3"]], "s-", label="LS on (V2 -> V3)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LF off", "LF on"])
    ax.set_ylabel("Rata-rata fitness (lebih kecil lebih baik)")
    ax.set_title(f"Plot interaksi LF x LS - {title}"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "plot_interaction.png"), dpi=200)
    plt.close(fig)


def print_summary(runs, contrasts, base, title):
    print(f"\n--- Ringkasan {title} (rata-rata +- sd per konfigurasi) ---")
    agg = runs.groupby("variant").agg(
        fitness=("fitness", "mean"), fitness_sd=("fitness", "std"),
        macro_f1=("macro_f1", "mean"), n_sel=("n_selected", "mean"),
        fpr=("fpr", "mean"), time_s=("search_wall_s", "mean"))
    print(agg.round(5).to_string())
    print(f"ALL features: macro_f1={base['macro_f1']:.4f}, fpr={base['fpr']:.4f}")
    print(f"\n--- Kontras ablasi (fitness = primer; lainnya eksploratif) ---")
    cols = ["metric", "contrast", "mean", "ci95_low", "ci95_high", "wins", "ties", "losses",
            "p_primary", "p_holm", "rank_biserial"]
    print(contrasts[cols].to_string(index=False, float_format=lambda x: f"{x:.5g}"))


def main():
    args = parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        t0 = time.time()
        if args.dataset in ("unsw", "both"):
            run_dataset("unsw", load_unsw(args.unsw_train, args.unsw_test), args,
                        os.path.join(args.out, "UNSW-NB15"))
        if args.dataset in ("cicids", "both"):
            run_dataset("cicids", load_cicids(args.cicids, split_seed=2026), args,
                        os.path.join(args.out, "CIC-IDS2017"))
        print(f"\nSelesai dalam {(time.time() - t0) / 60:.1f} menit. Hasil: {args.out}")


if __name__ == "__main__":
    main()
