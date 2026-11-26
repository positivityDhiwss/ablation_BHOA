"""
bhoa.py
=======
Implementasi bersama (shared implementation) Binary Horse Herd Optimization Algorithm (BHOA)
dengan dua saklar komponen untuk studi ablasi faktorial 2x2:

    V0 = BHOA dasar            (LF=off, LS=off)
    V1 = BHOA + Lévy Flight    (LF=on,  LS=off)
    V2 = BHOA + Local Search   (LF=off, LS=on)
    V3 = BHOA + LF + LS        (LF=on,  LS=on)   ~ BinHOA (Elmanakhly dkk., 2022)

Prinsip pengendalian eksperimen yang diterapkan (sesuai Bab III proposal):
  1. Satu kode dasar; saklar LF/LS hanya mengaktifkan operator terkait.
  2. Anggaran evaluasi fitness (jumlah *permintaan* fitness) setara untuk semua varian.
     Setiap permintaan -- termasuk cache hit -- mengurangi anggaran satu unit.
     Operator dihentikan segera saat anggaran habis (BudgetExhausted).
  3. Populasi awal identik antar-varian untuk seed yang sama.
  4. Generator acak dipisah per komponen (init, hoa, binarisasi, LF, LS, repair) agar
     aktivasi suatu operator tidak menggeser urutan bilangan acak operator lain.
  5. Jadwal peluruhan koefisien HOA dinyatakan terhadap PROPORSI anggaran yang terpakai,
     bukan nomor iterasi, sehingga varian dengan LS (yang memakai evaluasi ekstra)
     tidak memperoleh jadwal yang berbeda.

CATATAN PENTING (Lampiran A proposal):
  Nilai default koefisien HOA, skala LF, dan detail LS di bawah ini adalah nilai AWAL
  untuk pilot. Seluruhnya WAJIB dicocokkan dengan persamaan/tabel pada sumber primer
  [MiarNaeimi dkk., 2021] dan [Elmanakhly dkk., 2022] lalu dikunci sebelum eksperimen utama.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional

import numpy as np


# =============================================================================
# 1. PARAMETER
# =============================================================================
@dataclass
class BHOAParams:
    pop_size: int = 20

    # --- Pembagian kelompok usia (urut fitness terbaik -> terburuk) ---
    frac_alpha: float = 0.10
    frac_beta: float = 0.20
    frac_gamma: float = 0.30          # sisanya = delta (40%)

    # --- Koefisien awal enam perilaku HOA (VERIFIKASI ke sumber primer) ---
    g0: float = 1.5                   # grazing
    u_graz: float = 1.05              # batas atas ruang grazing
    l_graz: float = 0.95              # batas bawah ruang grazing
    h_beta: float = 0.9               # hierarchy
    h_gamma: float = 0.5
    s_beta: float = 0.2               # sociability
    s_gamma: float = 0.1
    i_gamma: float = 0.3              # imitation
    i_delta: float = 0.3
    d_alpha: float = 0.5              # defense mechanism
    d_beta: float = 0.2
    d_gamma: float = 0.1
    r_gamma: float = 0.05             # roam
    r_delta: float = 0.1
    frac_imitation: float = 0.10      # pN: proporsi kuda terbaik untuk imitasi
    frac_defense: float = 0.20        # qN: proporsi kuda terburuk untuk defense

    # --- Jadwal peluruhan terhadap proporsi anggaran p in [0,1] ---
    decay: str = "exponential"        # "exponential": c0 * omega^(p*T_ref) | "linear": c0*(1-p)
    omega: float = 0.95

    # --- Domain posisi kontinu ---
    x_init: float = 2.0               # inisialisasi U(-x_init, x_init)
    x_bound: float = 4.0              # clipping posisi ke [-x_bound, x_bound] (adaptasi, didokumentasikan)
    #   Catatan pilot: grazing G = g(u + rho*l)X bersifat multiplikatif sehingga |X| cenderung
    #   membesar hingga batas. S(4)=0.982 -> masih ada peluang flip bit ~1.8%; S(6)=0.9975.
    #   Nilai ini adalah keputusan teknis yang harus dikunci melalui pilot (Lampiran A).

    # --- Lévy Flight (Mantegna) ---
    lf_beta: float = 1.5
    lf_scale: float = 1.0             # skala perturbasi
    lf_step_clip: float = 10.0        # batas |langkah Lévy| agar tidak meledak (adaptasi)

    # --- Local Search (RWS + Uniform Crossover) ---
    ls_trials: int = 1                # jumlah kandidat LS per iterasi
    ls_crossover_p: float = 0.5       # P(gen diambil dari x_best) pada uniform crossover
    ls_encode: float = 2.0            # kekuatan pengodean mask -> posisi kontinu untuk x_best

    def to_dict(self) -> dict:
        return asdict(self)


VARIANTS: Dict[str, Dict[str, bool]] = {
    "V0": {"use_lf": False, "use_ls": False},
    "V1": {"use_lf": True,  "use_ls": False},
    "V2": {"use_lf": False, "use_ls": True},
    "V3": {"use_lf": True,  "use_ls": True},
}
VARIANT_NAMES = {
    "V0": "BHOA dasar",
    "V1": "BHOA + LF",
    "V2": "BHOA + LS",
    "V3": "BHOA + LF + LS",
}


# =============================================================================
# 2. ANGGARAN EVALUASI + CACHE
# =============================================================================
class BudgetExhausted(Exception):
    """Dilempar ketika anggaran permintaan fitness telah habis."""


class BudgetedFitness:
    """
    Pembungkus fungsi fitness dengan penghitung anggaran terpusat dan cache per-run.

    - `requests`  : jumlah permintaan fitness (termasuk cache hit) -> mengurangi anggaran
    - `evals`     : jumlah evaluasi classifier aktual
    - `cache_hits`: jumlah permintaan yang dilayani cache
    - `trace`     : best-so-far fitness setelah setiap permintaan (panjang = anggaran)
    """

    def __init__(self, fitness_fn: Callable[[np.ndarray], float], budget: int, use_cache: bool = True):
        self.fitness_fn = fitness_fn
        self.budget = int(budget)
        self.use_cache = use_cache
        self.cache: Dict[bytes, float] = {}
        self.requests = 0
        self.evals = 0
        self.cache_hits = 0
        self.best_fitness = math.inf
        self.best_mask: Optional[np.ndarray] = None
        self.trace: List[float] = []
        self.source_trace: List[str] = []   # operator asal tiap permintaan

    @property
    def remaining(self) -> int:
        return self.budget - self.requests

    @property
    def progress(self) -> float:
        return min(1.0, self.requests / self.budget)

    def __call__(self, mask: np.ndarray, source: str) -> float:
        if self.requests >= self.budget:
            raise BudgetExhausted
        self.requests += 1
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if self.use_cache and key in self.cache:
            self.cache_hits += 1
            f = self.cache[key]
        else:
            f = float(self.fitness_fn(mask))
            self.evals += 1
            if self.use_cache:
                self.cache[key] = f
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_mask = mask.copy()
        self.trace.append(self.best_fitness)
        self.source_trace.append(source)
        return f


# =============================================================================
# 3. OPERATOR DASAR
# =============================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Fungsi transfer S-shaped: S(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def binarize(x: np.ndarray, rng_bin: np.random.Generator) -> np.ndarray:
    """b_j = 1 jika U(0,1) < S(x_j), selain itu 0."""
    return (rng_bin.random(x.shape) < sigmoid(x)).astype(np.uint8)


def repair_empty(mask: np.ndarray, rng_repair: np.random.Generator) -> np.ndarray:
    """Jika mask kosong, pilih satu fitur secara acak (prosedur identik untuk semua varian)."""
    if mask.sum() == 0:
        mask = mask.copy()
        mask[rng_repair.integers(0, mask.size)] = 1
    return mask


def mantegna_sigma(beta: float) -> float:
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
    return (num / den) ** (1.0 / beta)


def levy_step(dim: int, beta: float, rng_lf: np.random.Generator, sigma_u: float) -> np.ndarray:
    """Langkah Lévy metode Mantegna: L = u / |v|^(1/beta), u~N(0, sigma_u^2), v~N(0,1)."""
    u = rng_lf.normal(0.0, sigma_u, dim)
    v = rng_lf.normal(0.0, 1.0, dim)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-12)


# =============================================================================
# 4. OPTIMIZER BHOA DENGAN SAKLAR LF / LS
# =============================================================================
class BHOA:
    def __init__(self, dim: int, fitness: BudgetedFitness, params: BHOAParams,
                 use_lf: bool, use_ls: bool, seed: int):
        self.dim = dim
        self.f = fitness
        self.p = params
        self.use_lf = use_lf
        self.use_ls = use_ls
        self.seed = seed

        # Generator acak terpisah per komponen (SeedSequence -> stream independen)
        ss = np.random.SeedSequence(seed)
        streams = ss.spawn(6)
        self.rng_init = np.random.default_rng(streams[0])
        self.rng_hoa = np.random.default_rng(streams[1])
        self.rng_bin = np.random.default_rng(streams[2])
        self.rng_lf = np.random.default_rng(streams[3])
        self.rng_ls = np.random.default_rng(streams[4])
        self.rng_repair = np.random.default_rng(streams[5])

        self.sigma_u = mantegna_sigma(params.lf_beta)
        self.t_ref = max(1.0, fitness.budget / params.pop_size)   # iterasi nominal V0

        # Log operator & diagnostik
        self.log = {"hoa_updates": 0, "lf_applied": 0, "ls_trials": 0, "ls_improved": 0,
                    "iterations": 0}
        self.diag: List[dict] = []
        self.initial_population: Optional[np.ndarray] = None

    # --------------------------------------------------------------- helpers
    def _decay(self) -> float:
        prog = self.f.progress
        if self.p.decay == "linear":
            return max(0.0, 1.0 - prog)
        return self.p.omega ** (prog * self.t_ref)

    def _make_mask(self, x: np.ndarray) -> np.ndarray:
        return repair_empty(binarize(x, self.rng_bin), self.rng_repair)

    def _age_groups(self, order: np.ndarray):
        n = self.p.pop_size
        na = max(1, int(round(self.p.frac_alpha * n)))
        nb = max(1, int(round(self.p.frac_beta * n)))
        nc = max(1, int(round(self.p.frac_gamma * n)))
        group = np.empty(n, dtype="<U1")
        group[order[:na]] = "a"
        group[order[na:na + nb]] = "b"
        group[order[na + nb:na + nb + nc]] = "c"
        group[order[na + nb + nc:]] = "d"
        return group

    # --------------------------------------------------------------- main
    def run(self) -> dict:
        p, n, D = self.p, self.p.pop_size, self.dim
        t0_wall, t0_cpu = time.perf_counter(), time.process_time()

        X = self.rng_init.uniform(-p.x_init, p.x_init, size=(n, D))
        self.initial_population = X.copy()
        B = np.zeros((n, D), dtype=np.uint8)
        F = np.full(n, np.inf)
        best_x = X[0].copy()
        best_f = math.inf

        try:
            # ---- Inisialisasi & evaluasi populasi awal ----
            for m in range(n):
                B[m] = self._make_mask(X[m])
                F[m] = self.f(B[m], "init")
                if F[m] < best_f:
                    best_f, best_x = F[m], X[m].copy()

            # ---- Loop utama sampai anggaran habis ----
            while True:
                self.log["iterations"] += 1
                w = self._decay()
                order = np.argsort(F, kind="stable")
                group = self._age_groups(order)
                mean_all = X.mean(axis=0)
                pN = max(1, int(round(p.frac_imitation * n)))
                qN = max(1, int(round(p.frac_defense * n)))
                mean_best = X[order[:pN]].mean(axis=0)
                mean_worst = X[order[-qN:]].mean(axis=0)
                self._record_diag(X, B, best_f)

                for m in range(n):
                    xm = X[m]
                    rho = self.rng_hoa.random(D)
                    G = (p.g0 * w) * (p.u_graz + rho * p.l_graz) * xm
                    g = group[m]
                    if g == "a":
                        V = G - (p.d_alpha * w) * (mean_worst - xm)
                    elif g == "b":
                        V = (G + (p.h_beta * w) * (best_x - xm)
                               + (p.s_beta * w) * (mean_all - xm)
                               - (p.d_beta * w) * (mean_worst - xm))
                    elif g == "c":
                        V = (G + (p.h_gamma * w) * (best_x - xm)
                               + (p.s_gamma * w) * (mean_all - xm)
                               + (p.i_gamma * w) * (mean_best - xm)
                               - (p.d_gamma * w) * (mean_worst - xm)
                               + (p.r_gamma * w) * self.rng_hoa.random(D) * xm)
                    else:
                        V = ((G + (p.i_delta * w) * (mean_best - xm))
                             + (p.r_delta * w) * self.rng_hoa.random(D) * xm)
                    x_new = xm + V
                    self.log["hoa_updates"] += 1

                    # ---- Operator Lévy Flight (saklar LF) ----
                    if self.use_lf:
                        L = np.clip(levy_step(D, p.lf_beta, self.rng_lf, self.sigma_u),
                                    -p.lf_step_clip, p.lf_step_clip)
                        r = self.rng_lf.random()
                        sgn = np.sign(self.rng_lf.random(D) - 0.5)
                        x_new = x_new + p.lf_scale * r * sgn * L
                        self.log["lf_applied"] += 1

                    X[m] = np.clip(x_new, -p.x_bound, p.x_bound)
                    B[m] = self._make_mask(X[m])
                    F[m] = self.f(B[m], "lf" if self.use_lf else "hoa")
                    if F[m] < best_f:
                        best_f, best_x = F[m], X[m].copy()

                # ---- Operator Local Search: RWS + Uniform Crossover (saklar LS) ----
                if self.use_ls:
                    best_mask = self.f.best_mask
                    for _ in range(p.ls_trials):
                        quality = 1.0 / (F + 1e-12)
                        prob = quality / quality.sum()
                        k = int(self.rng_ls.choice(n, p=prob))
                        take_best = self.rng_ls.random(D) < p.ls_crossover_p
                        cand = np.where(take_best, best_mask, B[k]).astype(np.uint8)
                        cand = repair_empty(cand, self.rng_repair)
                        self.log["ls_trials"] += 1
                        fc = self.f(cand, "ls")
                        if fc < best_f:
                            # Penerimaan: kandidat menggantikan solusi terbaik; posisi kontinu
                            # dikodekan dari mask agar mempengaruhi perilaku hierarchy.
                            best_f = fc
                            best_x = np.where(cand == 1, p.ls_encode, -p.ls_encode).astype(float)
                            best_mask = cand
                            self.log["ls_improved"] += 1
                            # ganti kuda terburuk dengan kandidat LS
                            worst = int(np.argmax(F))
                            X[worst], B[worst], F[worst] = best_x.copy(), cand, fc
        except BudgetExhausted:
            pass

        wall = time.perf_counter() - t0_wall
        cpu = time.process_time() - t0_cpu
        mask = self.f.best_mask
        return {
            "best_mask": mask,
            "best_fitness": self.f.best_fitness,
            "n_selected": int(mask.sum()),
            "trace": np.asarray(self.f.trace),
            "requests": self.f.requests,
            "evals": self.f.evals,
            "cache_hits": self.f.cache_hits,
            "search_wall_s": wall,
            "search_cpu_s": cpu,
            "log": dict(self.log),
            "diag": self.diag,
        }

    def _record_diag(self, X: np.ndarray, B: np.ndarray, best_f: float) -> None:
        """Diagnostik eksploratif: saturasi posisi & keragaman mask (Hamming rata-rata)."""
        n = B.shape[0]
        Bf = B.astype(np.float32)
        ham = (Bf.sum(1)[:, None] + Bf.sum(1)[None, :] - 2 * Bf @ Bf.T) / B.shape[1]
        diversity = float(ham[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0
        self.diag.append({
            "requests": self.f.requests,
            "best_f": best_f,
            "frac_at_bound": float(np.mean(np.abs(X) >= self.p.x_bound - 1e-9)),
            "mean_abs_x": float(np.mean(np.abs(X))),
            "diversity": diversity,
        })


def run_variant(variant: str, dim: int, fitness_fn: Callable[[np.ndarray], float],
                budget: int, params: BHOAParams, seed: int, use_cache: bool = True) -> dict:
    """Menjalankan satu run untuk varian V0..V3 dengan anggaran dan seed tertentu."""
    sw = VARIANTS[variant]
    bf = BudgetedFitness(fitness_fn, budget, use_cache=use_cache)
    opt = BHOA(dim, bf, params, use_lf=sw["use_lf"], use_ls=sw["use_ls"], seed=seed)
    res = opt.run()
    res["variant"] = variant
    res["seed"] = seed
    res["initial_population"] = opt.initial_population
    res["source_trace"] = bf.source_trace
    return res
