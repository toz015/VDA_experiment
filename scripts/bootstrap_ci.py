"""scripts/bootstrap_ci.py — bootstrap CIs + McNemar tests for the main results.

For each subset:
  1. Per-method accuracy with 95% bootstrap CI.
  2. Paired bootstrap CIs and McNemar exact tests for key A-vs-B comparisons.
     - Bootstrap reports the percentile 95% CI on (acc_A - acc_B).
     - McNemar (exact) is the standard test for paired binary outcomes and
       is usually more powerful than bootstrap on small samples; we report
       both. The two should agree on direction but McNemar's p-value is
       the one to cite.

Run from repo root:
    python scripts/bootstrap_ci.py
    python scripts/bootstrap_ci.py --n-boot 20000   # tighter CIs
"""

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.allocation import solve_allocation
from vcg.allocation_huber import solve_allocation_huber
from vcg.allocation_l1huber import solve_allocation_l1huber


PROB_KEYS    = ["theta_hat", "probs", "prob", "p", "scores", "theta", "raw"]
GT_STEP_KEYS = ["mistake_step", "gt_step", "true_step", "label_step"]


def _first_present(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def _safe_int(npz, key):
    if not key:
        return None
    try:
        val = np.asarray(npz[key]).item()
    except (ValueError, AttributeError):
        return None
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def load_trace(path):
    npz = np.load(path, allow_pickle=True)
    pk = _first_present(npz, PROB_KEYS)
    probs = np.asarray(npz[pk], dtype=float)
    gt_step = _safe_int(npz, _first_present(npz, GT_STEP_KEYS))
    return probs, gt_step


# ---------- predictors ----------

def pred_simple(P, kind, eps):
    Pc = np.clip(P, eps, 1 - eps)
    if kind == "mean":
        s = Pc.mean(axis=0)
    elif kind == "median":
        s = np.median(Pc, axis=0)
    elif kind == "mean_logit":
        s = np.log(Pc / (1 - Pc)).mean(axis=0)
    elif kind == "max":
        s = Pc.max(axis=0)
    return int(np.argmax(s))


def pred_vcg_original(P, eps=1e-6):
    return int(np.argmax(solve_allocation(P, eps=eps).d))


def pred_vcg_huber(P, c=0.05, eps=1e-6):
    return int(np.argmax(solve_allocation_huber(P, c=c, eps=eps).d))


def pred_vcg_l1huber(P, c=0.05, eps=1e-6):
    return int(np.argmax(solve_allocation_l1huber(P, c=c, eps=eps).d))


def pred_vcg_select_confidence(P):
    k_star = int(np.argmax(P.max(axis=1)))
    return int(np.argmax(P[k_star]))


# ---------- statistical machinery ----------

def compute_correctness(traces, predict_fn):
    """Return (n_valid,) int array of 0/1 per valid trace, in trace order."""
    out = []
    for P, gt in traces:
        if gt is None or not (0 <= gt < P.shape[1]):
            continue
        try:
            pred = predict_fn(P)
            out.append(int(pred == gt))
        except Exception:
            out.append(0)
    return np.asarray(out, dtype=int)


def bootstrap_single_acc(correct, n_boot, rng):
    n = len(correct)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = correct[idx].mean()
    return {
        "acc":  float(correct.mean()),
        "lo":   float(np.quantile(boot, 0.025)),
        "hi":   float(np.quantile(boot, 0.975)),
    }


def bootstrap_paired_diff(cor_a, cor_b, n_boot, rng):
    """Bootstrap 95% CI on (acc_a - acc_b)."""
    assert len(cor_a) == len(cor_b)
    n = len(cor_a)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = cor_a[idx].mean() - cor_b[idx].mean()
    return {
        "diff": float(cor_a.mean() - cor_b.mean()),
        "lo":   float(np.quantile(diffs, 0.025)),
        "hi":   float(np.quantile(diffs, 0.975)),
    }


def mcnemar_exact(cor_a, cor_b):
    """Exact two-sided McNemar test for paired binary outcomes.

    H0: P(A correct, B wrong) == P(A wrong, B correct).
    Under H0, conditional on the n_disc = n10 + n01 disagreement pairs,
    n10 ~ Binomial(n_disc, 0.5). Two-sided p-value via the exact binomial.
    """
    n10 = int(np.sum((cor_a == 1) & (cor_b == 0)))
    n01 = int(np.sum((cor_a == 0) & (cor_b == 1)))
    n_disc = n10 + n01
    if n_disc == 0:
        return {"n10": 0, "n01": 0, "p_value": 1.0}
    k = min(n10, n01)
    # Two-sided exact: 2 * P(X <= k) where X ~ Binomial(n_disc, 0.5).
    tail = sum(comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
    p_two = min(1.0, 2.0 * tail)
    return {"n10": n10, "n01": n01, "p_value": float(p_two)}


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "  "))


# ---------- main ----------

METHODS = [
    ("VCG-original R=0",         pred_vcg_original),
    ("Mean(prob) eps=1e-6",      lambda P: pred_simple(P, "mean", 1e-6)),
    ("Mean(prob) eps=0.10",      lambda P: pred_simple(P, "mean", 0.10)),
    ("Median(prob)",             lambda P: pred_simple(P, "median", 1e-6)),
    ("VCG-L1Huber c=0.05",       lambda P: pred_vcg_l1huber(P, c=0.05)),
    ("VCG-Huber   c=0.05",       lambda P: pred_vcg_huber(P, c=0.05)),
    ("VCG-Select(max-prob)",     pred_vcg_select_confidence),
]


def run_subset(subset, data_root, n_boot, seed):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        return None
    traces = [load_trace(p) for p in files]
    valid = [t for t in traces if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    K = traces[0][0].shape[0]
    n_valid = len(valid)
    print(f"  {n_valid} valid traces, K = {K}")

    # Best single (oracle-pick).
    per_k = [(k, compute_correctness(traces, lambda P, kk=k: int(np.argmax(P[kk]))))
             for k in range(K)]
    best_k, best_cor = max(per_k, key=lambda x: x[1].mean())
    best_label = f"Best single (D{best_k})"

    cor = {best_label: best_cor}
    for name, fn in METHODS:
        cor[name] = compute_correctness(traces, fn)

    # 1. Single-method accuracy + bootstrap CI.
    rng = np.random.default_rng(seed)
    print(f"\n  Single-method accuracy   (95% bootstrap CI, B={n_boot}):")
    print(f"  {'Method':<28s}  {'Acc':>6s}    {'95% CI':<22s}")
    print(f"  {'-'*28}  {'-'*6}    {'-'*22}")
    order = [best_label] + [n for n, _ in METHODS]
    for name in order:
        r = bootstrap_single_acc(cor[name], n_boot, np.random.default_rng(seed))
        print(f"  {name:<28s}  {r['acc']:>5.1%}    "
              f"[{r['lo']:>5.1%}, {r['hi']:>5.1%}]")

    # 2. Paired comparisons.
    if subset == "Hand-Crafted":
        comparisons = [
            ("VCG-L1Huber c=0.05",   "VCG-original R=0"),
            ("VCG-L1Huber c=0.05",   "Mean(prob) eps=1e-6"),
            ("VCG-L1Huber c=0.05",   "Median(prob)"),
            ("VCG-L1Huber c=0.05",   best_label),
            ("Median(prob)",         "VCG-original R=0"),
        ]
    else:
        comparisons = [
            ("VCG-Select(max-prob)", "VCG-original R=0"),
            ("VCG-Select(max-prob)", "Mean(prob) eps=0.10"),
            ("VCG-Select(max-prob)", best_label),
            ("VCG-Huber   c=0.05",   "VCG-original R=0"),
            ("VCG-Huber   c=0.05",   "Mean(prob) eps=0.10"),
        ]

    print(f"\n  Paired comparisons (A vs B):  diff = acc_A - acc_B")
    print(f"  {'A':<22s} vs {'B':<22s}    "
          f"{'diff':>7s}   {'95% boot CI':<20s}   "
          f"{'n10/n01':>9s}  {'McNemar p':>9s}")
    print(f"  {'-'*22}    {'-'*22}    "
          f"{'-'*7}   {'-'*20}   {'-'*9}  {'-'*11}")
    for a, b in comparisons:
        if a not in cor or b not in cor:
            continue
        boot = bootstrap_paired_diff(cor[a], cor[b], n_boot,
                                     np.random.default_rng(seed + 1))
        mc   = mcnemar_exact(cor[a], cor[b])
        print(f"  {a:<22s} vs {b:<22s}    "
              f"{boot['diff']:>+6.1%}   "
              f"[{boot['lo']:>+5.1%}, {boot['hi']:>+5.1%}]   "
              f"{mc['n10']:>3d}/{mc['n01']:<3d}  "
              f"{mc['p_value']:>7.4f} {stars(mc['p_value'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for subset in ["Hand-Crafted", "Algorithm-Generated"]:
        print(f"\n{'='*86}\nSUBSET: {subset}\n{'='*86}")
        run_subset(subset, args.data_root, args.n_boot, args.seed)

    print("\n  Significance markers:  *** p < 0.01    ** p < 0.05    * p < 0.10")
    print("  McNemar (exact binomial on paired-disagreement counts) is the test")
    print("  to cite for paired binary outcomes; bootstrap CI is a sanity check.")
    print("  n10/n01: # traces where A right & B wrong / A wrong & B right.")


if __name__ == "__main__":
    main()