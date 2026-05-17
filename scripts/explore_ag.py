"""scripts/explore_ag.py (v2) — per-regime breakdown including VCG variants.

v2 changes: include VCG-original, VCG-Huber c=0.05, VCG-L1Huber c=0.05 in
the per-regime breakdown so we see WHERE our VCG variants win / lose.

Agreement regimes are defined by the K discriminators' argmax_t:
  - "Unanimous":     all K agree on the same step
  - "2-of-3 agree":  exactly two agree
  - "All differ":    all K argmax_t are distinct

Key diagnostic question: on the hardest regime (all differ), does VCG-L1Huber
on HC and Best-single on AG come from genuine within-regime improvement,
or are they riding on the trivial unanimous cases?

Run from repo root:
    python scripts/explore_ag.py
"""

import argparse
import sys
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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def load_trace(path):
    npz = np.load(path, allow_pickle=True)
    probs = np.asarray(npz[_first_present(npz, PROB_KEYS)], dtype=float)
    gt = _safe_int(npz, _first_present(npz, GT_STEP_KEYS))
    return probs, gt


# ---------- VCG predictors ----------

def pred_vcg_original(P, eps=1e-6):
    return int(np.argmax(solve_allocation(P, eps=eps).d))


def pred_vcg_huber(P, c=0.05, eps=1e-6):
    return int(np.argmax(solve_allocation_huber(P, c=c, eps=eps).d))


def pred_vcg_l1huber(P, c=0.05, eps=1e-6):
    return int(np.argmax(solve_allocation_l1huber(P, c=c, eps=eps).d))


# ---------- selection mechanisms ----------

def sel_max_prob(P):
    return int(np.argmax(P.max(axis=1)))


def sel_neg_entropy(P, eps=1e-12):
    p_norm = P / P.sum(axis=1, keepdims=True)
    entropy = -(p_norm * np.log(p_norm + eps)).sum(axis=1)
    return int(np.argmin(entropy))


def sel_peak_minus_mean(P):
    return int(np.argmax(P.max(axis=1) - P.mean(axis=1)))


def sel_peak_separation(P):
    sorted_desc = -np.sort(-P, axis=1)
    if sorted_desc.shape[1] < 2:
        return int(np.argmax(sorted_desc[:, 0]))
    sep = sorted_desc[:, 0] - sorted_desc[:, 1]
    return int(np.argmax(sep))


def sel_gini(P):
    K, T = P.shape
    p_norm = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    gini = np.zeros(K)
    for k in range(K):
        sorted_p = np.sort(p_norm[k])
        i = np.arange(1, T + 1)
        gini[k] = (2 * (i * sorted_p).sum() - (T + 1) * sorted_p.sum()) \
                  / max(T * sorted_p.sum(), 1e-12)
    return int(np.argmax(gini))


# ---------- per-discriminator normalised aggregators ----------

def agg_raw_mean(P):
    return P.mean(axis=0)


def agg_zscore_mean(P, eps=1e-6):
    mu = P.mean(axis=1, keepdims=True)
    sd = P.std(axis=1, keepdims=True) + eps
    return ((P - mu) / sd).mean(axis=0)


def agg_minmax_mean(P, eps=1e-6):
    lo = P.min(axis=1, keepdims=True)
    hi = P.max(axis=1, keepdims=True)
    return ((P - lo) / (hi - lo + eps)).mean(axis=0)


def agg_rank_mean(P):
    ranks = np.argsort(np.argsort(P, axis=1), axis=1).astype(float)
    return ranks.mean(axis=0)


# ---------- predictors ----------

def pred_select(P, sel_fn):
    k = sel_fn(P)
    return int(np.argmax(P[k]))


def pred_agg(P, agg_fn):
    return int(np.argmax(agg_fn(P)))


def pred_consensus_or(P, fallback_pred):
    argmax_per_k = P.argmax(axis=1)
    unique = set(int(x) for x in argmax_per_k)
    if len(unique) == 1:
        return int(argmax_per_k[0])
    return fallback_pred(P)


# ---------- evaluation ----------

def evaluate_with_regimes(traces, predict_fn):
    regimes = {"unanimous": [0, 0], "two_of_three": [0, 0], "all_differ": [0, 0]}
    total_hit = total_n = 0
    for P, gt in traces:
        if gt is None or not (0 <= gt < P.shape[1]):
            continue
        argmax_per_k = P.argmax(axis=1)
        unique = set(int(x) for x in argmax_per_k)
        if len(unique) == 1:
            regime = "unanimous"
        elif len(unique) == 2:
            regime = "two_of_three"
        else:
            regime = "all_differ"
        try:
            pred = predict_fn(P)
        except Exception:
            pred = -1
        hit = int(pred == gt)
        regimes[regime][0] += hit
        regimes[regime][1] += 1
        total_hit += hit
        total_n += 1
    return total_hit, total_n, regimes


def fmt_pct(h, n):
    if n == 0:
        return "  -  "
    return f"{h/n:>5.1%}"


def run_subset(subset, data_root):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] no .npz under {subset_dir}", file=sys.stderr)
        return
    traces = [load_trace(p) for p in files]
    valid = [t for t in traces if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    n_valid = len(valid)
    K = traces[0][0].shape[0]
    print(f"  {n_valid} valid traces, K = {K}")

    # Agreement regime diagnostic.
    regime_counts = {"unanimous": 0, "two_of_three": 0, "all_differ": 0}
    unanimous_correct = 0
    for P, gt in valid:
        argmax_per_k = P.argmax(axis=1)
        unique = set(int(x) for x in argmax_per_k)
        if len(unique) == 1:
            regime_counts["unanimous"] += 1
            if argmax_per_k[0] == gt:
                unanimous_correct += 1
        elif len(unique) == 2:
            regime_counts["two_of_three"] += 1
        else:
            regime_counts["all_differ"] += 1
    print(f"\n  Agreement regimes (across K={K} discriminators' argmax_t):")
    print(f"    Unanimous:     {regime_counts['unanimous']:>3d}/{n_valid}  "
          f"({regime_counts['unanimous']/n_valid:>5.1%})  "
          f"-- of these, {unanimous_correct} correct "
          f"({unanimous_correct/max(regime_counts['unanimous'],1):.1%})")
    print(f"    2-of-3 agree:  {regime_counts['two_of_three']:>3d}/{n_valid}  "
          f"({regime_counts['two_of_three']/n_valid:>5.1%})")
    print(f"    All differ:    {regime_counts['all_differ']:>3d}/{n_valid}  "
          f"({regime_counts['all_differ']/n_valid:>5.1%})")

    methods = []

    # Simple baselines.
    methods.append(("Mean(prob)        ",       lambda P: pred_agg(P, agg_raw_mean)))
    methods.append(("Median(prob)      ",       lambda P: int(np.argmax(np.median(P, axis=0)))))

    # VCG family (the focus of the project).
    methods.append(("VCG-original   R=0", pred_vcg_original))
    methods.append(("VCG-Huber      c=0.05", pred_vcg_huber))
    methods.append(("VCG-L1Huber    c=0.05", pred_vcg_l1huber))

    # Per-discriminator argmax baselines.
    for k in range(K):
        methods.append((f"Discriminator D{k}    ",
                        lambda P, kk=k: int(np.argmax(P[kk]))))

    # Selection mechanisms.
    sel_list = [
        ("Select max-prob   ",  sel_max_prob),
        ("Select neg-entropy",  sel_neg_entropy),
        ("Select peak-mean  ",  sel_peak_minus_mean),
        ("Select peak-sep   ",  sel_peak_separation),
        ("Select Gini       ",  sel_gini),
    ]
    for name, sf in sel_list:
        methods.append((name, lambda P, sf_=sf: pred_select(P, sf_)))

    # Normalised aggregators.
    for name, af in [("Agg z-score mean  ", agg_zscore_mean),
                     ("Agg min-max mean  ", agg_minmax_mean),
                     ("Agg rank mean     ", agg_rank_mean)]:
        methods.append((name, lambda P, af_=af: pred_agg(P, af_)))

    # Consensus hybrids.
    for name, sf in sel_list:
        methods.append((
            f"Cons-or [{name.strip()}]",
            (lambda P, sf_=sf:
                pred_consensus_or(P, lambda Q, sf_inner=sf_: pred_select(Q, sf_inner))),
        ))

    # Evaluate.
    print(f"\n  {'Method':<32s}    {'Overall':<8s}  "
          f"{'Unan.':<6s}  {'2-of-3':<6s}  {'Diff.':<6s}")
    print(f"  {'-'*32}    {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name, fn in methods:
        h, n, reg = evaluate_with_regimes(valid, fn)
        overall = f"{h/max(n,1):>5.1%} ({h:>3d}/{n:<3d})"
        u  = fmt_pct(reg["unanimous"][0],     reg["unanimous"][1])
        t  = fmt_pct(reg["two_of_three"][0],  reg["two_of_three"][1])
        d  = fmt_pct(reg["all_differ"][0],    reg["all_differ"][1])
        print(f"  {name:<32s}    {overall:<8s}  {u:<6s}  {t:<6s}  {d:<6s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="both",
                    choices=["Hand-Crafted", "Algorithm-Generated", "both"])
    args = ap.parse_args()

    subsets = (["Hand-Crafted", "Algorithm-Generated"]
               if args.subset == "both" else [args.subset])
    for s in subsets:
        print(f"\n{'='*86}\nSUBSET: {s}\n{'='*86}")
        run_subset(s, args.data_root)


if __name__ == "__main__":
    main()