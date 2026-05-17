"""scripts/cross_trace_prior.py — two-pass mechanism with a cross-trace prior.

Pass 1: For each trace, compute the L1-Huber VCG payment Pi_k for each
        discriminator k (leave-one-out: solve allocation with k removed,
        measure the change in others' welfare).

Pass 2: For each trace i, form a leave-one-out cross-trace prior
            pi_k = aggregate(Pi_k over all traces j != i)
        and use it as either
          - a hard selection signal:
                k^* = argmax_k pi_k,  predict argmax_t theta_hat[k^*, t]
          - or a multiplicative augmentation of cross-step reputation:
                w_k^t  ->  pi_k * w_k^t
            in the L1-Huber Gauss--Seidel allocation.

The leave-one-out aggregation avoids in-sample contamination: every test
trace is predicted from a prior that excludes its own payment information.

Self-contained: defines its own L1-Huber kernel/median helpers and only
imports the public solver from vcg/allocation_l1huber.py.

Run from repo root:
    python scripts/cross_trace_prior.py
    python scripts/cross_trace_prior.py --subset both
    python scripts/cross_trace_prior.py --c 0.10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


# -------- inline L1-Huber helpers (self-contained) --------

def l1h_factors(d, theta, c):
    """kappa_c(d - theta) = 1 - min(|d - theta|, c), elementwise.

    Broadcasts: d in shape (T,), theta in shape (K, T) -> output (K, T)."""
    return 1.0 - np.minimum(np.abs(d - theta), c)


def weighted_median(values, weights):
    """Weighted median with the usual (lower median) tie-break.

    Both arguments are 1-D arrays of equal length. Returns the smallest
    value v such that the cumulative weight up to v is >= total/2."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v_s = v[order]
    w_s = w[order]
    cumw = np.cumsum(w_s)
    total = cumw[-1] if cumw.size > 0 else 0.0
    if total <= 0:
        return float(np.median(v))
    idx = int(np.searchsorted(cumw, total / 2.0))
    if idx >= v_s.size:
        idx = v_s.size - 1
    return float(v_s[idx])


# -------- L1-Huber payment (leave-one-out) --------

def l1huber_payments(theta, c=0.05, eps=1e-6):
    """Return (payments[K], avg_reputation[K], d_full[T])."""
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1 - eps)
    K, T = theta.shape

    res_full = solve_allocation_l1huber(theta, c=c, eps=eps)
    d_full = np.asarray(res_full.d, dtype=np.float64)

    factors_full = l1h_factors(d_full, theta, c)   # (K, T)
    v_full = factors_full.prod(axis=1)              # (K,)

    # Average reputation: mean over t of w_k^t = product_{t' != t} kappa_{k,t'}.
    safe = factors_full > 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        w_full = np.where(safe,
                          v_full[:, None] / np.where(safe, factors_full, 1.0),
                          0.0)
    avg_repu = w_full.mean(axis=1)                  # (K,)

    payments = np.zeros(K)
    for k in range(K):
        mask = np.ones(K, dtype=bool)
        mask[k] = False
        theta_mk = theta[mask]
        # Warm-start from d_full for speed.
        try:
            res_mk = solve_allocation_l1huber(theta_mk, c=c, eps=eps,
                                              d_init=d_full)
        except TypeError:
            # solver may not accept d_init kwarg; fall back.
            res_mk = solve_allocation_l1huber(theta_mk, c=c, eps=eps)
        d_mk = np.asarray(res_mk.d, dtype=np.float64)
        factors_mk = l1h_factors(d_mk, theta_mk, c)
        v_mk = factors_mk.prod(axis=1)
        s_at_dmk   = float(v_mk.sum())
        s_at_dfull = float(v_full[mask].sum())
        payments[k] = s_at_dmk - s_at_dfull

    return payments, avg_repu, d_full


# -------- L1-Huber allocation with multiplicative prior pi --------

def solve_l1huber_with_prior(theta, pi, c=0.05, L=50, tau=1e-10, eps=1e-6):
    """Gauss--Seidel L1-Huber with reputation weight multiplied by pi[k].

    pi should be normalised so that uniform pi reproduces the un-augmented
    behaviour. We do not enforce a particular normalisation: the argmax
    of the final allocation is invariant to a global scalar."""
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1 - eps)
    K, T = theta.shape
    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape != (K,):
        raise ValueError(f"pi shape {pi.shape} != ({K},)")

    d = np.median(theta, axis=0).copy()

    def V_pi(d_arr):
        factors = l1h_factors(d_arr, theta, c)
        return float((pi * factors.prod(axis=1)).sum())

    V_old = V_pi(d)
    for _ in range(L):
        for t in range(T):
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            factors_excl = l1h_factors(d[mask], theta[:, mask], c)
            w_t = factors_excl.prod(axis=1) * pi    # AUGMENTED weight

            dev = np.abs(d[t] - theta[:, t])
            active = dev < c
            if active.any():
                d_new = weighted_median(theta[active, t], w_t[active])
                d_new = float(np.clip(d_new, eps, 1 - eps))
                d_old_t = float(d[t])
                d[t] = d_new
                if V_pi(d) < V_old - 1e-9:
                    d[t] = d_old_t

        V_new = V_pi(d)
        if abs(V_new - V_old) < tau:
            break
        V_old = V_new
    return d


# -------- predictors --------

def pred_l1huber_baseline(P, c):
    return int(np.argmax(np.asarray(solve_allocation_l1huber(P, c=c).d)))


def pred_max_prob_select(P):
    k_star = int(np.argmax(P.max(axis=1)))
    return int(np.argmax(P[k_star]))


def pred_pi_select(P, pi):
    k_star = int(np.argmax(pi))
    return int(np.argmax(P[k_star]))


def pred_pi_weighted(P, pi, c):
    d = solve_l1huber_with_prior(P, pi, c=c)
    return int(np.argmax(d))


def pred_best_single(P, k):
    return int(np.argmax(P[k]))


# -------- evaluation --------

def evaluate(traces, predict_fn):
    hits = total = 0
    for P, gt in traces:
        if gt is None or not (0 <= gt < P.shape[1]):
            continue
        total += 1
        try:
            if predict_fn(P) == gt:
                hits += 1
        except Exception:
            pass
    return hits, total


def normalize_for_prior(pi_raw, K_for_floor=None, floor_frac=1e-2):
    """Clip negatives, add a small floor (so every k > 0), normalise so
    sum(pi) == K. (Sum == K means a uniform pi gives back pi[k] == 1 for
    all k, i.e. the un-augmented behaviour.)"""
    pi_clip = np.maximum(pi_raw, 0.0)
    K = len(pi_clip)
    if pi_clip.sum() <= 0:
        return np.ones_like(pi_clip)
    s = pi_clip.sum()
    pi = pi_clip + (floor_frac * s / K)
    return pi * (K / pi.sum())


def run_subset(subset, data_root, c):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] no .npz under {subset_dir}", file=sys.stderr)
        return
    traces = [load_trace(p) for p in files]
    valid = [(t[0], t[1]) for t in traces
             if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    n = len(valid)
    K = traces[0][0].shape[0]
    print(f"  {n} valid / {len(traces)} traces, K = {K}, c = {c}")

    # -------- PASS 1 --------
    print(f"\n  Pass 1: computing L1-Huber payment for each trace...")
    t0 = time.time()
    all_payments = np.zeros((n, K))
    all_avg_repu = np.zeros((n, K))
    for i, (P, _gt) in enumerate(valid):
        pay, repu, _ = l1huber_payments(P, c=c)
        all_payments[i] = pay
        all_avg_repu[i] = repu
    print(f"  Pass 1 done in {time.time() - t0:.1f}s")

    # Diagnostic.
    print(f"\n  Cross-trace aggregated statistics (n={n} traces):")
    print(f"    Sum of payments per k:    "
          f"{np.array2string(all_payments.sum(axis=0), precision=3)}")
    print(f"    Mean of payments per k:   "
          f"{np.array2string(all_payments.mean(axis=0), precision=4)}")
    print(f"    Frac of traces Pi_k > 0:  "
          f"{np.array2string((all_payments > 0).mean(axis=0), precision=3)}")
    print(f"    Mean avg-reputation:      "
          f"{np.array2string(all_avg_repu.mean(axis=0), precision=4)}")

    # Which discriminator does each prior nominate?
    print(f"\n  Prior nomination  (k = argmax_k pi):")
    priors = [
        ("sum-payment",    all_payments.sum(axis=0)),
        ("mean-payment",   all_payments.mean(axis=0)),
        ("avg-reputation", all_avg_repu.mean(axis=0)),
    ]
    for name, pi in priors:
        print(f"    {name:<22s} -> argmax = D{int(np.argmax(pi))}    "
              f"values = {np.array2string(pi, precision=3)}")

    # Per-discriminator argmax accuracy for reference.
    print(f"\n  Per-discriminator argmax-step accuracy (oracle reference):")
    for k in range(K):
        h, _ = evaluate(valid, lambda P, kk=k: pred_best_single(P, kk))
        print(f"    D{k}: {h/max(n,1):>5.1%}   ({h}/{n})")

    # -------- PASS 2 --------
    print(f"\n  Pass 2: evaluating methods with leave-one-out prior...")

    def eval_loo(predict_with_pi, agg_kind):
        hits = 0
        for i, (P, gt) in enumerate(valid):
            if agg_kind == "sum-payment":
                pi_raw = all_payments.sum(axis=0) - all_payments[i]
            elif agg_kind == "mean-payment":
                pi_raw = (all_payments.sum(axis=0) - all_payments[i]) / max(n - 1, 1)
            elif agg_kind == "avg-reputation":
                pi_raw = (all_avg_repu.sum(axis=0) - all_avg_repu[i]) / max(n - 1, 1)
            else:
                raise ValueError(agg_kind)
            pi = normalize_for_prior(pi_raw)
            try:
                if predict_with_pi(P, pi) == gt:
                    hits += 1
            except Exception:
                pass
        return hits

    # Best single (oracle, ground-truth needed to pick best k).
    per_k_hits = [(k, evaluate(valid,
                               lambda P, kk=k: pred_best_single(P, kk))[0])
                  for k in range(K)]
    bk, bh = max(per_k_hits, key=lambda x: x[1])

    print(f"\n  {'Method':<46s}  Acc       Hits        Time")
    print(f"  {'-'*46}  ------    --------    ------")
    print(f"  {'Best single (ORACLE, D'+str(bk)+')':<46s}  "
          f"{bh/max(n,1):>5.1%}    {bh}/{n}")

    # Baselines.
    t1 = time.time()
    h, _ = evaluate(valid, lambda P: pred_l1huber_baseline(P, c))
    print(f"  {'L1-Huber (baseline, no prior)':<46s}  "
          f"{h/max(n,1):>5.1%}    {h}/{n}    [{time.time()-t1:.1f}s]")

    t1 = time.time()
    h, _ = evaluate(valid, pred_max_prob_select)
    print(f"  {'VCG-Select(max-prob)  (current AG winner)':<46s}  "
          f"{h/max(n,1):>5.1%}    {h}/{n}    [{time.time()-t1:.1f}s]")

    # Pi-Select and Pi-Weighted for each aggregation.
    for agg in ["sum-payment", "mean-payment", "avg-reputation"]:
        t1 = time.time()
        h = eval_loo(pred_pi_select, agg)
        print(f"  {'π-Select ['+agg+']':<46s}  "
              f"{h/max(n,1):>5.1%}    {h}/{n}    [{time.time()-t1:.1f}s]")

    for agg in ["sum-payment", "mean-payment", "avg-reputation"]:
        t1 = time.time()
        h = eval_loo(lambda P, pi: pred_pi_weighted(P, pi, c), agg)
        print(f"  {'π-Weighted L1-Huber ['+agg+']':<46s}  "
              f"{h/max(n,1):>5.1%}    {h}/{n}    [{time.time()-t1:.1f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="Algorithm-Generated",
                    choices=["Hand-Crafted", "Algorithm-Generated", "both"])
    ap.add_argument("--c", type=float, default=0.05,
                    help="L1-Huber bandwidth (default 0.05)")
    args = ap.parse_args()

    subsets = (["Hand-Crafted", "Algorithm-Generated"]
               if args.subset == "both" else [args.subset])
    for s in subsets:
        print(f"\n{'='*82}\nSUBSET: {s}\n{'='*82}")
        run_subset(s, args.data_root, args.c)


if __name__ == "__main__":
    main()