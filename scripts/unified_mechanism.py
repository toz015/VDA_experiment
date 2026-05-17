"""scripts/unified_mechanism.py — unified within-trace adaptive VCG.

Tests whether a SINGLE parametric mechanism works well on BOTH HC and AG.

The mechanism is L1-Huber multiplicative VCG with a *within-trace* evidence
prior pi_k inserted into the reputation:

    w_k^t = (pi_k(theta_hat))^alpha  *  prod_{t' != t} kappa_c(d_{t'} - theta_hat^k_{t'}).

Within-trace evidence priors tested (no cross-trace data needed):
  - uniform       : pi_k = 1                                  (baseline L1-Huber)
  - peak          : pi_k = max_t theta_hat[k, t]              (Gemini-favouring on AG)
  - peaksep       : pi_k = top1 - top2 within k's own row     (concentration)
  - negent        : pi_k = exp(-H_k)  with H_k entropy of    (concentration)
                            theta_hat[k, :] / sum

Decision rules at the end:
  - raw           : argmax_t d_t
  - gap-adjusted  : argmax_t (d_t + beta * g_t)
                    where g_t = top1 - top2 across k at step t
                    (per-step cross-discriminator disagreement)

The alpha parameter sharpens the prior. alpha=0 recovers L1-Huber.
alpha->infinity recovers hard selection on pi.

Run from repo root:
    python scripts/unified_mechanism.py
    python scripts/unified_mechanism.py --alpha 2 --beta 0.5
    python scripts/unified_mechanism.py --scan-alpha
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


# ---- inline L1-Huber helpers ----

def l1h_factors(d, theta, c):
    return 1.0 - np.minimum(np.abs(d - theta), c)


def weighted_median(values, weights):
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v_s, w_s = v[order], w[order]
    cumw = np.cumsum(w_s)
    total = cumw[-1] if cumw.size > 0 else 0.0
    if total <= 0:
        return float(np.median(v))
    idx = int(np.searchsorted(cumw, total / 2.0))
    if idx >= v_s.size:
        idx = v_s.size - 1
    return float(v_s[idx])


# ---- within-trace evidence priors ----

def prior_uniform(P):
    return np.ones(P.shape[0])


def prior_peak(P):
    return P.max(axis=1)


def prior_peaksep(P):
    """top-1 minus top-2 within k's own predictions, per-discriminator."""
    sorted_desc = -np.sort(-P, axis=1)
    if sorted_desc.shape[1] < 2:
        return np.ones(P.shape[0])
    sep = sorted_desc[:, 0] - sorted_desc[:, 1]
    return sep + 1e-6


def prior_negent(P):
    """exp(-H_k); H_k = entropy of (theta_hat[k, :] / sum_t theta_hat[k, t])."""
    P_norm = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    H = -(P_norm * np.log(P_norm + 1e-12)).sum(axis=1)
    # Map small H to large prior. Subtract min(H) for numerical stability.
    H_shift = H - H.min()
    return np.exp(-H_shift)


PRIOR_FUNCS = {
    "uniform":  prior_uniform,
    "peak":     prior_peak,
    "peaksep":  prior_peaksep,
    "negent":   prior_negent,
}


# ---- L1-Huber with within-trace prior ----

def solve_with_prior(theta, pi, c=0.05, alpha=1.0, L=50, tau=1e-10, eps=1e-6):
    """Gauss-Seidel L1-Huber with reputation weight multiplied by pi^alpha."""
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1 - eps)
    K, T = theta.shape

    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape != (K,):
        raise ValueError(f"pi shape {pi.shape} != ({K},)")
    if alpha == 0.0:
        pi_eff = np.ones(K)
    else:
        # Power, then normalise so mean is 1 (preserves argmax of V).
        pi_eff = np.maximum(pi, 1e-12) ** alpha
        pi_eff = pi_eff / (pi_eff.mean() + 1e-12)

    d = np.median(theta, axis=0).copy()

    def V(d_arr):
        return float((pi_eff * l1h_factors(d_arr, theta, c).prod(axis=1)).sum())

    V_old = V(d)
    for _ in range(L):
        for t in range(T):
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            factors_excl = l1h_factors(d[mask], theta[:, mask], c)
            w_t = factors_excl.prod(axis=1) * pi_eff

            dev = np.abs(d[t] - theta[:, t])
            active = dev < c
            if active.any():
                d_new = weighted_median(theta[active, t], w_t[active])
                d_new = float(np.clip(d_new, eps, 1 - eps))
                d_old_t = float(d[t])
                d[t] = d_new
                if V(d) < V_old - 1e-9:
                    d[t] = d_old_t

        V_new = V(d)
        if abs(V_new - V_old) < tau:
            break
        V_old = V_new
    return d


# ---- decision rules ----

def per_step_gap(theta):
    """g_t = top1 - top2 across k, per step. Shape (T,)."""
    sorted_desc = -np.sort(-theta, axis=0)
    if sorted_desc.shape[0] < 2:
        return np.zeros(theta.shape[1])
    return sorted_desc[0] - sorted_desc[1]


def decide_raw(d, theta, beta=0.0):
    return int(np.argmax(d))


def decide_gap(d, theta, beta=0.5):
    g = per_step_gap(theta)
    return int(np.argmax(d + beta * g))


# ---- evaluation ----

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


def make_predictor(prior_name, alpha, decide_name, beta, c):
    prior_fn = PRIOR_FUNCS[prior_name]
    decide_fn = decide_raw if decide_name == "raw" else decide_gap

    def predict(P):
        pi = prior_fn(P)
        d = solve_with_prior(P, pi, c=c, alpha=alpha)
        return decide_fn(d, P, beta=beta)
    return predict


def run_subset(subset, data_root, c, alpha, beta, scan_alpha=False):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] no .npz under {subset_dir}", file=sys.stderr)
        return
    traces_all = [load_trace(p) for p in files]
    valid = [(t[0], t[1]) for t in traces_all
             if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    n = len(valid)
    K = traces_all[0][0].shape[0] if traces_all else 0
    print(f"  {n} valid traces, K = {K}, c = {c}")

    if scan_alpha:
        # Just scan alpha for peak prior with raw decision.
        print(f"\n  Alpha scan for prior=peak, decision=raw:")
        print(f"  {'alpha':<8s}  Acc       Hits")
        print(f"  {'-'*8}  ------    --------")
        for a in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]:
            pred = make_predictor("peak", a, "raw", 0.0, c)
            h, _ = evaluate(valid, pred)
            print(f"  {a:<8.1f}  {h/max(n,1):>5.1%}    {h}/{n}")
        print(f"\n  Alpha scan for prior=negent, decision=raw:")
        print(f"  {'alpha':<8s}  Acc       Hits")
        print(f"  {'-'*8}  ------    --------")
        for a in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]:
            pred = make_predictor("negent", a, "raw", 0.0, c)
            h, _ = evaluate(valid, pred)
            print(f"  {a:<8.1f}  {h/max(n,1):>5.1%}    {h}/{n}")
        return

    # Standard test: all (prior, decision) combinations at given (alpha, beta).
    print(f"\n  alpha = {alpha}, beta = {beta}")
    print(f"\n  {'Method':<48s}  Acc       Hits        Time")
    print(f"  {'-'*48}  ------    --------    ------")
    for prior_name in ["uniform", "peak", "peaksep", "negent"]:
        for decide_name in ["raw", "gap"]:
            pred = make_predictor(prior_name, alpha, decide_name, beta, c)
            t0 = time.time()
            h, _ = evaluate(valid, pred)
            dt = time.time() - t0
            label = f"{prior_name:<8s} pi + {decide_name:<3s} decide"
            print(f"  L1H + {label:<40s}  {h/max(n,1):>5.1%}    {h}/{n}    [{dt:.1f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="both",
                    choices=["Hand-Crafted", "Algorithm-Generated", "both"])
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="prior sharpening exponent (default 2.0)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="gap weight in gap-adjusted decision (default 0.5)")
    ap.add_argument("--scan-alpha", action="store_true",
                    help="scan alpha for prior=peak (and =negent) instead")
    args = ap.parse_args()

    subsets = (["Hand-Crafted", "Algorithm-Generated"]
               if args.subset == "both" else [args.subset])
    for s in subsets:
        print(f"\n{'='*82}\nSUBSET: {s}\n{'='*82}")
        run_subset(s, args.data_root, args.c, args.alpha, args.beta,
                   scan_alpha=args.scan_alpha)


if __name__ == "__main__":
    main()