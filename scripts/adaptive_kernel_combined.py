"""scripts/adaptive_kernel_combined.py — smooth two-feature adaptive selector.

Combines trace length T and report kurtosis as the score:

    score(theta)  =  log(T)  +  alpha * kurt(theta)

Both features are theoretically motivated:
  - log(T) measures cross-step signal strength (longer trace -> more
    reliable reputation product -> safe to use robust low-p kernel).
  - kurt(theta) measures marginal report noise heaviness (heavy tails ->
    small p; light tails -> large p), directly from M-estimator literature
    on optimal loss exponents.

The mapping is smooth via sigmoid:
    p   = p_L * sigmoid((score - mid)/scale) + p_S * (1 - sigmoid(...))
similarly for c and eps. Anchors:
  LONG_CFG  = (p=1.00, c=0.05, eps=1e-6)
  SHORT_CFG = (p=1.75, c=1.00, eps=0.10)

Sliding alpha from 0 (T-only) to large (kurtosis-only) and tuning (mid,
scale) gives a 3D scan. We also report the best alpha=0 row (recovers
T-only result 32.6%) and the best alpha->infinity row (recovers
kurtosis-only result 32.0%) for direct comparison.

Run from repo root:
    python scripts/adaptive_kernel_combined.py --scan
    python scripts/adaptive_kernel_combined.py --alpha 0.5 --mid 2.5 --scale 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.allocation_power import solve_allocation_power


PROB_KEYS    = ["theta_hat", "probs", "prob", "p", "scores", "theta", "raw"]
GT_STEP_KEYS = ["mistake_step", "gt_step", "true_step", "label_step"]


LONG_CFG  = {"p": 1.00, "c": 0.05, "eps": 1e-6}
SHORT_CFG = {"p": 1.75, "c": 1.00, "eps": 0.10}


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


def kurtosis(theta):
    arr = theta.flatten()
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-10:
        return 0.0
    z = (arr - mu) / sd
    return float((z ** 4).mean() - 3.0)


def score_logT_plus_alphakurt(theta, alpha):
    T = theta.shape[1]
    return float(np.log(max(T, 1)) + alpha * kurtosis(theta))


def sigmoid(z):
    # Clip z to avoid harmless overflow warnings; saturating values
    # already give sigmoid output 0 or 1 to numerical precision.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def smooth_cfg(score, mid, scale):
    z = (score - mid) / max(scale, 1e-9)
    w = float(sigmoid(z))
    return (
        LONG_CFG["p"]   * w + SHORT_CFG["p"]   * (1 - w),
        LONG_CFG["c"]   * w + SHORT_CFG["c"]   * (1 - w),
        LONG_CFG["eps"] * w + SHORT_CFG["eps"] * (1 - w),
        w,
    )


def predict(theta, alpha, mid, scale):
    score = score_logT_plus_alphakurt(theta, alpha)
    p, c, eps, w = smooth_cfg(score, mid, scale)
    P = np.clip(theta, eps, 1.0 - eps)
    d = solve_allocation_power(P, c=c, p=p, eps=eps).d
    return int(np.argmax(d)), score, p, c, eps, w


def evaluate(traces, alpha, mid, scale):
    hits = total = 0
    for P, gt in traces:
        if gt is None or not (0 <= gt < P.shape[1]):
            continue
        total += 1
        pred, *_ = predict(P, alpha, mid, scale)
        if pred == gt:
            hits += 1
    return hits, total


def evaluate_both(data_root, alpha, mid, scale):
    out = {}
    pooled_h = pooled_n = 0
    for s in ["Hand-Crafted", "Algorithm-Generated"]:
        files = sorted((Path(data_root) / "reports" /
                        f"{s}_hybrid_v2").glob("*.npz"))
        traces = [load_trace(f) for f in files]
        valid = [(t[0], t[1]) for t in traces
                 if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
        h, n = evaluate(valid, alpha, mid, scale)
        out[s] = (h, n)
        pooled_h += h
        pooled_n += n
    return out, (pooled_h, pooled_n)


def scan(data_root):
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    # mid range derived from log(T) range plus expected kurtosis contribution.
    # Useful score range is roughly 1.6 to 7-ish for these alpha values.
    mid_values = np.linspace(1.6, 5.5, 9)
    scale_values = [0.01, 0.05, 0.1, 0.3]

    print(f"  Scanning alpha x mid x scale...")
    print(f"  {'alpha':<6s} {'mid':<6s} {'scale':<6s} {'HC':<14s} {'AG':<14s} {'pooled':<14s}")
    print("  " + "-" * 75)

    best = (-1, None)
    rows_by_alpha = {a: [] for a in alphas}
    for alpha in alphas:
        for mid in mid_values:
            for scale in scale_values:
                per_subset, pooled = evaluate_both(
                    data_root, alpha, float(mid), float(scale)
                )
                h_hc, n_hc = per_subset["Hand-Crafted"]
                h_ag, n_ag = per_subset["Algorithm-Generated"]
                h_p, n_p   = pooled
                rows_by_alpha[alpha].append(
                    (float(mid), float(scale), h_hc, n_hc, h_ag, n_ag, h_p, n_p)
                )
                if h_p > best[0]:
                    best = (h_p, (alpha, float(mid), float(scale)))

    # Print best (mid, scale) per alpha.
    print(f"\n  Best (mid, scale) per alpha:")
    print(f"  {'alpha':<6s} {'mid':<8s} {'scale':<6s} {'HC':<14s} {'AG':<14s} {'pooled':<14s}")
    print("  " + "-" * 75)
    for alpha in alphas:
        rows = rows_by_alpha[alpha]
        best_row = max(rows, key=lambda r: r[6])
        mid_v, scale_v, h_hc, n_hc, h_ag, n_ag, h_p, n_p = best_row
        print(f"  {alpha:<6.2f} {mid_v:<8.3f} {scale_v:<6.3f} "
              f"{h_hc/max(n_hc,1):>5.1%} ({h_hc:>3d}/{n_hc:<3d})  "
              f"{h_ag/max(n_ag,1):>5.1%} ({h_ag:>3d}/{n_ag:<3d})  "
              f"{h_p/max(n_p,1):>5.1%} ({h_p:>3d}/{n_p:<3d})")

    print(f"\n  Global best pooled: {best[0]}/{n_p} at "
          f"(alpha, mid, scale) = {best[1]}")

    # Show the routing characteristics at the global optimum.
    alpha, mid, scale = best[1]
    print(f"\n  Routing weight distribution at the optimum:")
    for s in ["Hand-Crafted", "Algorithm-Generated"]:
        files = sorted((Path(data_root) / "reports" /
                        f"{s}_hybrid_v2").glob("*.npz"))
        traces = [load_trace(f) for f in files]
        valid = [t[0] for t in traces
                 if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
        weights = []
        for P in valid:
            _, _, _, _, _, w = predict(P, alpha, mid, scale)
            weights.append(w)
        w = np.array(weights)
        print(f"    {s:<22s}: "
              f"min={w.min():.3f}  q25={np.percentile(w,25):.3f}  "
              f"median={np.median(w):.3f}  q75={np.percentile(w,75):.3f}  "
              f"max={w.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--mid",   type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    if args.scan:
        scan(args.data_root)
        return

    if args.alpha is None or args.mid is None or args.scale is None:
        print("Use --scan or specify --alpha, --mid, --scale.")
        return
    per_subset, pooled = evaluate_both(
        args.data_root, args.alpha, args.mid, args.scale
    )
    for s, (h, n) in per_subset.items():
        print(f"  {s}: {h/max(n,1):.1%}  ({h}/{n})")
    h_p, n_p = pooled
    print(f"  Pooled: {h_p/max(n_p,1):.1%}  ({h_p}/{n_p})")


if __name__ == "__main__":
    main()