"""scripts/regime_stats.py — compute multiple within-trace statistics and see
which one cleanly separates HC and AG.

We want a within-trace observable s(theta) such that the empirical
distribution over s differs between HC and AG enough that a threshold rule
on s would route most HC traces to one config and most AG traces to another.

Candidates evaluated:
  - T            : trace length (deterministic: HC has variable T, AG always 10)
  - saturation   : fraction of (k,t) cells with theta <= 0.05 or theta >= 0.95
  - bimodality   : fraction in the extremes minus fraction in [0.4, 0.6]
  - max_prob_max : max_k max_t theta[k, t]  (= Confidence-Select score)
  - peak_spread  : max_k max_t - max_k median_t  (peakiness across k)
  - argmax_disagree : 1 if all K argmax_t differ, 0.5 if 2 agree, 0 if all agree
  - mean_active_size : average over t of #{k : |median_k theta - theta_k_t| < 0.1}
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def stat_T(theta):
    return float(theta.shape[1])


def stat_saturation(theta, lo=0.05, hi=0.95):
    return float(((theta <= lo) | (theta >= hi)).mean())


def stat_bimodality(theta, lo=0.10, hi=0.90):
    extremes = ((theta <= lo) | (theta >= hi)).mean()
    middle = ((theta > 0.4) & (theta < 0.6)).mean()
    return float(extremes - middle)


def stat_max_prob_max(theta):
    return float(theta.max())


def stat_peak_spread(theta):
    per_k_max = theta.max(axis=1)
    return float(per_k_max.max() - per_k_max.min())


def stat_argmax_disagree(theta):
    argmaxes = theta.argmax(axis=1)
    unique = set(int(x) for x in argmaxes)
    if len(unique) == 1:
        return 0.0
    if len(unique) == 2:
        return 0.5
    return 1.0


def stat_mean_active(theta, c=0.10):
    med = np.median(theta, axis=0)
    dev = np.abs(theta - med[None, :])
    active_per_t = (dev < c).sum(axis=0)
    return float(active_per_t.mean()) / theta.shape[0]


STATS = {
    "T":              stat_T,
    "saturation":     stat_saturation,
    "bimodality":     stat_bimodality,
    "max_prob_max":   stat_max_prob_max,
    "peak_spread":    stat_peak_spread,
    "argmax_disagr":  stat_argmax_disagree,
    "mean_active":    stat_mean_active,
}


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    return {
        "min":  float(v.min()),
        "q25":  float(np.percentile(v, 25)),
        "med":  float(np.median(v)),
        "q75":  float(np.percentile(v, 75)),
        "max":  float(v.max()),
    }


def separability(hc_vals, ag_vals):
    """How well does a single threshold on this stat separate HC from AG?

    For each candidate threshold (taken from the sorted union of values),
    compute the misclassification rate (fraction of HC above + fraction of AG
    below, assuming HC is the high-stat class; flip if direction reversed).
    Return the minimum error rate and the threshold that achieves it.
    """
    hc = np.asarray(hc_vals)
    ag = np.asarray(ag_vals)
    candidates = np.unique(np.concatenate([hc, ag]))
    best = (1.0, None, None)   # (error, threshold, "hc_above" or "hc_below")
    for thr in candidates:
        # Case 1: HC above threshold, AG below
        err_above = (hc <= thr).mean() * 0.5 + (ag > thr).mean() * 0.5
        # Case 2: HC below threshold, AG above
        err_below = (hc > thr).mean() * 0.5 + (ag <= thr).mean() * 0.5
        if err_above < best[0]:
            best = (err_above, float(thr), "hc_above")
        if err_below < best[0]:
            best = (err_below, float(thr), "hc_below")
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    subsets_data = {}
    for s in ["Hand-Crafted", "Algorithm-Generated"]:
        files = sorted((Path(args.data_root) / "reports" /
                        f"{s}_hybrid_v2").glob("*.npz"))
        traces = [load_trace(f) for f in files]
        valid = [t[0] for t in traces if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
        subsets_data[s] = valid
        print(f"  {s}: {len(valid)} valid traces")

    print(f"\n  {'stat':<16s} | "
          + " | ".join(f"{q:<7s}" for q in ["HC min", "HC q25", "HC med", "HC q75", "HC max",
                                            "AG min", "AG q25", "AG med", "AG q75", "AG max"])
          + " | best-thr (err, direction)")
    print("  " + "-" * 165)
    for name, fn in STATS.items():
        hc_vals = [fn(P) for P in subsets_data["Hand-Crafted"]]
        ag_vals = [fn(P) for P in subsets_data["Algorithm-Generated"]]
        hc_d = describe(hc_vals)
        ag_d = describe(ag_vals)
        err, thr, direction = separability(hc_vals, ag_vals)
        print(f"  {name:<16s} | "
              + " | ".join(f"{hc_d[k]:<7.3f}" for k in ["min", "q25", "med", "q75", "max"])
              + " | "
              + " | ".join(f"{ag_d[k]:<7.3f}" for k in ["min", "q25", "med", "q75", "max"])
              + f" | thr={thr:.3f} err={err:.1%} ({direction})")


if __name__ == "__main__":
    main()