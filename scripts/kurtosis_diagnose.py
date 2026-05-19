"""scripts/kurtosis_diagnose.py — characterise kurtosis as a regime separator.

Goal: see whether the trace-level kurtosis statistic can serve as a clean,
theoretically-motivated alternative to trace length T as the regime
selector.

Outputs:
  1. Per-subset distribution of kurtosis values.
  2. Best threshold-based separability (analogous to regime_stats.py).
  3. Per-trace correlation between kurtosis and T (should be moderate, not
     perfect; we want kurtosis to add signal beyond T, not be redundant).
  4. Joint scatter of (T, kurtosis) by subset, printed as a simple table.
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


def kurtosis(theta):
    arr = theta.flatten()
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-10:
        return 0.0
    z = (arr - mu) / sd
    return float((z ** 4).mean() - 3.0)


def separability_error(hc_vals, ag_vals):
    """Best threshold-based misclassification error rate."""
    hc = np.asarray(hc_vals)
    ag = np.asarray(ag_vals)
    candidates = np.unique(np.concatenate([hc, ag]))
    best = (1.0, None, None)
    for thr in candidates:
        err_above = (hc <= thr).mean() * 0.5 + (ag > thr).mean() * 0.5
        err_below = (hc > thr).mean() * 0.5 + (ag <= thr).mean() * 0.5
        if err_above < best[0]:
            best = (err_above, float(thr), "hc_above")
        if err_below < best[0]:
            best = (err_below, float(thr), "hc_below")
    return best


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    return {
        "min":  float(v.min()),
        "q10":  float(np.percentile(v, 10)),
        "q25":  float(np.percentile(v, 25)),
        "med":  float(np.median(v)),
        "q75":  float(np.percentile(v, 75)),
        "q90":  float(np.percentile(v, 90)),
        "max":  float(v.max()),
    }


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

    print("=" * 78)
    print("Kurtosis distribution per subset")
    print("=" * 78)
    print(f"  {'subset':<22s} | {'min':>7s} | {'q10':>7s} | {'q25':>7s} | "
          f"{'med':>7s} | {'q75':>7s} | {'q90':>7s} | {'max':>7s}")
    print("  " + "-" * 80)
    per_subset_kurt = {}
    per_subset_T    = {}
    for s in subsets_data:
        kurts = np.array([kurtosis(P) for P in subsets_data[s]])
        Ts    = np.array([float(P.shape[1]) for P in subsets_data[s]])
        per_subset_kurt[s] = kurts
        per_subset_T[s]    = Ts
        d = describe(kurts)
        print(f"  {s:<22s} | "
              + " | ".join(f"{d[k]:>7.3f}" for k in
                           ["min", "q10", "q25", "med", "q75", "q90", "max"]))

    print(f"\n  Separability of HC vs AG by kurtosis threshold:")
    err, thr, direction = separability_error(
        per_subset_kurt["Hand-Crafted"],
        per_subset_kurt["Algorithm-Generated"],
    )
    print(f"  best thr = {thr:.4f}, error = {err:.1%} ({direction})")

    # For comparison, T separator.
    print(f"\n  For reference, separability by trace length T:")
    err_T, thr_T, direction_T = separability_error(
        per_subset_T["Hand-Crafted"],
        per_subset_T["Algorithm-Generated"],
    )
    print(f"  best thr = {thr_T:.1f}, error = {err_T:.1%} ({direction_T})")

    # Correlation between T and kurtosis within each subset.
    print(f"\n  Within-subset correlation between T and kurtosis:")
    for s in subsets_data:
        Ts    = per_subset_T[s]
        kurts = per_subset_kurt[s]
        if len(Ts) > 1 and Ts.std() > 0 and kurts.std() > 0:
            corr = float(np.corrcoef(Ts, kurts)[0, 1])
            print(f"    {s:<22s}: corr(T, kurtosis) = {corr:+.3f}")
        else:
            print(f"    {s:<22s}: degenerate (T or kurtosis constant)")

    # Show low-T HC traces — the ones that look most like AG by T.
    print(f"\n  HC traces with shortest T (where T-rule could go wrong):")
    Ts = per_subset_T["Hand-Crafted"]
    kurts = per_subset_kurt["Hand-Crafted"]
    order = np.argsort(Ts)
    print(f"    {'rank':<6s} | {'T':<6s} | {'kurtosis':<10s}")
    for i in order[:8]:
        print(f"    {i:<6d} | {int(Ts[i]):<6d} | {kurts[i]:<10.3f}")

    # And the highest kurtosis AG traces.
    print(f"\n  AG traces with highest kurtosis (where kurtosis-rule could go wrong):")
    kurts_ag = per_subset_kurt["Algorithm-Generated"]
    Ts_ag    = per_subset_T["Algorithm-Generated"]
    order = np.argsort(-kurts_ag)
    print(f"    {'rank':<6s} | {'T':<6s} | {'kurtosis':<10s}")
    for i in order[:8]:
        print(f"    {i:<6d} | {int(Ts_ag[i]):<6d} | {kurts_ag[i]:<10.3f}")


if __name__ == "__main__":
    main()