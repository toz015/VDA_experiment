"""scripts/adaptive_kernel_smooth.py — continuous smooth (p, c, eps)
selection driven by interpretable observable statistics.

Motivation: the hard threshold T > 10 in adaptive_kernel.py is ad hoc and
not theoretically motivated. We replace it with a smooth function of
within-trace observables that have direct connections to the M-estimator
literature on optimal loss exponents:

  - Trace length T (longer trace -> richer cross-step signal -> safer to
    use robust low-p kernel)
  - Per-step inter-discriminator dispersion (higher dispersion -> more
    disagreement -> heavier-tail noise -> small p; lower dispersion ->
    cleaner Gaussian-like noise -> larger p)

The smooth mapping is

    score(theta) = alpha * log(T)  +  beta * (median_t std_k(theta_{k,t}))
                                  +  gamma     [bias term]

  p(theta) = p_L + (p_S - p_L) * sigmoid(-score / scale)
  c(theta) = c_L + (c_S - c_L) * sigmoid(-score / scale)
  eps(theta) = eps_L + (eps_S - eps_L) * sigmoid(-score / scale)

where (p_L, c_L, eps_L) = (1.0, 0.05, 1e-6)  [long-trace optimum]
and   (p_S, c_S, eps_S) = (1.75, 1.0, 0.10)  [short-trace optimum].

When score is large (long T, high dispersion), the mechanism is close to
the L1-Huber operating point. When score is small (short T, low
dispersion), it's close to the smooth-large-c operating point.

Several scan modes:

  --mode score          Use a tunable linear score with bias.
  --mode disp-only      Use ONLY per-step inter-discriminator dispersion
                        (no trace length). This isolates whether the
                        statistical noise structure alone can route.
  --mode T-only         Use ONLY log(T). This is a smooth analogue of the
                        hard T-threshold rule in adaptive_kernel.py.
  --mode kurtosis       Use sample kurtosis of report values within each
                        trace. From robust-stats theory, high kurtosis
                        favours small p.

Each mode is implemented as a 1-parameter (or 2-parameter) family that can
be scanned to find the operating point with best pooled accuracy without
explicit ground-truth labels at the subset level.

Run from repo root:
    python scripts/adaptive_kernel_smooth.py --mode score --scan
    python scripts/adaptive_kernel_smooth.py --mode disp-only --scan
    python scripts/adaptive_kernel_smooth.py --mode T-only --scan
    python scripts/adaptive_kernel_smooth.py --mode kurtosis --scan
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.allocation_power import solve_allocation_power


PROB_KEYS    = ["theta_hat", "probs", "prob", "p", "scores", "theta", "raw"]
GT_STEP_KEYS = ["mistake_step", "gt_step", "true_step", "label_step"]


# Two anchor configurations from the (p, c) scan.
LONG_CFG  = {"p": 1.00, "c": 0.05, "eps": 1e-6}     # HC optimum
SHORT_CFG = {"p": 1.75, "c": 1.00, "eps": 0.10}     # AG optimum


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


# -------- Observable statistics --------

def stat_T(theta):
    return float(theta.shape[1])


def stat_disp(theta):
    """Median over t of std across discriminators k at step t.

    Measures how strongly LLM discriminators disagree on average within
    this trace. Higher = noisier reports = heavier-tail aggregate noise.
    """
    return float(np.median(np.std(theta, axis=0)))


def stat_kurtosis(theta):
    """Excess kurtosis of all entries in theta (a single scalar per trace).

    From robust-statistics theory, high kurtosis (heavy-tail noise) calls
    for a smaller p, low kurtosis (Gaussian-like) calls for larger p.
    """
    arr = theta.flatten()
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-10:
        return 0.0
    z = (arr - mu) / sd
    return float((z ** 4).mean() - 3.0)


def stat_logT(theta):
    return float(np.log(max(theta.shape[1], 1)))


# -------- Smooth interpolating maps --------

def sigmoid(z):
    # Clip z to avoid harmless overflow warnings; saturating values
    # already give sigmoid output 0 or 1 to numerical precision.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def smooth_cfg(score, score_mid, score_scale):
    """Map a scalar score to a (p, c, eps) interpolation.

    High score (positive z)  ->  weight ~ 1 (LONG_CFG)
    Low score (negative z)   ->  weight ~ 0 (SHORT_CFG)
    """
    z = (score - score_mid) / max(score_scale, 1e-9)
    w = float(sigmoid(z))
    p   = LONG_CFG["p"]   * w + SHORT_CFG["p"]   * (1 - w)
    c   = LONG_CFG["c"]   * w + SHORT_CFG["c"]   * (1 - w)
    eps = LONG_CFG["eps"] * w + SHORT_CFG["eps"] * (1 - w)
    return p, c, eps, w


def predict_smooth(theta, score_fn, score_mid, score_scale):
    score = score_fn(theta)
    p, c, eps, w = smooth_cfg(score, score_mid, score_scale)
    P = np.clip(theta, eps, 1.0 - eps)
    d = solve_allocation_power(P, c=c, p=p, eps=eps).d
    return int(np.argmax(d)), score, p, c, eps, w


def evaluate_with_score(traces, score_fn, score_mid, score_scale):
    hits = total = 0
    weights = []
    for P, gt in traces:
        if gt is None or not (0 <= gt < P.shape[1]):
            continue
        total += 1
        pred, score, p, c, eps, w = predict_smooth(
            P, score_fn, score_mid, score_scale
        )
        weights.append(w)
        if pred == gt:
            hits += 1
    return hits, total, weights


# -------- Modes --------

def make_score_fn(mode):
    if mode == "score":
        # Linear combination of log T and dispersion, tuned via bias.
        def _fn(theta):
            return 1.0 * np.log(max(theta.shape[1], 1)) \
                 + 5.0 * stat_disp(theta)
        return _fn
    if mode == "disp-only":
        return stat_disp
    if mode == "T-only":
        return stat_logT
    if mode == "kurtosis":
        return stat_kurtosis
    raise ValueError(mode)


def evaluate_subsets(data_root, score_fn, score_mid, score_scale, label=""):
    subsets = ["Hand-Crafted", "Algorithm-Generated"]
    per_subset = {}
    pooled_hits = pooled_total = 0
    all_weights = {}
    for s in subsets:
        subset_dir = Path(data_root) / "reports" / f"{s}_hybrid_v2"
        files = sorted(subset_dir.glob("*.npz"))
        traces = [load_trace(f) for f in files]
        valid = [(t[0], t[1]) for t in traces
                 if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
        hits, total, weights = evaluate_with_score(
            valid, score_fn, score_mid, score_scale
        )
        per_subset[s] = (hits, total)
        pooled_hits += hits
        pooled_total += total
        all_weights[s] = weights
    return per_subset, (pooled_hits, pooled_total), all_weights


def scan_modes(data_root, mode):
    score_fn = make_score_fn(mode)

    # Sample range of score values to find the threshold.
    subsets = ["Hand-Crafted", "Algorithm-Generated"]
    all_scores = []
    for s in subsets:
        subset_dir = Path(data_root) / "reports" / f"{s}_hybrid_v2"
        files = sorted(subset_dir.glob("*.npz"))
        for f in files:
            npz = np.load(f, allow_pickle=True)
            theta = np.asarray(npz[_first_present(npz, PROB_KEYS)], dtype=float)
            all_scores.append(score_fn(theta))
    all_scores = np.array(all_scores)

    print(f"\n  Mode: {mode}")
    print(f"  Score distribution: "
          f"min={all_scores.min():.3f}  "
          f"q25={np.percentile(all_scores, 25):.3f}  "
          f"median={np.median(all_scores):.3f}  "
          f"q75={np.percentile(all_scores, 75):.3f}  "
          f"max={all_scores.max():.3f}")

    # Scan score_mid and score_scale.
    mid_values   = np.linspace(np.percentile(all_scores, 10),
                               np.percentile(all_scores, 90), 7)
    scale_values = [0.01, 0.05, 0.1, 0.3, 1.0]   # smaller scale = sharper

    print(f"\n  Scan (mid, scale) -> (HC acc, AG acc, pooled acc):")
    print(f"  {'mid':<8s}  {'scale':<6s}  {'HC':<14s}  {'AG':<14s}  {'pooled':<14s}")
    print("  " + "-" * 70)
    best = (-1, None)
    for score_mid in mid_values:
        for score_scale in scale_values:
            per_subset, pooled, _ = evaluate_subsets(
                data_root, score_fn, float(score_mid), float(score_scale)
            )
            h_hc, n_hc = per_subset["Hand-Crafted"]
            h_ag, n_ag = per_subset["Algorithm-Generated"]
            h_p, n_p   = pooled
            if h_p > best[0]:
                best = (h_p, (float(score_mid), float(score_scale)))
            print(f"  {score_mid:<8.3f}  {score_scale:<6.3f}  "
                  f"{h_hc/max(n_hc,1):>5.1%} ({h_hc:>3d}/{n_hc:<3d})  "
                  f"{h_ag/max(n_ag,1):>5.1%} ({h_ag:>3d}/{n_ag:<3d})  "
                  f"{h_p/max(n_p,1):>5.1%} ({h_p:>3d}/{n_p:<3d})")

    print(f"\n  Best pooled: {best[0]}/{n_p} at (mid, scale) = {best[1]}")

    # Also report the detailed routing weight distribution at the optimum.
    sm, ss = best[1]
    per_subset, pooled, weights = evaluate_subsets(
        data_root, score_fn, sm, ss
    )
    print(f"\n  At optimum (mid={sm:.3f}, scale={ss:.3f}), routing-weight w distribution")
    print(f"  (w close to 1 means LONG_CFG, w close to 0 means SHORT_CFG):")
    for s in subsets:
        w = np.array(weights[s])
        print(f"    {s:<22s}: "
              f"min={w.min():.3f}  q25={np.percentile(w,25):.3f}  "
              f"median={np.median(w):.3f}  q75={np.percentile(w,75):.3f}  "
              f"max={w.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--mode", default="score",
                    choices=["score", "disp-only", "T-only", "kurtosis"],
                    help="which observable statistic to use as the score")
    ap.add_argument("--scan", action="store_true",
                    help="scan over (score_mid, score_scale)")
    ap.add_argument("--mid", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    if args.scan:
        scan_modes(args.data_root, args.mode)
        return

    if args.mid is None or args.scale is None:
        print("Either pass --scan or specify --mid and --scale.")
        return
    score_fn = make_score_fn(args.mode)
    per_subset, pooled, weights = evaluate_subsets(
        args.data_root, score_fn, args.mid, args.scale
    )
    for s, (h, n) in per_subset.items():
        print(f"  {s}: {h/max(n,1):.1%}  ({h}/{n})")
    h_p, n_p = pooled
    print(f"  Pooled: {h_p/max(n_p,1):.1%}  ({h_p}/{n_p})")


if __name__ == "__main__":
    main()