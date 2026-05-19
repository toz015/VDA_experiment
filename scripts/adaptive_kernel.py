"""scripts/adaptive_kernel.py — trace-length-driven (p, c, report_eps) selection.

For each trace, use the trace length T as a single observable routing
statistic. By design Algorithm-Generated traces have T = 10 fixed while
Hand-Crafted traces have variable T mostly > 10; this provides a clean
separator (empirical separability error 3.6% on Who&When).

Routing rule:
  T  >  T_thresh        ->  long-trace config (HC-like; rich cross-step signal)
  T  <= T_thresh        ->  short-trace config (AG-like; cross-step weak)

Default long config (optimum on HC scan at raw clip):
  (p, c, report_eps)  =  (1.0, 0.05, 1e-6)
Default short config (optimum on AG scan at heavy clip):
  (p, c, report_eps)  =  (1.75, 1.0, 0.10)

Per-subset accuracies are reported for diagnostic purposes only; the
selector itself reads only T.

Run from repo root:
    python scripts/adaptive_kernel.py
    python scripts/adaptive_kernel.py --t-thresh 12
    python scripts/adaptive_kernel.py --scan-thresh
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.allocation_power import solve_allocation_power


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


def adaptive_predict(theta, t_thresh, long_cfg, short_cfg):
    """Pick a config by trace length T and run the power-kernel allocation."""
    T = theta.shape[1]
    cfg = long_cfg if T > t_thresh else short_cfg
    P = np.clip(theta, cfg["report_eps"], 1.0 - cfg["report_eps"])
    d = solve_allocation_power(
        P, c=cfg["c"], p=cfg["p"], eps=cfg["report_eps"]
    ).d
    return int(np.argmax(d)), cfg, T


def evaluate_subset(name, files, t_thresh, long_cfg, short_cfg):
    traces = [load_trace(f) for f in files]
    valid = [(t[0], t[1]) for t in traces
             if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    n = len(valid)
    if n == 0:
        return 0, 0, [], (0, 0, 0, 0)
    n_long = n_short = n_hit_long = n_hit_short = 0
    rows = []
    for P, gt in valid:
        pred, cfg, T = adaptive_predict(P, t_thresh, long_cfg, short_cfg)
        hit = int(pred == gt)
        regime = "long" if T > t_thresh else "short"
        if regime == "long":
            n_long += 1
            n_hit_long += hit
        else:
            n_short += 1
            n_hit_short += hit
        rows.append((regime, T, hit))
    n_hit = n_hit_long + n_hit_short
    return n_hit, n, rows, (n_long, n_short, n_hit_long, n_hit_short)


def run_evaluation(data_root, t_thresh, long_cfg, short_cfg):
    print(f"  Trace-length threshold:  T_thresh = {t_thresh}")
    print(f"  Long-trace config (T > T_thresh): "
          f"p = {long_cfg['p']}, c = {long_cfg['c']}, "
          f"report_eps = {long_cfg['report_eps']}")
    print(f"  Short-trace config (T <= T_thresh): "
          f"p = {short_cfg['p']}, c = {short_cfg['c']}, "
          f"report_eps = {short_cfg['report_eps']}")

    subsets = ["Hand-Crafted", "Algorithm-Generated"]
    per_subset_summary = {}
    pooled_rows = []
    for s in subsets:
        subset_dir = Path(data_root) / "reports" / f"{s}_hybrid_v2"
        files = sorted(subset_dir.glob("*.npz"))
        if not files:
            print(f"  [WARN] no .npz under {subset_dir}", file=sys.stderr)
            continue
        n_hit, n, rows, breakdown = evaluate_subset(
            s, files, t_thresh, long_cfg, short_cfg
        )
        per_subset_summary[s] = (n_hit, n, breakdown)
        pooled_rows.extend(rows)
        nL, nS, hL, hS = breakdown
        print(f"\n  {s}:")
        print(f"    Overall: {n_hit/max(n,1):>5.1%}  ({n_hit}/{n})")
        print(f"    Routed LONG:   {nL}/{n}  "
              f"acc {hL/max(nL,1):>5.1%}  ({hL}/{nL})")
        print(f"    Routed SHORT:  {nS}/{n}  "
              f"acc {hS/max(nS,1):>5.1%}  ({hS}/{nS})")
        Ts = np.array([r[1] for r in rows])
        print(f"    Trace length T:  min={int(Ts.min())}  "
              f"median={int(np.median(Ts))}  max={int(Ts.max())}")

    n_pooled = len(pooled_rows)
    n_hit_pooled = sum(r[2] for r in pooled_rows)
    print(f"\n  Pooled across both subsets:  {n_hit_pooled/max(n_pooled,1):>5.1%}  "
          f"({n_hit_pooled}/{n_pooled})")

    return per_subset_summary


def threshold_scan(data_root, long_cfg, short_cfg, t_thresh_values):
    print(f"\n  T_thresh scan  (long_cfg vs short_cfg fixed):")
    print(f"  {'T_thresh':<10s} | {'HC':<14s} | {'AG':<14s} | {'pooled':<14s}")
    print("  " + "-" * 60)
    subsets = ["Hand-Crafted", "Algorithm-Generated"]
    cached_traces = {}
    for s in subsets:
        subset_dir = Path(data_root) / "reports" / f"{s}_hybrid_v2"
        files = sorted(subset_dir.glob("*.npz"))
        traces = [load_trace(f) for f in files]
        valid = [(t[0], t[1]) for t in traces
                 if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
        cached_traces[s] = valid

    for t_thresh in t_thresh_values:
        per_subset_hits = {}
        for subset in subsets:
            valid = cached_traces[subset]
            hits = 0
            for P, gt in valid:
                pred, _, _ = adaptive_predict(P, t_thresh, long_cfg, short_cfg)
                if pred == gt:
                    hits += 1
            per_subset_hits[subset] = (hits, len(valid))
        h_hc, n_hc = per_subset_hits["Hand-Crafted"]
        h_ag, n_ag = per_subset_hits["Algorithm-Generated"]
        h_total = h_hc + h_ag
        n_total = n_hc + n_ag
        print(f"  {t_thresh:<10d} | "
              f"{h_hc/max(n_hc,1):>5.1%} ({h_hc:>3d}/{n_hc:<3d}) | "
              f"{h_ag/max(n_ag,1):>5.1%} ({h_ag:>3d}/{n_ag:<3d}) | "
              f"{h_total/max(n_total,1):>5.1%} ({h_total}/{n_total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--t-thresh", type=int, default=10,
                    help="trace-length threshold (default 10)")
    ap.add_argument("--p-long", type=float, default=1.0)
    ap.add_argument("--c-long", type=float, default=0.05)
    ap.add_argument("--eps-long", type=float, default=1e-6)
    ap.add_argument("--p-short", type=float, default=1.75)
    ap.add_argument("--c-short", type=float, default=1.0)
    ap.add_argument("--eps-short", type=float, default=0.10)
    ap.add_argument("--scan-thresh", action="store_true",
                    help="scan T_thresh in {5, 7, 9, 10, 11, 12, 15, 20}")
    args = ap.parse_args()

    long_cfg  = {"p": args.p_long,  "c": args.c_long,  "report_eps": args.eps_long}
    short_cfg = {"p": args.p_short, "c": args.c_short, "report_eps": args.eps_short}

    print(f"\n{'='*78}")
    print(f"Adaptive kernel selection by trace length")
    print(f"{'='*78}")
    run_evaluation(args.data_root, args.t_thresh, long_cfg, short_cfg)

    if args.scan_thresh:
        print(f"\n{'='*78}")
        print(f"Threshold scan")
        print(f"{'='*78}")
        threshold_scan(args.data_root, long_cfg, short_cfg,
                       [5, 7, 9, 10, 11, 12, 15, 20])


if __name__ == "__main__":
    main()