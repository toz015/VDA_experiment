"""scripts/scan_pc.py — sweep (p, c) on the power-kernel VCG, R = 0, no OMD.

Kernel:  kappa_{c, p}(x) = 1 - min(|x|^p, c^p),   p >= 1.

This is an exploratory scan to identify whether any (p, c) gives a
single-mechanism win on both HC and AG. R = 0 is used throughout for
speed; if a promising configuration emerges, OMD calibration can be added
as a second-stage experiment.

For each subset and each (p, c) on a coarse grid, applies the same
report_eps pre-clip as Tong's pipeline (report_eps = 1e-6 to keep raw
saturation; can be overridden to 0.10 to mirror the AG baseline setup).

Run from repo root:
    python scripts/scan_pc.py                       # default report_eps=1e-6
    python scripts/scan_pc.py --report-eps 0.10
    python scripts/scan_pc.py --subset Algorithm-Generated
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


def run_subset(subset, data_root, report_eps, p_values, c_values):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] no .npz under {subset_dir}", file=sys.stderr)
        return
    traces = [load_trace(f) for f in files]
    valid = [(t[0], t[1]) for t in traces
             if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    n = len(valid)
    K = traces[0][0].shape[0]
    print(f"  {n} valid traces, K = {K}, report_eps = {report_eps}")

    def predict_fn(P, c_val, p_val):
        P_clipped = np.clip(P, report_eps, 1.0 - report_eps)
        return int(np.argmax(
            solve_allocation_power(P_clipped, c=c_val, p=p_val, eps=report_eps).d
        ))

    # Print grid header.
    print(f"\n  Step accuracy on {subset}, report_eps = {report_eps}, R = 0:")
    print(f"  {'p \\ c':<8s} | " + " | ".join(f"{c:<8.3g}" for c in c_values))
    print("  " + "-" * (10 + len(c_values) * 11))
    results = {}
    for p_val in p_values:
        row = f"  p={p_val:<5.2g} | "
        for c_val in c_values:
            h, _ = evaluate(valid, lambda P, c_=c_val, p_=p_val: predict_fn(P, c_, p_))
            results[(p_val, c_val)] = (h, n)
            row += f"{h/max(n,1):>6.1%} | "
        print(row.rstrip(" |"))

    # Highlight the best cell.
    best_pc = max(results, key=lambda pc: results[pc][0])
    h, _ = results[best_pc]
    print(f"\n  Best on {subset}: p = {best_pc[0]}, c = {best_pc[1]}: "
          f"{h/max(n,1):.1%} ({h}/{n})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="both",
                    choices=["Hand-Crafted", "Algorithm-Generated", "both"])
    ap.add_argument("--report-eps", type=float, default=1e-6,
                    help="external clip on raw reports (default 1e-6; "
                         "use 0.10 to match Tong's AG baseline)")
    ap.add_argument("--p-values", type=str, default="1.0,1.25,1.5,1.75,2.0",
                    help="comma-separated p values to scan")
    ap.add_argument("--c-values", type=str, default="0.05,0.10,0.20,0.50,1.00",
                    help="comma-separated c values to scan")
    args = ap.parse_args()

    p_values = [float(x) for x in args.p_values.split(",")]
    c_values = [float(x) for x in args.c_values.split(",")]

    subsets = (["Hand-Crafted", "Algorithm-Generated"]
               if args.subset == "both" else [args.subset])
    all_results = {}
    for s in subsets:
        print(f"\n{'='*78}\nSUBSET: {s}\n{'='*78}")
        all_results[s] = run_subset(
            s, args.data_root, args.report_eps, p_values, c_values
        )

    # If both subsets scanned, look for a (p, c) that is good on both.
    if len(subsets) == 2 and all(all_results.values()):
        print(f"\n{'='*78}")
        print(f"Cross-subset summary  (HC acc, AG acc) per (p, c) cell:")
        print(f"{'='*78}")
        print(f"  {'p \\ c':<8s} | " + " | ".join(f"{c:<14.3g}" for c in c_values))
        print("  " + "-" * (10 + len(c_values) * 17))
        for p_val in p_values:
            row = f"  p={p_val:<5.2g} | "
            for c_val in c_values:
                hh, nh = all_results["Hand-Crafted"][(p_val, c_val)]
                ha, na = all_results["Algorithm-Generated"][(p_val, c_val)]
                row += f"{hh/max(nh,1):>5.1%} / {ha/max(na,1):>5.1%} | "
            print(row.rstrip(" |"))

        # Best by sum of accuracies.
        def total(pc):
            hh, nh = all_results["Hand-Crafted"][pc]
            ha, na = all_results["Algorithm-Generated"][pc]
            return hh / max(nh, 1) + ha / max(na, 1)
        best_pc = max(all_results["Hand-Crafted"].keys(), key=total)
        hh, nh = all_results["Hand-Crafted"][best_pc]
        ha, na = all_results["Algorithm-Generated"][best_pc]
        print(f"\n  Best joint (HC + AG) at p = {best_pc[0]}, c = {best_pc[1]}:")
        print(f"    HC: {hh/max(nh,1):.1%} ({hh}/{nh})")
        print(f"    AG: {ha/max(na,1):.1%} ({ha}/{na})")


if __name__ == "__main__":
    main()