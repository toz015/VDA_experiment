"""scripts/eps_sweep.py — sweep external clip (report_eps) to check whether
clipping alone can bring L1-Huber up to Tong's 34.9% baseline on AG.

Background: Tong's reported baseline is VCG (squared kernel) with
  R = 5  OMD rounds  +  report_eps = 0.10  external clip,
which gives HC 9.1%, AG 34.9%.

Our earlier L1-Huber numbers (R = 0, report_eps = 1e-6) were:
  HC 23.6%,  AG 30.2%.

This script tests whether applying Tong's external clip (report_eps = 0.10)
to L1-Huber WITHOUT yet implementing the OMD pipeline closes the AG gap.

For each subset and each report_eps in {1e-6, 0.01, 0.05, 0.10, 0.20}:
  - Clip theta to [report_eps, 1 - report_eps]  (external clip)
  - Run L1-Huber allocation with c in {0.05, 0.10, 0.20}
  - Report step accuracy
  - Also run original VCG (squared kernel) as a sanity check that we can
    reproduce Tong's 34.9% at report_eps = 0.10.

Run from repo root:
    python scripts/eps_sweep.py
    python scripts/eps_sweep.py --subset Algorithm-Generated
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.allocation import solve_allocation
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


def run_subset(subset, data_root):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] no .npz under {subset_dir}", file=sys.stderr)
        return
    traces = [load_trace(p) for p in files]
    valid = [(t[0], t[1]) for t in traces
             if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    n = len(valid)
    print(f"  {n} valid traces, K = {traces[0][0].shape[0]}")

    eps_values = [1e-6, 0.01, 0.05, 0.10, 0.20]
    c_values   = [0.05, 0.10, 0.20]

    # --- Original VCG (squared) sanity check ---
    print(f"\n  Original VCG (squared kernel) — sanity check vs Tong's baseline")
    print(f"  ----------------------------------------------------------------")
    print(f"  {'report_eps':<12s} | {'Acc':<10s} | Hits")
    for re in eps_values:
        def predict_sq(P, re_=re):
            P_clipped = np.clip(P, re_, 1.0 - re_)
            return int(np.argmax(solve_allocation(P_clipped, eps=re_).d))
        h, _ = evaluate(valid, predict_sq)
        print(f"  {re:<12.4g} | {h/max(n,1):>7.1%} | {h}/{n}")

    # --- L1-Huber (linear) at various (report_eps, c) ---
    print(f"\n  L1-Huber kernel — (report_eps × c) grid")
    print(f"  ----------------------------------------")
    header = "  c \\ eps  | " + " | ".join(f"{re:<9.4g}" for re in eps_values)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in c_values:
        row = f"  c={c:<6.3g} | "
        for re in eps_values:
            def predict_l1h(P, re_=re, c_=c):
                P_clipped = np.clip(P, re_, 1.0 - re_)
                return int(np.argmax(
                    solve_allocation_l1huber(P_clipped, c=c_, eps=re_).d))
            h, _ = evaluate(valid, predict_l1h)
            row += f"{h/max(n,1):>7.1%}   | "
        print(row.rstrip(" |"))

    # --- Confidence-Select for reference (does not depend on eps/c) ---
    def predict_select(P):
        k_star = int(np.argmax(P.max(axis=1)))
        return int(np.argmax(P[k_star]))
    h, _ = evaluate(valid, predict_select)
    print(f"\n  Reference: Confidence-Select(max-prob) = {h/max(n,1):.1%}  ({h}/{n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="both",
                    choices=["Hand-Crafted", "Algorithm-Generated", "both"])
    args = ap.parse_args()

    subsets = (["Hand-Crafted", "Algorithm-Generated"]
               if args.subset == "both" else [args.subset])
    for s in subsets:
        print(f"\n{'='*78}\nSUBSET: {s}\n{'='*78}")
        run_subset(s, args.data_root)


if __name__ == "__main__":
    main()