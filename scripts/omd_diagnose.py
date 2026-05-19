"""scripts/omd_diagnose.py — diagnose OMD dynamics on individual traces.

For a small set of AG traces, track:
  - Initial argmax_t d_t vs. final argmax_t d_t
  - Whether OMD moved reports significantly
  - Whether sign convention of finite-diff gradient is consistent
    with theoretical expectations
  - Whether the "consensus drift" hypothesis is borne out per-trace

Run from repo root:
    python scripts/omd_diagnose.py
    python scripts/omd_diagnose.py --trace-id 0
    python scripts/omd_diagnose.py --n-traces 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.pipeline_power import run_pipeline_power
from vcg.allocation_power import solve_allocation_power
from vcg.gradient_power import compute_power_gradient_matrix
from vcg.payment_power import compute_power_payments


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


def diagnose_one(theta, gt, p=1.75, c=1.0, report_eps=0.10, R=5):
    """Run R=0 and R=5 versions and compare."""
    print(f"\n--- Trace gt={gt}, T={theta.shape[1]}, p={p}, c={c}, eps={report_eps} ---")
    print(f"  theta_hat raw (K x T):")
    for k in range(theta.shape[0]):
        print(f"    D{k}: " + "  ".join(f"{v:.3f}" for v in theta[k]))
    print(f"  argmax_t per discriminator: {theta.argmax(axis=1)}")

    res0 = run_pipeline_power(theta, c=c, p=p, R=0, report_eps=report_eps)
    res5 = run_pipeline_power(theta, c=c, p=p, R=R, report_eps=report_eps)

    init_argmax = int(np.argmax(res0.theta_bar))
    final_argmax = int(np.argmax(res5.theta_bar))
    print(f"  Initial d (R=0): " + "  ".join(f"{v:.3f}" for v in res0.theta_bar))
    print(f"  Final d   (R=5): " + "  ".join(f"{v:.3f}" for v in res5.theta_bar))
    print(f"  Initial argmax (R=0): {init_argmax}  (hit={init_argmax==gt})")
    print(f"  Final   argmax (R=5): {final_argmax}  (hit={final_argmax==gt})")
    print(f"  OMD converged: {res5.omd_converged}")

    # Theta drift per OMD round
    if len(res5.omd_history) > 1:
        drift = np.max(np.abs(res5.omd_history[-1] - res5.omd_history[0]))
        print(f"  Theta drift L_inf: {drift:.4f}")

    # Check finite-diff gradient at the initial theta
    theta_clipped = np.clip(theta, report_eps, 1.0 - report_eps)
    g = compute_power_gradient_matrix(theta_clipped, c=c, p=p)
    print(f"  Initial gradient sign per (k, t):")
    for k in range(theta.shape[0]):
        signs = "".join("+" if g[k, t] > 1e-8 else
                        "-" if g[k, t] < -1e-8 else "0"
                        for t in range(theta.shape[1]))
        print(f"    D{k}: {signs}    max|g|={np.max(np.abs(g[k])):.4e}")

    # Look at payments at initial alloc
    init_alloc = solve_allocation_power(theta_clipped, c=c, p=p)
    pay = compute_power_payments(theta_clipped, init_alloc, c=c, p=p)
    print(f"  Initial payments Pi_k: {pay.payments.round(4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="Algorithm-Generated")
    ap.add_argument("--trace-id", type=int, default=None,
                    help="specific trace id; if not set, inspect first --n-traces")
    ap.add_argument("--n-traces", type=int, default=5,
                    help="number of traces to inspect when --trace-id not set")
    ap.add_argument("--p", type=float, default=1.75)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--report-eps", type=float, default=0.10)
    ap.add_argument("--R", type=int, default=5)
    args = ap.parse_args()

    files = sorted((Path(args.data_root) / "reports" /
                    f"{args.subset}_hybrid_v2").glob("*.npz"),
                   key=lambda p: int(p.stem))

    if args.trace_id is not None:
        target = next((f for f in files if int(f.stem) == args.trace_id), None)
        if not target:
            print(f"trace id {args.trace_id} not found.")
            return
        theta, gt = load_trace(target)
        diagnose_one(theta, gt, p=args.p, c=args.c,
                     report_eps=args.report_eps, R=args.R)
        return

    # Inspect first N traces (or up to n_traces with valid gt)
    n_inspected = 0
    for f in files:
        if n_inspected >= args.n_traces:
            break
        theta, gt = load_trace(f)
        if gt is None or not (0 <= gt < theta.shape[1]):
            continue
        diagnose_one(theta, gt, p=args.p, c=args.c,
                     report_eps=args.report_eps, R=args.R)
        n_inspected += 1


if __name__ == "__main__":
    main()