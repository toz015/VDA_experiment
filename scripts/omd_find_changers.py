"""scripts/omd_find_changers.py — find AG traces where OMD changes the argmax.

Compare R=0 vs R=5 predictions per trace; show details of any trace where
the argmax flipped (either from hit -> miss or from miss -> hit).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.pipeline_power import run_pipeline_power


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
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def load_trace(path):
    npz = np.load(path, allow_pickle=True)
    probs = np.asarray(npz[_first_present(npz, PROB_KEYS)], dtype=float)
    gt = _safe_int(npz, _first_present(npz, GT_STEP_KEYS))
    return probs, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="Algorithm-Generated")
    ap.add_argument("--p", type=float, default=1.75)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--report-eps", type=float, default=0.10)
    ap.add_argument("--R", type=int, default=5)
    args = ap.parse_args()

    files = sorted((Path(args.data_root) / "reports" /
                    f"{args.subset}_hybrid_v2").glob("*.npz"),
                   key=lambda p: int(p.stem))

    changes_hit_to_miss = []
    changes_miss_to_hit = []
    changes_miss_to_miss = []
    n_hit_R0 = n_hit_R5 = 0
    n_total = 0

    print(f"  Comparing R=0 vs R={args.R} on {args.subset} ...")
    for f in files:
        theta, gt = load_trace(f)
        if gt is None or not (0 <= gt < theta.shape[1]):
            continue
        n_total += 1

        res0 = run_pipeline_power(theta, c=args.c, p=args.p, R=0,
                                  report_eps=args.report_eps)
        resR = run_pipeline_power(theta, c=args.c, p=args.p, R=args.R,
                                  report_eps=args.report_eps)

        pred0 = int(np.argmax(res0.theta_bar))
        predR = int(np.argmax(resR.theta_bar))
        hit0 = (pred0 == gt)
        hitR = (predR == gt)
        n_hit_R0 += int(hit0)
        n_hit_R5 += int(hitR)

        if pred0 != predR:
            entry = (int(f.stem), gt, pred0, predR, hit0, hitR, theta)
            if hit0 and not hitR:
                changes_hit_to_miss.append(entry)
            elif not hit0 and hitR:
                changes_miss_to_hit.append(entry)
            else:
                changes_miss_to_miss.append(entry)

    print(f"\n  Summary:")
    print(f"    n_total: {n_total}")
    print(f"    R=0 hits: {n_hit_R0}  ({n_hit_R0/n_total:.1%})")
    print(f"    R={args.R} hits: {n_hit_R5}  ({n_hit_R5/n_total:.1%})")
    print(f"    Changes hit -> miss:  {len(changes_hit_to_miss)}")
    print(f"    Changes miss -> hit:  {len(changes_miss_to_hit)}")
    print(f"    Changes miss -> miss: {len(changes_miss_to_miss)}")

    def print_change(entry, kind):
        trace_id, gt, p0, pR, h0, hR, theta = entry
        print(f"\n  Trace {trace_id} ({kind}):  gt={gt}, R=0 pred={p0} (hit={h0}), R=5 pred={pR} (hit={hR})")
        print(f"    Raw reports (K x T):")
        for k in range(theta.shape[0]):
            row = "  ".join(f"{v:.3f}" for v in theta[k])
            print(f"      D{k}: {row}")
        print(f"    Argmax per disc: {theta.argmax(axis=1)}")

    print(f"\n  --- HIT -> MISS  (OMD broke a correct prediction) ---")
    for e in changes_hit_to_miss:
        print_change(e, "OMD broke it")

    print(f"\n  --- MISS -> HIT  (OMD fixed an incorrect prediction) ---")
    for e in changes_miss_to_hit:
        print_change(e, "OMD fixed it")

    if changes_miss_to_miss:
        print(f"\n  --- MISS -> MISS  (OMD changed prediction but still wrong) ---")
        for e in changes_miss_to_miss:
            print_change(e, "OMD changed it, still wrong")


if __name__ == "__main__":
    main()