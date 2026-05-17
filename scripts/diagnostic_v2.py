"""
diagnostic_v2.py — Aggregator comparison + margin + GT-step inspection.

What this script adds beyond diagnostic_v1:
  [1] Per-discriminator-only argmax-step accuracy (cross-check the email).
  [2] Multiple aggregators (mean / median / max / logit-mean) under several
      eps-clipping levels — isolates "is clipping hurting?" from "is the
      aggregation rule wrong?".
  [3] Ceilings (oracle, best single).
  [4] Per-trace margin distribution under the best aggregator — tells you
      how 'protected' the argmax decision actually is.
  [5] Per-discriminator P(GT step) statistics — what probability does each
      model assign to the actual mistake step? This is the hard upper bound
      that no aggregation can exceed.

Run from repo root:
    python scripts/diagnostic_v2.py --subset Hand-Crafted
    python scripts/diagnostic_v2.py --subset Algorithm-Generated
"""

import argparse
import sys
from pathlib import Path
import numpy as np


PROB_KEYS     = ["theta_hat", "probs", "prob", "p", "scores", "theta", "raw"]
GT_STEP_KEYS  = ["mistake_step", "gt_step", "true_step", "label_step"]
DISC_KEYS     = ["model_ids", "discriminators", "models", "disc_names"]
FALLBACK_KEYS = ["fallback_counts"]


def first_present(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def _safe_item(npz, key, cast):
    if not key:
        return None
    try:
        val = np.asarray(npz[key]).item()
    except (ValueError, AttributeError):
        return None
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        return cast(val)
    except (ValueError, TypeError):
        return None


def load_trace(path):
    npz = np.load(path, allow_pickle=True)
    pk = first_present(npz, PROB_KEYS)
    if pk is None:
        raise RuntimeError(f"No probability matrix in {path.name}")
    probs = np.asarray(npz[pk], dtype=float)
    gt_step = _safe_item(npz, first_present(npz, GT_STEP_KEYS), int)
    dk = first_present(npz, DISC_KEYS)
    discs = ([str(x) for x in np.asarray(npz[dk]).flatten()]
             if dk else [f"D{i}" for i in range(probs.shape[0])])
    fk = first_present(npz, FALLBACK_KEYS)
    fallback = (np.asarray(npz[fk]).flatten().astype(int)
                if fk else np.zeros(probs.shape[0], dtype=int))
    return {"path": path, "probs": probs, "gt_step": gt_step,
            "discs": discs, "fallback": fallback}


# ---------- aggregators ----------
# Each takes a (K, T) probability matrix and returns a (T,) score vector.

def agg_mean_prob(P):
    return P.mean(axis=0)


def agg_median_prob(P):
    return np.median(P, axis=0)


def agg_max_prob(P):
    return P.max(axis=0)


def agg_mean_logit(P):
    """Geometric-odds mean.  P must already be clipped strictly inside (0, 1)."""
    logit = np.log(P / (1.0 - P))
    return 1.0 / (1.0 + np.exp(-logit.mean(axis=0)))


def clip_eps(P, eps):
    return np.clip(P, eps, 1.0 - eps)


# ---------- evaluation helpers ----------

def argmax_accuracy(traces, score_fn):
    hits = total = 0
    for tr in traces:
        if tr["gt_step"] is None:
            continue
        T_ = tr["probs"].shape[1]
        if not (0 <= tr["gt_step"] < T_):
            continue
        total += 1
        scores = score_fn(tr["probs"])
        if int(np.argmax(scores)) == tr["gt_step"]:
            hits += 1
    return (hits / max(total, 1), total)


def margins_under(traces, score_fn):
    all_m, ok_m, bad_m = [], [], []
    for tr in traces:
        if tr["gt_step"] is None:
            continue
        T_ = tr["probs"].shape[1]
        if T_ < 2 or not (0 <= tr["gt_step"] < T_):
            continue
        scores = score_fn(tr["probs"])
        s_sorted = np.sort(scores)[::-1]
        m = s_sorted[0] - s_sorted[1]
        all_m.append(m)
        if int(np.argmax(scores)) == tr["gt_step"]:
            ok_m.append(m)
        else:
            bad_m.append(m)
    return np.array(all_m), np.array(ok_m), np.array(bad_m)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="Hand-Crafted",
                    choices=["Hand-Crafted", "Algorithm-Generated"])
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    subset_dir = Path(args.data_root) / "reports" / f"{args.subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] No .npz under {subset_dir}", file=sys.stderr)
        sys.exit(1)

    traces = [load_trace(p) for p in files]
    valid = [t for t in traces
             if t["gt_step"] is not None
             and 0 <= t["gt_step"] < t["probs"].shape[1]]
    K = traces[0]["probs"].shape[0]
    disc_names = traces[0]["discs"]

    print(f"Subset: {args.subset}")
    print(f"Total traces: {len(traces)}    "
          f"valid (with in-range GT step): {len(valid)}")
    print(f"K = {K}    discriminators = {disc_names}")

    # Fallback summary
    total_fallback = np.zeros(K, dtype=int)
    total_calls = 0
    for t in traces:
        total_fallback += t["fallback"]
        total_calls += t["probs"].shape[1]
    print(f"\nTotal LLM calls per discriminator: {total_calls}")
    print("Fallback (failed-parse) counts:")
    for n, fc in zip(disc_names, total_fallback):
        rate = fc / max(total_calls, 1)
        print(f"  {n:<40s}  {fc:>5d}  ({rate:>5.1%})")

    # ----- 1. Per-discriminator-only argmax accuracy -----
    print("\n" + "=" * 72)
    print("[1] PER-DISCRIMINATOR ARGMAX-STEP ACCURACY  (cross-check the email)")
    print("=" * 72)
    for k, name in enumerate(disc_names):
        acc, n = argmax_accuracy(valid, lambda P, kk=k: P[kk])
        hits = int(round(acc * n))
        print(f"  {name:<40s}  {acc:>6.1%}   ({hits}/{n})")

    # ----- 2. Aggregator x eps -----
    print("\n" + "=" * 72)
    print("[2] AGGREGATOR x EPS  (step-argmax accuracy)")
    print("=" * 72)
    eps_list = [1e-6, 0.01, 0.05, 0.10]
    header = "  Aggregator               " + \
             "  ".join(f"eps={e:<7g}" for e in eps_list)
    print(header)
    for agg_name, agg_fn in [
        ("Mean   (prob space)",  agg_mean_prob),
        ("Median (prob space)",  agg_median_prob),
        ("Max    (prob space)",  agg_max_prob),
        ("Mean   (logit space)", agg_mean_logit),
    ]:
        row = []
        for eps in eps_list:
            fn = (lambda P, fn_=agg_fn, e_=eps: fn_(clip_eps(P, e_)))
            acc, _ = argmax_accuracy(valid, fn)
            row.append(acc)
        cells = "  ".join(f"{a:>10.1%}" for a in row)
        print(f"  {agg_name:<23s}  {cells}")
    print()
    print("  Notes:")
    print("  - 'Mean prob @ eps=0.10' is the closest one-shot analogue to your")
    print("    current pipeline (without OMD/VCG).")
    print("  - If 'Mean logit @ eps=1e-6' beats every row-1 cell, saturation is")
    print("    the bottleneck -- switch to logit-space aggregation.")
    print("  - 'Max prob' is a heuristic ceiling for any-agreement aggregators.")

    # ----- 3. Ceilings -----
    print("\n" + "=" * 72)
    print("[3] CEILINGS")
    print("=" * 72)
    oracle = 0
    for tr in valid:
        if (tr["probs"].argmax(axis=1) == tr["gt_step"]).any():
            oracle += 1
    print(f"  Oracle (some disc's argmax == gt):   {oracle/len(valid):>6.1%}   "
          f"({oracle}/{len(valid)})")
    best_single = max(argmax_accuracy(valid, lambda P, kk=k: P[kk])[0]
                      for k in range(K))
    print(f"  Best single discriminator:           {best_single:>6.1%}")

    # ----- 4. Margin distribution under best aggregator (Mean logit, eps=1e-6) -----
    print("\n" + "=" * 72)
    print("[4] MARGIN DISTRIBUTION  under Mean(logit), eps=1e-6")
    print("=" * 72)
    fn_best = (lambda P: agg_mean_logit(clip_eps(P, 1e-6)))
    m_all, m_ok, m_bad = margins_under(valid, fn_best)
    print(f"  All     ({len(m_all):>4d}):  "
          f"mean={m_all.mean():.3f}  median={np.median(m_all):.3f}  "
          f"std={m_all.std():.3f}")
    if len(m_ok):
        print(f"  Correct ({len(m_ok):>4d}):  "
              f"mean={m_ok.mean():.3f}  median={np.median(m_ok):.3f}")
    if len(m_bad):
        print(f"  Wrong   ({len(m_bad):>4d}):  "
              f"mean={m_bad.mean():.3f}  median={np.median(m_bad):.3f}")
    print("\n  Histogram of margins (all traces):")
    bins = [0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((m_all >= lo) & (m_all < hi)).sum())
        bar = "#" * int(40 * n / max(len(m_all), 1))
        print(f"    [{lo:>5.2f}, {hi:>5.2f}):  {n:>4d}  {bar}")
    print()
    print("  Reading the table:")
    print("  - If correct-margin << wrong-margin, no useful signal at trace level.")
    print("  - If correct-margin >> wrong-margin, argmax is well-protected.")
    print("  - If most mass sits in [0, 0.05), small noise can flip the answer.")

    # ----- 5. P(GT step) per discriminator -----
    print("\n" + "=" * 72)
    print("[5] P(GT step) PER DISCRIMINATOR  -- how confident on the truth?")
    print("=" * 72)
    gt_probs = np.zeros((K, len(valid)))
    for i, tr in enumerate(valid):
        gt_probs[:, i] = tr["probs"][:, tr["gt_step"]]
    name_row = "  " + " " * 24 + "  ".join(f"{n:>14s}" for n in disc_names)
    print(name_row)
    for stat_name, fn in [
        ("mean P(GT step)",       np.mean),
        ("median P(GT step)",     np.median),
        ("std P(GT step)",        np.std),
        ("frac > 0.5",            lambda x: np.mean(x > 0.5)),
        ("frac > 0.1",            lambda x: np.mean(x > 0.1)),
        ("frac < 0.01 (bad)",     lambda x: np.mean(x < 0.01)),
    ]:
        cells = "  ".join(f"{fn(gt_probs[k]):>14.3f}" for k in range(K))
        print(f"  {stat_name:<25s} {cells}")
    print()
    print("  Reading the table:")
    print("  - 'mean P(GT step)' is the average probability assigned to the actual")
    print("    mistake step. Higher = more signal.")
    print("  - 'frac > 0.5' is the upper bound on step accuracy obtainable by")
    print("    *thresholding* that single discriminator at 0.5.")
    print("  - 'frac < 0.01' counts traces where the discriminator confidently")
    print("    fails to flag the truth -- no aggregator can recover these.")


if __name__ == "__main__":
    main()