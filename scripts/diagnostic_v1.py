"""
diagnostic_v1.py — Anchoring / oracle / saturation / correlation diagnostics
for the multiplicative-VCG failure-attribution pipeline.

Run from the repo root:

    python scripts/diagnostic_v1.py --inspect --subset Hand-Crafted
        # first pass: print the structure of one .npz file so you can
        # confirm the field-name guesses below

    python scripts/diagnostic_v1.py --subset Hand-Crafted
    python scripts/diagnostic_v1.py --subset Algorithm-Generated

Reads:
    data/reports/<subset>_hybrid_v2/*.npz
        Each file is one trace and is assumed to contain
            - a (K, T) array of raw probabilities (key tried in PROB_KEYS)
            - a scalar ground-truth step index (key tried in GT_STEP_KEYS)
            - optionally a ground-truth agent name and discriminator names

If the field names in your npz files differ, either rename them or extend
the *_KEYS lists at the top.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Field-name guesses — adjust if your npz keys differ
# ---------------------------------------------------------------------------
PROB_KEYS     = ["probs", "prob", "p", "scores", "theta", "theta_hat", "raw"]
GT_STEP_KEYS  = ["gt_step", "mistake_step", "true_step", "label_step"]
GT_AGENT_KEYS = ["gt_agent", "mistake_agent", "true_agent", "label_agent"]
DISC_KEYS     = ["discriminators", "models", "disc_names", "names"]


def first_present(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def _safe_item(npz, key, cast):
    """Read 0-d numpy entry; return cast(value), or None if missing/None/NaN/uncastable."""
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


def load_trace(path: Path):
    """Return dict with probs (K, T) float, gt_step int|None, gt_agent str|None."""
    npz = np.load(path, allow_pickle=True)
    pk = first_present(npz, PROB_KEYS)
    if pk is None:
        for k in npz.files:
            arr = npz[k]
            if getattr(arr, "ndim", 0) == 2 and np.issubdtype(arr.dtype, np.floating):
                pk = k
                break
    if pk is None:
        raise RuntimeError(f"No probability matrix found in {path.name}; "
                           f"keys = {list(npz.files)}")
    probs = np.asarray(npz[pk], dtype=float)
    if probs.ndim != 2:
        raise RuntimeError(f"{path.name}: expected 2-D probs, got shape {probs.shape}")

    gt_step  = _safe_item(npz, first_present(npz, GT_STEP_KEYS),  int)
    gt_agent = _safe_item(npz, first_present(npz, GT_AGENT_KEYS), str)

    dk = first_present(npz, DISC_KEYS)
    if dk:
        discs = [str(x) for x in np.asarray(npz[dk]).flatten()]
    else:
        discs = [f"D{i}" for i in range(probs.shape[0])]

    return {"path": path, "probs": probs,
            "gt_step": gt_step, "gt_agent": gt_agent, "discs": discs}

def inspect_one(path: Path):
    print(f"\n=== Structure of {path.name} ===")
    npz = np.load(path, allow_pickle=True)
    print(f"Keys: {list(npz.files)}")
    for k in npz.files:
        arr = npz[k]
        try:
            print(f"  {k!r:>20s} : shape={arr.shape}  dtype={arr.dtype}")
        except AttributeError:
            print(f"  {k!r:>20s} : {type(arr).__name__} value={arr!r}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def oracle_upper_bound(traces):
    """Fraction of traces where SOME discriminator's argmax-over-steps hits gt."""
    hit = total = 0
    for tr in traces:
        if tr["gt_step"] is None:
            continue
        total += 1
        if (tr["probs"].argmax(axis=1) == tr["gt_step"]).any():
            hit += 1
    return hit / max(total, 1)


def anchoring_profile(traces, eta_list, frac_threshold):
    """
    For each (eta, k): fraction of traces in which D_k satisfies
        |p_t^k - y_t| <= eta  on at least  frac_threshold * T  steps,
    where y_t = 1[t == gt_step].

    Theorem 4.X's hypothesis requires a *majority* of discriminators to
    pass this test at the chosen (eta, frac_threshold).
    """
    K = traces[0]["probs"].shape[0]
    out = {}
    for eta in eta_list:
        passed = np.zeros(K)
        n = 0
        for tr in traces:
            if tr["gt_step"] is None:
                continue
            n += 1
            P = tr["probs"]
            T_ = P.shape[1]
            y = np.zeros(T_)
            if 0 <= tr["gt_step"] < T_:
                y[tr["gt_step"]] = 1.0
            within = np.abs(P - y) <= eta
            frac_within = within.mean(axis=1)
            passed += (frac_within >= frac_threshold).astype(float)
        out[eta] = passed / max(n, 1)
    return out


def saturation_summary(traces, eps_list):
    """Fraction of raw probability mass clipped at each eps."""
    flat = np.concatenate([tr["probs"].reshape(-1) for tr in traces])
    rows = []
    for eps in eps_list:
        rows.append((eps,
                     float((flat <= eps).mean()),
                     float((flat >= 1 - eps).mean()),
                     float(((flat > eps) & (flat < 1 - eps)).mean())))
    return rows, float(flat.mean()), float(flat.std())


def pairwise_correlation(traces):
    """Pearson correlation between discriminators across all (trace, step)."""
    P = np.concatenate([tr["probs"] for tr in traces], axis=1)  # (K, sum T)
    Pc = P - P.mean(axis=1, keepdims=True)
    cov = Pc @ Pc.T / Pc.shape[1]
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return cov / np.outer(sd, sd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="Hand-Crafted",
                    choices=["Hand-Crafted", "Algorithm-Generated"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--eta", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--frac", type=float, default=0.5,
                    help="gamma: fraction of steps that must be accurate")
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.01, 0.05, 0.10])
    ap.add_argument("--inspect", action="store_true",
                    help="Print npz structure for the first file and exit")
    args = ap.parse_args()

    subset_dir = Path(args.data_root) / "reports" / f"{args.subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] No .npz under {subset_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} traces under {subset_dir}")

    if args.inspect:
        inspect_one(files[0])
        return

    traces = [load_trace(p) for p in files]
    K = traces[0]["probs"].shape[0]
    names = traces[0]["discs"]
    n_with_gt = sum(1 for t in traces if t["gt_step"] is not None)
    Ts = [t["probs"].shape[1] for t in traces]
    print(f"K = {K}   discriminators = {names}")
    print(f"Traces with gt step: {n_with_gt}/{len(traces)}")
    print(f"T distribution: min={min(Ts)} median={int(np.median(Ts))} max={max(Ts)}")

    # 1. Oracle ceiling
    print("\n" + "=" * 70)
    print("[1] ORACLE UPPER BOUND")
    print("=" * 70)
    oracle = oracle_upper_bound(traces)
    print(f"Fraction of traces where at least one discriminator's "
          f"argmax-step == gt step: {oracle:.1%}")
    print("This is the ceiling any per-trace aggregator can hit. If this is")
    print("close to your best single-model number, no aggregator will help.")

    # 2. Anchoring
    print("\n" + "=" * 70)
    print(f"[2] ANCHORING PROFILE  (frac threshold gamma = {args.frac:.0%})")
    print("=" * 70)
    profile = anchoring_profile(traces, args.eta, args.frac)
    print("Frac of traces where |p - y| <= eta on at least gamma fraction of steps:")
    print("  eta    " + "  ".join(f"{n:>12}" for n in names))
    for eta, row in profile.items():
        cells = "  ".join(f"{v:>12.1%}" for v in row)
        print(f"  {eta:>4.2f}   {cells}")
    print("\nTheorem 4.X needs a majority of discriminators with high values")
    print("at moderate eta. If columns are uniformly low, anchoring fails")
    print("and the multiplicative mechanism has no theoretical guarantee.")

    # 3. Saturation
    print("\n" + "=" * 70)
    print("[3] SATURATION CHECK")
    print("=" * 70)
    rows, mean_p, std_p = saturation_summary(traces, args.eps)
    print(f"Pooled raw p: mean = {mean_p:.3f}, std = {std_p:.3f}")
    print("  eps    p <= eps    p >= 1-eps    in middle")
    for eps, lo, hi, mid in rows:
        print(f"  {eps:>4.2f}   {lo:>8.1%}    {hi:>10.1%}    {mid:>9.1%}")
    print("\nLarge fractions at the boundaries indicate hard clipping is")
    print("destroying signal -- switch to logit-space (temperature-scaled)")
    print("aggregation. This is what your email's diagnostic-2 was after.")

    # 4. Correlation
    print("\n" + "=" * 70)
    print("[4] PAIRWISE CORRELATION (Pearson, pooled over all (trace, step))")
    print("=" * 70)
    corr = pairwise_correlation(traces)
    print("        " + "  ".join(f"{n:>8}" for n in names))
    for i, n in enumerate(names):
        cells = "  ".join(f"{c:>8.3f}" for c in corr[i])
        print(f"  {n:>5}  {cells}")
    print("\nIf all off-diagonal r > ~0.7, common-noise dominates and the")
    print("'cross-step consistency' signal is mostly shared bias, not truth.")
    print("Multiplicative weighting will amplify the shared bias.")


if __name__ == "__main__":
    main()