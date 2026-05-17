"""scripts/run_ablation_huber.py (v4) — disambiguate the MAD-adaptive win.

The v3 results showed VCG-L1Huber with c = 1.4826 * MAD achieving 23.6%
on HC, far above any fixed-c run (best fixed was 18.2% at c in [0.40, 0.60]).

Question this script answers:
    Is the win due to adaptivity (different c per trace), or merely because
    MAD usually hits the c=0.05 floor of the clip range, and a small fixed
    c was simply not in our previous grid?

It scans:
  - L1-Huber fixed c in {0.03, 0.05, 0.08, 0.10, 0.20, 0.50} (cover small c)
  - L1-Huber MAD with scale factors {0.5, 1.0, 1.4826, 2.0, 3.0}
  - L1-Huber with PER-STEP MAD (different c at each step within a trace)
  - Huber with MAD-adaptive c (does adaptivity also help squared-loss VCG?)
  - VCG-Select-by-confidence (pick k with highest max-prob; targets AG)
  - Keeps simple baselines for reference

Diagnostic printed at the top of each subset: MAD distribution across traces
and what fraction of traces hit the c=0.05 floor at scale=1.4826.

Run from repo root:
    python scripts/run_ablation_huber.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcg.allocation import solve_allocation
from vcg.allocation_huber import solve_allocation_huber
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
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def load_trace(path):
    npz = np.load(path, allow_pickle=True)
    pk = _first_present(npz, PROB_KEYS)
    probs = np.asarray(npz[pk], dtype=float)
    gt_step = _safe_int(npz, _first_present(npz, GT_STEP_KEYS))
    return probs, gt_step


# ---------- bandwidth helpers ----------

def mad_global(P):
    """Single MAD scalar over all (k, t) entries, around per-step median."""
    med = np.median(P, axis=0, keepdims=True)
    return float(np.median(np.abs(P - med)))


def mad_per_step(P):
    """Per-step MAD vector (T,)."""
    med = np.median(P, axis=0, keepdims=True)
    return np.median(np.abs(P - med), axis=0)


# ---------- predictors ----------

def pred_simple(P, kind, eps):
    Pc = np.clip(P, eps, 1 - eps)
    if kind == "mean":
        s = Pc.mean(axis=0)
    elif kind == "median":
        s = np.median(Pc, axis=0)
    elif kind == "mean_logit":
        s = np.log(Pc / (1 - Pc)).mean(axis=0)
    elif kind == "max":
        s = Pc.max(axis=0)
    return int(np.argmax(s))


def pred_vcg_original(P, eps=1e-6):
    return int(np.argmax(solve_allocation(P, eps=eps).d))


def pred_vcg_huber(P, c, eps=1e-6):
    return int(np.argmax(solve_allocation_huber(P, c=c, eps=eps).d))


def pred_vcg_l1huber(P, c, eps=1e-6):
    return int(np.argmax(solve_allocation_l1huber(P, c=c, eps=eps).d))


def pred_vcg_l1huber_mad(P, scale=1.4826, eps=1e-6,
                         c_min=0.05, c_max=0.95):
    """Trace-level MAD adaptive c."""
    c = float(np.clip(scale * mad_global(P), c_min, c_max))
    return int(np.argmax(solve_allocation_l1huber(P, c=c, eps=eps).d))


def pred_vcg_l1huber_mad_perstep(P, scale=1.4826, eps=1e-6,
                                 c_min=0.05, c_max=0.95):
    """Per-step MAD: different c at each step. Inline implementation since
    solve_allocation_l1huber takes a single scalar c."""
    P_clip = np.clip(P, eps, 1 - eps)
    K, T = P_clip.shape
    c_t = np.clip(scale * mad_per_step(P_clip), c_min, c_max)        # (T,)
    d = np.median(P_clip, axis=0).copy()

    for _ in range(50):
        d_old = d.copy()
        for t in range(T):
            dev = np.abs(d[t] - P_clip[:, t])
            active = dev < c_t[t]
            if not active.any():
                continue
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            dev_other = np.abs(d[mask][None, :] - P_clip[:, mask])   # (K, T-1)
            ct_other = c_t[mask][None, :]                            # (1, T-1)
            factors = 1.0 - np.minimum(dev_other, ct_other)
            w_t = factors.prod(axis=1)                                # (K,)

            theta_a = P_clip[active, t]
            w_a = w_t[active]
            order = np.argsort(theta_a)
            v_sorted = theta_a[order]
            w_sorted = w_a[order]
            cumw = np.cumsum(w_sorted)
            total = cumw[-1] if cumw.size > 0 else 0.0
            if total <= 0:
                continue
            idx = int(np.searchsorted(cumw, total / 2.0))
            d_new = float(v_sorted[-1] if idx >= v_sorted.size else v_sorted[idx])
            d[t] = float(np.clip(d_new, eps, 1 - eps))
        if np.max(np.abs(d - d_old)) < 1e-8:
            break
    return int(np.argmax(d))


def pred_vcg_huber_mad(P, scale=1.4826, eps=1e-6, c_min=0.05, c_max=0.95):
    """Trace-level MAD adaptive c, applied to squared Huber (not L1)."""
    c = float(np.clip(scale * mad_global(P), c_min, c_max))
    return int(np.argmax(solve_allocation_huber(P, c=c, eps=eps).d))


def pred_vcg_select_confidence(P):
    """Pick discriminator with the highest max-probability vote; return its argmax."""
    k_star = int(np.argmax(P.max(axis=1)))
    return int(np.argmax(P[k_star]))


def pred_vcg_select_range(P):
    """Pick discriminator with the largest probability range (max - min)."""
    k_star = int(np.argmax(P.max(axis=1) - P.min(axis=1)))
    return int(np.argmax(P[k_star]))


# ---------- runner ----------

def evaluate(traces, predict_fn):
    hits = total = 0
    for P, gt in traces:
        if gt is None or not (0 <= gt < P.shape[1]):
            continue
        total += 1
        try:
            pred = predict_fn(P)
        except Exception as exc:
            print(f"  [warn] predict failed: {exc}", file=sys.stderr)
            continue
        if pred == gt:
            hits += 1
    return hits, total


def build_methods():
    methods = [
        ("best_single",  "Best single (oracle-pick)", None),
        ("mean",         "Mean(prob)        eps=1e-6", lambda P: pred_simple(P, "mean", 1e-6)),
        ("mean_eps10",   "Mean(prob)        eps=0.10", lambda P: pred_simple(P, "mean", 0.10)),
        ("median",       "Median(prob)             ", lambda P: pred_simple(P, "median", 1e-6)),
        ("vcg_orig",     "VCG-original      R=0     ", pred_vcg_original),
    ]
    # Fixed-c L1-Huber, ESPECIALLY small c (to disambiguate MAD).
    for c in [0.03, 0.05, 0.08, 0.10, 0.20, 0.50]:
        methods.append((
            f"l1h_fix_c{c:.2f}",
            f"VCG-L1Huber  fix c={c:<4.2f}",
            lambda P, c_=c: pred_vcg_l1huber(P, c_),
        ))
    # MAD scale scan (L1-Huber).
    for s in [0.5, 1.0, 1.4826, 2.0, 3.0]:
        methods.append((
            f"l1h_mad_s{s:.2f}",
            f"VCG-L1Huber  MAD x{s:<4.2f}",
            lambda P, s_=s: pred_vcg_l1huber_mad(P, scale=s_),
        ))
    methods.append((
        "l1h_mad_perstep",
        "VCG-L1Huber  MAD per-step",
        pred_vcg_l1huber_mad_perstep,
    ))
    for s in [0.5, 1.4826, 3.0]:
        methods.append((
            f"huber_mad_s{s:.2f}",
            f"VCG-Huber    MAD x{s:<4.2f}",
            lambda P, s_=s: pred_vcg_huber_mad(P, scale=s_),
        ))
    methods.append((
        "sel_confidence",
        "VCG-Select(max-prob)     ",
        pred_vcg_select_confidence,
    ))
    methods.append((
        "sel_range",
        "VCG-Select(range)        ",
        pred_vcg_select_range,
    ))
    return methods


def run_subset(subset, data_root):
    subset_dir = Path(data_root) / "reports" / f"{subset}_hybrid_v2"
    files = sorted(subset_dir.glob("*.npz"))
    if not files:
        print(f"[ERR] no .npz under {subset_dir}", file=sys.stderr)
        return None
    traces = [load_trace(p) for p in files]
    valid = [t for t in traces if t[1] is not None and 0 <= t[1] < t[0].shape[1]]
    K = traces[0][0].shape[0]
    print(f"  {len(valid)} valid / {len(traces)} total; K = {K}")

    # MAD distribution diagnostic.
    mad_values = np.array([mad_global(np.clip(t[0], 1e-6, 1 - 1e-6)) for t in valid])
    print(f"  MAD distribution across traces: "
          f"min={mad_values.min():.4f}  median={np.median(mad_values):.4f}  "
          f"max={mad_values.max():.4f}  mean={mad_values.mean():.4f}")
    c_at_default = np.clip(1.4826 * mad_values, 0.05, 0.95)
    floor_hit = float(np.mean(c_at_default == 0.05))
    cap_hit = float(np.mean(c_at_default == 0.95))
    print(f"  c at scale=1.4826 (after clip): "
          f"{floor_hit:.0%} traces at floor=0.05, "
          f"{cap_hit:.0%} at cap=0.95, "
          f"unique c values: {len(np.unique(c_at_default.round(3)))}")

    methods = build_methods()
    rows = {}

    accs = []
    for k in range(K):
        h, n = evaluate(traces, lambda P, kk=k: int(np.argmax(P[kk])))
        accs.append((k, h, n))
    bk, bh, bn = max(accs, key=lambda x: x[1] / max(x[2], 1))
    rows["best_single"] = (f"Best single (D{bk})", bh, bn, 0.0)

    for key, label, fn in methods:
        if key == "best_single":
            continue
        t0 = time.time()
        h, n = evaluate(traces, fn)
        rows[key] = (label, h, n, time.time() - t0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--subset", default="both",
                    choices=["Hand-Crafted", "Algorithm-Generated", "both"])
    args = ap.parse_args()

    subsets = (["Hand-Crafted", "Algorithm-Generated"]
               if args.subset == "both" else [args.subset])

    all_results = {}
    for s in subsets:
        print(f"\n{'='*82}\nSUBSET: {s}\n{'='*82}")
        rows = run_subset(s, args.data_root)
        if rows is None:
            continue
        all_results[s] = rows
        label_w = max(len(v[0]) for v in rows.values())
        print(f"\n  {'Method'.ljust(label_w)}     Acc      Hits        Time")
        print(f"  {'-'*label_w}     -----    --------    -----")
        methods = build_methods()
        order = ["best_single"] + [k for k, _, _ in methods if k != "best_single"]
        for k in order:
            if k not in rows:
                continue
            label, h, n, dt = rows[k]
            acc = h / max(n, 1)
            time_s = "" if dt == 0 else f"{dt:>5.1f}s"
            print(f"  {label.ljust(label_w)}   {acc:>6.1%}    {h:>3d}/{n:<4d}   {time_s}")

    if len(all_results) == 2:
        print(f"\n{'='*82}\nSIDE-BY-SIDE SUMMARY\n{'='*82}")
        hc = all_results["Hand-Crafted"]
        ag = all_results["Algorithm-Generated"]
        methods = build_methods()
        order = ["best_single"] + [k for k, _, _ in methods if k != "best_single"]
        clean_label = {k: hc[k][0] for k in hc}
        clean_label["best_single"] = "Best single"
        label_w = max(len(clean_label.get(k, k)) for k in order if k in hc or k in ag)
        print(f"  {'Method'.ljust(label_w)}     HC                  AG")
        print(f"  {'-'*label_w}     ----------------    ----------------")
        for k in order:
            hc_row = hc.get(k)
            ag_row = ag.get(k)
            def cell(row):
                if row is None:
                    return " " * 16
                _, h, n, _ = row
                acc = h / max(n, 1)
                return f"{acc:>5.1%} ({h:>3d}/{n:<3d})"
            label = clean_label.get(k, k)
            print(f"  {label.ljust(label_w)}   {cell(hc_row)}    {cell(ag_row)}")


if __name__ == "__main__":
    main()