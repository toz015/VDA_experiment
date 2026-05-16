"""Compare top-K accuracy of the 3 organization methods (v2 ensemble).

Method 1: per-message, action-only (36 traces, target = mistake_step in action-only index)
Method 2: hybrid full-trace + LLM step classification (58 traces, target = mistake_step in full-history index)
Method 3: DoVer trial-level (58 traces, target = gt_trial; for hybrid-equivalent message accuracy
          we score within the predicted trial via argmax of per-message theta if available, but
          here Method 3 reports are trial-level theta only — we measure trial-level top-K).

For each method, report per-discriminator and ensemble (mean) top-1/top-3 hit rate against
ground truth, plus the gemini-only / qwen-only / gpt-oss-only breakdown.

Usage:
    python scripts/compare_v2_methods.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    ("Method 1: per-message (action-only)",      "data/reports/Hand-Crafted_v2",                      "msg"),
    ("Method 2: hybrid full-trace",              "data/reports/Hand-Crafted_hybrid_v2",               "msg"),
    ("Method 3: DoVer-trial",                    "data/reports/Hand-Crafted_dover_v2",                "trial"),
    ("Method 4: full-history + action priors",   "data/reports/Hand-Crafted_method4_v2",              "msg"),
    ("Method 4 (AG): full-history + action priors", "data/reports/Algorithm-Generated_method4_v2",   "msg"),
    ("Method 2 (AG): hybrid full-trace",         "data/reports/Algorithm-Generated_hybrid_v2",        "msg"),
]


def topk_hit(scores, target, k):
    if target is None or target < 0:
        return None
    order = np.argsort(-scores)
    return int(target in order[:k].tolist())


def evaluate_method(name, dir_rel, mode, summary=None):
    dir_path = ROOT / dir_rel
    files = sorted(dir_path.glob("*.npz"))
    if not files:
        print(f"\n{name}: NO REPORTS in {dir_path}")
        if summary is not None:
            summary[name] = None
        return

    rows = []
    K = None
    model_ids = None
    for f in files:
        data = np.load(f, allow_pickle=True)
        theta = data["theta_hat"]                  # (K, T) or (K, n_trials)
        if K is None:
            K = theta.shape[0]
            model_ids = [str(x) for x in data["model_ids"].tolist()]
        if mode == "msg":
            raw = data["mistake_step"]
            try:
                target = int(raw)
            except (TypeError, ValueError):
                target = -1
        else:
            target = int(data["gt_trial"])
        rows.append((theta, target, theta.shape[1]))

    print(f"\n{name}")
    print(f"  dir: {dir_rel}")
    print(f"  traces: {len(rows)}, K={K}, model_ids={model_ids}")

    skipped = sum(1 for _, t, _ in rows if t is None or t < 0)
    if skipped:
        print(f"  skipped (no valid target): {skipped}")

    valid = [(theta, t, T) for theta, t, T in rows if t is not None and t >= 0]
    n = len(valid)
    if n == 0:
        return

    Ks = [1, 3, 5]

    # Accumulate hit counts so we can both print per-method and feed the
    # consolidated summary table.
    per_disc_pct = {}    # model_id -> {k: pct}
    ens_raw_pct = {}     # k -> pct
    ens_znorm_pct = {}   # k -> pct
    ens_mm_pct = {}      # k -> pct

    print(f"  --- per-discriminator ---")
    for k_idx in range(K):
        line = f"  {model_ids[k_idx]:60s}"
        per_disc_pct[model_ids[k_idx]] = {}
        for kk in Ks:
            hits = sum(topk_hit(theta[k_idx], t, kk) for theta, t, _ in valid)
            pct = hits / n * 100
            per_disc_pct[model_ids[k_idx]][kk] = pct
            line += f"  top-{kk}={pct:5.1f}%"
        print(line)

    print(f"  --- ensemble (mean) ---")
    line = f"  {'mean (raw)':60s}"
    for kk in Ks:
        hits = 0
        for theta, t, _ in valid:
            scores = theta.mean(axis=0)
            hits += topk_hit(scores, t, kk)
        pct = hits / n * 100
        ens_raw_pct[kk] = pct
        line += f"  top-{kk}={pct:5.1f}%"
    print(line)

    # Per-trace z-score normalization: each discriminator's row becomes
    # mean=0, std=1 within the trace. Then average. This neutralizes the
    # calibration mismatch where one model is conservative (low-mean) and
    # another is confident (high-mean) — both get equal weight in the
    # combined ranking.
    line = f"  {'mean (z-norm per trace, per-disc)':60s}"
    for kk in Ks:
        hits = 0
        for theta, t, _ in valid:
            mu = theta.mean(axis=1, keepdims=True)
            sd = theta.std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-9, 1.0, sd)
            znorm = (theta - mu) / sd
            scores = znorm.mean(axis=0)
            hits += topk_hit(scores, t, kk)
        pct = hits / n * 100
        ens_znorm_pct[kk] = pct
        line += f"  top-{kk}={pct:5.1f}%"
    print(line)

    # Per-trace min-max normalization: each row rescaled to [0, 1] before averaging.
    line = f"  {'mean (min-max per trace, per-disc)':60s}"
    for kk in Ks:
        hits = 0
        for theta, t, _ in valid:
            lo = theta.min(axis=1, keepdims=True)
            hi = theta.max(axis=1, keepdims=True)
            rng = np.where(hi - lo < 1e-9, 1.0, hi - lo)
            mm = (theta - lo) / rng
            scores = mm.mean(axis=0)
            hits += topk_hit(scores, t, kk)
        pct = hits / n * 100
        ens_mm_pct[kk] = pct
        line += f"  top-{kk}={pct:5.1f}%"
    print(line)

    if summary is not None:
        summary[name] = {
            "n": n,
            "K": K,
            "model_ids": model_ids,
            "per_disc": per_disc_pct,
            "ens_raw": ens_raw_pct,
            "ens_znorm": ens_znorm_pct,
            "ens_mm": ens_mm_pct,
        }

    # Per-discriminator distributional sanity (mean theta, min, max)
    print(f"  --- distribution ---")
    all_theta_k = [[] for _ in range(K)]
    for theta, _, _ in valid:
        for k_idx in range(K):
            all_theta_k[k_idx].extend(theta[k_idx].tolist())
    for k_idx in range(K):
        arr = np.array(all_theta_k[k_idx])
        print(f"  {model_ids[k_idx]:60s}  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")


def evaluate_two_stage():
    """Method 3 trial argmax -> Method 2 within-trial argmax -> message-level top-K.

    For each trace present in BOTH dover_v2 and hybrid_v2:
      1. Predict the trial via argmax of dover theta (per discriminator + ensemble).
      2. Within the predicted trial span, argmax the hybrid theta to pick a message.
      3. Score against the full-history mistake_step.
    """
    dover_dir  = ROOT / "data/reports/Hand-Crafted_dover_v2"
    hybrid_dir = ROOT / "data/reports/Hand-Crafted_hybrid_v2"
    print(f"\nMethod 3 -> Method 2 two-stage pipeline (trial argmax then within-trial argmax)")
    print(f"  dover dir:  {dover_dir}")
    print(f"  hybrid dir: {hybrid_dir}")

    Ks = [1, 3, 5]
    K_disc = None
    model_ids = None
    per_disc_hits = None
    ens_hits = None
    n_eval = 0

    for fdov in sorted(dover_dir.glob("*.npz")):
        tid = fdov.stem
        fhyb = hybrid_dir / f"{tid}.npz"
        if not fhyb.exists():
            continue

        dov = np.load(fdov, allow_pickle=True)
        hyb = np.load(fhyb, allow_pickle=True)
        try:
            target_msg = int(hyb["mistake_step"])
        except (TypeError, ValueError):
            continue

        theta_dov = dov["theta_hat"]              # (K, n_trials)
        theta_hyb = hyb["theta_hat"]              # (K, T_full)
        spans = dov["trial_spans"]                # (n_trials, 2)
        if K_disc is None:
            K_disc = theta_dov.shape[0]
            model_ids = [str(x) for x in dov["model_ids"].tolist()]
            per_disc_hits = {(k, kk): 0 for k in range(K_disc) for kk in Ks}
            ens_hits = {kk: 0 for kk in Ks}

        n_eval += 1

        # Per-discriminator: pick that disc's argmax trial, then argmax within span
        for k in range(K_disc):
            dov_order = np.argsort(-theta_dov[k])
            for kk in Ks:
                trials = dov_order[:kk]
                preds = []
                for ti in trials:
                    s, e = int(spans[ti, 0]), int(spans[ti, 1])
                    if e > s:
                        local = theta_hyb[k, s:e]
                        preds.append(s + int(np.argmax(local)))
                if target_msg in preds:
                    per_disc_hits[(k, kk)] += 1

        # Ensemble: mean over K, argmax trial, then argmax within span
        ens_dov = theta_dov.mean(axis=0)
        ens_hyb = theta_hyb.mean(axis=0)
        ens_order = np.argsort(-ens_dov)
        for kk in Ks:
            trials = ens_order[:kk]
            preds = []
            for ti in trials:
                s, e = int(spans[ti, 0]), int(spans[ti, 1])
                if e > s:
                    local = ens_hyb[s:e]
                    preds.append(s + int(np.argmax(local)))
            if target_msg in preds:
                ens_hits[kk] += 1

    if n_eval == 0:
        print("  no overlap")
        return
    print(f"  evaluated traces: {n_eval}")
    print(f"  --- per-discriminator (top-K trials, message argmax inside each) ---")
    for k in range(K_disc):
        line = f"  {model_ids[k]:60s}"
        for kk in Ks:
            line += f"  top-{kk}={per_disc_hits[(k, kk)]/n_eval*100:5.1f}%"
        print(line)
    print(f"  --- ensemble ---")
    line = f"  {'mean':60s}"
    for kk in Ks:
        line += f"  top-{kk}={ens_hits[kk]/n_eval*100:5.1f}%"
    print(line)


SHORT_NAME_PATTERNS = [
    ("gemini-3-flash", "gemini-3-flash"),
    ("qwen3-next-80b", "qwen3-next-80b"),
    ("gpt-oss-120b",   "gpt-oss-120b"),
    ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("gpt-4o-mini",     "gpt-4o-mini"),
]


def _short(model_id):
    for needle, label in SHORT_NAME_PATTERNS:
        if needle in model_id:
            return label
    return model_id


def print_summary_table(summary):
    """Wide consolidated table: rows = methods, columns = each disc + 3 ensembles, x top-1/3/5."""
    rows = [(name, m) for name, m in summary.items() if m is not None]
    if not rows:
        return

    # Use the first non-empty entry's model_ids to define column order
    ref_models = rows[0][1]["model_ids"]
    Ks = [1, 3, 5]

    # Build header
    method_w = max(len(name) for name, _ in rows) + 2
    method_w = max(method_w, 30)

    cell_w = 8  # e.g. " 41.7% "
    col_groups = [_short(m) for m in ref_models] + ["ens raw", "ens z", "ens mm"]
    group_w = cell_w * len(Ks)

    print("\n" + "=" * 80)
    print("CONSOLIDATED TOP-K SUMMARY (rows = method, columns = scorer)")
    print("=" * 80)

    # Header row 1: group names
    line = f"{'method':<{method_w}}{'n':>5}  "
    for g in col_groups:
        line += f"{g:^{group_w}}|"
    print(line)
    # Header row 2: top-k labels
    line = f"{'':<{method_w}}{'':>5}  "
    for _ in col_groups:
        for kk in Ks:
            line += f"{'t'+str(kk):>{cell_w}}"
        line += "|"
    print(line)
    print("-" * len(line))

    for name, m in rows:
        line = f"{name:<{method_w}}{m['n']:>5}  "
        for mid in ref_models:
            disc = m["per_disc"].get(mid)
            for kk in Ks:
                if disc is None:
                    line += f"{'-':>{cell_w}}"
                else:
                    line += f"{disc[kk]:>{cell_w-1}.1f}%"
            line += "|"
        for grp_key in ("ens_raw", "ens_znorm", "ens_mm"):
            for kk in Ks:
                pct = m[grp_key][kk]
                line += f"{pct:>{cell_w-1}.1f}%"
            line += "|"
        print(line)


def main():
    summary = {}
    for name, rel, mode in METHODS:
        evaluate_method(name, rel, mode, summary=summary)
    evaluate_two_stage()
    print_summary_table(summary)


if __name__ == "__main__":
    main()
