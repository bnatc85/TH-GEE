"""Reproduce the TH-GEE empirical results in the paper (Section: Results).

Runs the synthetic construct tests (RQ1-RQ4) and the RedNote-Vibe observational
analysis, and writes ``results.json`` with every number cited in the paper.

Data
----
The RedNote-Vibe corpus is not redistributed here (CC BY-NC 4.0). Download the
three ``.jsonl`` files from the dataset's Google Drive folder (linked from
https://github.com/ydli-ai/RedNote-Vibe) into a directory and pass it via
``--data-dir`` (or set ``REDNOTE_DIR``). Only ``training_set_human.jsonl`` and
``exploring_set.jsonl`` are used; the AIGC split has no timestamps or engagement.

Usage
-----
    python analysis/rednote_analysis.py --data-dir /path/to/rednote_data
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import geometric_persistence, thgee, reemergence_index  # noqa: E402

# Engagement weights for RedNote interaction types (documented in the paper).
W_LIKE, W_COLLECT, W_COMMENT = 0.3, 0.5, 1.0
WINDOW = timedelta(days=30)
GAMMA = 0.10
DORMANCY = 2
MIN_TAG_POSTS = 50
TAG_RE = re.compile(r"#([^#\[\]\s]{1,30})(?:\[[^\]]*\])?#")


def parse_local_time(lt):
    s = str(lt)
    if len(s) != 10 or not s.isdigit():
        return None
    try:
        return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]),
                        tzinfo=timezone.utc)
    except ValueError:
        return None


def load(path, require_engagement):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = parse_local_time(r.get("local_time"))
            if t is None:
                continue
            lk, cl, cm = r.get("liked_count"), r.get("collected_count"), r.get("comments_count")
            if require_engagement and (lk is None or cl is None or cm is None):
                continue
            rows.append({
                "time": t,
                "domain": r.get("domain") or "?",
                "like": lk or 0, "collect": cl or 0, "comment": cm or 0,
                "text": (r.get("note_title") or "") + " " + (r.get("desc") or ""),
            })
    return rows


def weighted_engagement(r):
    return W_LIKE * r["like"] + W_COLLECT * r["collect"] + W_COMMENT * r["comment"]


def build_trace(posts, horizon_start, horizon_end, window):
    """Fixed-width tiling (matches streamlit_app): E_tau = sum(weighted eng)/n_posts."""
    span = horizon_end - horizon_start
    T = max(1, int(math.ceil(span / window)))
    num = np.zeros(T)
    den = np.zeros(T)
    for p in posts:
        idx = int((p["time"] - horizon_start) / window)
        if 0 <= idx < T:
            num[idx] += weighted_engagement(p)
            den[idx] += 1
    trace = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return trace, den


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "std": float(a.std()), "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max()),
    }


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def synthetic_tests():
    out = {}
    # RQ1: matched-volume distributed vs concentrated
    persistent = np.array([10, 11, 9, 10, 12, 11], dtype=float)
    acute = np.zeros(6); acute[0] = persistent.sum()
    out["rq1"] = {
        "total_volume": float(persistent.sum()),
        "persistent_Ph": geometric_persistence(persistent),
        "acute_Ph": geometric_persistence(acute),
        "persistent_mean": float(persistent.mean()), "acute_mean": float(acute.mean()),
    }
    # RQ2: permutation invariance of Ph vs order sensitivity of TH-GEE
    rng = np.random.default_rng(20260716)
    base = np.array([20, 10, 5, 0, 3, 8], dtype=float)
    perms, Ph_vals, thgee_vals = set(), [], []
    for _ in range(5000):
        p = tuple(rng.permutation(base))
        if p in perms:
            continue
        perms.add(p)
        arr = np.array(p)
        Ph_vals.append(geometric_persistence(arr))
        thgee_vals.append(thgee(arr, 0.5))
    out["rq2"] = {
        "n_unique_perms": len(perms),
        "Ph_std_across_perms": float(np.std(Ph_vals)),
        "thgee_std_across_perms": float(np.std(thgee_vals)),
        "thgee_range": [float(min(thgee_vals)), float(max(thgee_vals))],
        "reduction_identity_max_abs_diff": float(max(
            abs(geometric_persistence(np.array(p)) - thgee(np.array(p), 0.0))
            for p in list(perms)[:500])),
    }
    gamma_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    early = np.array([20, 12, 8, 5, 2, 1], dtype=float)
    late = np.array([1, 2, 5, 8, 12, 20], dtype=float)
    out["rq2_gamma_direction"] = {
        "early_thgee_by_gamma": {g: thgee(early, g) for g in gamma_grid},
        "late_thgee_by_gamma": {g: thgee(late, g) for g in gamma_grid},
    }
    # RQ3: continuous vs formerly-active vs re-emergent
    out["rq3"] = {
        k: reemergence_index(np.array(v, float), 5.0, 3)[0] for k, v in {
            "continuous_R": [10, 11, 9, 10, 12, 11],
            "reemergent_R": [12, 10, 0, 0, 0, 14],
            "formerly_active_R": [12, 10, 8, 0, 0, 0],
            "late_first_appearance_R": [0, 0, 0, 0, 0, 14],
        }.items()
    }
    # RQ4: d and q sensitivity
    reem = np.array([12, 10, 0, 14, 0, 0, 0, 15], dtype=float)
    out["rq4_d"] = {"trace": reem.tolist(),
                    "R_by_d": {dd: reemergence_index(reem, 5.0, dd)[0] for dd in [1, 2, 3, 4]}}
    out["rq4_q"] = {"trace": reem.tolist(),
                    "R_by_q": {qq: reemergence_index(reem, float(qq), 2)[0]
                               for qq in [1, 3, 5, 9, 11, 13, 16]}}
    return out


def cohort_metrics(posts, hstart, hend):
    tr, counts = build_trace(posts, hstart, hend, WINDOW)
    active = tr[tr > 0]
    qh = float(np.percentile(active, 25)) if active.size else 1.0
    Ph = geometric_persistence(tr)
    G = thgee(tr, GAMMA)
    R, _ = reemergence_index(tr, max(qh, 1e-6), DORMANCY)
    return Ph, G, R, float(counts.sum()), int((tr > 0).sum()), len(tr), tr


def run_period(posts, label):
    hstart = min(p["time"] for p in posts)
    hend = max(p["time"] for p in posts)

    by_dom = defaultdict(list)
    for p in posts:
        by_dom[p["domain"]].append(p)
    dom_rows, Ph_all, G_all, R_all, vol_all = {}, [], [], [], []
    for dom, ps in by_dom.items():
        if dom == "?" or len(ps) < 30:
            continue
        Ph, G, R, vol, nz, T, _ = cohort_metrics(ps, hstart, hend)
        dom_rows[dom] = {"n_posts": len(ps), "Ph": Ph, "thgee": G, "R": R,
                         "nonzero_windows": nz, "T": T}
        Ph_all.append(Ph); G_all.append(G); R_all.append(R); vol_all.append(math.log1p(vol))

    by_tag = defaultdict(list)
    for p in posts:
        for tag in set(TAG_RE.findall(p["text"])):
            by_tag[tag].append(p)
    tag_Ph, tag_G, tag_R, tag_vol, tag_traces = [], [], [], [], []
    for tag, ps in by_tag.items():
        if len(ps) < MIN_TAG_POSTS:
            continue
        Ph, G, R, vol, nz, T, tr = cohort_metrics(ps, hstart, hend)
        tag_Ph.append(Ph); tag_G.append(G); tag_R.append(R)
        tag_vol.append(math.log1p(vol)); tag_traces.append((len(ps), tr))

    perm_block = None
    if tag_traces:
        _, big = max(tag_traces, key=lambda t: t[0])
        rng = np.random.default_rng(7)
        php = [geometric_persistence(pr := rng.permutation(big)) for _ in range(2000)]
        rng = np.random.default_rng(7)
        ghp = [thgee(rng.permutation(big), GAMMA) for _ in range(2000)]
        perm_block = {
            "trace_len": int(big.size), "nonzero": int((big > 0).sum()),
            "Ph_std": float(np.std(php)),
            "thgee_range": [float(min(ghp)), float(max(ghp))],
        }

    hashtag_block = None
    if len(tag_Ph) >= 5:
        tag_R_arr = np.asarray(tag_R)
        hashtag_block = {
            "n": len(tag_Ph), "Ph": summarize(tag_Ph), "thgee": summarize(tag_G),
            "R": summarize(tag_R), "spearman_Ph_thgee": spearman(tag_Ph, tag_G),
            "pearson_Ph_logvol": float(np.corrcoef(tag_Ph, tag_vol)[0, 1]),
            "frac_R_gt_0": float(np.mean(tag_R_arr > 1e-9)),
            "frac_R_gt_0p05": float(np.mean(tag_R_arr > 0.05)),
            "permutation_test": perm_block,
        }
    return {
        "label": label, "horizon": [hstart.isoformat(), hend.isoformat()],
        "n_domain_cohorts": len(dom_rows), "domain_rows": dom_rows,
        "Ph": summarize(Ph_all), "thgee": summarize(G_all), "R": summarize(R_all),
        "spearman_Ph_thgee": spearman(Ph_all, G_all) if len(Ph_all) >= 3 else None,
        "pearson_Ph_logvol": float(np.corrcoef(Ph_all, vol_all)[0, 1]) if len(Ph_all) >= 3 else None,
        "n_hashtag_cohorts_kept": len(tag_Ph), "min_tag_posts": MIN_TAG_POSTS,
        "hashtag": hashtag_block,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.getenv("REDNOTE_DIR", "rednote_data"),
                    help="Directory with training_set_human.jsonl and exploring_set.jsonl")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    args = ap.parse_args()

    results = {"weights": {"like": W_LIKE, "collect": W_COLLECT, "comment": W_COMMENT},
               "params": {"window_days": 30, "gamma": GAMMA, "dormancy": DORMANCY}}
    results.update(synthetic_tests())

    data = Path(args.data_dir)
    human_path = data / "training_set_human.jsonl"
    expl_path = data / "exploring_set.jsonl"
    if human_path.exists() and expl_path.exists():
        human = load(human_path, require_engagement=False)
        expl = load(expl_path, require_engagement=True)
        results["dataset"] = {
            "human_pre_llm": len(human), "exploration_post_llm": len(expl),
            "total": len(human) + len(expl),
        }
        results["pre_llm"] = run_period(human, "pre-LLM (human)")
        results["post_llm"] = run_period(expl, "post-LLM (exploration)")
        print(f"dataset: {results['dataset']}")
    else:
        print(f"[warn] corpus not found under {data}; wrote synthetic results only.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
