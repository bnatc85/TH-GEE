"""Emit the Online Resource 1 tables as LaTeX, straight from the result JSONs.

Transcribing forty numbers by hand into a manuscript is how manuscripts acquire
numbers that no longer match their code, so the tables are generated. Writes
``reproducibility/or1_tables.tex``, which Online Resource 1 \\input's.

Run after wikipedia_analysis.py and sequence_baselines.py:
    python analysis/make_wikipedia_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wikipedia_collect import TOPICS  # noqa: E402

OUT = HERE.parent / "reproducibility" / "or1_tables.tex"

SPECS = [
    ("wikipedia_results.json", "90-day", "common", "quartile"),
    ("wikipedia_results_90d_own.json", "90-day", "own", "quartile"),
    ("wikipedia_results_30d_common.json", "30-day", "common", "quartile"),
    ("wikipedia_results_30d_own.json", "30-day", "own", "quartile"),
    ("wikipedia_results_7d_common.json", "7-day", "common", "quartile"),
    ("wikipedia_results_7d_own.json", "7-day", "own", "quartile"),
    ("wikipedia_results_7d_common_elevation.json", "7-day", "common", "elevation"),
]

METRIC_LABEL = {"ga_ratio": "$P_h/\\bar{E}$", "recency_ratio": "$\\THGEE/P_h$",
                "R": "$R_h$", "n_reactivations": "react.\\ count"}
ESC = str.maketrans({"&": "\\&", "_": "\\_", "%": "\\%", "#": "\\#"})


def tex(s: str) -> str:
    return str(s).translate(ESC)


def load(name: str) -> dict | None:
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def table_mapping() -> str:
    res = load("wikipedia_results.json")
    rows = []
    for topic, spec in TOPICS.items():
        w = res["wikipedia"][topic]
        rows.append(f"{tex(topic)} & {tex(spec['article'])} & {spec['fidelity']} & "
                    f"{w['total_views']:,} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Topic-to-article mapping. \\emph{{Fidelity}} grades how closely the
article's subject matches the YouTube search term (Section~\\ref{{sec:or1_data}}).
Views are the 2015-07-01--2026-06-30 totals for the \\texttt{{user}} agent class.}}
\\label{{tab:or1_mapping}}
\\small
\\begin{{tabular}}{{llll}}
\\toprule
\\textbf{{Topic (YouTube search term)}} & \\textbf{{Wikipedia article}} &
\\textbf{{Fidelity}} & \\textbf{{Pageviews}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_agreement() -> str:
    rows = []
    for fname, win, span, qh in SPECS:
        res = load(fname)
        if res is None:
            continue
        b = res["agreement"]["youtube_matched"]["all"]
        h = res["agreement"]["youtube_matched"]["high_fidelity"]
        cells = []
        for m in ("ga_ratio", "recency_ratio", "R", "n_reactivations"):
            cells.append(f"{b[m]['rho']:+.2f} & {b[m]['p']:.3f}")
        star = "$^{\\dagger}$" if qh == "elevation" else ""
        rows.append(f"{win} & {span} & {qh}{star} & " + " & ".join(cells)
                    + f" & {h['ga_ratio']['rho']:+.2f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Rank agreement between the event-level Wikipedia profiles and the
publication-cohort YouTube profiles on the matched horizon, across
specifications. Each cell is Spearman $\\rho$ over the 37 topics with a
two-sided permutation $p$ (20{{,}}000 permutations). The final column repeats the
persistence correlation on the 31 high-fidelity mappings only. For reference,
the YouTube profile's agreement \\emph{{with itself}} under the two time axes is
$\\rho = 0.47$ for both $P_h/\\bar{{E}}$ and $\\THGEE/P_h$, and $\\rho = 0.89$
and $0.84$ for $R_h$ and the reactivation count.}}
\\label{{tab:or1_agreement}}
\\small
\\begin{{tabular}}{{lll rr rr rr rr r}}
\\toprule
& & & \\multicolumn{{2}}{{c}}{{$P_h/\\bar{{E}}$}}
& \\multicolumn{{2}}{{c}}{{$\\THGEE/P_h$}}
& \\multicolumn{{2}}{{c}}{{$R_h$}}
& \\multicolumn{{2}}{{c}}{{react.}} & high-fid. \\\\
\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-9}}\\cmidrule(lr){{10-11}}
\\textbf{{Window}} & \\textbf{{Span}} & \\textbf{{$q_h$}}
& $\\rho$ & $p$ & $\\rho$ & $p$ & $\\rho$ & $p$ & $\\rho$ & $p$
& $\\rho$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
\\footnotesize $^{{\\dagger}}$ elevation rule: $q_h = 2\\times$ the topic's median
window score (Section~\\ref{{sec:or1_qh}}).
\\end{{table*}}
"""


def table_profiles() -> str:
    res = load("wikipedia_results.json")
    order = ["persistent", "acute", "reemergent", "recent", "debunked"]
    rows = []
    for pat in order:
        sel = sorted(((t, p) for t, p in res["wikipedia"].items()
                      if p["pattern"] == pat), key=lambda kv: -kv[1]["ga_ratio"])
        for t, p in sel:
            rows.append(f"{tex(t)} & {pat} & {p['Ph']:.1f} & {p['ga_ratio']:.2f} & "
                        f"{p['recency_ratio']:.2f} & {p['R']:.3f} & "
                        f"{p['n_reactivations']} & {p['coverage']*100:.0f}\\% \\\\")
        if pat != order[-1]:
            rows.append("\\addlinespace")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Event-level temporal profiles for all 37 topics (Wikipedia pageviews,
90-day windows on the common 2015-07-01--2026-06-30 horizon, $\\gamma = 0.1$,
$d = 2$, quartile $q_h$). $P_h$ is on the mean-daily-pageview scale and is not
comparable across topics; $P_h/\\bar{{E}}$, $\\THGEE/P_h$ and $R_h$ are.
Coverage is at or near 100\\% for nearly every topic, which is the degeneracy
discussed in Section~\\ref{{sec:or1_indicators}}.}}
\\label{{tab:or1_profiles}}
\\small
\\begin{{tabular}}{{llrrrrrr}}
\\toprule
\\textbf{{Topic}} & \\textbf{{Expected}} & $P_h$ & $P_h/\\bar{{E}}$ &
\\textbf{{recency}} & $R_h$ & \\textbf{{react.}} & \\textbf{{cov.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def table_baselines() -> str:
    rows = []
    for fname, label in (("sequence_baseline_results.json",
                          "90-day windows, quartile $q_h$"),
                         ("sequence_baseline_results_7d.json",
                          "7-day windows, quartile $q_h$"),
                         ("sequence_baseline_results_7d_elevation.json",
                          "7-day windows, elevation $q_h$")):
        res = load(fname)
        if res is None:
            continue
        for key, name in (("kleinberg_burst_onset", "Kleinberg burst onset"),
                          ("pelt_changepoint", "PELT change point")):
            L = res["localisation"][key]
            rank_key = ("n_reactivations~n_bursts" if "kleinberg" in key
                        else "n_reactivations~n_changepoints")
            rk = res["rank_agreement"][rank_key]
            rows.append(
                f"{label} & {name} & "
                f"{L['reactivation_windows_marked']}/{L['reactivation_windows_total']} & "
                f"{L['precision']:.3f} & {L['base_rate_mean']:.3f} & "
                f"{L['lift_aggregate']:.1f}$\\times$ & {L['permutation_p']:.4f} & "
                f"{L['recall_of_marks']:.3f} & {rk['rho']:+.2f} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Agreement between the Re-Emergence Index and the sequence-oriented
baselines on the daily pageview series. \\emph{{Hit}} counts reactivation windows
containing a detected burst onset or change point; \\emph{{base}} is the
probability that a window chosen uniformly at random contains one;
\\emph{{lift}} is precision over base; $p$ is a permutation test that reshuffles
each topic's reactivation windows uniformly (10{{,}}000 draws). \\emph{{Recall}} is
the share of all detected marks that fall inside a reactivation window. The final
column is Spearman $\\rho$ across the 37 topics between the reactivation count and
the detector's event count.}}
\\label{{tab:or1_baselines}}
\\small
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
\\textbf{{Specification}} & \\textbf{{Baseline}} & \\textbf{{Hit}} &
\\textbf{{Prec.}} & \\textbf{{Base}} & \\textbf{{Lift}} & \\textbf{{$p$}} &
\\textbf{{Recall}} & \\textbf{{$\\rho$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def table_groups() -> str:
    rows = []
    for fname, label in (("wikipedia_results_7d_common.json", "quartile $q_h$"),
                         ("wikipedia_results_7d_common_elevation.json",
                          "elevation $q_h$")):
        res = load(fname)
        if res is None:
            continue
        for pat in ["persistent", "acute", "reemergent", "recent", "debunked"]:
            g = res["wikipedia_group_medians"][pat]
            rows.append(f"{label} & {pat} & {g['n']} & {g['ga_ratio']:.2f} & "
                        f"{g['recency_ratio']:.2f} & {g['R']:.3f} & "
                        f"{g['n_reactivations']:.0f} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Group medians on the event-level data at weekly resolution, under the
paper's quartile $q_h$ and under the elevation rule. Only the elevation rule
recovers the \\emph{{a priori}} ordering of reactivation counts
(re-emergent $>$ acute $>$ persistent).}}
\\label{{tab:or1_groups}}
\\small
\\begin{{tabular}}{{llrrrrr}}
\\toprule
\\textbf{{$q_h$ rule}} & \\textbf{{Expected}} & $n$ & $P_h/\\bar{{E}}$ &
\\textbf{{recency}} & $R_h$ & \\textbf{{react.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main() -> None:
    parts = [
        "% Generated by analysis/make_wikipedia_tables.py -- do not edit by hand.\n",
        table_mapping(), table_profiles(), table_agreement(),
        table_groups(), table_baselines(),
    ]
    OUT.write_text("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
