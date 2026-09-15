#!/usr/bin/env bash
# Reproduce the whole Wikipedia event-level replication (Online Resource 1).
#
# Collection needs network but no API key; everything after it runs from the
# frozen wikipedia_raw.json. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f analysis/wikipedia_raw.json ]; then
  echo "== collecting daily pageview series (one request per article-year)"
  python analysis/wikipedia_collect.py --alts
else
  echo "== using frozen analysis/wikipedia_raw.json"
fi

echo "== ground-truth checks"
python analysis/test_sequence_baselines.py

echo "== profiles: primary specification (90-day, common horizon, quartile q_h)"
python analysis/wikipedia_analysis.py

echo "== profiles: window-length and span robustness"
for w in 7 30; do
  for s in common own; do
    python analysis/wikipedia_analysis.py --window-days "$w" --span "$s" >/dev/null
  done
done
python analysis/wikipedia_analysis.py --span own >/dev/null
python analysis/wikipedia_analysis.py --window-days 7 --qh-rule elevation >/dev/null
python analysis/wikipedia_analysis.py --qh-rule elevation >/dev/null

echo "== sequence baselines (Kleinberg + PELT)"
python analysis/sequence_baselines.py
python analysis/sequence_baselines.py --window-days 7 \
  --out analysis/sequence_baseline_results_7d.json >/dev/null
python analysis/sequence_baselines.py --window-days 7 --qh-rule elevation \
  --out analysis/sequence_baseline_results_7d_elevation.json >/dev/null

echo "== figures and tables"
python analysis/make_wikipedia_figures.py
python analysis/make_wikipedia_tables.py

echo "done."
