"""
aggregate_seeds.py
==================
Aggregate several seed runs into a mean +/- std table per stage.

Two input kinds are supported (chosen by file extension):

  *.log   Output logs of the v3 script.  The ISO-ACCURACY and FINAL tables
          printed at the end of each run are parsed; the deltas shown in the
          tables (accuracy/F1/precision/recall in pp, everything else in
          relative %) are aggregated directly.
  *.json  results_*_seed<n>.json written by the v4 script (full-precision
          stage records).

Usage:  python aggregate_seeds.py [glob_pattern] [output_csv]
        default pattern: output80_LeNet_500_300_CIFAR10_v3_seed*.log
        default output : aggregate_seeds_summary.csv  (give a name to avoid overwriting)

The seed is taken from "seed<n>" in the file name.  Logs without a finished
table (e.g. a run still in progress) are skipped with a warning.
std is the sample std (ddof=1) over seeds.
"""

import csv
import glob
import json
import re
import statistics as st
import sys
from collections import OrderedDict

pattern = sys.argv[1] if len(sys.argv) > 1 else "output80_LeNet_500_300_CIFAR10_v3_seed*.log"
out_csv = sys.argv[2] if len(sys.argv) > 2 else "aggregate_seeds_summary.csv"
files = sorted(glob.glob(pattern))
if not files:
    sys.exit(f"No files match {pattern!r}")

# (key, label) -- values are deltas vs baseline: pp for the first four, % for the rest
METRICS = [
    ("accuracy",  "Acc (pp)"),
    ("f1",        "F1 (pp)"),
    ("cost",      "WCTG %"),
    ("sp_cost",   "CrossGrp %"),
    ("edges",     "Edges %"),
    ("flops",     "FLOPs %"),
    ("size",      "Size %"),
]
# table column index -> metric key (column 0 is the stage name)
COLS = {1: "accuracy", 2: "cost", 3: "sp_cost", 4: "edges", 5: "neurons",
        6: "batch", 7: "flops", 8: "size", 9: "f1", 10: "precision", 11: "recall"}

_DELTA = re.compile(r"\(\s*([+-]?[\d.]+)\s*(?:pp|%)\s*\)")
_NUM   = re.compile(r"[\d,]+(?:\.\d+)?")


def _seed_of(path: str):
    m = re.search(r"seed(\d+)", path, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_log(path: str):
    """Return (stage -> {metric: value}, tolerance_text) from a v3 log."""
    stages = OrderedDict()
    tol = None
    in_tables = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "ISO-ACCURACY TABLE" in line or "FINAL TABLE" in line:
                in_tables = True
            if not in_tables:
                continue
            if tol is None and line.strip().startswith("Tolerance:"):
                tol = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 12 or cells[0] in ("Stage", ""):
                continue
            name = cells[0]
            if "(baseline)" in cells[1] or name in stages:
                continue
            rec = {}
            for idx, key in COLS.items():
                m = _DELTA.search(cells[idx])
                if key == "neurons":
                    n = _NUM.search(cells[idx])
                    rec[key] = float(n.group().replace(",", "")) if n else float("nan")
                elif m:
                    rec[key] = float(m.group(1))
            if "accuracy" in rec:
                stages[name] = rec
    return stages, tol


def parse_json(path: str):
    with open(path) as f:
        r = json.load(f)
    base = r["stages"][0]
    out = OrderedDict()
    for s in r["stages"][1:]:
        rec = {"neurons": float(s["neurons"])}
        for key, src, mode in [("accuracy", "accuracy", "pp"), ("f1", "f1", "pp"),
                               ("cost", "cost", "rel"), ("sp_cost", "sp_cost", "rel"),
                               ("edges", "edges", "rel"), ("flops", "flops", "rel"),
                               ("size", "state_size_kb", "rel")]:
            b, c = base[src], s[src]
            rec[key] = (c - b) * 100.0 if mode == "pp" else ((c - b) / b * 100.0 if b else float("nan"))
        out[s["name"]] = rec
    return out, f"Tolerance: {r.get('tolerance')}"


runs = []   # (seed, stages)
for fp in files:
    if fp.endswith(".json"):
        stages, tol = parse_json(fp)
    else:
        stages, tol = parse_log(fp)
    if not stages:
        print(f"WARNING: no finished tables in {fp} -- skipped (run still in progress?)")
        continue
    runs.append((_seed_of(fp), stages))
    print(f"  {fp}: seed={_seed_of(fp)}  {len(stages)} stages  {tol or ''}")

if not runs:
    sys.exit("No usable runs.")
print(f"\nSeeds used: {[s for s, _ in runs]}  (n={len(runs)})")


def fmt(vals):
    if not vals:
        return "n/a"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:+.2f} ± {s:.2f}"


by_stage = OrderedDict()
for _, stages in runs:
    for name, rec in stages.items():
        d = by_stage.setdefault(name, {k: [] for k, _ in METRICS} | {"neurons": []})
        for key, _ in METRICS:
            if key in rec:
                d[key].append(rec[key])
        d["neurons"].append(rec["neurons"])

header = ["Stage", "n"] + [lbl for _, lbl in METRICS] + ["Neurons"]
rows = []
for name, d in by_stage.items():
    row = [name, str(len(d["neurons"]))]
    row += [fmt(d[k]) for k, _ in METRICS]
    row.append(f"{st.mean(d['neurons']):.0f}")
    rows.append(row)

widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
print()
print(line)
print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
print(line)
for row in rows:
    print("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
print(line)
print("Values are mean ± sample std over seeds (n = seeds that recorded the stage).")
print("Log-based values are the 2-decimal deltas printed in the tables.")

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"Saved {out_csv}")
