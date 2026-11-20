"""
iso_compare.py -- compare IterCC (branch A) and Taylor (branch D) at EQUAL accuracy drop.

The snapshot rows of the logs ("A: ... @ drop<=X%") are the model state just BEFORE the step that
crossed the threshold, so two branches' rows with the same label sit at different real accuracy
drops.  This script reads every state (baseline, post-L1, all snapshots, final) from the logs,
and interpolates each metric linearly against the real accuracy drop on a common grid.

Usage:
    python iso_compare.py "output_LeNet5_CIFAR10_v8_seq15fine_seed*.log"
    python iso_compare.py "output_LeNet5_CIFAR10_v8_seq15fine_seed*.log" 0.5,0.75,1.0,1.25,1.5,1.75

Output: per grid point and metric, the mean +- std over seeds of the REDUCTION (%, positive = better)
for A (IterCC) and D (Taylor), the mean difference A - D, and the per-seed differences.
Positive difference = IterCC better.  Linear interpolation between two distant states is an
approximation (the trajectories are step functions), so read it together with the raw rows.
B and G are ignored (identical to A and D with L2_LONELY=1).
"""
import glob
import re
import statistics
import sys
from collections import defaultdict

METRICS = [("WCTG", 2), ("CrossGrp", 3), ("Edges", 4), ("Nodes", 5), ("FLOPs", 7)]
PCT = re.compile(r"\(([+-]?\d+(?:\.\d+)?)%\)")
PP = re.compile(r"\(([+-]?\d+(?:\.\d+)?) pp\)")
SEED = re.compile(r"seed(\d+)", re.I)
GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


def _reductions(cols):
    out = {}
    for name, idx in METRICS:
        m = PCT.search(cols[idx])
        if not m:
            return None
        out[name] = -float(m.group(1))          # positive = reduction
    return out


def parse(path):
    """-> {branch letter: sorted list of (drop_pp, {metric: reduction %})}"""
    anchors = [(0.0, {n: 0.0 for n, _ in METRICS})]
    per = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) < 10:
                continue
            name = cols[0]
            pp = PP.search(cols[1])
            if not pp:
                continue
            red = _reductions(cols)
            if red is None:
                continue
            drop = -float(pp.group(1))
            if name.startswith("After L1"):
                anchors.append((drop, red))
                continue
            m = re.match(r"^([A-Z]):", name)
            if m and ("@ drop" in name or name.endswith("[final]")):
                per[m.group(1)].append((drop, red))
    out = {}
    for b, lst in per.items():
        seen, uniq = set(), []
        for d, r in sorted(anchors + lst, key=lambda x: x[0]):
            key = (round(d, 4), round(r["WCTG"], 2), round(r["Edges"], 2))
            if key not in seen:
                seen.add(key)
                uniq.append((d, r))
        out[b] = uniq
    return out


def interp(states, g, metric):
    for (d0, r0), (d1, r1) in zip(states, states[1:]):
        if d0 <= g <= d1:
            if d1 == d0:
                return r1[metric]
            t = (g - d0) / (d1 - d0)
            return r0[metric] + t * (r1[metric] - r0[metric])
    return None


def _ms(xs):
    if not xs:
        return "n/a"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):6.1f} +- {sd:4.1f}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    grid = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else GRID
    files = sorted(glob.glob(sys.argv[1]))
    if not files:
        sys.exit(f"no files match {sys.argv[1]!r}")
    data = {}
    for f in files:
        m = SEED.search(f)
        data[m.group(1) if m else f] = parse(f)
    print(f"Seeds: {sorted(data)}   (reduction %, positive = better; diff = IterCC - Taylor)")
    print(f"{'drop pp':>8} {'metric':<9} {'n':>2} {'IterCC (A)':>16} {'Taylor (D)':>16} {'diff':>7}  per-seed diff")
    for g in grid:
        for metric, _ in METRICS:
            a, d, diffs = [], [], []
            for seed in sorted(data):
                br = data[seed]
                if "A" not in br or "D" not in br:
                    continue
                ra, rd = interp(br["A"], g, metric), interp(br["D"], g, metric)
                if ra is None or rd is None:
                    continue
                a.append(ra)
                d.append(rd)
                diffs.append(ra - rd)
            if not a:
                print(f"{g:8.2f} {metric:<9} {0:>2} {'n/a':>16} {'n/a':>16}")
                continue
            per_seed = "/".join(f"{x:+.1f}" for x in diffs)
            print(f"{g:8.2f} {metric:<9} {len(a):>2} {_ms(a):>16} {_ms(d):>16} "
                  f"{statistics.mean(diffs):+7.1f}  {per_seed}")
        print()


if __name__ == "__main__":
    main()
