"""
Side-by-side summary of several --eval results:

    python eval_table.py NAME=eval-a.json NAME=eval-b.json ...   (several files per name are pooled: NAME=a.json+b.json)

Per eval: runs, median, mean, share past each threshold (with 95% interval), and where the runs died
(share of runs per 1% stretch, stretches where at least 3% of some eval's runs died).
"""
import collections
import json
import statistics
import sys

from compare_evals import wilson

THRESHOLDS = [10, 20, 25, 26.5, 27, 30, 32]


def main():
    evals = {}
    for arg in sys.argv[1:]:
        name, files = arg.split("=", 1)
        runs = []
        for f in files.split("+"):
            runs += json.load(open(f, encoding="utf-8"))["runs"]
        evals[name] = runs
    width = max(len(n) for n in evals) + 2
    print("".ljust(width) + "runs  median  mean  " + "  ".join(f"past {t:g}%".rjust(15) for t in THRESHOLDS))
    for name, runs in evals.items():
        cells = []
        for t in THRESHOLDS:
            k = sum(p >= t for p in runs)
            lo, hi = wilson(k, len(runs))
            cells.append(f"{k / len(runs):4.0%} [{lo:3.0%}-{hi:3.0%}]".rjust(15))
        print(name.ljust(width) + f"{len(runs):4d}  {statistics.median(runs):6.1f}  {statistics.mean(runs):4.1f}  " + "  ".join(cells))
    bins = {name: collections.Counter(int(p) for p in runs if p < 100) for name, runs in evals.items()}
    shown = sorted({b for name, c in bins.items() for b, k in c.items() if k / len(evals[name]) >= 0.03})
    print("\nwhere runs died (share of runs):")
    print("".ljust(width) + "".join(f"{b:>5d}%" for b in shown))
    for name, c in bins.items():
        print(name.ljust(width) + "".join(f"{c[b] / len(evals[name]):6.0%}" for b in shown))


if __name__ == "__main__":
    main()
