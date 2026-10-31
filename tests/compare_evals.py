"""
Compares --eval results (eval-*.json files):

    python compare_evals.py BEFORE.json AFTER.json [THRESHOLD% ...]

For each threshold: share of full runs that got past it in each eval, with a 95% interval, and a
two-sided Fisher exact test of whether the difference could be chance. Also mean progress with
a bootstrap 95% interval for the difference.
"""
import json
import math
import random
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    observed = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= observed * (1 + 1e-9)))


def main():
    before, after = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
    thresholds = [float(t) for t in sys.argv[3:]] or [10, 20, 25, 26.5, 27, 30, 32]
    rb, ra = before["runs"], after["runs"]
    print(f"before: {len(rb)} runs (round {before.get('training_rounds')}) | after: {len(ra)} runs (round {after.get('training_rounds')})")
    for t in thresholds:
        kb, ka = sum(p >= t for p in rb), sum(p >= t for p in ra)
        lb, ub = wilson(kb, len(rb))
        la, ua = wilson(ka, len(ra))
        pv = fisher(kb, len(rb) - kb, ka, len(ra) - ka)
        print(f"  past {t:5.1f}%: {kb / len(rb):6.1%} [{lb:.0%}-{ub:.0%}] -> {ka / len(ra):6.1%} [{la:.0%}-{ua:.0%}]  p={pv:.3g}")
    rng = random.Random(0)
    diffs = sorted(sum(rng.choice(ra) for _ in ra) / len(ra) - sum(rng.choice(rb) for _ in rb) / len(rb)
                   for _ in range(4000))
    mb, ma = sum(rb) / len(rb), sum(ra) / len(ra)
    print(f"  mean progress {mb:.1f}% -> {ma:.1f}% (difference {ma - mb:+.1f}, 95% interval "
          f"{diffs[100]:+.1f} to {diffs[3899]:+.1f}); best {max(rb):.1f}% -> {max(ra):.1f}%")


if __name__ == "__main__":
    main()
