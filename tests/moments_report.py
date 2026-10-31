"""
Summarizes moments.jsonl (the last second before deaths near hard spots):

    python moments_report.py BRAIN LEVEL_ID [FROM% TO%]

Groups deaths by where they happened and prints, for each group, how the players moved
(mode, gravity, on ground, height, vertical speed), what was near, the clicks in the last
half second, and whether the attempts that died there had similar inputs.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
KINDS = {1: "spike", 2: "block", 3: "jump pad", 4: "gravity pad", 5: "jump orb", 6: "gravity orb",
         7: "gravity portal", 8: "mode portal", 9: "other portal", 10: "speed portal"}
TYPES = {0: "block", 2: "spike", 25: "slope", 47: "animated spike", -1: "nothing"}


def player(p):
    if p is None:
        return None
    mode = MODES[int(np.argmax(p[3:11]))] if max(p[3:11]) > 0 else "?"
    return {"mode": mode, "vy": p[0] * 20, "ground": p[1] > 0.5, "upside_down": p[2] > 0.5, "mini": p[11] > 0.5,
            "height": p[13] * 300 + 105, "to_ceiling_mid": p[14] * 300}


def main():
    brain, level_id = sys.argv[1], int(sys.argv[2])
    lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.0, 100.0)
    level = next((HERE.parent / "brain" / "brains2" / brain / "levels").glob(f"{level_id}-*"))
    records = [json.loads(l) for l in (level / "moments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [r for r in records if lo <= r["death"]["percent"] < hi]
    groups = collections.defaultdict(list)
    for r in records:
        groups[round(r["death"]["percent"], 1)].append(r)
    for pct, rs in sorted(groups.items(), key=lambda g: -len(g[1]))[:6]:
        d = rs[0]["death"]
        print(f"\n=== {pct}%: {len(rs)} deaths, {MODES[d['mode']]}, killed by {TYPES.get(d['object_type'], d['object_type'])} "
              f"(id {d['object_id']}), y {np.mean([r['death']['y'] for r in rs]):.0f} | kinds "
              f"{dict(collections.Counter(r['kind'] for r in rs))}, eval {sum(r.get('eval', False) for r in rs)}")
        last = [r["steps"][-1] for r in rs if r["steps"]]
        p1 = [player(s["p1"]) for s in last]
        print(f"  last decision, player 1: modes {dict(collections.Counter(p['mode'] for p in p1))}, "
              f"upside down {sum(p['upside_down'] for p in p1)}/{len(p1)}, on ground {sum(p['ground'] for p in p1)}, "
              f"height {np.mean([p['height'] for p in p1]):.0f}±{np.std([p['height'] for p in p1]):.0f}, "
              f"vy {np.mean([p['vy'] for p in p1]):+.1f}±{np.std([p['vy'] for p in p1]):.1f}")
        dual = [s for s in last if s["p2"] is not None]
        if dual:
            p2 = [player(s["p2"]) for s in dual]
            print(f"  DUAL in {len(dual)}/{len(last)}: player 2 modes {dict(collections.Counter(p['mode'] for p in p2))}, "
                  f"height {np.mean([p['height'] for p in p2]):.0f}, upside down {sum(p['upside_down'] for p in p2)}")
        things = collections.Counter(k for s in last for k in s["things"])
        print(f"  in view: {dict(things)}  values: "
              + ", ".join(f"{k} {sorted({v for s in last for v in s['things'].get(k, [])})}" for k in things))
        near = collections.Counter(KINDS.get(round(n[4] * 10), "?") for s in last for n in s["nearest"][:2])
        print(f"  two nearest things ahead: {dict(near)}")
        # Clicks over the last 0.5 s (30 decisions), as hold(H)/free(.) per decision, for a few deaths.
        for r in rs[:5]:
            st = r["steps"][-30:]
            row = "".join(("H" if s["action"] >= 4 else ".") if s["action"] % 4 == 0 else ("h" if s["action"] >= 4 else ",")
                          for s in st)
            print(f"  {r['kind']:8s} start {r.get('start_x', 0):7.1f}  last 0.5s: {row}  (h/, = change 1-3 ticks late)")
        # Do the deaths come from similar inputs? Compare player-1 numbers 0.25 s before death.
        if len(rs) >= 3:
            at = [np.array(r["steps"][-15]["p1"]) for r in rs if len(r["steps"]) >= 15]
            spread = np.std(np.stack(at), axis=0) if at else None
            if spread is not None:
                print(f"  0.25s before death, spread across deaths: height ±{spread[13] * 300:.1f}px, vy ±{spread[0] * 20:.2f}")


if __name__ == "__main__":
    main()
