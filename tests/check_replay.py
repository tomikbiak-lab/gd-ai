"""
Checks in the real game that the mod presses on exactly the ticks the brain asked for:

    python check_replay.py BRAIN LEVEL_ID

Needs a brain run with --trace-runs (decisions of every full run in the level's runs.jsonl) and
the mod's saved replay of that level (the exact ticks it pressed and released in its best run).
Finds the full run the replay was made from and compares tick by tick.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "brain"))
from brain2 import DECISION_TICKS, timeline  # noqa: E402

REPLAYS = Path(os.environ["LOCALAPPDATA"]) / "GeometryDash" / "geode" / "mods" / "takto.gd-ai" / "replays"


def switches(ticks):
    """(tick, held) wherever the button changes, like the mod records them."""
    out, held = [], False
    for t, h in enumerate(ticks):
        if h != held:
            out.append((t, h))
            held = h
    return out


def main():
    brain, level_id = sys.argv[1], int(sys.argv[2])
    replay = json.loads((REPLAYS / f"{level_id}.json").read_text(encoding="utf-8"))
    pressed = [(int(t), bool(h)) for t, h in replay["presses"]]
    level = next((HERE.parent / "brain" / "brains2" / brain / "levels").glob(f"{level_id}-*"))
    runs = [json.loads(l) for l in (level / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    played = replay["ticks"]
    matches = []
    for i, run in enumerate(runs):
        actions = [int(c) for c in run["actions"]]
        if not (len(actions) - 1) * DECISION_TICKS < played <= len(actions) * DECISION_TICKS:
            continue
        planned = switches(timeline(actions)[:played])
        if planned == pressed:
            matches.append(i)
    off_grid = sum(1 for t, _ in pressed if t % DECISION_TICKS)
    print(f"replay of {replay['levelName']}: {replay['percent']:.1f}%, {played} ticks, {len(pressed)} presses/releases, "
          f"{off_grid} of them on ticks 1-3 of a decision window")
    print(f"{len(runs)} full runs traced; runs whose decisions give exactly these ticks: {matches}")
    if not matches:
        close = min(runs, key=lambda r: abs(len(r["actions"]) * DECISION_TICKS - played))
        planned = switches(timeline([int(c) for c in close["actions"]])[:played])
        diff = next((k for k, (a, b) in enumerate(zip(planned, pressed)) if a != b), min(len(planned), len(pressed)))
        print(f"MISMATCH. Closest run ({close['percent']}%): first difference at press #{diff}: "
              f"brain {planned[diff:diff + 3]} vs mod {pressed[diff:diff + 3]}")
        sys.exit(1)
    print("OK  the mod pressed and released on exactly the ticks the brain decided")


if __name__ == "__main__":
    main()
