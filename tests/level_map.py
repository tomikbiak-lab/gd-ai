"""
Prints the objects of a main level around a spot, decoded from GD's own level file:

    python level_map.py LEVEL_ID FROM_X TO_X [--all]

Shows object id, position, rotation/flip and a name for the ones that matter to the AI
(spikes, blocks, pads, orbs, portals). Read-only: it only reads GD's Resources folder.
"""
import base64
import collections
import gzip
import sys
import zlib
from pathlib import Path

LEVELS = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash\Resources\levels")
NAMES = {
    8: "spike", 39: "small spike", 103: "medium spike", 392: "tiny spike", 9: "ground spike",
    35: "yellow pad", 67: "blue pad", 140: "pink pad", 1332: "red pad", 3005: "spider pad",
    36: "yellow orb", 84: "blue orb", 141: "pink orb", 1022: "green orb", 1330: "black orb", 1333: "red orb",
    1704: "green dash orb", 1751: "pink dash orb", 3004: "spider orb", 3027: "teleport orb",
    10: "blue gravity portal", 11: "yellow gravity portal", 2926: "green gravity portal",
    12: "cube portal", 13: "ship portal", 47: "ball portal", 111: "ufo portal", 660: "wave portal",
    745: "robot portal", 1331: "spider portal", 1933: "swing portal",
    45: "mirror portal", 46: "normal mirror portal", 99: "normal size portal", 101: "mini portal",
    286: "dual portal", 287: "solo portal", 747: "teleport portal",
    200: "0.5x speed", 201: "1x speed", 202: "2x speed", 203: "3x speed", 1334: "4x speed",
    1: "block", 2: "block", 3: "block", 4: "block", 5: "block", 6: "block", 7: "block",
    83: "block", 69: "block", 70: "block", 71: "block", 72: "block", 74: "block", 75: "block",
    40: "slab", 62: "block", 63: "block", 64: "block", 65: "block", 66: "block", 68: "block",
    88: "saw", 89: "saw", 98: "saw", 183: "saw", 184: "saw", 185: "saw", 186: "saw",
    187: "saw", 188: "saw", 397: "saw", 398: "saw", 399: "saw", 678: "saw", 679: "saw", 680: "saw",
    740: "saw", 741: "saw", 742: "saw", 918: "monster/animated hazard",
}


def decode(data):
    data = data.strip()
    raw = base64.urlsafe_b64decode(data + b"=" * (-len(data) % 4))
    try:
        return gzip.decompress(raw).decode("utf-8", "replace")
    except OSError:
        return zlib.decompress(raw, 15 | 32).decode("utf-8", "replace")


def objects(level_id):
    text = decode((LEVELS / f"{level_id}.txt").read_bytes())
    parts = text.split(";")
    out = []
    for part in parts[1:]:
        kv = part.split(",")
        if len(kv) < 4:
            continue
        props = dict(zip(kv[0::2], kv[1::2]))
        try:
            out.append({"id": int(props["1"]), "x": float(props.get("2", 0)), "y": float(props.get("3", 0)),
                        "flip_x": props.get("4") == "1", "flip_y": props.get("5") == "1",
                        "rot": float(props.get("6", 0)), "scale": float(props.get("32", 1) or 1)})
        except (KeyError, ValueError):
            continue
    return parts[0], out


def main():
    level_id, lo, hi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    show_all = "--all" in sys.argv
    header, objs = objects(level_id)
    print(f"level {level_id}: {len(objs)} objects, last x {max(o['x'] for o in objs):.0f}")
    near = sorted((o for o in objs if lo <= o["x"] <= hi), key=lambda o: (o["x"], o["y"]))
    counts = collections.Counter(NAMES.get(o["id"], f"other {o['id']}") for o in near)
    print("in range:", dict(counts.most_common(25)))
    for o in near:
        name = NAMES.get(o["id"])
        if name is None and not show_all:
            continue
        extra = (" flipY" if o["flip_y"] else "") + (" flipX" if o["flip_x"] else "") + \
                (f" rot {o['rot']:.0f}" if o["rot"] else "") + (f" scale {o['scale']:.2f}" if o["scale"] != 1 else "")
        print(f"  x {o['x']:8.1f}  y {o['y']:7.1f}  {name or 'id ' + str(o['id'])}{extra}")


if __name__ == "__main__":
    main()
