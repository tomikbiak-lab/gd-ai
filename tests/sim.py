"""
A simplified GD level that talks to the brain exactly like the mod does (protocol 4, the same
vision layout), so the brain can be tested and benchmarked without the game. Deterministic like
GD: the same start and the same clicks always play out the same.

    python sim.py PORT SECONDS          # plays until SECONDS are up, then prints a JSON summary
    python sim.py PORT SECONDS TRACE    # also writes the button state of every tick, per attempt

SIM_LEVEL=precise picks a tiny level that needs a press on tick 1-3 of a decision window;
SIM_NO_PRACTICE=1 always starts from the beginning (like the mod with practice off).

The level: cube spikes, a yellow pad over a wall, a spike pit that needs the yellow orb, a ship
section with pillars, and more cube spikes. Practice mode places checkpoints like the mod does.
"""
import json
import os
import socket
import struct
import sys
import time

KINDS = 10
CHANNELS, ROWS, COLS = 3 * KINDS, 11, 16
NEAR_P1, NEAR_P2, FAR_P1 = 0, KINDS, 2 * KINDS
NEAREST, NEAREST_VALUES = 5, 7
EXTRA = 33 + NEAREST * NEAREST_VALUES
CELL, FAR_CELL, TICK = 30.0, 60.0, 1 / 240
DECISION_TICKS = 4
CHECKPOINT_SPACING, DOOMED_TICKS = 90.0, 240
SPEED = 311.58
GROUND = 90.0
V0, G = 619.0, 2948.0
HAZARD, SOLID, JUMP_PAD, GRAVITY_PAD, JUMP_ORB, GRAVITY_ORB, GRAVITY_PORTAL, MODE_PORTAL, OTHER_PORTAL, SPEED_PORTAL = range(10)

objs = []  # (channel, value, x, y, w, h, kind)


def spike(x, y=GROUND):
    objs.append((HAZARD, 255, x + 11, y, 8, 14, "spike"))


def block(x, y, w=30, h=30):
    objs.append((SOLID, 255, x, y, w, h, "block"))


LEVEL = os.environ.get("SIM_LEVEL", "full")
NO_PRACTICE = os.environ.get("SIM_NO_PRACTICE") == "1"  # like the mod with practice turned off
SHIP_START = 0.0
FLOOR, CEIL = GROUND, GROUND + 300  # ship floor and ceiling
if LEVEL == "precise":
    # A low ceiling (no jumping before x 540), then two spikes 74.5 px apart: a jump only clears
    # both when pressed on tick 453, 454 or 455 of the attempt, i.e. on tick 1-3 of a decision
    # window. No way of pressing only on tick 0 of windows gets through (checked by brute force,
    # holding through a landing included).
    block(150, GROUND + 36, 390, 30)
    spike(601.25)
    spike(601.25 + 74.5)
    LENGTH = 601.25 + 74.5 + 250
else:
    x = 600
    for n, gap in [(1, 330), (1, 170), (2, 250), (1, 160), (3, 300), (1, 175), (2, 210), (1, 165), (3, 280)]:
        for i in range(n):
            spike(x + i * 30)
        x += gap
    # Yellow pad launches you over a tall wall; spikes before it punish jumping early.
    spike(x - 60)
    objs.append((JUMP_PAD, 170, x, GROUND, 30, 6, "pad"))
    block(x + 90, GROUND, 30, 90)
    x += 450
    for i in range(2):
        spike(x + i * 30)
    x += 300
    # Spike pit too wide to jump: needs the yellow orb in the middle.
    PIT = 6
    for i in range(PIT):
        spike(x + i * 30)
    objs.append((JUMP_ORB, 170, x + 60, GROUND + 50, 30, 30, "orb"))
    x += PIT * 30 + 300
    SHIP_START = x
    objs.append((MODE_PORTAL, 64, x, GROUND, 30, 90, "ship"))
    x += 200
    for gap_y in [150, 250, 130, 220, 180, 260, 140]:
        # Pillars from floor and ceiling leaving a gap around gap_y.
        block(x, FLOOR, 30, gap_y - 50 - FLOOR)
        block(x, gap_y + 50, 30, CEIL - gap_y - 50)
        x += 260
    objs.append((MODE_PORTAL, 32, x, GROUND, 30, 90, "cube"))
    x += 300
    for n, gap in [(1, 200), (2, 260), (1, 170), (3, 300)]:
        for i in range(n):
            spike(x + i * 30)
        x += gap
    LENGTH = x + 300
objs.sort(key=lambda o: o[2])


def start_state():
    return dict(px=0.0, py=GROUND + 15, vy=0.0, ship=False, ground=True, used=set())


def copy(s):
    return dict(s, used=set(s["used"]))


def mark_grid(grid, s, offset, cell):
    px, py = s["px"], s["py"]
    left, bottom = px - cell, py - cell * ROWS / 2
    right, top = left + cell * COLS, bottom + cell * ROWS
    for ch, v, ox, oy, w, h, kind in objs:
        if ox + w < left:
            continue
        if ox > right:
            break
        if oy + h < bottom or oy > top:
            continue
        for r in range(max(0, int((oy - bottom) // cell)), min(ROWS - 1, int((oy + h - bottom) // cell)) + 1):
            for c in range(max(0, int((ox - left) // cell)), min(COLS - 1, int((ox + w - left) // cell)) + 1):
                i = ((offset + ch) * ROWS + r) * COLS + c
                grid[i] = max(grid[i], v)


def observe(s, held, since_switch):
    grid = bytearray(CHANNELS * ROWS * COLS)
    mark_grid(grid, s, NEAR_P1, CELL)
    mark_grid(grid, s, FAR_P1, FAR_CELL)
    px, py = s["px"], s["py"]
    extra = [0.0] * EXTRA
    extra[0] = s["vy"] / 600
    extra[1] = 1.0 if s["ground"] else 0.0
    extra[3 + (1 if s["ship"] else 0)] = 1.0
    extra[12] = 0.9 / 1.6
    extra[13] = (py - 105) / 300
    if s["ship"]:
        extra[14] = ((FLOOR + CEIL) / 2 - py) / 300
    extra[15] = 1.0 if held else 0.0
    extra[32] = min(since_switch, 120) / 120
    found = []
    for ch, v, ox, oy, w, h, kind in objs:
        if ox < px - 120:
            continue
        if ox >= px + 600:
            break
        if ox + w < px - 15:
            continue
        dx, dy = max(0.0, ox - px), oy + h / 2 - py
        if abs(dy) > 300:
            continue
        found.append((dx + 0.7 * abs(dy), ch, ox, oy, w, h))
    for i, (_, ch, ox, oy, w, h) in enumerate(sorted(found)[:NEAREST]):
        extra[33 + i * NEAREST_VALUES:33 + (i + 1) * NEAREST_VALUES] = [
            (ox - px) / 300, (oy + h / 2 - py) / 150, w / 60, h / 60, (ch + 1) / 10, 0.0, 0.0]
    return extra, bytes(grid)


def step(s, hold):
    """Advances one tick. Returns ('dead', kind), ('win', None) or None."""
    s["px"] += SPEED * TICK
    if s["ship"]:
        s["vy"] = max(-420.0, min(420.0, s["vy"] + (1600.0 if hold else -1600.0) * TICK))
    else:
        if hold and s["ground"]:
            s["vy"], s["ground"] = V0, False
        s["vy"] -= G * TICK
    s["py"] += s["vy"] * TICK
    floor, ceil = FLOOR + 15, (CEIL - 15 if s["ship"] else 1e9)
    px, py = s["px"], s["py"]
    for ch, v, ox, oy, w, h, kind in objs:
        if ox > px + 15:
            break
        if ox + w < px - 15:
            continue
        if kind == "block" and not s["ship"] and py - 15 >= oy + h - 10:
            floor = max(floor, oy + h + 15)
    if py <= floor:
        s["py"], s["vy"], s["ground"] = floor, 0.0, True
    elif py >= ceil:
        s["py"], s["vy"], s["ground"] = ceil, 0.0, True
    else:
        s["ground"] = False
    py = s["py"]
    for i, (ch, v, ox, oy, w, h, kind) in enumerate(objs):
        if ox > px + 15:
            break
        if not (ox < px + 15 and px - 15 < ox + w and oy < py + 15 and py - 15 < oy + h):
            continue
        if kind == "spike":
            return "dead", 2
        if kind == "block" and (s["ship"] or py - 15 < oy + h - 10):
            return "dead", 0
        if kind == "pad" and i not in s["used"]:
            s["vy"], s["ground"] = V0 * 1.45, False
            s["used"].add(i)
        if kind == "orb" and hold and i not in s["used"] and not s["ground"]:
            s["vy"] = V0
            s["used"].add(i)
        if kind == "ship":
            s["ship"] = True
        if kind == "cube":
            s["ship"] = False
    return ("win", None) if px >= LENGTH else None


def main():
    conn = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = conn.makefile("rb")

    def read(n):
        data = f.read(n)
        if len(data) < n:
            raise ConnectionError("brain closed the connection")
        return data

    name = b"Sim Level"
    conn.sendall(b"H" + struct.pack("<3i", 4, 9999, len(name)) + name
                 + struct.pack("<4if", CHANNELS, ROWS, COLS, EXTRA, LENGTH))
    started = time.time()
    # runs: full runs from the start as (seconds, percent, decisions asked so far)
    stats = {"attempts": 0, "decisions": 0, "wins": {"normal": 0, "practice": 0}, "runs": []}
    trace = open(sys.argv[3], "w", encoding="utf-8") if len(sys.argv) > 3 else None
    try:
        play(conn, read, started + float(sys.argv[2]), started, stats, trace)
    except (ConnectionError, OSError):
        print("The brain stopped.", flush=True)
    # makefile() retains the socket until the file is closed too. Disconnect
    # before writing a potentially large summary to the parent process.
    f.close()
    conn.close()
    if trace:
        trace.close()
    print("SIM " + json.dumps(stats | {"ship_start": round(100 * SHIP_START / LENGTH, 1)}), flush=True)


def play(conn, read, deadline, started, stats, trace=None):
    practice, checkpoints = False, []  # checkpoints: (x, state, attempt, tick)
    doomed = None
    runs, wins = stats["runs"], stats["wins"]
    while time.time() < deadline:
        s = start_state()
        conn.sendall(b"A" + struct.pack("<fB", s["px"], practice))
        kind, flags, target, _level = struct.unpack("<BBfi", read(10))
        if doomed is not None:
            # Like the mod: checkpoints from the last second before a death are thrown away.
            dead_attempt, death_tick = doomed
            while checkpoints and checkpoints[-1][2] == dead_attempt and checkpoints[-1][3] >= death_tick - DOOMED_TICKS:
                checkpoints.pop()
            doomed = None
        want = kind == 1 and not NO_PRACTICE
        if want != practice:
            practice, checkpoints = want, []
        if practice:
            if target <= 0:
                checkpoints = []
            while checkpoints and checkpoints[-1][0] > target:
                checkpoints.pop()
            if checkpoints:
                s = copy(checkpoints[-1][1])
        stats["attempts"] += 1
        attempt = stats["attempts"]
        conn.sendall(b"B" + struct.pack("<fBB", s["px"], practice, 1 if practice and checkpoints else 0))
        tick, held, since_switch = 0, False, 120
        plan_hold = plan_tick = start_held = None
        ticks = []  # button state on every tick actually played
        while True:
            if tick % DECISION_TICKS == 0:
                extra, grid = observe(s, held, since_switch)
                conn.sendall(b"S" + struct.pack(f"<2f{EXTRA}f", s["px"], 100 * s["px"] / LENGTH, *extra) + grid)
                a = read(1)[0]
                stats["decisions"] += 1
                plan_hold, plan_tick, start_held = bool(a & 4), a & 3, held
            hold = plan_hold if tick % DECISION_TICKS >= plan_tick else start_held
            if hold != held:
                held, since_switch = hold, 0
            ticks.append(held)
            result = step(s, held)
            tick += 1
            since_switch += 1
            if result and result[0] == "dead":
                conn.sendall(b"D" + struct.pack("<3f4iB", 100 * s["px"] / LENGTH, s["px"], s["py"], 8, result[1],
                                                1 if s["ship"] else 0, tick, practice))
                doomed = (attempt, tick)
                if not practice:
                    runs.append((round(time.time() - started, 2), round(100 * s["px"] / LENGTH, 2), stats["decisions"]))
                break
            if result and result[0] == "win":
                conn.sendall(b"W" + struct.pack("<iB", tick, practice))
                wins["practice" if practice else "normal"] += 1
                if not practice:
                    runs.append((round(time.time() - started, 2), 100.0, stats["decisions"]))
                break
            if practice and (s["ship"] or s["ground"]):
                last = checkpoints[-1][0] if checkpoints else 0.0
                if s["px"] >= last + CHECKPOINT_SPACING:
                    checkpoints.append((s["px"], copy(s), attempt, tick))
        if trace:
            trace.write(json.dumps({"attempt": attempt, "result": result[0],
                                    "held": "".join("1" if h else "0" for h in ticks)}) + "\n")


if __name__ == "__main__":
    main()
