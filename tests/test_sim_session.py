"""
The brain playing and learning the simulated level for real (no game needed):

    python test_sim_session.py [SECONDS] [--pause-to-train]      # default 60 s, rounds train in the background

Checks that every decision it recorded is exactly what the level then played tick by tick (up to
the death, which always falls in the last recorded decision's window), that what PPO trains on
is that same recorded decision, and that every training round only had decisions of one network
version. Prints how long playing and training took per round and the decisions per second.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "brain"))
import brain2  # noqa: E402
from brain2 import DECISION_TICKS, timeline  # noqa: E402


class Recording(brain2.Session):
    """A session that keeps every attempt's recorded decisions and everything it says."""

    def __init__(self, folder, pause_to_train=False):
        super().__init__(folder, play_only=False, review=False, pause_to_train=pause_to_train)
        self.log, self.attempts, self.rounds = [], [], []
        self.at_start = None

    def say(self, text):
        self.log.append(text)

    def on_begin(self, start_x, practice, from_checkpoint):
        super().on_begin(start_x, practice, from_checkpoint)
        a = self.agent
        self.at_start = (a.rollout.n, a.unused, a.updates, a.training())

    def attempt_over(self):
        actions = [a for a, _ in self.trace]
        r = self.agent.rollout
        # What PPO will train on is the same decision (when the whole attempt went into one round).
        a = self.agent
        whole = self.at_start == (r.n - len(actions), a.unused, a.updates, False) and not a.training()
        if whole:
            trained_on = [int(a) for a in r.actions[r.n - len(actions):r.n]]
            assert trained_on == actions, "the rollout holds other actions than were sent to the level"
        self.attempts.append(actions)

    def on_death(self, info, practice):
        self.attempt_over()
        super().on_death(info, practice)

    def on_win(self, practice):
        self.attempt_over()
        super().on_win(practice)

    def report(self, stats):
        self.rounds.append(stats)
        super().report(stats)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    pause = "--pause-to-train" in sys.argv
    torch.set_num_threads(brain2.TRAIN_THREADS)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        session = Recording(tmp / "brain", pause_to_train=pause)
        (tmp / "brain").mkdir()
        conn_server = brain2.Connection(0)
        port = conn_server.server.getsockname()[1]
        sim = subprocess.Popen([sys.executable, str(HERE / "sim.py"), str(port), str(seconds), str(tmp / "trace.jsonl")],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        conn, _ = conn_server.server.accept()
        try:
            conn_server.handle(conn, session)
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()
        sim_out = sim.communicate()[0]
        played = [json.loads(line) for line in (tmp / "trace.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(played) == len(session.attempts), f"{len(played)} attempts played, {len(session.attempts)} recorded"
    ticks_checked = 0
    for i, (p, actions) in enumerate(zip(played, session.attempts)):
        held = [c == "1" for c in p["held"]]
        planned = timeline(actions)
        assert (len(actions) - 1) * DECISION_TICKS < len(held) <= len(actions) * DECISION_TICKS, \
            f"attempt {i + 1}: ended outside its last recorded decision's window"
        assert planned[:len(held)] == held, f"attempt {i + 1}: played ticks differ from the recorded decisions"
        ticks_checked += len(held)
    print(f"OK  {len(played)} attempts, {ticks_checked:,} ticks: what the level played = the recorded decisions")

    rounds = session.rounds
    assert len(rounds) >= 3, f"only {len(rounds)} training rounds; run longer"
    print(f"OK  {len(rounds)} training rounds, each from one network version (checked every round)")
    collect = sum(r["collect_seconds"] for r in rounds)
    train = sum(r["seconds"] for r in rounds)
    waited = sum(r["seconds"] for r in rounds if r["paused"])
    decisions = sum(r["decisions"] for r in rounds)
    unused = sum(r["unused"] for r in rounds)
    print(f"    per round: played {collect / len(rounds):.1f}s, trained {train / len(rounds):.1f}s "
          f"(longest {max(r['seconds'] for r in rounds):.1f}s), {decisions / len(rounds):,.0f} decisions learned from")
    print(f"    {decisions / (collect + train):,.0f} learned decisions/s; game waited {waited / (collect + train):.0%} "
          f"of the time; {unused:,} decisions played during training not learned from")
    for line in session.log:
        if line.startswith("  WARNING"):
            print(line)
    print(next(l for l in sim_out.splitlines() if l.startswith("SIM "))[:120] + " ...")


if __name__ == "__main__":
    main()
