"""
Brain 2.0 for GD AI.

Run this, then open a level in GD. The AI learns with PPO: one neural network
trained from every moment of every attempt, so a death mostly teaches it about
what it did just before. It gets "carrots" for every bit of progress and extra
ones for ground it has never reached, and a built-in coach decides by itself
when to practice a hard part and when to try the whole level again.

    python brain2.py              # learn (continues from the last save)
    python brain2.py --play       # just play with what it knows, no learning
    python brain2.py --fresh      # back up the brain and start over
    python brain2.py --brain name # use a separate brain folder
    python brain2.py --no-review  # only train the level you open (no reviews of old levels)
    python brain2.py --eval 20    # measure: 20 full runs without learning, then print how it went

One brain learns every level you open, so what it learns on Stereo Madness
(pads, orbs, ship...) carries over to Back on Track.
"""
import argparse
import base64
import collections
import json
import math
import random
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROTOCOL = 4
BLOCK = 30.0
MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
TYPES = {0: "block", 2: "spike", 25: "slope", 47: "animated spike", -1: "nothing (fell / crashed)"}

# Learning (PPO)
DECISION_TICKS = 4   # decide every 4 physics ticks = 60 times per game second (must match the mod)
HORIZON_SECONDS = 1.7  # how far ahead it cares, in game seconds
INIT_HOLD_BIAS = -2.5  # start out rarely holding (~8%), otherwise it jumps non-stop and never learns to wait
GAE_LAMBDA = 0.95
ROLLOUT = 4096       # decisions collected before each training round
EPOCHS = 4
MINIBATCH = 512
LEARNING_RATE = 3e-4
CLIP = 0.2
ENTROPY_BONUS = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TRAIN_THREADS = 3
PAUSE_WARNING = 60.0  # s: with --pause-to-train the mod waits at most 120 s for an answer, warn well before that

# Rewards ("carrots")
PROGRESS_REWARD = 0.1   # per block moved forward
FRONTIER_REWARD = 0.2   # extra per block of ground never reached before on this level
DEATH_PENALTY = -1.0
WIN_REWARD = 10.0
TELEPORT_JUMP = 60.0    # px between two decisions that can't be real movement

# Coach
RUN_WINDOW = 12          # full runs looked at when deciding what to practice
MIN_RUNS = 8             # full runs needed before it judges
HOTSPOT_RANGE = 150.0    # px: deaths this close together count as "the same spot"
HOTSPOT_SHARE = 0.4      # practice a spot once this share of recent full runs die there
PRACTICE_LEAD = 700.0    # practice starts this far before the hard part
PASS_MARGIN = 90.0       # px past the hard part that counts as getting through it
PASS_RATE = 0.7          # practice until this share of the last 10 practice tries get through
MIN_PRACTICE = 6
MAX_PRACTICE = 40        # then go back to full runs for a while anyway
REVIEW_CHANCE = 0.1      # practice attempts that start further back, so earlier parts stay fresh
FAIL_STREAK = 8          # practice tries in a row dying at the spot before it changes its approach
LEAD_STEP = 400.0        # ...then each streak starts practice this much further back (a different checkpoint)
MAX_LEAD = 2400.0
EXPLORE_PER_FAIL = 0.02  # chance of a random move near the spot, per failed practice try in a row
MAX_EXPLORE = 0.3
EXPLORE_BEFORE = 600.0   # random moves only from this far before the spot...
EXPLORE_AFTER = 150.0    # ...to this far after it

# Review of levels it has already beaten, so learning new ones doesn't wipe them out
REVIEW_EVERY = 40        # attempts on the current level between reviews
REVIEW_ATTEMPTS = 6      # quick check: full runs on the old level
REVIEW_GOOD = 0.9        # a run this far through the level means it still knows it
REVIEW_MAX = 30          # practice a rusty level at most this many attempts, then go back

# Plan kinds sent to the mod
PLAN_RUN, PLAN_PRACTICE, PLAN_SWITCH = 0, 1, 2

# Actions: hold or release, starting on tick 0-3 of the decision window (index = hold * 4 + tick),
# so a click can land on the exact physics tick although the brain decides 60 times a second.
ACTIONS = 2 * DECISION_TICKS
LATE_BIAS = -3.0  # starting preference against doing it 1-3 ticks late, relative to right away
ALLOWED_TICKS = DECISION_TICKS  # test knob (--ticks): only the first N ticks of a window can be picked

# Memory (--memory): numbers it carries from one decision to the next during an attempt (emptied
# at each start). Off by default: on the simulated level it trained slower and didn't play better.
MEMORY = 64
SEQUENCE = 16     # decisions in a row it trains its memory on

# Experiments after death (--experiments): same start + same clicks = exactly the same run, so it can
# replay its furthest try and move one of the last clicks before the death a few ticks earlier or
# later. Off by default: on the simulated level they took a fifth of the playing time and rarely helped.
EXPERIMENT_SHARE = 0.0          # share of practice tries at a stubborn spot that are experiments
EXPERIMENTS_ON = 0.5            # ...with --experiments
SHIFTS = (-1, 1, -2, 2, -4, 4)  # ticks a click or release gets moved
RECENT_SWITCHES = 4             # clicks/releases before the death that get tried...
LOOK_BACK = 1.5                 # ...if they were at most this many game seconds before it
MAX_OUT_OF_SYNC = 3             # replays of a try that may go differently before it's dropped
MAX_RECORDS = 16                # tries kept per level (one per start)
DEMO_BEFORE = 8                 # decisions before the changed click that go into the demo too
DEMO_LEAD = 300.0               # px before the spot where a demo of getting through it starts
DEMO_COEF = 0.5                 # how strongly it copies demos while training
DEMO_BATCH = 64
DEMO_CAP = 4096                 # decisions kept as demos (oldest go first)


class PolicyNet(nn.Module):
    """Looks at the grid with a small conv net, decides 'hold jump?' and guesses how well it's doing.
    With `memory` > 0, a small memory (GRU) carries what it saw and did over the last moments."""

    def __init__(self, channels, rows, cols, extra, memory=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, channels, rows, cols)).shape[1]
        self.body = nn.Sequential(nn.Linear(n + extra, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.memory = nn.GRU(256, memory, batch_first=True) if memory else None
        self.policy = nn.Linear(256 + memory, ACTIONS)
        self.value = nn.Linear(256 + memory, 1)
        nn.init.orthogonal_(self.policy.weight, 0.01)
        with torch.no_grad():
            self.policy.bias.copy_(action_bias(0.0, INIT_HOLD_BIAS))

    def features(self, grid, extra):
        return self.body(torch.cat([self.conv(grid), extra], dim=1))

    def heads(self, features, memory):
        h = torch.cat([features, memory], dim=1)
        logits = self.policy(h)
        if ALLOWED_TICKS < DECISION_TICKS:
            late = torch.tensor([t >= ALLOWED_TICKS for t in range(DECISION_TICKS)] * 2)
            logits = logits.masked_fill(late, -1e9)
        return logits, self.value(h).squeeze(-1)

    def forward(self, grid, extra, memory):
        """One decision: returns action logits, value guess and the updated memory."""
        f = self.features(grid, extra)
        if self.memory is not None:
            out, _ = self.memory(f.unsqueeze(1), memory.unsqueeze(0))
            memory = out[:, 0]
        return (*self.heads(f, memory), memory)

    def sequence(self, grid, extra, memory):
        """Decisions in a row ([batch, time, ...]), each row from one attempt, the memory starting
        from `memory` ([batch, memory size], as it was when playing)."""
        b, t = grid.shape[:2]
        f = self.features(grid.flatten(0, 1), extra.flatten(0, 1)).view(b, t, -1)
        if self.memory is None:
            return self.heads(f.flatten(0, 1), f.new_zeros(b * t, 0))
        out, _ = self.memory(f, memory.unsqueeze(0))
        return self.heads(f.flatten(0, 1), out.flatten(0, 1))


def allowed_actions():
    """Actions it may pick: all 8, or with --ticks N only those on the first N ticks of a window."""
    return [a for a in range(ACTIONS) if a % DECISION_TICKS < ALLOWED_TICKS]


def mixture_logp(own_logp, explore):
    """Log-chance of an action under what was really played: with chance `explore` a random action
    (each 1/len(allowed_actions())), otherwise the network's own choice (log-chance `own_logp`).
    Playing and training both use this, so the PPO ratio compares like with like. explore = 0 gives
    exactly `own_logp`, explore = 1 gives log(1/8); log(0) parts are -inf, which logaddexp handles."""
    explore = torch.as_tensor(explore, dtype=own_logp.dtype)
    return torch.logaddexp(torch.log(explore / len(allowed_actions())), torch.log1p(-explore) + own_logp)


def advantages(b, last_value, gamma, lam):
    """GAE over one round. An attempt's last decision looks ahead to its `bootstraps` value: 0 after a
    death or a win, its own value guess when the attempt was cut short (level reopened, disconnect).
    Nothing flows from one attempt into the one before it. The round's last decision, if its attempt
    goes on, looks ahead to `last_value`: the next state's value from the same network."""
    n = len(b["rewards"])
    adv = np.zeros(n, np.float32)
    running = 0.0
    for t in reversed(range(n)):
        if b["dones"][t]:
            next_value, running = b["bootstraps"][t], 0.0
        else:
            next_value = last_value if t == n - 1 else b["values"][t + 1]
        delta = b["rewards"][t] + gamma * next_value - b["values"][t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


def action_bias(release, hold):
    """Bias per action: doing it right away gets `release` / `hold`; each later tick adds LATE_BIAS."""
    late = [LATE_BIAS] * (DECISION_TICKS - 1)
    return torch.tensor([release, *(release + b for b in late), hold, *(hold + b for b in late)])


def split_head(state):
    """Turns an old 'hold?' head (one output) into the 8-action head without changing what it does:
    every hold action copies the old hold weights (later ticks less likely), every release action
    gets zero like the old 'not hold', so the chance of holding stays exactly what it was."""
    w, b = state["policy.weight"], state["policy.bias"]
    zeros = torch.zeros_like(w[0])
    out = dict(state)
    out["policy.weight"] = torch.stack([zeros] * DECISION_TICKS + [w[0]] * DECISION_TICKS)
    out["policy.bias"] = action_bias(0.0, float(b[0]))
    return out


def add_memory(state, net):
    """Gives a brain from before the memory existed a fresh one. The heads get zero weights from the
    memory, so it decides exactly as before until it learns what's worth remembering."""
    out = dict(state)
    for k, v in net.state_dict().items():
        if k.startswith("memory."):
            out[k] = v.clone()
    for head in ("policy", "value"):
        w = state[f"{head}.weight"]
        out[f"{head}.weight"] = torch.cat([w, torch.zeros(w.shape[0], net.memory.hidden_size)], dim=1)
    return out


class Rollout:
    """Decisions collected for training, all from one version of the network. Holds up to twice a
    round, so an attempt still going when a round is due can finish before it trains."""

    def __init__(self, layout, size, memory):
        c, r, w, e = layout
        self.grids = np.zeros((size, c, r, w), np.uint8)
        self.extras = np.zeros((size, e), np.float32)
        self.actions = np.zeros(size, np.float32)
        self.logps = np.zeros(size, np.float32)
        self.values = np.zeros(size, np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)       # last decision of its attempt
        self.bootstraps = np.zeros(size, np.float32)  # value looked ahead to after that (0 = died or won)
        self.explores = np.zeros(size, np.float32)    # share of random moves mixed in when it was made
        self.versions = np.zeros(size, np.int64)      # training rounds the network had done
        self.memories = np.zeros((size, memory), np.float32)  # memory it had going into the decision
        self.fresh = np.zeros(size, bool)  # doesn't follow on from the decision before it
        self.size = size
        self.n = 0

    def add(self, grid, extra, action, logp, value, memory, fresh, explore, version):
        i = self.n
        self.grids[i], self.extras[i] = grid, extra
        self.actions[i], self.logps[i], self.values[i] = action, logp, value
        self.memories[i], self.fresh[i] = memory, fresh
        self.explores[i], self.versions[i] = explore, version
        self.rewards[i] = self.dones[i] = self.bootstraps[i] = 0.0
        self.n += 1
        return i

    def take(self):
        n, self.n = self.n, 0
        return {k: getattr(self, k)[:n].copy()
                for k in ("grids", "extras", "actions", "logps", "values", "rewards", "dones", "bootstraps",
                          "explores", "versions", "memories", "fresh")}


class Demos:
    """Decisions from experiments that got past a spot it kept dying at; training nudges it to copy them."""

    def __init__(self, layout, path, memory):
        c, r, w, e = layout
        self.path = path
        self.grids = np.zeros((DEMO_CAP, c, r, w), np.uint8)
        self.extras = np.zeros((DEMO_CAP, e), np.float32)
        self.memories = np.zeros((DEMO_CAP, memory), np.float32)
        self.actions = np.zeros(DEMO_CAP, np.int64)
        self.n = 0     # how many are filled
        self.next = 0  # where the next one goes (oldest is overwritten)
        if path.exists():
            saved = np.load(path)
            if (saved["grids"].shape[1:] == self.grids.shape[1:] and saved["extras"].shape[1] == e
                    and saved["memories"].shape[1] == memory):
                k = min(DEMO_CAP, len(saved["actions"]))
                for name in ("grids", "extras", "memories", "actions"):
                    getattr(self, name)[:k] = saved[name][-k:]
                self.n, self.next = k, k % DEMO_CAP

    def add(self, steps):
        for grid, extra, memory, action in steps:
            i = self.next
            self.grids[i], self.extras[i], self.memories[i], self.actions[i] = grid, extra, memory, action
            self.next = (i + 1) % DEMO_CAP
            self.n = min(DEMO_CAP, self.n + 1)
        self.save()

    def snapshot(self):
        """A copy for the training thread, or None when there are no demos yet."""
        if self.n == 0:
            return None
        return {k: getattr(self, k)[:self.n].copy() for k in ("grids", "extras", "memories", "actions")}

    def save(self):
        # Oldest first, so loading keeps the newest when there are too many.
        order = np.r_[self.next:self.n, 0:self.next] if self.n == DEMO_CAP else np.arange(self.n)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez_compressed(tmp, **{k: getattr(self, k)[order] for k in ("grids", "extras", "memories", "actions")})
        tmp.replace(self.path)


def sequences(fresh, length):
    """Cuts the decisions into runs of at most `length` in a row that never cross the start of an
    attempt (or a gap of replayed decisions). Returns each run's decision indices ([runs, length],
    padded with its last one) and a mask of the real ones."""
    starts = []
    for i in range(len(fresh)):
        if not starts or fresh[i] or i - starts[-1] >= length:
            starts.append(i)
    starts.append(len(fresh))
    rows = np.zeros((len(starts) - 1, length), np.int64)
    mask = np.zeros((len(starts) - 1, length), np.float32)
    for r, (s, e) in enumerate(zip(starts, starts[1:])):
        rows[r] = e - 1
        rows[r, :e - s] = np.arange(s, e)
        mask[r, :e - s] = 1.0
    return rows, mask


def widen(state, old_layout, new_layout, net):
    """Fits a brain trained on a smaller vision into a bigger one. The mod only ever appends
    channels and numbers, so the old weights keep their meaning; the new ones start at zero,
    which makes the widened brain behave exactly like the old one until it learns to use them."""
    (oc, orows, ocols, oe), (nc, nrows, ncols, ne) = old_layout, new_layout
    if (orows, ocols) != (nrows, ncols) or oc > nc or oe > ne:
        sys.exit(f"The brain was trained with vision {old_layout}, the mod sends {new_layout}, "
                 f"and it can't be widened to that. Use --fresh or another --brain.")
    out = {k: v.clone() for k, v in state.items()}
    w = state["conv.0.weight"]
    out["conv.0.weight"] = torch.zeros(w.shape[0], nc, *w.shape[2:])
    out["conv.0.weight"][:, :oc] = w
    n_conv = net.body[0].in_features - ne
    w = state["body.0.weight"]
    out["body.0.weight"] = torch.zeros(w.shape[0], n_conv + ne)
    out["body.0.weight"][:, :n_conv + oe] = w
    return out


class Agent:
    def __init__(self, layout, path, memory=False):
        self.layout = tuple(layout)
        self.path = path
        saved = torch.load(path, weights_only=False) if path.exists() else None
        # A brain that has a memory keeps it; one without gets it only when asked for.
        had_memory = saved is not None and "memory.weight_hh_l0" in saved["net"]
        self.memory_size = saved["net"]["memory.weight_hh_l0"].shape[1] if had_memory else (MEMORY if memory else 0)
        # `net` plays; `learner` trains while the game waits and is copied into `net` when a round is done.
        self.net = PolicyNet(*self.layout, self.memory_size)
        self.learner = PolicyNet(*self.layout, self.memory_size)
        self.opt = torch.optim.Adam(self.learner.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.steps = 0
        self.updates = 0
        if saved is not None:
            old_layout = tuple(saved["layout"])
            state = saved["net"]
            changed = False
            if old_layout != self.layout:
                state = widen(state, old_layout, self.layout, self.learner)
                print(f"Brain widened from vision {old_layout} to {self.layout}: new inputs start at zero, "
                      f"everything it learned is kept")
                changed = True
            if state["policy.weight"].shape[0] == 1:
                state = split_head(state)
                print("Brain upgraded to tick-exact clicks: it decides the same as before and learns "
                      "when a click 1-3 ticks later works better")
                changed = True
            if self.memory_size and not had_memory:
                state = add_memory(state, self.learner)
                print("Brain got a memory: it decides the same as before and learns what's worth "
                      "remembering from the last moments")
                changed = True
            self.learner.load_state_dict(state)
            if not changed:
                self.opt.load_state_dict(saved["opt"])
            self.steps, self.updates = saved["steps"], saved["updates"]
        self.net.load_state_dict(self.learner.state_dict())
        self.rollout = Rollout(self.layout, 2 * ROLLOUT, self.memory_size)
        self.demos = Demos(self.layout, path.with_name("demos.npz"), self.memory_size)
        self.collect_started = time.time()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = None           # round training in the background
        self.job_info = {}
        self.unused = 0           # decisions played while it trained (not learned from)

    def empty_memory(self):
        return torch.zeros(1, self.memory_size)

    @torch.inference_mode()
    def act(self, grid, extra, memory, explore=0.0):
        """Samples an action (hold/release on tick 0-3). With `explore` > 0, that share of moves is
        random instead. Returns the action, the log-chance it had, the value guess, P(hold) and
        the updated memory."""
        g = torch.from_numpy(grid).unsqueeze(0).float().div_(255.0)
        e = torch.from_numpy(extra).unsqueeze(0)
        logits, value, memory = self.net(g, e, memory)
        dist = torch.distributions.Categorical(logits=logits[0])
        probs = dist.probs.tolist()
        if random.random() < explore:
            action = random.choice(allowed_actions())
        else:
            action = random.choices(range(ACTIONS), weights=probs)[0]
        # Training needs the chance of the move actually made, random moves included.
        logp = mixture_logp(dist.log_prob(torch.tensor(action)), explore)
        return action, float(logp), float(value), sum(probs[DECISION_TICKS:]), memory

    @torch.inference_mode()
    def value(self, grid, extra, memory):
        g = torch.from_numpy(grid).unsqueeze(0).float().div_(255.0)
        return float(self.net(g, torch.from_numpy(extra).unsqueeze(0), memory)[1])

    def due(self):
        """Enough decisions for a round (it trains at the next start of an attempt)."""
        return self.rollout.n >= ROLLOUT

    def full(self):
        """No room left: an attempt that long gets cut and trains right away."""
        return self.rollout.n >= self.rollout.size

    def take_round(self):
        b = self.rollout.take()
        if not (b["versions"] == self.updates).all():
            raise RuntimeError("a training round got decisions from an older network")
        info = {"collect_seconds": time.time() - self.collect_started, "decisions": len(b["actions"])}
        return b, info

    def train(self, last_value=0.0):
        """One PPO round on everything collected, while the game waits for the answer (--pause-to-train).
        Every decision in it came from the network as it is now; afterwards the playing network is the
        trained one and the buffer is empty, so the next round only ever sees the next version's decisions.
        `last_value`: value of the state after the round's last decision, if its attempt goes on."""
        b, info = self.take_round()
        stats = self._train(b, last_value, self.demos.snapshot())
        self.net.load_state_dict(self.learner.state_dict())
        self.collect_started = time.time()
        return stats | info | {"paused": True, "unused": 0}

    def training(self):
        """A round is training in the background."""
        return self.job is not None

    def start_training(self, last_value=0.0):
        """Starts a round in the background, so the game doesn't stop. Until it's done, the game goes
        on with the network as it was; those decisions aren't learned from (`skip`), so every round
        still only learns from the version it starts from."""
        b, self.job_info = self.take_round()
        self.unused = 0
        self.job = self.pool.submit(self._train, b, last_value, self.demos.snapshot())

    def skip(self):
        self.unused += 1

    def finish_training(self, wait=False):
        """If the background round is done (or `wait`), the playing network becomes the trained one.
        Returns the round's stats, or None while it's still training."""
        if self.job is None or not (wait or self.job.done()):
            return None
        stats = self.job.result()
        self.job = None
        self.net.load_state_dict(self.learner.state_dict())
        self.collect_started = time.time()
        return stats | self.job_info | {"paused": False, "unused": self.unused}

    def _train(self, b, last_value, demos=None):
        """One PPO round on the collected decisions. The memory trains on runs of SEQUENCE decisions
        in a row; demos (if any) are copied on the side."""
        started = time.time()
        n = len(b["actions"])
        gamma = math.exp(-DECISION_TICKS / (240.0 * HORIZON_SECONDS))
        adv = advantages(b, last_value, gamma, GAE_LAMBDA)
        returns = adv + b["values"]

        # With a memory it trains on runs of decisions in a row; without, on single decisions.
        length = SEQUENCE if self.memory_size else 1
        rows, mask = sequences(b["fresh"], length)
        rows, mask = torch.from_numpy(rows), torch.from_numpy(mask)
        padded = length > 1
        grids = torch.from_numpy(b["grids"]).float().div_(255.0)
        extras, memories = torch.from_numpy(b["extras"]), torch.from_numpy(b["memories"])
        actions, old_logps = torch.from_numpy(b["actions"]).long(), torch.from_numpy(b["logps"])
        explores = torch.from_numpy(b["explores"])
        adv_t = torch.from_numpy(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = torch.from_numpy(returns.astype(np.float32))
        if demos is not None:
            demo_grids = torch.from_numpy(demos["grids"])
            demo_extras, demo_memories = torch.from_numpy(demos["extras"]), torch.from_numpy(demos["memories"])
            demo_actions = torch.from_numpy(demos["actions"])

        stats = collections.defaultdict(float)
        batches = 0
        per_batch = max(1, MINIBATCH // length)
        for _ in range(EPOCHS):
            order = torch.randperm(len(rows))
            for start in range(0, len(rows), per_batch):
                pick_rows = order[start:start + per_batch]
                idx = rows[pick_rows]
                flat = idx.flatten()
                logits, values = self.learner.sequence(grids[idx], extras[idx], memories[idx[:, 0]])
                m = mask[pick_rows].flatten()
                count = m.sum()
                # Runs padded to SEQUENCE only count their real decisions.
                mean = (lambda x: (x * m).sum() / count) if padded else (lambda x: x.mean())
                dist = torch.distributions.Categorical(logits=logits)
                # Same mix of random and own moves as when it was played, so unchanged weights give ratio 1.
                logp = mixture_logp(dist.log_prob(actions[flat]), explores[flat])
                ratio = torch.exp(logp - old_logps[flat])
                a = adv_t[flat]
                policy_loss = -mean(torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a))
                value_loss = 0.5 * mean((values - returns_t[flat]).pow(2))
                # Regularizes the network's own choice (not the mix with random moves).
                entropy = mean(dist.entropy())
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_BONUS * entropy
                if demos is not None:
                    pick = torch.randint(len(demo_actions), (DEMO_BATCH,))
                    demo_logits, _, _ = self.learner(demo_grids[pick].float().div_(255.0), demo_extras[pick],
                                                     demo_memories[pick])
                    demo_loss = nn.functional.cross_entropy(demo_logits, demo_actions[pick])
                    loss = loss + DEMO_COEF * demo_loss
                    stats["demo_loss"] += float(demo_loss.detach())
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.learner.parameters(), MAX_GRAD_NORM)
                self.opt.step()
                stats["entropy"] += float(entropy.detach())
                stats["value_loss"] += float(value_loss.detach())
                batches += 1

        self.steps += n
        self.updates += 1
        self.save()
        # Only decisions that really press or release count: "hold" while already holding does nothing.
        changes = (b["actions"] >= DECISION_TICKS) != (b["extras"][:, 15] > 0.5)
        late = float(np.mean(b["actions"][changes] % DECISION_TICKS != 0)) if changes.any() else 0.0
        return {k: v / batches for k, v in stats.items()} | {
            "seconds": time.time() - started, "late": late, "changes": int(changes.sum()),
            "demos": 0 if demos is None else len(demos["actions"])}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        torch.save({"net": self.learner.state_dict(), "opt": self.opt.state_dict(), "layout": self.layout,
                    "steps": self.steps, "updates": self.updates}, tmp)
        tmp.replace(self.path)

class Coach:
    """Decides by itself, from how full runs go, when to practice a spot and when to run the whole level.
    When practice keeps dying at the spot, it tries something different: random moves near it, and
    starting further back so it reaches it from a different checkpoint and a different approach."""

    def __init__(self, state, say):
        self.s = state
        self.say = say
        for key, default in [("phase", "runs"), ("target_x", 0.0), ("run_ends", []), ("practice_results", []),
                             ("practice_count", 0), ("best_run_x", 0.0), ("wins", 0), ("fail_streak", 0),
                             ("spot_streaks", {})]:
            self.s.setdefault(key, default)

    def lead(self):
        return min(MAX_LEAD, PRACTICE_LEAD + LEAD_STEP * (self.s["fail_streak"] // FAIL_STREAK))

    def stubborn(self):
        """Practicing a spot it has died at many tries in a row."""
        return self.s["phase"] == "practice" and self.s["fail_streak"] >= FAIL_STREAK

    def explore_rate(self, x):
        """Share of random moves at position x: only near the spot it keeps failing."""
        s = self.s
        if s["phase"] != "practice" or not s["target_x"] - EXPLORE_BEFORE <= x <= s["target_x"] + EXPLORE_AFTER:
            return 0.0
        return min(MAX_EXPLORE, EXPLORE_PER_FAIL * s["fail_streak"])

    def plan(self):
        """Returns (kind, practice, start_x). kind is 'run' (from the start) or 'practice'."""
        s = self.s
        if s["phase"] != "practice":
            return "run", False, 0.0
        start = max(0.0, s["target_x"] - self.lead())
        if random.random() < REVIEW_CHANCE:
            start = random.uniform(0.0, start)
        return "practice", True, start

    def spot_key(self):
        return str(int(self.s["target_x"] // 300))

    def leave_practice(self, why):
        s = self.s
        self.say(why)
        # Remember how stubborn this spot was, so coming back to it starts from there.
        s["spot_streaks"][self.spot_key()] = s["fail_streak"]
        s["phase"], s["run_ends"] = "runs", []

    def attempt_done(self, kind, end_x, won, length):
        s = self.s
        pct = lambda x: 100.0 * x / max(length, 1.0)
        if kind == "practice":
            passed = bool(won or end_x > s["target_x"] + PASS_MARGIN)
            s["practice_results"] = (s["practice_results"] + [passed])[-10:]
            s["practice_count"] += 1
            if passed:
                s["fail_streak"] = 0
            elif end_x > s["target_x"] - EXPLORE_BEFORE:
                s["fail_streak"] += 1
                if s["fail_streak"] % FAIL_STREAK == 0:
                    self.say(f"Coach: {s['fail_streak']} practice tries in a row died at {pct(s['target_x']):.1f}% "
                             f"-> trying something different: {self.explore_rate(s['target_x']):.0%} random moves "
                             f"there, starting further back (~{pct(max(0.0, s['target_x'] - self.lead())):.1f}%)")
            r = s["practice_results"]
            if len(r) >= MIN_PRACTICE and sum(r) / len(r) >= PASS_RATE:
                s["fail_streak"] = 0
                self.leave_practice(f"Coach: gets through {pct(s['target_x']):.1f}% in {sum(r)}/{len(r)} "
                                    f"practice tries -> back to full runs")
            elif s["practice_count"] >= MAX_PRACTICE:
                self.leave_practice(f"Coach: {pct(s['target_x']):.1f}% is still hard after {MAX_PRACTICE} "
                                    f"practice tries -> full runs for a while")
            return

        if won:
            s["wins"] += 1
            self.say(f"Coach: full run BEAT THE LEVEL (win #{s['wins']})")
        s["best_run_x"] = max(s["best_run_x"], end_x)
        s["run_ends"] = (s["run_ends"] + [end_x])[-RUN_WINDOW:]
        ends = s["run_ends"]
        if len(ends) < MIN_RUNS:
            return
        # Where do recent full runs die most? That spot is worth practicing,
        # whether it's new ground or a jump it only makes sometimes.
        spot, count = max(((e, sum(abs(o - e) <= HOTSPOT_RANGE / 2 for o in ends)) for e in ends if e < length),
                          key=lambda t: t[1], default=(0.0, 0))
        if count / len(ends) >= HOTSPOT_SHARE:
            near = [o for o in ends if abs(o - spot) <= HOTSPOT_RANGE / 2]
            s["phase"], s["target_x"] = "practice", float(np.median(near))
            s["practice_results"], s["practice_count"] = [], 0
            s["fail_streak"] = s["spot_streaks"].get(self.spot_key(), 0)
            again = " (again)" if s["fail_streak"] else ""
            self.say(f"Coach: {count} of the last {len(ends)} full runs died around {pct(s['target_x']):.1f}% "
                     f"-> practice mode{again} from ~{pct(max(0.0, s['target_x'] - self.lead())):.1f}%")

    def status(self, length):
        s = self.s
        pct = lambda x: 100.0 * x / max(length, 1.0)
        if s["phase"] == "practice":
            r = s["practice_results"]
            explore = self.explore_rate(s["target_x"])
            trying = f", trying {explore:.0%} random moves" if explore else ""
            return f"practicing {pct(s['target_x']):.1f}% ({sum(r)}/{len(r)} got through{trying})"
        return "full runs"


def timeline(actions):
    """Whether the button is held on each tick, from per-decision actions (played like the mod does)."""
    held, ticks = False, []
    for a in actions:
        hold, at = a >= DECISION_TICKS, a % DECISION_TICKS
        start = held
        for t in range(DECISION_TICKS):
            held = hold if t >= at else start
            ticks.append(held)
    return ticks


def to_actions(ticks):
    """Per-decision actions that play `ticks` (whole decision windows), or None if a window
    would need two changes."""
    held, actions = False, []
    for w in range(0, len(ticks), DECISION_TICKS):
        part = ticks[w:w + DECISION_TICKS]
        changes = [t for t in range(DECISION_TICKS) if part[t] != (part[t - 1] if t else held)]
        if len(changes) > 1:
            return None
        actions.append(int(part[changes[0]]) * DECISION_TICKS + changes[0] if changes else int(held) * DECISION_TICKS)
        held = part[-1]
    return actions


def changes_to_try(actions):
    """What's worth trying on a try that died at its end: each of the last clicks and releases moved
    a few ticks, or a click left out. Yields (description, ticks, first changed tick, last tick to play)."""
    ticks = timeline(actions)
    end = len(ticks)
    switches = [t for t in range(end) if ticks[t] != (ticks[t - 1] if t else False)]
    for i in reversed(range(max(0, len(switches) - RECENT_SWITCHES), len(switches))):
        s = switches[i]
        if s < end - LOOK_BACK * 240:
            break
        lo = switches[i - 1] if i else -1                       # stays after the change before it...
        hi = switches[i + 1] if i + 1 < len(switches) else end  # ...and before the one after it
        what = "click" if ticks[s] else "release"
        for d in SHIFTS:
            to = s + d
            if not lo < to < hi:
                continue
            new = list(ticks)
            if d < 0:
                new[to:s] = [ticks[s]] * -d
            else:
                new[s:to] = [not ticks[s]] * d
            yield f"{what} {abs(d)} tick{'s' if abs(d) > 1 else ''} {'earlier' if d < 0 else 'later'}", new, min(s, to), max(s, to)
        if ticks[s]:
            new = list(ticks)
            new[s:hi] = [False] * (hi - s)
            yield "no click", new, s, min(hi, end - 1)


class Experiments:
    """Experiments after death. From the same start, the same clicks play out exactly the same, so it
    keeps the clicks of its furthest try from each start. At a spot it keeps dying at, it replays that
    try with one of the last clicks or releases before the death moved a few ticks (or left out), then
    plays on by itself. A change that gets it past where the try died becomes a demo it learns to copy."""

    def __init__(self, folder, say):
        self.file = folder / "experiments.json"
        self.say = say
        self.records = {}
        if self.file.exists():
            try:
                self.records = json.loads(self.file.read_text(encoding="utf-8"))
            except ValueError:
                pass
        self.tried = self.better = self.shown = self.out_of_sync = 0

    @staticmethod
    def key(start_x):
        return f"{start_x:.1f}"

    def pick(self, start_x, spot_x):
        """What to replay on a try starting at start_x, for the spot at spot_x it keeps dying at:
        - its furthest try from there got through the spot once: replay it through the spot (once),
          which makes a demo of how to get through;
        - otherwise, if that try got near the spot: replay it with one change before its death.
        Returns the script, or None."""
        rec = self.records.get(self.key(start_x))
        if rec is None or rec["end"] < spot_x - EXPLORE_BEFORE or rec["bad"] >= MAX_OUT_OF_SYNC:
            return None
        actions = [int(c) for c in rec["actions"]]
        prints = np.frombuffer(base64.b64decode(rec["prints"]), np.float32).reshape(-1, 3)
        script = {"key": self.key(start_x), "prints": prints}
        if rec["end"] > spot_x + PASS_MARGIN and not rec.get("shown"):
            rec["shown"] = True
            rec["used"] = time.time()
            xs = prints[:, 0]
            through = int(np.searchsorted(xs, spot_x + PASS_MARGIN)) + 1
            return script | {"kind": "demo", "actions": actions[:through], "check_until": through - 1,
                             "demo_from": int(np.searchsorted(xs, spot_x - DEMO_LEAD)), "old_end": spot_x}
        options = list(changes_to_try(actions))
        while rec["next"] < len(options):
            what, ticks, first, last = options[rec["next"]]
            rec["next"] += 1
            changed = to_actions(ticks[:(last // DECISION_TICKS + 1) * DECISION_TICKS])
            if changed is None:
                continue
            rec["used"] = time.time()
            return script | {"kind": "change", "actions": changed, "what": what,
                             "check_until": first // DECISION_TICKS,
                             "demo_from": max(0, first // DECISION_TICKS - DEMO_BEFORE), "old_end": rec["end"]}
        return None

    def attempt_done(self, start_x, end_x, actions, prints, script):
        """Keeps the clicks of the furthest try from each start and counts how experiments go."""
        key = self.key(start_x)
        rec = self.records.get(key)
        if script is not None:
            if script.get("out_of_sync"):
                self.out_of_sync += 1
                if rec is not None:
                    rec["bad"] += 1
            if script["kind"] == "demo":
                self.shown += not script.get("out_of_sync")
            else:
                self.tried += 1
                self.better += end_x > script["old_end"] + BLOCK and not script.get("out_of_sync")
        if actions and (rec is None or end_x > rec["end"] + 1.0):
            self.records[key] = {
                "start": start_x, "end": end_x, "actions": "".join(map(str, actions)),
                "prints": base64.b64encode(np.asarray(prints, np.float32).tobytes()).decode(),
                "next": 0, "bad": rec["bad"] if rec else 0, "used": time.time()}
            if len(self.records) > MAX_RECORDS:
                del self.records[min(self.records, key=lambda k: self.records[k]["used"])]
        elif script is None:
            return
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.records), encoding="utf-8")
        tmp.replace(self.file)


MOMENTS_SECONDS = 1.0     # how much before each death goes into moments.jsonl
MOMENTS_MAX_MB = 50       # moments.jsonl stops growing at this size
GRID_THINGS = {2: "jump_pad", 3: "gravity_pad", 4: "jump_orb", 5: "gravity_orb", 6: "gravity_portal",
               7: "mode_portal", 8: "other_portal", 9: "speed_portal"}  # near-grid channel -> name


def log_moments(path, history, record):
    """Appends the last MOMENTS_SECONDS before a death to moments.jsonl: per decision the position,
    what it chose, both players' numbers, the nearest things ahead and which kinds of pads, orbs and
    portals (by their grid value) are in view. black-box.txt only keeps the furthest death; this
    keeps many, so deaths at the same spot can be compared."""
    if path.exists() and path.stat().st_size > MOMENTS_MAX_MB * 1e6:
        return False
    steps = []
    for grid, extra, action, p, x in list(history)[-int(MOMENTS_SECONDS * 240 / DECISION_TICKS):]:
        r = lambda v: [round(float(a), 3) for a in v]
        dual = bool(extra[16] > 0.5)
        steps.append({
            "x": round(float(x), 1), "action": int(action), "held": bool(extra[15] > 0.5), "p_hold": round(p, 3),
            "p1": r(extra[0:15]), "p2": r(extra[17:32]) if dual else None,
            "ticks_since_switch": int(round(extra[32] * 120)),
            "nearest": [r(extra[33 + 7 * i:40 + 7 * i]) for i in range(5) if extra[33 + 7 * i:40 + 7 * i].any()],
            "things": {name: sorted({int(v) for v in grid[ch][grid[ch] > 0]}) for ch, name in GRID_THINGS.items()
                       if grid[ch].any()},
        })
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record | {"steps": steps}) + "\n")
    return True


def dump_black_box(path, history, layout, info, length):
    """Writes what the AI saw and did in its last moments of a full run, as ASCII grids."""
    c, rows, cols, _ = layout
    lines = [
        f"Full run death at {100 * info['x'] / max(length, 1):.1f}% (x={info['x']:.0f}, y={info['y']:.0f}), "
        f"{MODES[info['mode']]}, killed by {TYPES.get(info['object_type'], info['object_type'])} (object id {info['object_id']})",
        "Legend: ^ spike  # block  p pad  o orb  | portal  P = player row (player sits on the left edge)",
        "",
    ]
    frames = list(history)
    for i, (grid, extra, action, p, _) in enumerate(frames):
        if (len(frames) - 1 - i) % 6 != 0:
            continue
        lines.append(f"t={(i - len(frames) + 1) * DECISION_TICKS / 240:+.2f}s  {'hold' if action >= DECISION_TICKS else 'free'} from tick {action % DECISION_TICKS} "
                     f"(wanted {p:.0%})  on ground: {'yes' if extra[1] > 0.5 else 'no '}  y-speed: {extra[0] * 20:+.1f}  "
                     f"mode: {MODES[int(np.argmax(extra[3:11]))]}")
        for r in reversed(range(rows)):
            row = ""
            for col in range(cols):
                cell = grid[:, r, col]
                row += ("^" if cell[0] else "#" if cell[1] else "p" if cell[2] or cell[3] else
                        "o" if cell[4] or cell[5] else "|" if cell[6:10].any() else ".")
            lines.append(("P " if r == rows // 2 else "  ") + row)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


class Level:
    """Everything remembered about one level: coach state, best distances, death log."""

    def __init__(self, folder, level_id, name, length, say):
        self.folder = folder / "levels" / f"{level_id}-{''.join(ch if ch.isalnum() else '_' for ch in name)}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = self.folder / "state.json"
        self.state = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}
        self.state.update(name=name, id=level_id, length=length)
        self.id = level_id
        self.state.setdefault("best_x", 0.0)
        self.state.setdefault("attempts", 0)
        self.coach = Coach(self.state.setdefault("coach", {}), say)
        self.experiments = Experiments(self.folder, say)
        self.name, self.length = name, length

    def log_death(self, record):
        with (self.folder / "deaths.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.file)


class Reviewer:
    """Every so often takes the AI back to a level it has already beaten for a quick check, and
    practices it there if it got rusty, so learning new levels doesn't wipe out the old ones."""

    def __init__(self, folder, say, enabled=True):
        self.folder = folder
        self.say = say
        self.enabled = enabled
        self.home = None        # the level being learned (the one you opened)
        self.review = None      # the old level being checked: id, name, length, attempts, best, rusty, done
        self.switch_to = None   # level asked from the mod that hasn't opened yet
        self.since = 0          # attempts on the home level since the last review
        self.unavailable = set()

    def on_level(self, lvl):
        if self.switch_to is not None and lvl.id == self.switch_to:
            self.switch_to = None
            if self.review and self.review["id"] == lvl.id:
                self.say(f"Review: back on {lvl.name} for a quick check")
            else:
                self.review = None
            return
        if self.in_review(lvl) or (lvl.id == self.home and self.review is None):
            # The same level opened again (GD sometimes rebuilds it in practice): nothing changes.
            return
        # Any other level means you opened it: that's what it learns now.
        self.home, self.review, self.switch_to = lvl.id, None, None

    def in_review(self, lvl):
        return self.review is not None and self.review["id"] == lvl.id

    def quick_check(self, lvl):
        """During the first review attempts, plain full runs show what it still knows."""
        return self.in_review(lvl) and not self.review["rusty"]

    def next_switch(self, lvl):
        """Level to switch to before this attempt, or None to stay."""
        if not self.enabled:
            return None
        if self.review is not None:
            if self.review["done"] and self.home is not None:
                self.switch_to = self.home
                return self.home
            return None
        if lvl.id != self.home:
            return None
        self.since += 1
        if self.since < REVIEW_EVERY:
            return None
        self.since = 0
        old = self.pick()
        if old is None:
            return None
        self.review = dict(old, attempts=0, best=0.0, rusty=False, done=False)
        self.switch_to = old["id"]
        self.say(f"Review: time for a refresher on {old['name']} (beaten before)")
        return old["id"]

    def pick(self):
        """The beaten level (other than the current one) that was reviewed longest ago."""
        best = None
        for path in (self.folder / "levels").glob("*/state.json"):
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            length = s.get("length", 0) or 0
            beaten = s.get("coach", {}).get("wins", 0) >= 1 or (length > 0 and s.get("best_x", 0) >= length)
            if not beaten or s.get("id") in (self.home, None) or s["id"] in self.unavailable:
                continue
            if best is None or s.get("last_review", 0) < best["last_review"]:
                best = {"id": s["id"], "name": s.get("name", "?"), "length": s.get("length", 1.0),
                        "last_review": s.get("last_review", 0)}
        return best

    def attempt_done(self, lvl, end_x, won):
        if not self.in_review(lvl) or self.review["done"]:
            return
        r = self.review
        r["attempts"] += 1
        r["best"] = max(r["best"], end_x)
        pct = 100.0 * r["best"] / max(lvl.length, 1.0)
        if won or r["best"] >= REVIEW_GOOD * lvl.length:
            self.say(f"Review: {lvl.name} {'is back' if r['rusty'] else 'still solid'} ({pct:.0f}%) -> back to learning")
            r["done"] = True
        elif r["attempts"] == REVIEW_ATTEMPTS:
            r["rusty"] = True
            self.say(f"Review: {lvl.name} got rusty (best {pct:.0f}% in {REVIEW_ATTEMPTS} runs) -> practicing it for a bit")
        elif r["attempts"] >= REVIEW_MAX:
            self.say(f"Review: {lvl.name} still rusty after {REVIEW_MAX} tries ({pct:.0f}%) -> back for now, again later")
            r["done"] = True
        if r["done"]:
            lvl.state["last_review"] = time.time()

    def switch_failed(self, level_id):
        self.unavailable.add(level_id)
        if self.review and self.review["id"] == level_id:
            self.say(f"Review: GD can't open {self.review['name']} (open it once yourself), skipping it")
            self.review = None
        self.switch_to = None


class Session:
    """Handles the mod's messages for one connection."""

    def __init__(self, folder, play_only, review=False, eval_runs=0, memory=False, go_to=None, trace_runs=False,
                 minutes=0.0, rounds=0, pause_to_train=False):
        self.folder = folder
        self.pause_to_train = pause_to_train  # old way: the game waits while a round trains
        self.use_memory = memory
        self.go_to = go_to            # --level: switch GD to this level first
        self.target_level = go_to     # ...and stop if someone opens another one after that
        self.reached_target = False
        self.trace_runs = trace_runs
        self.stop_at = time.time() + minutes * 60 if minutes else None
        self.rounds_left = rounds or None  # --rounds: stop after this many training rounds
        self.start = {}
        self.play_only = play_only or eval_runs > 0
        self.eval_runs = eval_runs  # > 0: measure this many full runs, then stop
        self.eval_results = []
        self.agent = None
        self.level = None
        self.kind = None          # current attempt: 'run' (from the start), 'practice' or None
        self.pending = None       # rollout index waiting for its reward
        self.reward = 0.0
        self.last_x = 0.0
        self.attempt_best_x = 0.0
        self.history = collections.deque(maxlen=120)
        self.reviewer = Reviewer(folder, self.say, enabled=review)
        self.memory = None        # what the network carries between decisions
        self.continued = False    # the last decision went into the rollout (the next one follows on)
        self.start_x = 0.0
        self.trace = []           # (action, fingerprint) of every decision this attempt
        self.script = None        # experiment being replayed this attempt
        self.demo_steps = []
        self.demo_done = False

    def say(self, text):
        print(text, flush=True)

    # --- messages -------------------------------------------------------
    def on_hello(self, level_id, name, layout, length):
        self.end_attempt(truncated=True)
        if self.agent is None:
            self.agent = Agent(layout, self.folder / "model.pt", memory=self.use_memory)
            with_memory = " (with memory)" if self.agent.memory_size else ""
            self.say(f"Brain loaded{with_memory}: {self.agent.updates} training rounds, {self.agent.steps:,} decisions so far")
        elif tuple(layout) != self.agent.layout:
            sys.exit("The mod's vision layout changed mid-session. Restart the brain.")
        if self.target_level is not None:
            if level_id == self.target_level:
                self.reached_target = True
            elif self.reached_target:
                # Someone opened another level: let them play it, and don't learn or measure the wrong one.
                self.say(f"Another level was opened ({name}): stopping (--level {self.target_level}).")
                raise SystemExit(0)
        self.level = Level(self.folder, level_id, name, length, self.say)
        best = 100 * self.level.state["best_x"] / max(length, 1)
        self.say(f"Level: {name} (best ever {best:.1f}%, {self.level.coach.status(length)})")
        self.reviewer.on_level(self.level)

    def on_ask(self, x, practice):
        """Returns the plan for the mod: (kind, flags, start_x, level_id). Flag 1 = don't stop on a win."""
        self.end_attempt(truncated=True)
        if self.stop_at is not None and time.time() > self.stop_at:
            self.stop("Time's up (--minutes): stopping. The brain is saved.")
        if self.go_to is not None and self.level is not None and self.level.id != self.go_to:
            level, self.go_to = self.go_to, None
            self.say(f"Switching GD to level {level} (--level)")
            return PLAN_SWITCH, 0, 0.0, level
        if not self.play_only and self.agent is not None:
            stats = self.agent.finish_training()
            if stats:
                self.report(stats)
            if self.agent.due() and not self.agent.training():
                if self.pause_to_train:
                    # Between attempts: the mod waits for the plan, so the game stands still while it trains.
                    self.report(self.agent.train())
                else:
                    self.agent.start_training()
        if self.rounds_left is not None and self.rounds_left <= 0:
            self.stop("Training budget used up (--rounds): stopping. The brain is saved.")
        if self.play_only:
            self.kind = "run"
            return PLAN_RUN, 1 if self.eval_runs else 0, 0.0, 0
        lvl = self.level
        switch = self.reviewer.next_switch(lvl)
        if switch is not None:
            return PLAN_SWITCH, 0, 0.0, switch
        # The first win gets the end screen; after that (and in reviews) it just keeps training.
        quiet = 1 if self.reviewer.in_review(lvl) or lvl.state["coach"].get("wins", 0) >= 1 else 0
        if self.reviewer.quick_check(lvl):
            self.kind, want_practice, target = "run", False, 0.0
        else:
            self.kind, want_practice, target = lvl.coach.plan()
        return (PLAN_PRACTICE if want_practice else PLAN_RUN), quiet, target, 0

    def stop(self, why):
        """Ends the session cleanly: a round still training finishes and gets saved first."""
        if self.agent is not None and self.agent.training():
            self.report(self.agent.finish_training(wait=True))
        self.say(why)
        raise SystemExit(0)

    def on_switch_failed(self, level_id):
        self.reviewer.switch_failed(level_id)

    def on_begin(self, start_x, practice, from_checkpoint):
        self.last_x = start_x
        self.attempt_best_x = start_x
        self.history.clear()
        self.pending = None
        self.reward = 0.0
        self.memory = self.agent.empty_memory()
        self.continued = False
        self.start_x = start_x
        self.start = {"start_x": round(start_x, 1), "practice": bool(practice), "from_checkpoint": bool(from_checkpoint)}
        self.trace = []
        self.script = None
        self.demo_steps, self.demo_done = [], False
        if self.kind == "practice" and not self.play_only:
            coach = self.level.coach
            if coach.stubborn() and random.random() < EXPERIMENT_SHARE:
                self.script = self.level.experiments.pick(start_x, coach.s["target_x"])

    def on_state(self, x, percent, extra, grid_bytes):
        # The mod only asks at the start of each decision window.
        if self.kind is None:
            return 0
        grid = np.frombuffer(grid_bytes, np.uint8).reshape(self.agent.layout[:3]).copy()
        extra = np.array(extra, np.float32)
        self.collect_progress(x)
        i = len(self.trace)
        fingerprint = (x, float(extra[0]), float(extra[13]))
        script = self.script
        scripted = script is not None and i < len(script["actions"])
        if scripted and i <= script["check_until"] and not (
                i < len(script["prints"]) and np.allclose(fingerprint, script["prints"][i], rtol=0, atol=1e-3)):
            # It didn't play out like the try it replays (the start wasn't exactly the same): play on as usual.
            script["out_of_sync"] = True
            script["actions"] = script["actions"][:i]
            scripted = False
        learning = self.kind in ("run", "practice") and not self.play_only and not scripted
        if learning:
            if self.pending is not None:
                self.agent.rollout.rewards[self.pending] = self.reward
                self.pending = None
            stats = self.agent.finish_training()
            if stats:
                self.report(stats)
            if self.agent.full() and not self.agent.training():
                # An attempt too long for the buffer: cut it here. This state (after the last decision
                # in the buffer) stands in for the rest, valued by the same network that played it.
                last_value = self.agent.value(grid, extra, self.memory)
                if self.pause_to_train:
                    self.report(self.agent.train(last_value))
                else:
                    self.agent.start_training(last_value)
            if self.agent.training():
                # Played by the network a round is being trained from: not learned from.
                self.agent.skip()
                learning = False
        self.reward = 0.0
        explore = self.level.coach.explore_rate(x) if self.kind == "practice" and script is None else 0.0
        memory = self.memory
        action, logp, value, p, self.memory = self.agent.act(grid, extra, memory, explore)
        if scripted:
            action = script["actions"][i]
        if learning:
            # Replayed decisions stay out: PPO only learns from what the network chose itself.
            self.pending = self.agent.rollout.add(grid, extra, float(action), logp, value, memory[0].numpy(),
                                                  not self.continued, explore, self.agent.updates)
        self.continued = learning
        if script is not None and i >= script["demo_from"]:
            self.collect_demo(x, (grid, extra, memory[0].numpy(), action))
        self.trace.append((action, fingerprint))
        self.history.append((grid, extra, action, p, x))
        return action

    def collect_demo(self, x, step):
        """Experiment decisions from just before the change on; once it gets past where the
        replayed try died, they become a demo."""
        script = self.script
        if self.demo_done or script.get("out_of_sync"):
            return
        self.demo_steps.append(step)
        if x >= script["old_end"] + PASS_MARGIN:
            self.demo_done = True
            self.agent.demos.add(self.demo_steps)
            pct = lambda v: 100 * v / max(self.level.length, 1)
            if script["kind"] == "demo":
                self.say(f"Experiment: replayed its one try from here that got through {pct(script['old_end']):.1f}% "
                         f"-> learning to copy it ({self.agent.demos.n} demo decisions)")
            else:
                self.say(f"Experiment: {script['what']} at {pct(x):.1f}% got past where it died "
                         f"({pct(script['old_end']):.1f}%) -> learning to copy that ({self.agent.demos.n} demo decisions)")
        elif len(self.demo_steps) > 400:
            self.demo_done = True

    def on_death(self, info, practice):
        self.collect_progress(info["x"])
        kind = self.kind
        self.finish(DEATH_PENALTY)
        lvl = self.level
        lvl.state["attempts"] += 1
        lvl.log_death({"kind": kind, **info})
        if kind == "run" and info["x"] >= lvl.state["coach"]["best_run_x"] - HOTSPOT_RANGE:
            dump_black_box(lvl.folder / "black-box.txt", self.history, self.agent.layout, info, lvl.length)
        coach = lvl.state["coach"]
        near_spot = coach["phase"] == "practice" and coach["target_x"] - EXPLORE_BEFORE <= info["x"] <= coach["target_x"] + EXPLORE_AFTER
        near_best = info["x"] >= max(coach["best_run_x"], lvl.state["best_x"]) - 2 * HOTSPOT_RANGE
        if kind and (near_spot or near_best or self.eval_runs):
            log_moments(lvl.folder / "moments.jsonl", self.history, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": self.agent.updates, "kind": kind,
                "attempt": lvl.state["attempts"], "phase": coach["phase"], "spot_x": round(coach["target_x"], 1),
                "eval": bool(self.eval_runs), **self.start, "death": info})
        self.trace_run(info["x"], False)
        if self.play_only:
            self.say(f"Died at {info['percent']:.1f}%")
            if self.eval_runs:
                self.eval_result(info["percent"])
        elif kind:
            self.attempt_done(kind, info["x"], False)
        lvl.save()
        self.kind = None

    def on_win(self, practice):
        kind = self.kind
        self.finish(WIN_REWARD)
        lvl = self.level
        lvl.state["attempts"] += 1
        lvl.state["best_x"] = max(lvl.state["best_x"], lvl.length)
        self.trace_run(lvl.length, True)
        if self.play_only:
            self.say("LEVEL COMPLETE")
            if self.eval_runs:
                self.eval_result(100.0)
        elif kind:
            if kind == "practice":
                self.say("Reached the end in a practice run!")
            self.attempt_done(kind, lvl.length, True)
        lvl.save()
        self.kind = None

    def trace_run(self, end_x, won):
        """--trace-runs: keeps the decisions of every full run from the start in runs.jsonl."""
        if not self.trace_runs or self.kind != "run" or self.start_x > 5.0:
            return
        with (self.level.folder / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": self.agent.updates,
                                "end_x": round(end_x, 1), "percent": round(100 * end_x / max(self.level.length, 1), 2),
                                "won": won, "actions": "".join(str(a) for a, _ in self.trace)}) + "\n")

    def attempt_done(self, kind, end_x, won):
        lvl = self.level
        if EXPERIMENT_SHARE or self.script is not None:
            # Only worth keeping the furthest try per start when experiments can replay it.
            lvl.experiments.attempt_done(self.start_x, end_x, [a for a, _ in self.trace], [f for _, f in self.trace],
                                         self.script)
        if self.script is None:
            # Experiments ride on a replayed try, so they don't tell the coach how well it plays.
            lvl.coach.attempt_done(kind, end_x, won, lvl.length)
            self.reviewer.attempt_done(lvl, end_x, won)

    def eval_result(self, percent):
        """--eval: one more measured full run; after the last one, prints how it went and stops."""
        r = self.eval_results
        r.append(float(percent))
        if len(r) < self.eval_runs:
            return
        lvl, a = self.level, np.array(r)
        past = {m: float(np.mean(a >= m)) for m in range(10, 101, 10)}
        result = {"level": lvl.name, "id": lvl.id, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "training_rounds": self.agent.updates, "runs": r, "median": float(np.median(a)),
                  "mean": float(a.mean()), "best": float(a.max()), "wins": int(np.sum(a >= 100)), "past": past}
        (lvl.folder / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        self.say(f"EVAL {lvl.name}, {len(r)} full runs after {self.agent.updates} training rounds: "
                 f"median {result['median']:.1f}%, mean {result['mean']:.1f}%, best {result['best']:.1f}%, "
                 f"{result['wins']} wins")
        self.say("  got past: " + ", ".join(f"{m}% {s:.0%}" for m, s in past.items()))
        raise SystemExit(0)

    # --- helpers --------------------------------------------------------
    def collect_progress(self, x):
        """Carrots: reward for moving forward, extra for ground never reached before."""
        if self.level is None:
            return
        best = self.level.state["best_x"]
        if x - self.last_x > TELEPORT_JUMP:
            # Moved further than physics allows between two decisions: a
            # checkpoint load or teleport, not progress the AI earned.
            self.last_x = x
            return
        if x > self.last_x:
            self.reward += (x - self.last_x) / BLOCK * PROGRESS_REWARD
            new_ground = x - max(self.last_x, best)
            if new_ground > 0:
                self.reward += new_ground / BLOCK * FRONTIER_REWARD
        self.last_x = x
        self.attempt_best_x = max(self.attempt_best_x, x)
        self.level.state["best_x"] = max(best, x)

    def finish(self, terminal_reward, truncated=False):
        """Closes the attempt on its last decision actually played (even if it died partway
        through that decision's 4 ticks): the death/win reward goes there, nothing after it."""
        if self.pending is not None:
            r = self.agent.rollout
            r.rewards[self.pending] = self.reward + terminal_reward
            r.dones[self.pending] = 1.0
            # Died or won: nothing comes after. Cut short: guess the rest from its own value.
            r.bootstraps[self.pending] = r.values[self.pending] if truncated else 0.0
            self.pending = None
        self.reward = 0.0

    def end_attempt(self, truncated):
        # An attempt cut short (level reopened, disconnect): close it without a penalty.
        if self.pending is not None:
            self.finish(0.0, truncated=True)
        self.kind = None

    def report(self, stats):
        if self.rounds_left is not None:
            self.rounds_left -= 1
        lvl = self.level
        length = max(lvl.length, 1.0)  # online levels can report 0 when they open
        ex = lvl.experiments
        experiments = (f" | experiments: {ex.tried} tried, {ex.better} got further, {ex.shown} replayed as demos"
                       + (f", {ex.out_of_sync} out of sync" if ex.out_of_sync else "")) if ex.tried or ex.shown else ""
        demos = (f" | copying {stats['demos']} demo decisions ({math.exp(-stats['demo_loss']):.0%} sure of them)"
                 if stats["demos"] else "")
        self.say(f"[{lvl.name}] training round {self.agent.updates} | {self.agent.steps:,} decisions | "
                 f"best ever {100 * lvl.state['best_x'] / length:.1f}% | best full run "
                 f"{100 * lvl.state['coach']['best_run_x'] / length:.1f}% | {lvl.coach.status(length)} | "
                 f"{lvl.state['attempts']} attempts | {stats['changes']:,} presses/releases, {stats['late']:.0%} of them "
                 f"1-3 ticks late{experiments}{demos}")
        total = stats["collect_seconds"] + stats["seconds"]  # one whole round: playing, then training
        how = ("with the game waiting" if stats["paused"] else
               f"while the game played on ({stats['unused']:,} decisions meanwhile not learned from)")
        self.say(f"  timing: played {stats['decisions']:,} decisions in {stats['collect_seconds']:.1f}s, "
                 f"trained {stats['seconds']:.1f}s {how} | {stats['decisions'] / max(total, 1e-9):,.0f} "
                 f"learned decisions/s")
        if stats["paused"] and stats["seconds"] > PAUSE_WARNING:
            self.say(f"  WARNING: training took {stats['seconds']:.0f}s; the mod gives up waiting after 120s")


class Connection:
    """TCP server; the mod connects as a client each time the game starts."""

    def __init__(self, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", port))
        self.server.listen(1)
        self.port = port

    def serve(self, session):
        print(f"Waiting for GD on port {self.port}... open a level.", flush=True)
        while True:
            conn, _ = self.server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("GD connected, the AI is playing.", flush=True)
            try:
                self.handle(conn, session)
            except (ConnectionError, OSError) as e:
                print(f"GD disconnected ({e}). Waiting for it to come back...", flush=True)
                session.end_attempt(truncated=True)
            finally:
                conn.close()

    def handle(self, conn, session):
        stream = conn.makefile("rb", buffering=1 << 16)

        def read(n):
            data = stream.read(n)
            if len(data) < n:
                raise ConnectionError("connection closed")
            return data

        def unpack(fmt):
            return struct.unpack(fmt, read(struct.calcsize(fmt)))

        layout = None
        state_fmt = None
        grid_size = 0
        while True:
            tag = read(1)
            if tag == b"S":
                x, percent, *extra = unpack(state_fmt)
                action = session.on_state(x, percent, extra, read(grid_size))
                conn.sendall(bytes((action,)))
            elif tag == b"H":
                version, level_id, name_len = unpack("<3i")
                if version != PROTOCOL:
                    sys.exit(f"The mod speaks protocol {version}, this brain speaks {PROTOCOL}. Rebuild the mod.")
                name = read(name_len).decode("utf-8", "replace")
                channels, rows, cols, n_extra = unpack("<4i")
                (length,) = unpack("<f")
                layout = (channels, rows, cols, n_extra)
                state_fmt = f"<2f{n_extra}f"
                grid_size = channels * rows * cols
                session.on_hello(level_id, name, layout, length)
            elif tag == b"A":
                x, practice = unpack("<fB")
                conn.sendall(struct.pack("<BBfi", *session.on_ask(x, practice)))
            elif tag == b"F":
                (level_id,) = unpack("<i")
                session.on_switch_failed(level_id)
            elif tag == b"B":
                start_x, practice, from_checkpoint = unpack("<fBB")
                session.on_begin(start_x, practice, from_checkpoint)
            elif tag == b"D":
                percent, x, y, obj_id, obj_type, mode, steps, practice = unpack("<3f4iB")
                session.on_death(dict(percent=percent, x=x, y=y, object_id=obj_id, object_type=obj_type,
                                      mode=mode, steps=steps), practice)
            elif tag == b"W":
                steps, practice = unpack("<iB")
                session.on_win(practice)
            else:
                raise ConnectionError(f"unknown message {tag!r}")


AFTERFLOW_STAGES = ["2 ticks", "4 ticks", "8 ticks", "16 ticks (8 wave)", "24 ticks (12 wave)", "32 ticks (16 wave)",
                    "40 ticks (20 wave)", "off"]


def report_afterflow():
    """Afterflow (another Geode mod) holds back a click that would kill you to a later tick that
    survives. It hooks the same button function the AI's mod presses, so the AI's clicks go through
    it whenever it's installed and not set to 'off' (F6 in game cycles its stages)."""
    gd_mods = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash\geode\mods")
    saved = Path.home() / "AppData" / "Local" / "GeometryDash" / "geode" / "mods" / "afterflow.prsevend" / "saved.json"
    if not (gd_mods / "afterflow.prsevend.geode").exists():
        return
    try:
        stage = int(json.loads(saved.read_text(encoding="utf-8")).get("stage", -1))
    except (OSError, ValueError):
        stage = -1
    what = (f"stage {stage + 1}: {AFTERFLOW_STAGES[stage]}" if 0 <= stage < len(AFTERFLOW_STAGES)
            else "stage unknown")
    if stage == len(AFTERFLOW_STAGES) - 1:
        print(f"Afterflow is installed but set to off ({what}); press F6 in GD to turn its buffer on.")
    else:
        print(f"Afterflow buffer on ({what}, as last saved): the AI's clicks go through it. F6 in GD changes it.")


def main():
    global INIT_HOLD_BIAS, ENTROPY_BONUS, EXPERIMENT_SHARE, LATE_BIAS, ALLOWED_TICKS
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default="main", help="brain folder name (one brain learns all levels)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--play", action="store_true", help="just play, no learning")
    parser.add_argument("--fresh", action="store_true", help="back up the current brain and start from scratch")
    parser.add_argument("--review", action="store_true",
                        help="every 40 attempts, go back to a level it has beaten for a quick check (switches levels)")
    parser.add_argument("--no-review", action="store_true", help=argparse.SUPPRESS)  # the default now
    parser.add_argument("--pause-to-train", action="store_true",
                        help="stop the game while a round trains (the old way) instead of training in the background")
    parser.add_argument("--eval", type=int, default=0, metavar="N",
                        help="measure: N full runs without learning, then print median/mean progress and stop")
    parser.add_argument("--level", type=int, default=None, metavar="ID",
                        help="switch GD to this level first (main levels 1-22, or an online level opened once)")
    parser.add_argument("--minutes", type=float, default=0.0, help="stop after this many minutes (at an attempt start)")
    parser.add_argument("--rounds", type=int, default=0, help="stop after this many training rounds (at an attempt start)")
    parser.add_argument("--trace-runs", action="store_true",
                        help="save the decisions of every full run in the level's runs.jsonl")
    parser.add_argument("--memory", action="store_true",
                        help="give the brain a memory of the last moments (a brain that has one keeps it)")
    parser.add_argument("--experiments", type=float, nargs="?", const=EXPERIMENTS_ON, default=EXPERIMENT_SHARE,
                        metavar="SHARE", help="at spots it keeps dying at, replay its furthest try with one click "
                        f"moved a few ticks (SHARE of practice tries, default {EXPERIMENTS_ON})")
    # Tuning knobs for experiments.
    parser.add_argument("--init-bias", type=float, default=INIT_HOLD_BIAS, help=argparse.SUPPRESS)
    parser.add_argument("--entropy", type=float, default=ENTROPY_BONUS, help=argparse.SUPPRESS)
    # Preference against clicking 1-3 ticks late when an old one-output brain is converted (very low = only tick 0).
    parser.add_argument("--late-bias", type=float, default=LATE_BIAS, help=argparse.SUPPRESS)
    parser.add_argument("--ticks", type=int, default=ALLOWED_TICKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    INIT_HOLD_BIAS, ENTROPY_BONUS, EXPERIMENT_SHARE, LATE_BIAS = args.init_bias, args.entropy, args.experiments, args.late_bias
    ALLOWED_TICKS = max(1, min(DECISION_TICKS, args.ticks))

    torch.set_num_threads(TRAIN_THREADS)
    folder = HERE / "brains2" / args.brain
    if args.fresh and folder.exists():
        backup = folder.with_name(f"{args.brain}-backup-{time.strftime('%Y%m%d-%H%M%S')}")
        folder.rename(backup)
        print(f"Old brain backed up to {backup.name}")
    folder.mkdir(parents=True, exist_ok=True)
    if args.eval:
        print(f"Measuring: {args.eval} full runs without learning, then it stops.")
    elif args.review:
        print("Reviews on: every 40 attempts it goes back to a level it has beaten for a quick check.")
    else:
        print("It plays the level you open and never switches levels (--review to check old levels).")
    report_afterflow()
    if EXPERIMENT_SHARE and not args.eval:
        print(f"Experiments after death on: {EXPERIMENT_SHARE:.0%} of practice tries at a stubborn spot.")
    Connection(args.port).serve(Session(folder, args.play, review=args.review, eval_runs=args.eval,
                                        memory=args.memory, go_to=args.level, trace_runs=args.trace_runs,
                                        minutes=args.minutes, rounds=args.rounds,
                                        pause_to_train=args.pause_to_train))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. The brain is saved and continues next time.")
