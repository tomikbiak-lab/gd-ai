"""
Checks of the PPO fix (no game needed):  python test_ppo.py

- the log-chance of a move mixes random moves the same way when playing and when training
  (ratio 1 with unchanged weights, explore = 0 is the plain network, no NaN at the edges);
- rewards and look-ahead values: a death or win ends the attempt on its last decision with
  nothing after it, an attempt cut short guesses the rest from its own value, a cut of a full
  buffer looks ahead to the next state valued by the same network, and nothing flows from one
  attempt into another;
- a round only ever trains on decisions of one network version, and the buffer starts empty after;
- with no random moves, a round gives exactly the same weights as PPO did before the fix
  (same start weights, optimizer state, decisions and minibatch order).
"""
import contextlib
import copy
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
import brain2  # noqa: E402
from brain2 import ACTIONS, Agent, PolicyNet, advantages, mixture_logp  # noqa: E402

LAYOUT = (30, 11, 16, 68)
LENGTH = 3000.0
GAMMA = math.exp(-brain2.DECISION_TICKS / (240.0 * brain2.HORIZON_SECONDS))


def state(x, seed):
    """One 'S' message's worth of input at position x."""
    rng = np.random.default_rng(seed)
    grid = ((rng.random(30 * 11 * 16) < 0.1) * 255).astype(np.uint8).tobytes()
    extra = rng.standard_normal(68).astype(np.float32).tolist()
    return x, 100 * x / LENGTH, extra, grid


def death(x):
    return dict(percent=100 * x / LENGTH, x=x, y=105.0, object_id=8, object_type=2, mode=0, steps=0)


def session(tmp, pause_to_train=False):
    s = brain2.Session(Path(tmp), play_only=False, review=False, pause_to_train=pause_to_train)
    s.log = []
    s.say = s.log.append
    s.on_hello(9999, "Test", LAYOUT, LENGTH)
    return s


def check_mixture():
    torch.manual_seed(0)
    logits = torch.randn(200, ACTIONS) * 3
    actions = torch.randint(ACTIONS, (200,))
    for eps in (0.0, 1e-6, 0.3, 1.0):
        x = logits.clone().requires_grad_(True)
        own = torch.distributions.Categorical(logits=x).log_prob(actions)
        mixed = mixture_logp(own, torch.full((200,), eps))
        assert torch.isfinite(mixed).all(), f"explore {eps}: not finite"
        mixed.sum().backward()
        assert torch.isfinite(x.grad).all(), f"explore {eps}: gradient not finite"
        p = torch.softmax(logits, 1).gather(1, actions[:, None])[:, 0].double()
        want = eps / ACTIONS + (1 - eps) * p
        assert torch.allclose(mixed.detach().double().exp(), want, rtol=1e-5, atol=1e-9), f"explore {eps}: wrong chance"
        if eps == 0.0:
            assert torch.equal(mixed.detach(), own.detach()), "explore 0 must be exactly the network's own log-chance"
        if eps == 1.0:
            assert torch.allclose(mixed.detach(), torch.full((200,), -math.log(ACTIONS))), "explore 1 must be log(1/8)"
            assert float(x.grad.abs().max()) == 0.0, "explore 1: the network can't change the chance"
    print("OK  mixed log-chance: explore 0 = network exactly, 1e-6 / 0.3 right, 1 = log(1/8), no NaN in values or gradients")


def check_ratio_one():
    """Unchanged weights: training scores every played move exactly as playing did."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(LAYOUT, Path(tmp) / "model.pt")
        rng = np.random.default_rng(1)
        worst = {}
        for eps in (0.0, 1e-6, 0.3, 1.0):
            grids, extras, actions, logps = [], [], [], []
            for _ in range(300):
                g = ((rng.random((30, 11, 16)) < 0.1) * 255).astype(np.uint8)
                e = rng.standard_normal(68).astype(np.float32)
                a, logp, _, _, _ = agent.act(g, e, agent.empty_memory(), eps)
                grids.append(g), extras.append(e), actions.append(a), logps.append(logp)
            with torch.no_grad():
                grid = torch.from_numpy(np.stack(grids)).float().div_(255.0)[:, None]
                logits, _ = agent.learner.sequence(grid, torch.from_numpy(np.stack(extras))[:, None], torch.zeros(300, 0))
                own = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(actions))
                new = mixture_logp(own, torch.full((300,), eps, dtype=torch.float32))
            ratio = torch.exp(new - torch.tensor(logps, dtype=torch.float32))
            worst[eps] = float((ratio - 1).abs().max())
            assert worst[eps] < 1e-5, f"explore {eps}: ratio off by {worst[eps]}"
    print("OK  unchanged weights give ratio 1: " + ", ".join(f"explore {e}: max |ratio-1| {w:.1e}" for e, w in worst.items()))


def check_tick_mask():
    """--ticks 1 (the 'tick 0 only' control): no late action ever, random moves included."""
    saved = brain2.ALLOWED_TICKS
    brain2.ALLOWED_TICKS = 1
    try:
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(LAYOUT, Path(tmp) / "model.pt")
            rng = np.random.default_rng(5)
            seen = set()
            for eps in (0.0, 0.3, 1.0):
                for _ in range(200):
                    g = ((rng.random((30, 11, 16)) < 0.1) * 255).astype(np.uint8)
                    a, logp, _, _, _ = agent.act(g, rng.standard_normal(68).astype(np.float32), agent.empty_memory(), eps)
                    seen.add(a)
                    if eps == 1.0:
                        assert abs(logp - math.log(0.5)) < 1e-6, "random moves are 1 of 2 allowed actions"
            assert seen <= {0, brain2.DECISION_TICKS}, f"late actions picked with --ticks 1: {seen}"
    finally:
        brain2.ALLOWED_TICKS = saved
    print("OK  --ticks 1: only tick-0 actions, also among random moves")


def check_gae():
    rng = np.random.default_rng(2)
    n = 12
    b = {"rewards": rng.standard_normal(n).astype(np.float32), "values": rng.standard_normal(n).astype(np.float32),
         "dones": np.zeros(n, np.float32), "bootstraps": np.zeros(n, np.float32)}
    b["dones"][[3, 7]] = 1.0          # attempt 1 dies at 3, attempt 2 is cut short at 7, attempt 3 goes on
    b["bootstraps"][7] = b["values"][7]
    adv = advantages(b, 0.5, GAMMA, 0.95)
    # A death looks ahead to nothing, a cut-short attempt to its own value, the round's end to last_value.
    assert np.isclose(adv[3], b["rewards"][3] - b["values"][3])
    assert np.isclose(adv[7], b["rewards"][7] + GAMMA * b["values"][7] - b["values"][7])
    assert np.isclose(adv[11], b["rewards"][11] + GAMMA * 0.5 - b["values"][11])
    # Nothing flows across attempts: change attempt 2 and 3 completely, attempt 1 stays the same.
    other = {k: v.copy() for k, v in b.items()}
    other["rewards"][4:] += 100.0
    other["values"][4:] -= 50.0
    assert np.array_equal(advantages(other, 9.0, GAMMA, 0.95)[:4], adv[:4]), "advantage leaked into the attempt before"
    print("OK  GAE: death/win look ahead to 0, cut short to own value, round end to the next state; no leaking")


def check_rewards_on_last_decision():
    with tempfile.TemporaryDirectory() as tmp:
        s = session(tmp)
        r = s.agent.rollout
        s.on_ask(0.0, 0)
        s.on_begin(0.0, 0, 0)
        for k, x in enumerate([0.0, 5.0, 10.0]):
            s.on_state(*state(x, k))
        s.on_death(death(12.0), 0)  # died partway through the third decision's 4 ticks
        assert r.n == 3, "a decision was added after the death"
        assert list(r.dones[:3]) == [0, 0, 1] and r.bootstraps[2] == 0.0
        carrots = brain2.PROGRESS_REWARD + brain2.FRONTIER_REWARD
        assert np.allclose(r.rewards[:3], [5 / 30 * carrots, 5 / 30 * carrots, 2 / 30 * carrots + brain2.DEATH_PENALTY])
        # An attempt cut short (a new one asked for without a death): its own value is the look-ahead.
        s.on_ask(0.0, 0)
        s.on_begin(0.0, 0, 0)
        for k, x in enumerate([0.0, 5.0]):
            s.on_state(*state(x, 10 + k))
        s.on_ask(0.0, 0)
        assert r.n == 5 and r.dones[4] == 1.0 and r.bootstraps[4] == r.values[4]
        # Ground already reached before: plain progress, no frontier carrots; no death penalty at the end.
        assert np.isclose(r.rewards[3], 5 / 30 * brain2.PROGRESS_REWARD) and r.rewards[4] == 0.0
    print("OK  death reward on the last decision actually played (even mid-window), cut-short attempt bootstraps itself")


def check_one_version_per_round():
    saved_rollout = brain2.ROLLOUT
    brain2.ROLLOUT = 16  # buffer holds 32
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s = session(tmp, pause_to_train=True)
            agent, r = s.agent, s.agent.rollout
            seen_values = []
            real_value, real_train = agent.value, agent.train

            def value(grid, extra, memory):
                v = real_value(grid, extra, memory)
                seen_values.append((grid.copy(), v))
                return v

            cuts = []

            def train(last_value=0.0):
                cuts.append({"last_value": last_value, "n": r.n, "versions": set(r.versions[:r.n].tolist()),
                             "last_done": r.dones[r.n - 1], "last_grid": r.grids[r.n - 1].copy(),
                             "updates": agent.updates})
                return real_train(last_value)

            agent.value, agent.train = value, train
            seed = 0
            # Short attempts: once 16 are in, the round trains at the next start of an attempt.
            for attempt in range(3):
                s.on_ask(0.0, 0)
                s.on_begin(0.0, 0, 0)
                for k in range(6):
                    s.on_state(*state(5.0 * k, seed))
                    seed += 1
                s.on_death(death(27.0), 0)
                assert not cuts, "trained in the middle of things"
            s.on_ask(0.0, 0)
            assert len(cuts) == 1 and cuts[0]["n"] == 18 and cuts[0]["versions"] == {0} and cuts[0]["last_done"] == 1
            assert cuts[0]["last_value"] == 0.0 and r.n == 0 and agent.updates == 1, "round at an attempt start"
            # A very long attempt: the buffer (32) fills up and gets cut mid-attempt.
            s.on_begin(0.0, 0, 0)
            for k in range(40):
                x = 5.0 * k
                msg = state(x, seed)
                seed += 1
                s.on_state(*msg)
                if len(cuts) == 2 and "grid" not in cuts[1]:
                    cuts[1]["grid"] = np.frombuffer(msg[3], np.uint8).reshape(30, 11, 16)
            cut = cuts[1]
            assert cut["n"] == 32 and cut["versions"] == {1} and cut["last_done"] == 0, "cut mid-attempt"
            # Look-ahead = value of the state after the last buffered decision, from the same network.
            assert len(seen_values) == 1, "the look-ahead value is only needed at the cut"
            grid_seen, value_seen = seen_values[0]
            assert np.array_equal(grid_seen, cut["grid"]), "bootstrap from the wrong state"
            assert not np.array_equal(grid_seen, cut["last_grid"]), "bootstrap from the state before the last decision"
            assert cut["last_value"] == value_seen and cut["updates"] == 1
            assert set(r.versions[:r.n].tolist()) == {2}, "the rest of the attempt belongs to the next round"
            # A round with an older decision in it is refused.
            r.versions[0] = 1
            try:
                real_train()
            except RuntimeError:
                pass
            else:
                raise AssertionError("a round with decisions from an older network was not refused")
    finally:
        brain2.ROLLOUT = saved_rollout
    print("OK  each round = one network version; trains at an attempt start, a full buffer is cut with the right look-ahead "
          "(--pause-to-train)")


def check_background_round():
    """Default: a round trains in the background, the game never waits, and what's played meanwhile
    (by the network the round started from) isn't learned from, so rounds stay one version each."""
    import threading
    saved_rollout = brain2.ROLLOUT
    brain2.ROLLOUT = 16
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s = session(tmp)
            agent, r = s.agent, s.agent.rollout
            gate = threading.Event()
            real_train = agent._train

            def slow_train(*args):
                gate.wait(30)
                return real_train(*args)

            agent._train = slow_train
            seed = 0
            for attempt in range(3):
                s.on_ask(0.0, 0)
                s.on_begin(0.0, 0, 0)
                for k in range(6):
                    s.on_state(*state(5.0 * k, seed))
                    seed += 1
                s.on_death(death(27.0), 0)
            started = time.time()
            s.on_ask(0.0, 0)  # 18 decisions in: the round starts, but in the background
            assert time.time() - started < 1.0 and agent.training(), "the game waited for the round"
            assert r.n == 0
            s.on_begin(0.0, 0, 0)
            for k in range(5):
                s.on_state(*state(5.0 * k, seed))
                seed += 1
            assert r.n == 0 and agent.unused == 5, "decisions played during the round were kept"
            gate.set()
            agent.job.result(timeout=30)
            s.on_state(*state(30.0, seed))  # the round is done: the next decision is the new version's
            assert agent.updates == 1 and r.n == 1 and r.versions[0] == 1 and r.fresh[0]
            assert any("not learned from" in line for line in s.log), "the round wasn't reported"
    finally:
        brain2.ROLLOUT = saved_rollout
    print("OK  default: rounds train in the background (no waiting); moves played meanwhile aren't learned from")


@contextlib.contextmanager
def fixed_order(perms):
    """Makes torch.randperm hand out these minibatch orders, so two implementations see the same."""
    real = torch.randperm
    queue = list(perms)

    def randperm(n, *args, **kwargs):
        p = queue.pop(0)
        assert len(p) == n
        return p.clone()

    torch.randperm = randperm
    try:
        yield
    finally:
        torch.randperm = real
    assert not queue, "minibatch orders left over"


def reference_round(net, opt, b, last_value):
    """PPO as it was before the fix (no memory, no random moves): the yardstick for the regression."""
    from brain2 import CLIP, ENTROPY_BONUS, EPOCHS, GAE_LAMBDA, MAX_GRAD_NORM, MINIBATCH, VALUE_COEF
    n = len(b["actions"])
    adv = np.zeros(n, np.float32)
    running = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else b["values"][t + 1]
        alive = 1.0 - b["dones"][t]
        delta = b["rewards"][t] + GAMMA * next_value * alive - b["values"][t]
        running = delta + GAMMA * GAE_LAMBDA * alive * running
        adv[t] = running
    returns = adv + b["values"]
    grids = torch.from_numpy(b["grids"]).float().div_(255.0)
    extras = torch.from_numpy(b["extras"])
    actions = torch.from_numpy(b["actions"]).long()
    old_logps = torch.from_numpy(b["logps"])
    adv_t = torch.from_numpy(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    returns_t = torch.from_numpy(returns)
    for _ in range(EPOCHS):
        order = torch.randperm(n)
        for start in range(0, n, MINIBATCH):
            idx = order[start:start + MINIBATCH]
            logits, values, _ = net(grids[idx], extras[idx], torch.zeros(len(idx), 0))
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(actions[idx]) - old_logps[idx])
            a = adv_t[idx]
            policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
            value_loss = 0.5 * (values - returns_t[idx]).pow(2).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_BONUS * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()


def batch_from(agent, rng, n=2048):
    """Decisions as if played by `agent` with no random moves, attempts of 150 decisions ending in a death."""
    b = {"grids": ((rng.random((n, 30, 11, 16)) < 0.08) * 255).astype(np.uint8),
         "extras": rng.standard_normal((n, 68)).astype(np.float32)}
    actions, logps, values = [], [], []
    for i in range(n):
        a, logp, v, _, _ = agent.act(b["grids"][i], b["extras"][i], agent.empty_memory())
        actions.append(a), logps.append(logp), values.append(v)
    b |= {"actions": np.array(actions, np.float32), "logps": np.array(logps, np.float32),
          "values": np.array(values, np.float32), "rewards": rng.standard_normal(n).astype(np.float32),
          "dones": (np.arange(n) % 150 == 149).astype(np.float32), "bootstraps": np.zeros(n, np.float32),
          "explores": np.zeros(n, np.float32), "versions": np.full(n, agent.updates),
          "memories": np.zeros((n, 0), np.float32), "fresh": np.arange(n) % 150 == 0}
    return b


def check_regression():
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(LAYOUT, Path(tmp) / "model.pt")
        ref = PolicyNet(*LAYOUT)
        ref.load_state_dict(agent.learner.state_dict())
        ref_opt = torch.optim.Adam(ref.parameters(), lr=brain2.LEARNING_RATE, eps=1e-5)
        ref_opt.load_state_dict(copy.deepcopy(agent.opt.state_dict()))
        rng = np.random.default_rng(3)
        g = torch.Generator().manual_seed(4)
        diffs = []
        for _ in range(2):  # the second round starts with Adam's running averages from the first
            b = batch_from(agent, rng)
            perms = [torch.randperm(len(b["actions"]), generator=g) for _ in range(brain2.EPOCHS)]
            with fixed_order(perms):
                agent._train({k: v.copy() for k, v in b.items()}, 0.0, None)
            with fixed_order(perms):
                reference_round(ref, ref_opt, {k: v.copy() for k, v in b.items()}, 0.0)
            agent.net.load_state_dict(agent.learner.state_dict())
            mine, theirs = agent.learner.state_dict(), ref.state_dict()
            diffs.append(max(float((mine[k] - theirs[k]).abs().max()) for k in mine))
        assert max(diffs) <= 1e-7, f"weights differ from PPO before the fix: {diffs}"
    print(f"OK  no random moves: 2 rounds give the same weights as PPO before the fix (max diff {max(diffs):.1e})")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_mixture()
    check_ratio_one()
    check_tick_mask()
    check_gae()
    check_rewards_on_last_decision()
    check_one_version_per_round()
    check_background_round()
    check_regression()
    print("All PPO checks passed.")
