"""
Quick checks of Brain 2.0 that don't need the game:  python test_brain.py
(Each check prints OK or stops with what went wrong.)
"""
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
import brain2  # noqa: E402
from brain2 import (ACTIONS, DECISION_TICKS, MEMORY, Agent, Experiments, PolicyNet, add_memory,  # noqa: E402
                    changes_to_try, split_head, timeline, to_actions, widen)

LAYOUT = (30, 11, 16, 68)
OLD_LAYOUT = (30, 11, 16, 32)


def inputs(layout, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    c, r, w, e = layout
    grid = (torch.rand(n, c, r, w, generator=g) < 0.1).float()
    extra = torch.randn(n, e, generator=g)
    return grid, extra


def old_brain_state(layout):
    """A brain like the ones saved before memory and tick-exact clicks: one 'hold?' output."""
    net = PolicyNet(*layout)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("memory.")}
    state["policy.weight"] = torch.randn(1, 256) * 0.05
    state["policy.bias"] = torch.tensor([-1.3])
    state["value.weight"] = state["value.weight"][:, :256]
    return state


def old_hold_chance(state, layout, grid, extra):
    """P(hold) the way the old network computed it."""
    net = PolicyNet(*layout)
    body = {k: v for k, v in state.items() if k.startswith(("conv.", "body."))}
    net.load_state_dict(body, strict=False)
    with torch.no_grad():
        f = net.features(grid, extra)
        return torch.sigmoid(f @ state["policy.weight"][0] + state["policy.bias"][0])


def check_old_brain_upgrade():
    """Widening + tick-exact head (+ memory): the old brain must decide exactly as before."""
    state = old_brain_state(OLD_LAYOUT)
    grid, extra = inputs(OLD_LAYOUT, 64)
    before = old_hold_chance(state, OLD_LAYOUT, grid, extra)
    # New inputs non-zero and any memory: still the same decisions.
    extra_new = torch.cat([extra, torch.randn(64, LAYOUT[3] - OLD_LAYOUT[3])], 1)
    for memory in (0, MEMORY):
        net = PolicyNet(*LAYOUT, memory)
        new = split_head(widen(state, OLD_LAYOUT, LAYOUT, net))
        if memory:
            new = add_memory(new, net)
        net.load_state_dict(new)
        with torch.no_grad():
            logits, _, _ = net(grid, extra_new, torch.randn(64, memory))
        after = torch.softmax(logits, 1)[:, DECISION_TICKS:].sum(1)
        diff = float((before - after).abs().max())
        assert diff < 1e-5, f"upgraded brain decides differently (max diff {diff}, memory {memory})"
        what = "wider vision, 8 actions" + (", memory" if memory else "")
        print(f"OK  old brain upgraded ({what}) decides the same (max diff {diff:.1e})")


def check_agent_load_upgrade():
    """Agent loading an old model.pt file does the upgrade itself."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        state = old_brain_state(OLD_LAYOUT)
        torch.save({"net": state, "opt": {}, "layout": OLD_LAYOUT, "steps": 5, "updates": 1}, path)
        grid, extra = inputs(OLD_LAYOUT, 16, seed=1)
        before = old_hold_chance(state, OLD_LAYOUT, grid, extra)
        g = np.zeros((30, 11, 16), np.uint8)
        for memory in (False, True):
            agent = Agent(LAYOUT, path, memory=memory)
            assert agent.memory_size == (MEMORY if memory else 0)
            diffs = []
            for i in range(16):
                g[:] = (grid[i] * 255).byte().numpy()
                e = np.concatenate([extra[i].numpy(), np.random.randn(36).astype(np.float32)])
                _, _, _, p, _ = agent.act(g, e, torch.randn(1, agent.memory_size))
                diffs.append(abs(p - float(before[i])))
            assert max(diffs) < 1e-5, f"loaded brain decides differently (max diff {max(diffs)}, memory {memory})"
        # A brain that has a memory keeps it even without --memory.
        agent.save()
        assert Agent(LAYOUT, path).memory_size == MEMORY, "a brain with a memory lost it"
    print("OK  Agent upgrades an old model.pt on load (with and without memory)")


def check_sequence_matches_steps():
    """Training's sequence pass gives the same outputs as playing one decision at a time."""
    torch.manual_seed(0)
    net = PolicyNet(*LAYOUT, MEMORY)
    b, t = 3, 10
    grid, extra = inputs(LAYOUT, b * t, seed=2)
    grid, extra = grid.view(b, t, *grid.shape[1:]), extra.view(b, t, -1)
    start = torch.randn(b, MEMORY)
    start[1] = 0  # a new attempt starts with an empty memory
    with torch.no_grad():
        seq_logits, seq_values = net.sequence(grid, extra, start)
        step_logits, step_values = [], []
        for i in range(b):
            h = start[i:i + 1]
            for j in range(t):
                logits, value, h = net(grid[i, j:j + 1], extra[i, j:j + 1], h)
                step_logits.append(logits)
                step_values.append(value)
    diff = max(float((seq_logits - torch.cat(step_logits)).abs().max()),
               float((seq_values - torch.cat(step_values)).abs().max()))
    assert diff < 1e-5, f"sequence and step outputs differ ({diff})"
    print(f"OK  memory over a sequence matches decision by decision (max diff {diff:.1e})")


def check_sequences_cut():
    """Training runs never cross an attempt start, cover every decision once and are at most SEQUENCE long."""
    fresh = np.zeros(100, bool)
    fresh[[0, 7, 40, 41, 99]] = True
    rows, mask = brain2.sequences(fresh, brain2.SEQUENCE)
    real = rows[mask > 0]
    assert sorted(real.tolist()) == list(range(100)), "decisions missing or repeated"
    for row, m in zip(rows, mask):
        r = row[m > 0]
        assert all(b == a + 1 for a, b in zip(r, r[1:])), "a run isn't decisions in a row"
        assert not fresh[r[1:]].any(), "a run crosses the start of an attempt"
    assert mask.shape[1] == brain2.SEQUENCE
    single, single_mask = brain2.sequences(fresh, 1)
    assert single.flatten().tolist() == list(range(100)) and single_mask.all(), "without memory: one decision each"
    print(f"OK  {len(rows)} training runs, none crossing an attempt start")


def check_timeline_round_trip():
    random.seed(1)
    for _ in range(200):
        actions = [random.randrange(ACTIONS) for _ in range(random.randint(1, 60))]
        ticks = timeline(actions)
        back = to_actions(ticks)
        assert back is not None and timeline(back) == ticks, "clicks don't round-trip"
    assert to_actions([True, False, False, False]) is None, "two changes in one window must be refused"
    print("OK  clicks per tick <-> decisions round-trip")


def check_changes_to_try():
    random.seed(2)
    # Cube-like run: hold for a few ticks every now and then.
    ticks = []
    while len(ticks) < 800:
        ticks += [False] * random.randint(20, 60) + [True] * random.randint(8, 30)
    ticks = ticks[:800]
    actions = to_actions(ticks)
    assert actions is not None
    options = list(changes_to_try(actions))
    assert options, "nothing to try"
    seen = set()
    for what, new, first, last in options:
        assert len(new) == len(ticks)
        changed = [i for i in range(len(ticks)) if new[i] != ticks[i]]
        assert changed and first <= changed[0] and changed[-1] <= last, f"{what}: changes outside {first}..{last}"
        assert first >= len(ticks) - brain2.LOOK_BACK * 240 - 4, f"{what}: too far back"
        seen.add(what.split()[0])
        script = to_actions(new[:(last // DECISION_TICKS + 1) * DECISION_TICKS])
        if script is not None:
            assert timeline(script) == new[:len(script) * DECISION_TICKS]
    assert {"click", "release", "no"} <= seen, f"missing kinds of changes: {seen}"
    print(f"OK  {len(options)} changes to try after a death, all within the last {brain2.LOOK_BACK}s")


def check_experiments_records():
    with tempfile.TemporaryDirectory() as tmp:
        ex = Experiments(Path(tmp), print)
        ticks = [False] * 100 + [True] * 12 + [False] * 60 + [True] * 10 + [False] * 18
        actions = to_actions(ticks)
        prints = [(float(i), 0.0, 0.0) for i in range(len(actions))]
        prints = [(120.0 + 15.5 * i, 0.0, 0.0) for i in range(len(actions))]  # reaches ~880 by the death at 900
        ex.attempt_done(120.0, 900.0, actions, prints, None)
        assert ex.pick(120.0, 2000.0) is None, "picked a try that didn't get far enough"
        # Died near the spot: replay with a change before the death.
        script = ex.pick(120.0, 870.0)
        assert script is not None and script["kind"] == "change" and script["old_end"] == 900.0
        assert script["actions"][:script["check_until"]] == actions[:script["check_until"]], "replay changed too early"
        assert np.allclose(script["prints"][5], prints[5])
        # Saved and loaded again, it carries on with the next change.
        ex.attempt_done(120.0, 850.0, actions, prints, script)
        again = Experiments(Path(tmp), print).pick(120.0, 870.0)
        assert again is not None and again["kind"] == "change" and again["what"] != script["what"]
        # Got through an earlier spot: replayed (once) through it as a demo.
        demo = ex.pick(120.0, 500.0)
        assert demo["kind"] == "demo" and demo["actions"] == actions[:len(demo["actions"])]
        assert prints[len(demo["actions"]) - 1][0] >= 500.0 + brain2.PASS_MARGIN
        assert prints[demo["demo_from"]][0] >= 500.0 - brain2.DEMO_LEAD
        assert ex.pick(120.0, 500.0)["kind"] == "change", "the demo replay should happen only once"
        # A further try from the same start replaces it.
        ex.attempt_done(120.0, 1200.0, actions + [0] * 10, prints + [(0.0, 0.0, 0.0)] * 10, None)
        assert ex.records[ex.key(120.0)]["end"] == 1200.0 and ex.records[ex.key(120.0)]["next"] == 0
    print("OK  experiments keep the furthest try per start and step through changes")


def check_training_round():
    for memory in (False, True):
        training_round(memory)


def training_round(with_memory):
    """One PPO round (with or without memory) with demos runs and changes the network."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(LAYOUT, Path(tmp) / "model.pt", memory=with_memory)
        n = 300
        rng = np.random.default_rng(0)
        r = agent.rollout
        memory = agent.empty_memory()
        for i in range(n):
            g = (rng.random((30, 11, 16)) < 0.1).astype(np.uint8) * 255
            e = rng.standard_normal(68).astype(np.float32)
            a, logp, v, _, new_memory = agent.act(g, e, memory)
            r.add(g, e, float(a), logp, v, memory[0].numpy(), i % 50 == 0, 0.0, agent.updates)
            memory = new_memory if i % 50 != 49 else agent.empty_memory()
            r.rewards[i] = rng.random()
            r.dones[i] = float(i % 50 == 49)
        agent.demos.add([(np.zeros((30, 11, 16), np.uint8), np.zeros(68, np.float32),
                          np.zeros(agent.memory_size, np.float32), 5)] * 20)
        before = {k: v.clone() for k, v in agent.learner.state_dict().items()}
        stats = agent._train(r.take(), 0.0, agent.demos.snapshot())
        assert all(np.isfinite(v) for v in stats.values()), stats
        assert stats["demos"] == 20 and stats["demo_loss"] > 0
        moved = [k for k, v in agent.learner.state_dict().items() if not torch.equal(v, before[k])]
        assert any(k.startswith("memory.") for k in moved) == with_memory, "memory didn't train"
        assert moved, "nothing trained"
        assert (Path(tmp) / "model.pt").exists() and (Path(tmp) / "demos.npz").exists()
        reloaded = Agent(LAYOUT, Path(tmp) / "model.pt")
        assert reloaded.demos.n == 20, "demos weren't kept"
    print(f"OK  a training round {'with' if with_memory else 'without'} memory, with demos "
          f"(entropy {stats['entropy']:.2f}, demo loss {stats['demo_loss']:.2f})")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_old_brain_upgrade()
    check_agent_load_upgrade()
    check_sequence_matches_steps()
    check_sequences_cut()
    check_timeline_round_trip()
    check_changes_to_try()
    check_experiments_records()
    check_training_round()
    print("All checks passed.")
