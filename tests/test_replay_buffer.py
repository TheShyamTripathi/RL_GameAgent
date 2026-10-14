"""Torch-free tests for the replay components."""
import numpy as np
import pytest

from replay_buffer import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer, SumTree


def test_sumtree_total_and_proportional_sampling():
    tree = SumTree(5)
    tree.set(np.arange(5), [1.0, 2.0, 3.0, 4.0, 10.0])
    assert tree.total == pytest.approx(20.0)
    np.random.seed(0)
    draws = tree.find(np.random.uniform(0, tree.total, size=20_000))
    freq = np.bincount(draws, minlength=5) / len(draws)
    assert freq == pytest.approx(np.array([1, 2, 3, 4, 10]) / 20, abs=0.02)


def test_sumtree_update_overwrites():
    tree = SumTree(8)
    tree.set(np.arange(8), np.ones(8))
    tree.set([3, 3], [5.0, 7.0])
    assert tree.total == pytest.approx(14.0)


def test_nstep_return_matches_manual_computation():
    acc = NStepAccumulator(n_step=3, gamma=0.5)
    out = []
    for t in range(4):
        out += acc.push(np.array([t]), 0, 1.0, np.array([t + 1]), False, False)
    assert len(out) == 2                              # windows starting at t=0 and t=1
    s0, a0, ret, s_n, term, disc = out[0]
    assert ret == pytest.approx(1 + 0.5 + 0.25)
    assert disc == pytest.approx(0.5 ** 3)
    assert s0[0] == 0 and s_n[0] == 3 and term == 0.0


def test_nstep_flushes_short_windows_at_episode_end():
    acc = NStepAccumulator(n_step=3, gamma=1.0)
    out = acc.push(0, 0, 1.0, 1, False, False)
    out += acc.push(1, 0, 1.0, 2, True, True)         # episode of length 2
    assert len(out) == 2
    assert [o[2] for o in out] == [2.0, 1.0]          # returns of the two partial windows
    assert [o[5] for o in out] == [1.0, 1.0]
    assert all(o[4] == 1.0 for o in out)              # terminal flag propagated
    assert len(acc.window) == 0


def test_truncation_still_bootstraps():
    acc = NStepAccumulator(n_step=1, gamma=0.99)
    (_, _, _, _, term, disc), = acc.push(0, 0, 1.0, 1, False, True)   # time limit, not terminal
    assert term == 0.0 and disc == pytest.approx(0.99)


def test_uniform_buffer_wraps_around():
    buf = ReplayBuffer(capacity=4, state_dim=2)
    for t in range(10):
        buf.add(np.full(2, t), t % 2, float(t), np.full(2, t + 1), False, False)
    assert len(buf) == 4 and buf.ptr == 2
    batch = buf.sample(8)
    assert batch["states"].shape == (8, 2) and batch["weights"].shape == (8, 1)
    assert set(batch["rewards"].ravel()) <= {6.0, 7.0, 8.0, 9.0}


def test_per_samples_high_priority_more_often_and_weights_are_normalised():
    np.random.seed(1)
    buf = PrioritizedReplayBuffer(capacity=64, state_dim=1, alpha=1.0)
    for t in range(64):
        buf.add(np.array([t]), 0, 0.0, np.array([t]), False, False)
    buf.update_priorities(np.arange(64), np.where(np.arange(64) == 7, 100.0, 0.01))
    idx = np.concatenate([buf.sample(32, beta=0.4)["indices"] for _ in range(50)])
    assert (idx == 7).mean() > 0.5
    w = buf.sample(32, beta=1.0)["weights"]
    assert w.max() == pytest.approx(1.0) and w.min() > 0.0


def test_per_new_transitions_get_max_priority():
    buf = PrioritizedReplayBuffer(capacity=8, state_dim=1)
    buf.add(np.zeros(1), 0, 0.0, np.zeros(1), False, False)
    buf.update_priorities([0], [50.0])
    buf.add(np.zeros(1), 0, 0.0, np.zeros(1), False, False)
    assert buf.tree.get([1])[0] == pytest.approx(buf.max_priority)
