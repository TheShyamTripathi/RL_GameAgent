"""End-to-end smoke tests (skipped automatically when PyTorch is not installed)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

import configs
import utils
from agent import DQNAgent
from train import Trainer


@pytest.fixture()
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(configs, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(configs, "MODELS_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("variant", ["dqn", "full"])
def test_training_smoke(tmp_dirs, variant):
    cfg = configs.make_config("CartPole-v1", variant, episodes=4, warmup_steps=32, batch_size=16,
                              eval_every=2, eval_episodes=1, final_eval_episodes=2, early_stop=False)
    result = Trainer(cfg).run()
    assert result["episodes_run"] == 4 and result["status"] == "finished"
    assert utils.load_run_log(cfg.run_name) is not None
    assert utils.list_runs("CartPole-v1")[0]["summary"]["episodes_run"] == 4

    agent = DQNAgent.load(result["best_model"])
    assert agent.act(np.zeros(4, dtype=np.float32), greedy=True) in (0, 1)


def test_learning_step_updates_priorities(tmp_dirs):
    cfg = configs.make_config("CartPole-v1", "full", warmup_steps=0, batch_size=8, n_step=2)
    agent = DQNAgent(4, 2, cfg)
    for t in range(40):
        agent.observe(np.random.randn(4).astype("float32"), t % 2, 1.0,
                      np.random.randn(4).astype("float32"), False, t % 20 == 19)
    before = agent.buffer.tree.total
    stats = agent.learn()
    assert stats is not None and np.isfinite(stats["loss"])
    assert agent.buffer.tree.total != before            # priorities were rewritten
