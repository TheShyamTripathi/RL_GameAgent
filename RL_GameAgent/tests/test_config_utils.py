"""Torch-free tests for configuration and the run registry."""
import pytest

import configs
import utils


def test_variant_flags_are_applied():
    cfg = configs.make_config("CartPole-v1", "dqn")
    assert (cfg.double_dqn, cfg.dueling, cfg.prioritized, cfg.n_step) == (False, False, False, 1)
    cfg = configs.make_config("CartPole-v1", "full")
    assert (cfg.double_dqn, cfg.dueling, cfg.prioritized, cfg.n_step) == (True, True, True, 3)


def test_overrides_and_validation():
    cfg = configs.make_config("Acrobot-v1", "ddqn", episodes=12, lr=None, hidden=[16, 16])
    assert cfg.episodes == 12 and cfg.lr == configs.GAMES["Acrobot-v1"].lr and cfg.hidden == (16, 16)
    with pytest.raises(ValueError):
        configs.make_config("Nope-v0")
    with pytest.raises(ValueError):
        configs.make_config("CartPole-v1", "nope")
    with pytest.raises(ValueError):
        configs.make_config("CartPole-v1", "full", not_an_option=1)


def test_config_roundtrip():
    cfg = configs.make_config("CartPole-v1", "full", seed=7)
    again = configs.TrainConfig.from_dict(cfg.to_dict())
    assert again == cfg and isinstance(again.hidden, tuple)


def test_run_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(configs, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(configs, "MODELS_DIR", str(tmp_path))
    for seed in (1, 2):
        cfg = configs.make_config("CartPole-v1", "ddqn", seed=seed)
        utils.write_meta(cfg.run_name, {"run_name": cfg.run_name, "game": cfg.game, "variant": cfg.variant,
                                        "seed": seed, "created": f"2026-01-0{seed}T10:00:00", "status": "finished"})
    runs = utils.list_runs("CartPole-v1")
    assert [r["seed"] for r in runs] == [2, 1]               # newest first
    assert utils.list_runs("Acrobot-v1") == []
    assert "DDQN" in utils.run_label(runs[0])


def test_summarize_scores():
    s = utils.summarize_scores([1.0, 3.0])
    assert s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0
