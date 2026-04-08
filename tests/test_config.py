from config import VDAConfig


def test_default_config_matches_note():
    cfg = VDAConfig()
    assert cfg.K == 3
    assert cfg.openai_model == "gpt-4o-mini"
    assert cfg.temperatures == [0.0, 0.7, 1.0]
    assert cfg.R == 15
    assert cfg.eta_0 == 0.3
    assert cfg.eta_p == 0.7
    assert cfg.L == 100
    assert cfg.tau == 1e-10
    assert cfg.delta == 1e-4
    assert cfg.eps == 1e-6
    assert cfg.c_t == 0.5
    assert cfg.prompt_max_tokens_per_prior_step == 500


def test_learning_rate_schedule():
    cfg = VDAConfig()
    # r is 0-indexed; formula is eta_0 / (r+1)^p
    assert abs(cfg.lr(0) - 0.3) < 1e-12                # 0.3 / 1^0.7
    assert abs(cfg.lr(1) - 0.3 / (2 ** 0.7)) < 1e-12
