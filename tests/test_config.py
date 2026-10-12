from dataclasses import replace

import pytest

from gvc.config import ConfigError, fingerprint, load_config


def test_defaults_use_full_preset():
    cfg = load_config()
    assert cfg.preset == "full"
    assert (cfg.caption.context, cfg.generate.init) == ("text", "chain")


def test_precedence_preset_file_overrides(tmp_path):
    path = tmp_path / "gvc.toml"
    path.write_text('preset = "literal"\n[generate]\ninit = "chain-ref"\nseed = 7\n')

    cfg = load_config(path)
    assert cfg.preset == "literal"
    assert cfg.caption.context == "none"  # from preset
    assert cfg.generate.init == "chain-ref"  # file beats preset
    assert cfg.generate.seed == 7

    cfg = load_config(path, preset="guided", overrides={"generate": {"seed": 1}})
    assert cfg.preset == "guided"
    assert cfg.caption.context == "text"  # CLI preset beats file preset
    assert cfg.generate.init == "chain-ref"  # explicit file setting still beats preset
    assert cfg.generate.seed == 1  # CLI override beats file


@pytest.mark.parametrize(
    "overrides",
    [
        {"caption": {"context": "everything"}},
        {"generate": {"init": "magic"}},
        {"generate": {"ref_method": "vibes"}},
        {"generate": {"strength": 1.5}},
        {"caption": {"window": 0}},
        {"caption": {"no_such_key": 1}},
    ],
)
def test_invalid_values_rejected(overrides):
    with pytest.raises(ConfigError):
        load_config(overrides=overrides)


def test_unknown_preset_rejected():
    with pytest.raises(ConfigError):
        load_config(preset="nope")


def test_fingerprint_ignores_runtime_fields():
    g = load_config().generate
    assert fingerprint(g) == fingerprint(replace(g, server_url="http://elsewhere:1", manage_server=False))
    assert fingerprint(g) != fingerprint(replace(g, strength=0.3))
