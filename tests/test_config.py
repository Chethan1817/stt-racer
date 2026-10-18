import pytest

from stt_fusion.config import Settings


def test_defaults_follow_the_brief():
    s = Settings.from_env({})
    assert s.stt_engines == ("deepgram", "openai", "groq")
    assert s.fusion_skip_if_unanimous is False  # the LLM pass runs on every turn
    assert s.race_grace_s == 2.0
    assert "English" in s.openai_stt_prompt


def test_env_overrides_and_parsing():
    s = Settings.from_env(
        {
            "STT_ENGINES": "groq, Deepgram",
            "FUSION_SKIP_IF_UNANIMOUS": "yes",
            "RACE_GRACE_S": "",
            "FUSION_TEMPERATURE": "",
            "FUSION_KEYTERMS": "Toit, LiveKit,,",
            "LOG_DIR": "",
        }
    )
    assert s.stt_engines == ("groq", "deepgram")
    assert s.fusion_skip_if_unanimous is True
    assert s.race_grace_s is None  # empty means "wait for every engine"
    assert s.fusion_temperature is None
    assert s.fusion_keyterms == ("Toit", "LiveKit")
    assert s.log_dir is None


@pytest.mark.parametrize(
    "env",
    [{"STT_ENGINES": "deepgram,azure"}, {"STT_ENGINES": ","}, {"TURN_DETECTION": "magic"}],
)
def test_bad_values_fail_fast(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)
