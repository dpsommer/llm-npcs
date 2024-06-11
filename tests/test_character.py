import pytest

import npcs.memory.character
from npcs.memory.character import Character


@pytest.fixture(autouse=True)
def patch_conf_dir(monkeypatch):
    monkeypatch.setattr(npcs.memory.character, "CHARACTER_CONFIG_PATH", pytest.DATA_DIR)


@pytest.fixture
def character():
    return Character(
        name="Hatsune Miku",
        age=16,
        race="vocaloid",
        gender="female",
        location="cyberspace",
        backstory="Popular musician and creator of Minecraft",
    )


def test_load_character():
    name = "Jimmy Buffett"
    character = Character.load_character(name)
    assert character.name == name
    assert "Margaritaville" in character.backstory


def test_save(character):
    try:
        filename = pytest.DATA_DIR / "hatsune_miku.yaml"
        character.save()
        assert filename.exists()
    finally:
        filename.unlink()


def test_character_sheet(character):
    assert character.name == "Hatsune Miku"
    assert character.age == 16
    assert character.race == "vocaloid"
    assert character.gender == "female"
    assert character.location == "cyberspace"
    assert "Minecraft" in character.backstory
