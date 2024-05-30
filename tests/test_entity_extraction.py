import pytest

from npcs.memory import entity_extraction


@pytest.fixture
def history():
    return """Although he was very busy with his work, Peter had had enough of it.
He and Nancy decided they needed a holiday.
They travelled to Spain because they loved the country very much."""


def test_entity_extraction():
    ents = entity_extraction("This inn's name is the Silver Fox.")
    assert ents == {'the Silver Fox'}


def test_repeated_entity_names():
    ents = entity_extraction("Bob says hello. Jane says 'Hi, Bob'.")
    assert ents == {'Bob', 'Jane'}


def test_coref_resolution(history):
    ents = entity_extraction(
        message="He spent most of the time on the beach, while she explored the town.",
        context=history
    )
    assert ents == {'Peter', 'Nancy', 'Spain'}


def test_plural_coref_resolution(history):
    ents = entity_extraction(
        message="They had a wonderful trip and returned home refreshed.",
        context=history
    )
    assert ents == {'Peter', 'Nancy'}
