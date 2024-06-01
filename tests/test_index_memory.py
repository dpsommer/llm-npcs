import pytest

from npcs.memory import Conversation


@pytest.fixture
def conversation(in_memory_index):
    yield Conversation(name="Bob", index=in_memory_index)


# Positive case: we want to test that, given memories of a specific entity,
# we can get a context-aware response.
@pytest.mark.functional
def test_memory_with_context(conversation):
    response = conversation.say("Tell me about the Silver Fox.")
    assert "inn" in response


# Negative case: we want to ensure that, if no memories of a specified entity
# are given in context, the response will not hallucinate.
@pytest.mark.functional
def test_memory_without_context():
    pass
