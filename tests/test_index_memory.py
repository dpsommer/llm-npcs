import re

import pytest

from npcs.memory import Conversation, IndexedMemory


@pytest.fixture
def mock_huggingface_auth(monkeypatch, requests_mock):
    requests_mock.get(
        "https://huggingface.co/api/whoami-v2",
        json={"auth": {"accessToken": {"role": "write"}}},
    )
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_mocktoken")


@pytest.fixture
def conversation(in_memory_index, nlp):
    yield Conversation(name="John Doe", index=in_memory_index, nlp=nlp)


def test_memory_loading(in_memory_index):
    memory = IndexedMemory(
        name="John Doe",
        index=in_memory_index,
        llm=None,
        human_prefix="Player",
        ai_prefix="NPC",
    )
    assert "inn's name is the Silver Fox" in memory.load_memory_variables({})["history"]


def test_mock_memory_response(mock_huggingface_auth, conversation, requests_mock):
    matcher = re.compile(r"https://api-inference.huggingface.co/.*")
    # mock huggingface login
    requests_mock.post(
        matcher,
        json=[
            {
                "generated_text": """The Silver Fox is an inn.
It charges 4 copper pieces for ale and 1 silver piece to stay a night."""
            }
        ],
    )
    response = conversation.say("Tell me about the Silver Fox.")
    assert "inn" in response


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
