import os

import pytest
from whoosh.filedb.filestore import RamStorage

from npcs.memory import NPCMemorySchema, NPCMemory, load_index, add_memories, search_memories

TEST_INDEX_DIR = os.path.join(pytest.TEST_DIR_BASEPATH, 'test_index')


@pytest.fixture
def memories():
    return [
        {
            "npc": "John Doe",
            "memory": "A mug of ale costs 4 copper pieces.",
            "entities": ["ale"],
            "sentiment_polarity": 0,
        },
        {
            "npc": "John Doe",
            "memory": "This inn's name is the Silver Fox.",
            "entities": ["Silver Fox"],
            "sentiment_polarity": 0,
        },
        {
            "npc": "John Doe",
            "memory": "It costs 1 silver piece a night to stay at the inn.",
            "entities": ["Silver Fox"],
            "sentiment_polarity": 0,
        },
    ]


# TODO: create a function in npcs.memory to abstract this so we don't rely
# directly on whoosh here (we may switch to FAISS or another index)
@pytest.fixture
def in_memory_index(memories):
    storage = RamStorage()
    index = storage.create_index(NPCMemorySchema)
    add_memories(index=index, data=[NPCMemory(**m) for m in memories])
    yield index
    index.close()


@pytest.mark.functional
@pytest.fixture
def file_based_index(memories, clean_test_subtree):
    test_index = load_index(TEST_INDEX_DIR, 'test')
    add_memories(index=test_index, data=memories)
    yield test_index
    clean_test_subtree(TEST_INDEX_DIR)


def test_search_by_npc_name(in_memory_index):
    results = search_memories(index=in_memory_index, search_string='npc:"John Doe"')
    assert len(results) == 3


def test_search_by_entity_name(in_memory_index):
    results = search_memories(index=in_memory_index, search_string='entities:"ale"')
    assert len(results) == 1
