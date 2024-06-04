from npcs.memory import search_memories


def test_search_by_npc_name(in_memory_index):
    results = search_memories(index=in_memory_index, search_string='npc:"John Doe"')
    assert len(results) == 4


def test_search_by_entity_name(in_memory_index):
    results = search_memories(index=in_memory_index, search_string='entities:"ale"')
    assert len(results) == 2


def test_search_by_multiple_entities(in_memory_index):
    results = search_memories(
        index=in_memory_index, search_string='entities:"ale" entities:"Silver Fox"'
    )
    assert len(results) == 1
