def test_search_by_npc_name(in_memory_index):
    results = in_memory_index.search_memories("", npc="John Doe")
    assert len(results) == 4


def test_search_by_entity_name(in_memory_index):
    results = in_memory_index.search_memories("", entities=["ale"])
    assert len(results) == 2


def test_search_by_multiple_entities(in_memory_index):
    results = in_memory_index.search_memories(
        "",
        entities=["ale", "Silver Fox"],
    )
    assert len(results) == 1
