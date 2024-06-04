import os
from typing import List

from whoosh.filedb.filestore import RamStorage
from whoosh.index import FileIndex, create_in, exists_in, open_dir
from whoosh.qparser import QueryParser

from npcs.memory.schema import NPCMemory, NPCMemorySchema
from npcs.utils.constants import ROOT_DIRECTORY

DEFAULT_SEARCH_RESULT_COUNT = 10


# TODO: abstract this interface from the underlying implementation
# so we can swap Whoosh for FAISS or another index
def default_index():
    index_path = os.path.join(ROOT_DIRECTORY, "index")
    return load_index(index_path=index_path, index_name="default")


def ram_index():
    return RamStorage().create_index(NPCMemorySchema)


def load_index(index_path: str, index_name: str) -> FileIndex:
    if exists_in(index_path, indexname=index_name):
        return open_dir(index_path, indexname=index_name)
    os.makedirs(index_path, exist_ok=True)
    return create_in(index_path, NPCMemorySchema, indexname=index_name)


def add_memories(index: FileIndex, data: List[NPCMemory]) -> None:
    # NB: consider using multisegment=True here for initial indexing
    # Not using procs to multithread as it breaks RamStorage (used in tests)
    w = index.writer(limitmb=512)
    for memory in data:
        w.add_document(**memory.as_dict())
    w.commit()


def search_memories(
    index: FileIndex, search_string: str, results_count=DEFAULT_SEARCH_RESULT_COUNT
) -> List[NPCMemory]:
    results = []
    with index.searcher() as searcher:
        parser = QueryParser("memory", index.schema)
        query = parser.parse(search_string.encode())
        search_results = searcher.search(query, limit=results_count)
        for result in search_results:
            results.append(NPCMemory.from_dict(dict(result)))
    return results
