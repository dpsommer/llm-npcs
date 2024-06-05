from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from npcs.utils import DEFAULT_SEARCH_RESULT_COUNT, INDEX_PATH, NPC_INDEX_NAME

from .schema import NPCMemory

Texts = Tuple[List[str], List[dict]]


class NPCMemoryVectorStore(ABC):
    def __init__(self, seed_memories: List[NPCMemory] | None = None) -> None:
        self._seed_memories = seed_memories
        self._embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self._vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self) -> VectorStore:
        try:
            print("Loading vector store...")
            return self._load_vector_store()
        except Exception:
            # if we can't load, create a new store
            print("No existing vector store, creating...")
            return self._create_vector_store()

    def _memories_to_texts(self, memories: List[NPCMemory]) -> Texts:
        mem_texts = []
        metadatas = []
        for mem in memories:
            mem = mem.as_dict()
            mem_texts.append(mem.pop("memory", ""))
            metadatas.append(mem)
        return (mem_texts, metadatas)

    @abstractmethod
    def _load_vector_store(self) -> VectorStore:
        pass

    @abstractmethod
    def _create_vector_store(self) -> VectorStore:
        pass

    @abstractmethod
    def add_memories(self, memories: List[NPCMemory]) -> None:
        pass

    @abstractmethod
    def search_memories(self, query: str, **kwargs) -> List[NPCMemory]:
        pass

    @abstractmethod
    def _cleanup(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # TODO: log an error
            print(exc_val)
            print(exc_tb)
        self._cleanup()


class FAISSVectorStore(NPCMemoryVectorStore):
    def _load_vector_store(self) -> FAISS:
        return FAISS.load_local(
            folder_path=INDEX_PATH,
            embeddings=self._embeddings,
            index_name=NPC_INDEX_NAME,
        )

    def _create_vector_store(self) -> FAISS:
        mem_texts, metadatas = self._memories_to_texts(self._seed_memories)
        return FAISS.from_texts(
            texts=mem_texts,
            embedding=self._embeddings,
            metadatas=metadatas,
        )

    def add_memories(self, memories: List[NPCMemory]) -> None:
        mem_texts, metadatas = self._memories_to_texts(memories)
        self._vector_store.add_texts(mem_texts, metadatas=metadatas)

    def search_memories(self, query: str, **kwargs) -> List[NPCMemory]:
        # define a custom filter function that, when passed a list, checks if
        # all values from the list are present in the comma-separated field.
        # this is slow, but only runs on fetch_k documents
        def filter_fn(metadata: Dict[str, Any]) -> bool:
            return all(
                (
                    all(v in metadata.get(key) for v in value)
                    if isinstance(value, list)
                    else metadata.get(key) == value
                )
                for key, value in kwargs.items()
            )

        docs = self._vector_store.max_marginal_relevance_search(
            query=query,
            k=DEFAULT_SEARCH_RESULT_COUNT,
            # fetch_k is the number of docs to fetch before filtering
            fetch_k=DEFAULT_SEARCH_RESULT_COUNT * 2,  # double the search space
            filter=filter_fn,
        )
        return [
            NPCMemory.from_dict({**d.metadata, "memory": d.page_content}) for d in docs
        ]

    def _cleanup(self):
        self._vector_store.save_local(
            folder_path=INDEX_PATH,
            index_name=NPC_INDEX_NAME,
        )


class RAMVectorStore(FAISSVectorStore):
    def _load_vector_store(self) -> FAISS:
        return self._create_vector_store()

    def _cleanup(self):
        pass  # no-op
