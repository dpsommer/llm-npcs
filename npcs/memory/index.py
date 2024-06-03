from typing import Dict, List

from langchain.schema import BaseMemory
# XXX: we need to pin pydantic to v1 as langchain internals have not migrated to v2
# see https://python.langchain.com/v0.1/docs/guides/development/pydantic_compatibility/
from pydantic.v1 import BaseModel, Field
from whoosh.index import FileIndex

from npcs.memory.nlp import NLPPipeline
from npcs.memory.schema import NPCMemory
from npcs.memory.search import default_index, add_memories, search_memories


class IndexedMemory(BaseMemory, BaseModel):
    name: str = ""
    index: FileIndex = Field(default_factory=default_index)
    nlp: NLPPipeline = Field(default_factory=NLPPipeline)
    memory_key: str = "history"

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        print('loading memories: ', inputs)
        # TODO: search on entities as well. we will need to run the NLP
        # pipeline when saving memories; can we avoid running it both times?
        search = f'npc:"{self.name}"'
        memories = "\n".join([mem.memory for mem in search_memories(self.index, search)])
        return {self.memory_key: memories}

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        print('saving memories: ', inputs, outputs)
        input_text = '\n'.join(inputs.values())
        output_text = '\n'.join(outputs.values())
        add_memories(self.index, [
            self._memory_from_text(input_text),
            self._memory_from_text(output_text),
        ])

    def _memory_from_text(self, text: str) -> NPCMemory:
        doc = self.nlp.run(text)
        return NPCMemory(
            npc=self.name,
            memory=doc.resolved_text,
            entities=doc.entities,
            sentiment_polarity=doc.sentiment_analysis.polarity,
            sentiment_subjectivity=doc.sentiment_analysis.subjectivity,
        )

    def clear(self) -> None:
        self.index.delete_by_term('npc', self.name)
