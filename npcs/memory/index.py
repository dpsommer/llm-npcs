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
    memory_key: str = "history"  #: :meta private:

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        print('loading memories: ', inputs)
        search = f'npc={self.name} entities='
        return {self.memory_key: mem['memory'] for mem in search_memories(self.index, search)}

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        print('saving memories: ', inputs, outputs)
        input_text = '\n'.join(inputs.values())
        output_text = '\n'.join(outputs.values())
        # TODO: sentiment analysis on the output?
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
