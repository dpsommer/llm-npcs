from typing import Any, Dict, List

from langchain.schema import BaseMemory
from pydantic import BaseModel, Field
from spacy import Language
from whoosh.index import FileIndex

from npcs.memory.nlp import default_pipeline, entity_extraction, sentiment_analysis
from npcs.memory.schema import NPCMemory
from npcs.memory.search import default_index, add_memories, search_memories


class IndexedMemory(BaseMemory, BaseModel):
    name: str = ""
    index: FileIndex = Field(default_factory=default_index)
    pipeline: Language = Field(default_factory=default_pipeline)
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
        # FIXME: need to extract text/context?
        polarity, subjectivity = sentiment_analysis(self.pipeline, text)
        return NPCMemory(
            npc=self.name,
            memory=text,
            entities=entity_extraction(self.pipeline, text),
            sentiment_polarity=polarity,
            sentiment_subjectivity=subjectivity,
        )

    def clear(self) -> None:
        self.index.delete_by_term('npc', self.name)
