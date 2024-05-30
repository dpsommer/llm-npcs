from dataclasses import dataclass, asdict
from typing import Set

from whoosh.fields import (
    SchemaClass,
    TEXT,
    KEYWORD,
    ID,
    NUMERIC
)


# is it better to store individual memories or whole conversations?
class NPCMemorySchema(SchemaClass):
    npc = ID(stored=True)
    # other characters involved in the memory
    characters = KEYWORD(commas=True)
    memory = TEXT(stored=True)
    entities = KEYWORD(commas=True, scorable=True)
    sentiment_polarity = NUMERIC
    sentiment_subjectivity = NUMERIC


@dataclass
class NPCMemory:
    npc: str
    characters: Set[str]
    memory: str
    entities: Set[str]
    sentiment_polarity: float
    sentiment_subjectivity: float

    def as_document(self):
        doc = asdict(self)
        doc['characters'] = ','.join(self.characters)
        doc['entities'] = ','.join(self.entities)
        return doc
