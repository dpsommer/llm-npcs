from dataclasses import dataclass, asdict, field
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
    memory: str
    characters: Set[str] = field(default_factory=list)
    entities: Set[str] = field(default_factory=list)
    sentiment_polarity: float = 0.0
    sentiment_subjectivity: float = 0.0

    def as_document(self):
        doc = asdict(self)
        doc['characters'] = ','.join(self.characters)
        doc['entities'] = ','.join(self.entities)
        return doc
