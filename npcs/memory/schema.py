from dataclasses import asdict, dataclass, field
from typing import Set

from whoosh.fields import ID, KEYWORD, NUMERIC, TEXT, SchemaClass


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
    memory: str  # XXX: this name could be improved; maybe memory.content?
    characters: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    sentiment_polarity: float = 0.0
    sentiment_subjectivity: float = 0.0

    def as_dict(self):
        doc = asdict(self)
        doc["characters"] = ",".join(self.characters)
        doc["entities"] = ",".join(self.entities)
        return doc

    @staticmethod
    def from_dict(doc: dict):
        doc["characters"] = set(doc.get("characters", "").split(","))
        doc["entities"] = set(doc.get("entities", "").split(","))
        return NPCMemory(**doc)
