from npcs.memory import Conversation, FAISSVectorStore, NPCMemory

# const values for testing the LLM
NPC_NAME = "Jimmy Buffett"
SEED_MEMORIES = [
    NPCMemory(
        npc="Jimmy Buffett",
        memory="A margarita costs 4 copper pieces.",
        entities={"margarita"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="The name of this inn is Margaritaville.",
        entities={"Margaritaville"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="The inn is in Havana.",
        entities={"Margaritaville"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="A margarita costs 4 copper pieces at Margaritaville.",
        entities={"margarita", "Margaritaville"},
    ),
]


def run():
    with FAISSVectorStore(SEED_MEMORIES) as idx:
        convo = Conversation(name=NPC_NAME, index=idx)

        while True:
            message = input("> ")
            print(convo.say(message.lstrip("> ")))


if __name__ == "__main__":
    run()
