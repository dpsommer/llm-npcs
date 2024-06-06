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
        entities={"Margaritaville", "Havana"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="It costs 1 silver piece to stay the night at Margaritaville.",
        entities={"Margaritaville"},
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
