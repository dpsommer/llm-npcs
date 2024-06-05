from npcs.memory import Conversation, FAISSVectorStore, NPCMemory

# const values for testing the LLM
NPC_NAME = "Jimmy Buffett"
SEED_MEMORIES = [
    NPCMemory(
        npc="Jimmy Buffett",
        memory="A mug of ale costs 4 copper pieces.",
        entities={"ale"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="This inn's name is the Silver Fox.",
        entities={"Silver Fox"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="It costs 1 silver piece a night to stay at the inn.",
        entities={"Silver Fox"},
    ),
    NPCMemory(
        npc="Jimmy Buffett",
        memory="A mug of ale costs 4 copper pieces at the Silver Fox.",
        entities={"ale", "Silver Fox"},
    ),
]


def run():
    with FAISSVectorStore(SEED_MEMORIES) as idx:
        convo = Conversation(name=NPC_NAME, index=idx)

        while True:
            message = input("Prompt: ")
            print(convo.say(message))


if __name__ == "__main__":
    run()
