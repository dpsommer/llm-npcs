from npcs.memory import Conversation, NPCMemory

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
    )
]


# pre-seed the index with some memories
def seed_index():
    from npcs.memory.search import default_index, search_memories
    idx = default_index()
    old_memories = search_memories(idx, f'npc:"{NPC_NAME}"')
    for memory in old_memories:
        print(memory.memory)
    w = idx.writer()
    w.delete_by_term("npc", NPC_NAME)
    for memory in SEED_MEMORIES:
        w.add_document(**memory.as_dict())
    w.commit()


def run():
    seed_index()
    convo = Conversation(name=NPC_NAME)

    while True:
        message = input("Prompt: ")
        print(convo.say(message))

if __name__ == '__main__':
    run()