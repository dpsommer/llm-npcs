from typing import List

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

from npcs.memory import IndexedMemory
from npcs.utils.constants import (
    OPENAI_TEMP,
    OPENAI_FREQUENCY_PENALTY,
    CONVERSATION_SUMMARY_TOKEN_LIMIT,
)

TEMPLATE = """The following is a conversation between a Player and an NPC.
The NPC is wary of the player but not hostile, and will provide the Player with contextual information.
The NPC ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Player: {input}
NPC:"""


def conversation(prompt):
    # player (or other npc?) starts a conversation with an initial prompt
    #
    pass


class NPC:
    def __init__(self, template: str, input_vars: List[str]) -> None:
        self.template = PromptTemplate(
            template=template,
            input_variables=input_vars,
        )
        self.conversation_chain = ConversationChain(
            llm=llm,
            verbose=False,
            prompt=prompt,
            memory=IndexedMemory()
        )


if __name__ == '__main__':
    # FIXME: is langchain even useful here..?
    llm = OpenAI(
        temperature=OPENAI_TEMP,
        frequency_penalty=OPENAI_FREQUENCY_PENALTY,
    )
    prompt = PromptTemplate(template=TEMPLATE, input_variables=["history", "input"])
    convo = ConversationChain(
        llm=llm,
        verbose=False,
        prompt=prompt,
        memory=ConversationSummaryBufferMemory(
            llm=llm,
            human_prefix='Player',
            ai_prefix='NPC',
            max_token_limit=CONVERSATION_SUMMARY_TOKEN_LIMIT,
        ),
    )

    while True:
        inp = input("Prompt: ")
        print(convo.predict(input=inp))
